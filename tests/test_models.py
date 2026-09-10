"""Architecture contracts, gradient flow, head supervision and CNN layout."""

from dataclasses import asdict
import io
import itertools
import json
import unittest

import torch
from torch_geometric.data import Batch, Data

from emulator.models import ModelConfig, build_model
from emulator.models.heads import ExceedanceHead, ForecastOutput
from emulator.models.spatial import SpatialEncoder
from emulator.training import dual_loss_terms


def graph_batch(steps=3, count=2):
    return Batch.from_data_list([Data(x=torch.randn(6, 3), x_hist=torch.randn(6, steps, 3),
        edge_index=torch.tensor([[0, 1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 0]]),
        y=torch.randn(1, 4), grid_H=2, grid_W=3, sample_id=torch.tensor([i])) for i in range(count)])


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def assert_checkpoint_roundtrip(self, config, model, batch, features=None):
        checkpoint = io.BytesIO()
        torch.save(dict(model_config=json.loads(json.dumps(asdict(config))),
                        model_state=model.state_dict()), checkpoint)
        checkpoint.seek(0)
        saved = torch.load(checkpoint, weights_only=False)
        restored = build_model(ModelConfig(**saved['model_config'])).eval()
        restored.load_state_dict(saved['model_state'], strict=True)
        with torch.no_grad():
            expected = model.eval()(batch, features)
            actual = restored(batch, features)
        for before, after in zip(expected, actual):
            if before is None:
                self.assertIsNone(after)
            else:
                torch.testing.assert_close(before, after, rtol=0, atol=0)

    def test_python_constructor_defaults_and_required_hidden_width(self):
        with self.assertRaisesRegex(TypeError, 'hidden_channels'):
            ModelConfig(3, 4)
        config = ModelConfig(3, 4, hidden_channels=8)
        self.assertEqual((config.dropout, config.head_dropout, config.temporal_dropout), (0., 0., .05))
        self.assertIsNone(config.head_hidden)
        self.assertIsNone(config.temporal_hidden)

    def test_metadata_minimum_width_and_checkpoint_roundtrip(self):
        for hidden, meta_hidden in ((8, 16), (128, 128)):
            with self.subTest(hidden=hidden):
                config = ModelConfig(3, 4, hidden_channels=hidden, station_feat_dim=6, head_type='single')
                model = build_model(config)
                self.assertEqual(tuple(model.station_meta[0].weight.shape), (meta_hidden, 6))
                self.assertEqual(tuple(model.station_meta[2].weight.shape), (hidden, meta_hidden))
                batch, features = graph_batch(), torch.randn(6, requires_grad=True)
                output = model(batch, features)
                self.assertEqual(output.prediction.shape, (2, 4))
                output.prediction.square().mean().backward()
                self.assertTrue(torch.isfinite(features.grad).all())
                self.assert_checkpoint_roundtrip(config, model, batch, features)

    def test_custom_head_widths_include_all_dual_branches(self):
        for head, ablation, width in itertools.product(('single', 'dual'), ('none', 'fixed_gate'), (None, 23)):
            if head == 'single' and ablation != 'none':
                continue
            with self.subTest(head=head, ablation=ablation, width=width):
                config = ModelConfig(3, 4, hidden_channels=8, head_type=head, head_hidden=width,
                                     dual_ablation=ablation, peak_threshold_norm=[1.] * 4)
                model = build_model(config)
                branches = ([model.head.regression] if head == 'single' else
                            [model.head.body, model.head.excess] + ([] if ablation == 'fixed_gate' else [model.head.gate]))
                for branch in branches:
                    self.assertEqual(branch[0].out_features, 16 if width is None else width)
                    self.assertEqual(branch[-1].in_features, branch[0].out_features)
                batch = graph_batch()
                output = model(batch)
                self.assertEqual(output.prediction.shape, (2, 4))
                loss = output.prediction.square().mean()
                if head == 'dual':
                    loss += sum(dual_loss_terms(output, batch.y, torch.ones(4)))
                loss.backward()
                self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
                self.assert_checkpoint_roundtrip(config, model, batch)

    def test_all_architectures_are_label_free_and_differentiable(self):
        for encoder, temporal, head, history in itertools.product(
                ("GraphSAGE", "CNN"), ("MLP", "LSTM", "GRU", "Transformer"), ("single", "dual"), (0, 8)):
            with self.subTest(encoder=encoder, temporal=temporal, head=head, history=history):
                config = ModelConfig(3, 4, encoder_type=encoder, temporal_block=temporal, head_type=head,
                    hidden_channels=16, node_read_heads=2, time_read_heads=2, temporal_layers=2,
                    history_steps=history, station_feat_dim=6, peak_threshold_norm=[1.] * 4)
                model = build_model(config)
                batch = graph_batch(history + 1)
                features = torch.randn(6)
                model.eval()
                with torch.no_grad():
                    expected = model(batch, features).prediction
                    batch.y.fill_(100000)
                    torch.testing.assert_close(model(batch, features).prediction, expected, rtol=0, atol=0)
                    del batch.y
                    torch.testing.assert_close(model(batch, features).prediction, expected, rtol=0, atol=0)
                model.train()
                output = model(batch, features)
                loss = output.prediction.square().mean()
                if head == "dual":
                    loss += sum(dual_loss_terms(output, torch.randn(2, 4), torch.ones(4)))
                loss.backward()
                self.assertEqual(output.prediction.shape, (2, 4))
                for name, param in model.named_parameters():
                    self.assertIsNotNone(param.grad, name)
                    self.assertTrue(torch.isfinite(param.grad).all(), name)

    def test_attention_never_requests_weights(self):
        model = build_model(ModelConfig(3, 4, head_type="single", hidden_channels=16,
                                        node_read_heads=2, time_read_heads=2))
        seen = []
        def check_attention(module, args, kwargs):
            seen.append(kwargs["need_weights"])
        for module in model.modules():
            if isinstance(module, torch.nn.MultiheadAttention):
                module.register_forward_pre_hook(check_attention, with_kwargs=True)
        model(graph_batch())
        self.assertEqual(seen, [False] * 6)  # 3 spatial reads + 2 temporal + 1 forecast.

    def test_positive_gains_have_finite_gradients(self):
        for temporal in ("Transformer", "MLP"):
            model = build_model(ModelConfig(3, 4, head_type="single", temporal_block=temporal,
                                hidden_channels=16, node_read_heads=2, time_read_heads=2))
            optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
            for _ in range(4):
                optimizer.zero_grad()
                model(graph_batch()).prediction.square().mean().backward()
                optimizer.step()
            gains = [p for name, p in model.named_parameters() if name.endswith("log_gain")]
            self.assertTrue(gains)
            for gain in gains:
                self.assertTrue(torch.isfinite(gain.grad))
                self.assertGreater(float(gain.exp()), 0)

    def test_dual_constraints_and_whole_batch_excess_loss(self):
        head = ExceedanceHead(8, 0, [1, 2], 0.05)
        result = head(torch.randn(3, 2, 8))
        self.assertTrue(torch.all(result.body <= result.threshold))
        self.assertTrue(torch.all(result.excess > 0))
        torch.testing.assert_close(result.gate_logits.sigmoid(), torch.full((3, 1), .05))
        target = torch.tensor([[2., 0.], [0., 0.], [0., 0.]])
        excess = torch.ones_like(target, requires_grad=True)
        output = ForecastOutput(torch.zeros_like(target), torch.zeros_like(target), excess,
                                torch.zeros(3, 1, requires_grad=True), torch.ones(1, 2))
        _, risk, _ = dual_loss_terms(output, target, torch.ones(2))
        self.assertAlmostEqual(float(risk), 1 / 6)
        risk.backward()
        torch.testing.assert_close(excess.grad[1:], torch.zeros(2, 2), rtol=0, atol=0)
        self.assertAlmostEqual(float(excess.grad[0, 1]), 1 / 3)

    def test_cnn_rectangular_layout_and_invalid_metadata(self):
        batch = graph_batch()
        encoder = SpatialEncoder(3, 5, 1, 0, "CNN")
        expected = encoder.layers[0](batch.x.reshape(2, 2, 3, 3).permute(0, 3, 1, 2))
        expected = torch.nn.functional.leaky_relu(expected, .1).permute(0, 2, 3, 1).reshape(12, 5)
        torch.testing.assert_close(encoder(batch.x, batch.edge_index, encoder.grid_shape(batch)), expected)
        batch.grid_H[1] = 3
        with self.assertRaisesRegex(ValueError, "uniform"):
            encoder.grid_shape(batch)

    def test_baseline_controls(self):
        for encoder, history, width in itertools.product(("GraphSAGE", "CNN"), (0, 2), (None, 23)):
            with self.subTest(encoder=encoder, history=history, temporal_hidden=width):
                config = ModelConfig(3, 4, model="baseline", head_type="single", encoder_type=encoder,
                                     history_steps=history, hidden_channels=16, temporal_hidden=width)
                model = build_model(config)
                expected_width = width if history and width is not None else 16
                self.assertEqual(model.head.in_features, expected_width)
                if history:
                    self.assertEqual(model.rnn.input_size, 16)
                    self.assertEqual(model.rnn.hidden_size, expected_width)
                    self.assertEqual(model.rnn.num_layers, 1)
                else:
                    self.assertIsNone(model.rnn)
                batch = graph_batch(history + 1)
                prediction = model(batch).prediction
                self.assertEqual(prediction.shape, (2, 4))
                prediction.square().mean().backward()
                self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
                self.assert_checkpoint_roundtrip(config, model, batch)
