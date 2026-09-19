"""Focused capacity/pooling contracts, legacy parity and checkpoint plumbing."""

import contextlib
from dataclasses import asdict
import io
import itertools
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

import infer
import train
from emulator.models import ModelConfig, build_model
from emulator.models.heads import ExceedanceHead, ExceedanceHead_Experiment
from emulator.training import ForecastLoss, LossConfig, dual_loss_terms
from test_config_interfaces import dry_commands
from test_models import graph_batch
from test_pipeline import make_fixture


VARIANTS = ("legacy", "c1", "c2", "c2r", "c3")
POOLS = ("mean", "learned")


class ExceedanceExperimentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        self.addCleanup(torch.use_deterministic_algorithms, torch.are_deterministic_algorithms_enabled())
        torch.use_deterministic_algorithms(True)
        torch.manual_seed(42)

    def assert_output_equal(self, expected, actual):
        for before, after in zip(expected, actual):
            if before is None:
                self.assertIsNone(after)
            else:
                torch.testing.assert_close(before, after, rtol=0, atol=0)

    def test_legacy_seed_and_copied_weight_parity_including_dropout(self):
        context = torch.randn(3, 4, 128)
        for width, fixed in itertools.product((None, 37), (False, True)):
            with self.subTest(width=width, fixed=fixed):
                torch.manual_seed(42)
                production = ExceedanceHead(128, .3, [-1., .2, 1., 2.], .17, fixed, width)
                torch.manual_seed(42)
                experiment = ExceedanceHead_Experiment(128, .3, [-1., .2, 1., 2.], .17, fixed, width,
                                                       variant="legacy", pooling="mean")
                self.assertEqual(production.state_dict().keys(), experiment.state_dict().keys())
                for key, value in production.state_dict().items():
                    torch.testing.assert_close(value, experiment.state_dict()[key], rtol=0, atol=0)
                # Nonconstant excess/gate outputs exercise more than initialization.
                with torch.no_grad():
                    production.excess[-1].weight.normal_(std=.1)
                    if production.gate is not None:
                        production.gate[-1].weight.normal_(std=.1)
                experiment.load_state_dict(production.state_dict(), strict=True)
                for training in (False, True):
                    production.train(training)
                    experiment.train(training)
                    torch.manual_seed(42)
                    expected = production(context)
                    torch.manual_seed(42)
                    self.assert_output_equal(expected, experiment(context))

    def test_all_ten_variants_forward_loss_backward_and_checkpoint(self):
        batch = graph_batch(steps=1)
        stats = dict(y_mean=torch.zeros(4), y_std=torch.tensor([.25, .5, 2., 5.]))
        criterion = ForecastLoss(LossConfig(), stats, 1., 1.)
        for variant, pooling in itertools.product(VARIANTS, POOLS):
            with self.subTest(variant=variant, pooling=pooling):
                config = ModelConfig(3, 4, hidden_channels=16, history_steps=0,
                    peak_threshold_norm=[-.4, .2, 1., 2.], peak_prior=.17,
                    exceedance_head_experiment=variant, exceedance_gate_pooling=pooling)
                model = build_model(config).eval()
                self.assertIs(type(model.head), ExceedanceHead_Experiment)
                output = model(batch)
                for value in (output.prediction, output.body, output.excess):
                    self.assertEqual(value.shape, (2, 4))
                for value in (output.gate_logits, output.gate_probability):
                    self.assertEqual(value.shape, (2, 1))
                self.assertEqual(output.threshold.shape, (1, 4))
                self.assertTrue(all(torch.isfinite(value).all() for value in output if value is not None))
                self.assertTrue((output.body <= output.threshold).all())
                self.assertTrue((output.excess >= 0).all())
                self.assertTrue(((output.gate_probability >= 0) & (output.gate_probability <= 1)).all())
                torch.testing.assert_close(output.excess, torch.full_like(output.excess, .1))
                torch.testing.assert_close(output.prediction, output.body + output.gate_probability * output.excess,
                                           rtol=0, atol=0)
                self.assertIsNone(output.severity_phys)
                self.assertIsNone(output.excess_shape)
                prediction = output.prediction * stats['y_std']
                loss = criterion(output, prediction, batch.y)
                expected = (prediction - batch.y).square().mean() + sum(
                    dual_loss_terms(output, batch.y / stats['y_std'], stats['y_std']))
                torch.testing.assert_close(loss, expected)
                loss.backward()
                self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
                checkpoint = io.BytesIO()
                torch.save(dict(model_config=asdict(config), model_state=model.state_dict()), checkpoint)
                checkpoint.seek(0)
                saved = torch.load(checkpoint, weights_only=False)
                restored = build_model(ModelConfig(**saved['model_config'])).eval()
                restored.load_state_dict(saved['model_state'], strict=True)
                self.assert_output_equal(output, restored(batch))

    def test_capacity_dimensions_norm_residual_and_horizon_locality(self):
        context = torch.randn(2, 4, 128) * 3 + 2
        for variant in VARIANTS:
            with self.subTest(variant=variant):
                head = ExceedanceHead_Experiment(128, .3, [1.] * 4, .17, variant=variant).eval()
                layers = list(head.excess_transform or []) + list(head.excess)
                dimensions = [(m.in_features, m.out_features) for m in layers if isinstance(m, nn.Linear)]
                self.assertEqual(dimensions, [(128, 512), (512, 1)] if variant == "c1" else
                                 [(128, 256), (256, 1)] if variant == "legacy" else
                                 [(128, 256), (256, 128), (128, 1)])
                self.assertEqual([type(m) for m in layers],
                    [nn.Linear, nn.LeakyReLU, nn.Dropout, nn.Linear] +
                    ([] if variant in ("legacy", "c1") else [nn.Linear]))
                self.assertEqual((layers[1].negative_slope, layers[2].p), (.1, .3))
                self.assertEqual(head.excess_norm is not None, variant == "c3")
                self.assertEqual(head.body[0].out_features, 256)
                self.assertEqual(head.gate[0].out_features, 256)
                self.assertIsNone(head.event_pool_norm)
                with torch.no_grad():
                    head.excess[-1].weight.normal_(std=.1)
                if head.excess_transform is not None:
                    branch_input = F.layer_norm(context, (128,)) if variant == "c3" else context
                    adapted = head.excess_transform(branch_input)
                    if variant in ("c2r", "c3"):
                        adapted = context + adapted
                    torch.testing.assert_close(head(context).excess, F.softplus(head.excess(adapted).squeeze(-1)))
                changed = context.clone()
                changed[:, 1] += torch.randn_like(changed[:, 1])
                torch.testing.assert_close(head(context).excess[:, [0, 2, 3]],
                                           head(changed).excess[:, [0, 2, 3]], rtol=0, atol=0)

    def test_learned_pool_uses_raw_context_and_independent_norm(self):
        head = ExceedanceHead_Experiment(8, 0, [1.] * 4, .17, variant="c3", pooling="learned")
        context = torch.randn(2, 4, 8) * torch.arange(1., 5.)[None, :, None] + 3
        self.assertIsNot(head.excess_norm, head.event_pool_norm)
        self.assertTrue(set(head.excess_norm.parameters()).isdisjoint(head.event_pool_norm.parameters()))
        scores = F.linear(F.layer_norm(context, (8,)), head.event_score.weight, head.event_score.bias).squeeze(-1)
        weights = scores.softmax(dim=1)
        torch.testing.assert_close(head._pool_event_context(context), (weights[..., None] * context).sum(dim=1))
        with torch.no_grad():
            head.gate[-1].weight.normal_(std=.1)
        head(context).gate_logits.square().mean().backward()
        self.assertGreater(head.event_score.weight.grad.abs().sum().item(), 0.)
        self.assertGreater(head.event_pool_norm.weight.grad.abs().sum().item(), 0.)
        with torch.no_grad():
            head.event_score.weight.zero_()
        torch.testing.assert_close(head._pool_event_context(context), context.mean(dim=1))

    def test_old_config_selection_and_invalid_combinations(self):
        values = dict(in_channels=3, out_channels=4, hidden_channels=8, peak_threshold_norm=[1.] * 4)
        old = build_model(ModelConfig(**values)).eval()
        self.assertIs(type(old.head), ExceedanceHead)
        production = build_model(ModelConfig(**values, exceedance_gate_pooling="learned")).eval()
        self.assertIs(type(production.head), ExceedanceHead)
        production.load_state_dict(old.state_dict(), strict=True)
        batch = graph_batch()
        self.assert_output_equal(old(batch), production(batch))
        for extra in (dict(exceedance_head_experiment="bad"), dict(exceedance_gate_pooling="attention"),
                      dict(exceedance_head_experiment="c1", head_type="single"),
                      dict(exceedance_head_experiment="c1", model="baseline", head_type="single"),
                      dict(exceedance_head_experiment="c1", excess_formulation="severity_shape", target_y_std=[1.] * 4)):
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                build_model(ModelConfig(**values, **extra))
        for variant, pooling in itertools.product(VARIANTS, POOLS):
            head = ExceedanceHead_Experiment(8, 0, [1.] * 4, .17, fixed_gate=True, variant=variant, pooling=pooling)
            output = head(torch.randn(2, 4, 8))
            self.assertIsNone(output.gate_logits)
            self.assertIsNone(head.event_score)
            torch.testing.assert_close(output.gate_probability, torch.full((2, 1), .17))

    def test_cli_and_shell_config_plumbing(self):
        defaults = train.parse_args(['--model', 'perceiver3'])
        self.assertEqual((defaults.exceedance_head_experiment, defaults.exceedance_gate_pooling), (None, 'mean'))
        self.assertEqual((defaults.deterministic, defaults.seed), (0, 42))
        single = train.parse_args(['--model', 'perceiver3', '--head_type', 'single'])
        self.assertEqual((single.loss_mode, single.dual_loss), ('mse', 0))
        with tempfile.TemporaryDirectory() as temporary:
            config = Path(temporary) / 'smoke.sh'
            for variant, pooling in itertools.product(VARIANTS, POOLS):
                with self.subTest(variant=variant, pooling=pooling):
                    config.write_text(f'MODEL=perceiver3\nHEAD_TYPE=dual\nEXCESS_FORMULATION=direct\n'
                        f'EXCEEDANCE_HEAD_EXPERIMENT={variant}\nEXCEEDANCE_GATE_POOLING={pooling}\n'
                        'DETERMINISTIC=1\nSEED=42\n')
                    commands = dry_commands(config)
                    self.assertEqual(len(commands), 1)
                    args = train.parse_args(commands[0])
                    self.assertEqual((args.exceedance_head_experiment, args.exceedance_gate_pooling), (variant, pooling))
                    self.assertEqual((args.deterministic, args.seed, args.dual_loss), (1, 42, 1))
                    for name in asdict(LossConfig()):
                        self.assertEqual(getattr(args, name), getattr(defaults, name))
        for options in (['--exceedance_head_experiment', 'bad'], ['--exceedance_gate_pooling', 'bad'],
                        ['--exceedance_head_experiment', 'c3', '--head_type', 'single'],
                        ['--exceedance_head_experiment', 'c3', '--excess_formulation', 'severity_shape']):
            with self.subTest(options=options), self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
                train.parse_args(['--model', 'perceiver3', *options])

    def test_deterministic_training_checkpoint_and_inference(self):
        with tempfile.TemporaryDirectory() as temporary, contextlib.redirect_stdout(io.StringIO()):
            root = Path(temporary)
            graphs, stations = make_fixture(root)
            states = []
            for run in range(2):
                output = root / f'run{run}'
                train.main(['--root_dir', str(graphs), '--station', 'Battery', '--station_json_dir', str(stations),
                    '--output_dir', str(output), '--model', 'perceiver3', '--head_type', 'dual',
                    '--hidden_channels', '8', '--node_read_heads', '2', '--time_read_heads', '2',
                    '--temporal_block', 'MLP', '--history_hours', '12', '--epochs', '1', '--batch_size', '4',
                    '--num_workers', '0', '--device', 'cpu', '--deterministic', '1', '--seed', '42',
                    '--exceedance_head_experiment', 'c3', '--exceedance_gate_pooling', 'learned'])
                checkpoint = next(output.glob('best_*.pth'))
                saved = torch.load(checkpoint, weights_only=False)
                self.assertEqual(saved['model_config']['exceedance_head_experiment'], 'c3')
                self.assertEqual(saved['model_config']['exceedance_gate_pooling'], 'learned')
                states.append(saved['model_state'])
                inference = root / f'infer{run}'
                infer.main(['--ckpt', str(checkpoint), '--root_dir', str(graphs), '--out_dir', str(inference),
                            '--device', 'cpu', '--num_workers', '0', '--batch_size', '4', '--save_npz'])
                with np.load(next(output.glob('test_preds_*.npz'))) as expected, \
                        np.load(inference / 'predictions.npz') as actual:
                    np.testing.assert_array_equal(expected['y_pred'], actual['y_pred'])
            self.assertEqual(states[0].keys(), states[1].keys())
            for key in states[0]:
                torch.testing.assert_close(states[0][key], states[1][key], rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
