"""Body-only cap selection, production parity and old/new checkpoint contracts."""

import contextlib
from dataclasses import asdict
import io
import json
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
from emulator.models.heads import ExceedanceHead, ForecastOutput
from test_config_interfaces import dry_commands
from test_models import graph_batch
from test_pipeline import make_fixture


def production_forward(head, context):
    """The pre-option production forward, including its operation/RNG order."""
    raw_body = head.body(context).squeeze(-1).float()
    body = head.threshold - F.softplus(head.threshold - raw_body)
    excess = F.softplus(head.excess(context).squeeze(-1).float())
    logits = head.gate(context.mean(dim=1)).float()
    probability = logits.sigmoid()
    return ForecastOutput(body + probability * excess, body, excess, logits,
                          head.threshold, probability)


class DualBodyCapTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def assert_output_equal(self, expected, actual):
        for before, after in zip(expected, actual):
            if before is None:
                self.assertIsNone(after)
            else:
                torch.testing.assert_close(before, after, rtol=0, atol=0)

    def test_default_and_explicit_soft_preserve_production_outputs_gradients_and_rng(self):
        torch.manual_seed(11)
        context = torch.randn(3, 4, 16)
        for training in (False, True):
            for options in ({}, dict(dual_body_cap="soft")):
                with self.subTest(training=training, options=options):
                    head = ExceedanceHead(16, .3, [-1., .2, 1., 2.], .17, **options).train(training)
                    with torch.no_grad():
                        head.excess[-1].weight.normal_()
                        head.gate[-1].weight.normal_()
                    torch.manual_seed(42)
                    expected = production_forward(head, context)
                    expected.prediction.square().sum().backward()
                    gradients = [p.grad.clone() for p in head.parameters()]
                    rng = torch.get_rng_state()
                    head.zero_grad(set_to_none=True)
                    torch.manual_seed(42)
                    actual = head(context)
                    actual.prediction.square().sum().backward()
                    self.assert_output_equal(expected, actual)
                    self.assertTrue(torch.equal(rng, torch.get_rng_state()))
                    for before, parameter in zip(gradients, head.parameters()):
                        torch.testing.assert_close(before, parameter.grad, rtol=0, atol=0)

    def test_exact_is_identity_below_threshold_and_attains_cap_at_or_above(self):
        head = ExceedanceHead(1, 0, [-1., .25, 2., 3.], .17, dual_body_cap="exact")
        head.body = nn.Identity()
        raw = torch.tensor([[-2., .25, 4., 2.], [0., -.5, 2., 5.]], requires_grad=True)
        output = head(raw.unsqueeze(-1))
        tau = head.threshold.expand_as(raw)
        self.assertTrue((output.body <= tau).all())
        torch.testing.assert_close(output.body[raw >= tau], tau[raw >= tau], rtol=0, atol=0)
        torch.testing.assert_close(output.body[raw < tau], raw[raw < tau], rtol=0, atol=0)
        torch.testing.assert_close(output.prediction, output.body + output.gate_probability * output.excess,
                                   rtol=0, atol=0)
        output.body.sum().backward()
        self.assertTrue(torch.isfinite(raw.grad).all())
        self.assertTrue((raw.grad[raw > tau] == 0).all())
        self.assertTrue((raw.grad[raw < tau] == 1).all())

    def test_exact_changes_only_body_with_identical_parameters_and_initialization(self):
        values = dict(in_channels=3, out_channels=4, hidden_channels=128, history_steps=0,
                      peak_threshold_norm=[-1., .2, 1., 2.], peak_prior=.17, head_dropout=.3)
        models, rngs = [], []
        for cap in ("soft", "exact"):
            torch.manual_seed(42)
            models.append(build_model(ModelConfig(**values, dual_body_cap=cap)))
            rngs.append(torch.get_rng_state())
        soft, exact = models
        self.assertTrue(torch.equal(*rngs))
        self.assertEqual(sum(p.numel() for p in soft.parameters()), sum(p.numel() for p in exact.parameters()))
        self.assertEqual(soft.state_dict().keys(), exact.state_dict().keys())
        for key, value in soft.state_dict().items():
            torch.testing.assert_close(value, exact.state_dict()[key], rtol=0, atol=0)
        with torch.no_grad():
            soft.head.excess[-1].weight.normal_(std=.1)
            soft.head.gate[-1].weight.normal_(std=.1)
        exact.load_state_dict(soft.state_dict(), strict=True)
        context = torch.randn(3, 4, 128)
        for training in (False, True):
            soft.train(training)
            exact.train(training)
            torch.manual_seed(73)
            a = soft.head(context)
            torch.manual_seed(73)
            b = exact.head(context)
            for key in ("excess", "gate_logits", "gate_probability", "threshold"):
                torch.testing.assert_close(getattr(a, key), getattr(b, key), rtol=0, atol=0)
            self.assertFalse(torch.equal(a.body, b.body))
            torch.testing.assert_close(b.prediction, b.body + b.gate_probability * b.excess, rtol=0, atol=0)

    def test_checkpoint_roundtrip_and_old_config_defaults_to_soft(self):
        batch = graph_batch()
        for cap in ("soft", "exact"):
            config = ModelConfig(3, 4, 8, peak_threshold_norm=[-.4, .2, 1., 2.], dual_body_cap=cap)
            model = build_model(config).eval()
            checkpoint = io.BytesIO()
            torch.save(dict(model_config=asdict(config), model_state=model.state_dict()), checkpoint)
            checkpoint.seek(0)
            saved = torch.load(checkpoint, weights_only=False)
            self.assertEqual(saved['model_config']['dual_body_cap'], cap)
            restored = build_model(ModelConfig(**saved['model_config'])).eval()
            restored.load_state_dict(saved['model_state'], strict=True)
            self.assert_output_equal(model(batch), restored(batch))
            if cap == "soft":
                saved['model_config'].pop('dual_body_cap')
                old = build_model(ModelConfig(**saved['model_config'])).eval()
                old.load_state_dict(saved['model_state'], strict=True)
                self.assertEqual(old.head.dual_body_cap, "soft")
                self.assert_output_equal(model(batch), old(batch))

    def test_cli_and_old_shell_configs_default_safely_to_soft(self):
        self.assertEqual(train.parse_args(['--model', 'perceiver3']).dual_body_cap, 'soft')
        with tempfile.TemporaryDirectory() as temporary:
            config = Path(temporary) / 'config.sh'
            for cap in (None, 'soft', 'exact'):
                config.write_text('MODEL=perceiver3\nHEAD_TYPE=dual\n' +
                                  (f'DUAL_BODY_CAP={cap}\n' if cap else ''))
                commands = dry_commands(config)
                self.assertEqual(len(commands), 1)
                self.assertEqual(train.parse_args(commands[0]).dual_body_cap, cap or 'soft')

    def test_invalid_or_unsupported_modes_fail_instead_of_being_ignored(self):
        values = dict(in_channels=3, out_channels=4, hidden_channels=8, peak_threshold_norm=[1.] * 4)
        for extra in (dict(dual_body_cap='bad'), dict(dual_body_cap='exact', head_type='single'),
                      dict(dual_body_cap='exact', exceedance_head_experiment='legacy'),
                      dict(dual_body_cap='exact', excess_formulation='severity_shape')):
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                build_model(ModelConfig(**values, **extra))
        for options in (['--dual_body_cap', 'bad'], ['--dual_body_cap', 'exact', '--head_type', 'single'],
                        ['--dual_body_cap', 'exact', '--exceedance_head_experiment', 'legacy'],
                        ['--dual_body_cap', 'exact', '--excess_formulation', 'severity_shape']):
            with self.subTest(options=options), self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
                train.parse_args(['--model', 'perceiver3', *options])

    def test_cpu_fixture_training_metadata_and_inference_roundtrip(self):
        with tempfile.TemporaryDirectory() as temporary, contextlib.redirect_stdout(io.StringIO()):
            root = Path(temporary)
            graphs, stations = make_fixture(root)
            for cap in ('soft', 'exact'):
                output = root / cap
                train.main(['--root_dir', str(graphs), '--station', 'Battery', '--station_json_dir', str(stations),
                    '--output_dir', str(output), '--model', 'perceiver3', '--head_type', 'dual',
                    '--hidden_channels', '8', '--node_read_heads', '2', '--time_read_heads', '2',
                    '--temporal_block', 'MLP', '--history_hours', '12', '--epochs', '1', '--batch_size', '4',
                    '--num_workers', '0', '--device', 'cpu', '--dual_body_cap', cap])
                path = next(output.glob('best_*.pth'))
                saved = torch.load(path, weights_only=False)
                for key in ('model_config', 'training_config', 'dual_metadata'):
                    self.assertEqual(saved[key]['dual_body_cap'], cap)
                self.assertEqual(json.loads(next(output.glob('config_*.json')).read_text())['dual_body_cap'], cap)
                self.assertIn(f'DUAL_BODY_CAP={cap}', (output / 'config_used.sh').read_text())
                inferred = root / f'infer_{cap}'
                infer.main(['--ckpt', str(path), '--root_dir', str(graphs), '--out_dir', str(inferred),
                            '--device', 'cpu', '--num_workers', '0', '--batch_size', '4', '--save_npz'])
                with np.load(next(output.glob('test_preds_*.npz'))) as expected, \
                        np.load(inferred / 'predictions.npz') as actual:
                    np.testing.assert_array_equal(expected['y_pred'], actual['y_pred'])


if __name__ == '__main__':
    unittest.main()
