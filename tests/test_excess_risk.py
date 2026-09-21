"""Opt-in aggregation of the existing direct-dual excess regression."""

import contextlib
from dataclasses import fields
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
import torch.nn.functional as F

import infer
import train
from emulator.common.dual import dual_excess_target
from emulator.data import ForcingGraphStore, fit_loss_thresholds
from emulator.models import ForecastOutput
from emulator.training import EpochResult, ForecastLoss, LossConfig, dual_loss_terms
from emulator.training.losses import relative_excess_weights
from test_config_interfaces import dry_commands
from test_pipeline import make_fixture


def fixture(dtype=torch.float32):
    # The first window is exactly at threshold: strict event semantics exclude it.
    target = torch.tensor([[1., 1., 1.], [1., 3., 2.], [2., 1., 1.]], dtype=dtype)
    std = torch.tensor([.5, 2., 4.], dtype=dtype)
    body = torch.full_like(target, .2, requires_grad=True)
    excess = torch.full_like(target, .7, requires_grad=True)
    logits = torch.tensor([[0.], [.2], [-.3]], dtype=dtype, requires_grad=True)
    output = ForecastOutput(body + logits.sigmoid() * excess, body=body, excess=excess,
                            gate_logits=logits, threshold=torch.ones(1, 3, dtype=dtype))
    return output, target, std


class ExcessRiskTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_default_exact_production_value_gradients_and_rng(self):
        for dtype in (torch.float32, torch.float64):
            output, target, std = fixture(dtype)
            event, excess_target = dual_excess_target(target, output.threshold)
            expected = (((output.excess - excess_target) * std).square() * event).mean()
            before = torch.get_rng_state().clone()
            with patch('emulator.training.losses.relative_excess_weights', side_effect=AssertionError('default allocated weights')):
                actual = dual_loss_terms(output, target, std)[1]
            self.assertTrue(torch.equal(before, torch.get_rng_state()))
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            actual_grad = torch.autograd.grad(actual, output.excess, retain_graph=True)[0]
            expected_grad = torch.autograd.grad(expected, output.excess)[0]
            torch.testing.assert_close(actual_grad, expected_grad, rtol=0, atol=0)
            stats = dict(y_mean=torch.zeros_like(std), y_std=std)
            prediction, physical_target = output.prediction * std, target * std
            body = ((output.body - torch.minimum(target, output.threshold)) * std).square().mean()
            gate = F.binary_cross_entropy_with_logits(output.gate_logits, event.float()) * std.square().mean()
            historical = (prediction - physical_target).square().mean() + (body + expected + gate)
            loss = ForecastLoss(LossConfig(), stats, 1., 1.)(output, prediction, physical_target)
            torch.testing.assert_close(loss, historical, rtol=0, atol=0)

    def test_fixed_train_prior_is_independent_of_batch_event_fraction(self):
        output, target, std = fixture()
        prior = .17  # Deliberately unlike either batch's prevalence or nominal .05.
        for indices in ([0, 1, 2], [0, 0, 0, 1], [1, 2], [0, 0]):
            batch = output._replace(body=output.body[indices], excess=output.excess[indices],
                                    gate_logits=output.gate_logits[indices])
            old = dual_loss_terms(batch, target[indices], std)[1]
            normalized = dual_loss_terms(batch, target[indices], std,
                excess_event_normalization="train_prior", event_prior=prior)[1]
            torch.testing.assert_close(normalized, old / prior, rtol=0, atol=0)

    def test_invalid_prior_rejected_only_when_requested(self):
        output, target, std = fixture()
        stats = dict(y_mean=torch.zeros_like(std), y_std=std)
        expected = dual_loss_terms(output, target, std)[1]
        for prior in (None, 0., -.1, float('nan'), float('inf'), -float('inf')):
            with self.subTest(prior=prior):
                with self.assertRaisesRegex(ValueError, 'positive TRAIN event_prior'):
                    dual_loss_terms(output, target, std, excess_event_normalization='train_prior', event_prior=prior)
                with self.assertRaisesRegex(ValueError, 'positive TRAIN event_prior'):
                    ForecastLoss(LossConfig(excess_event_normalization='train_prior'), stats, 1., 1., event_prior=prior)
                ForecastLoss(LossConfig(), stats, 1., 1., event_prior=prior)
                actual = dual_loss_terms(output, target, std, event_prior=prior)[1]
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_normalization_only_changes_excess_supervision(self):
        output, target, std = fixture()
        old = dual_loss_terms(output, target, std)
        new = dual_loss_terms(output, target, std, excess_event_normalization='train_prior', event_prior=.2)
        for index in (0, 2):
            torch.testing.assert_close(new[index], old[index], rtol=0, atol=0)
        gradients = torch.autograd.grad(new[1], (output.body, output.excess, output.gate_logits),
                                        allow_unused=True, retain_graph=True)
        self.assertIsNone(gradients[0])
        self.assertIsNone(gradients[2])
        self.assertTrue(torch.equal(gradients[1][0], torch.zeros_like(std)))
        stats = dict(y_mean=torch.zeros_like(std), y_std=std)
        prediction, physical_target = output.prediction * std, target * std
        for mode, terms in (('none', old), ('train_prior', new)):
            loss = ForecastLoss(LossConfig(excess_event_normalization=mode, excess_loss_weight=2., gate_loss_weight=.5),
                stats, 1., 1., event_prior=.2)(output, prediction, physical_target)
            expected = (prediction - physical_target).square().mean() + (terms[0] + 2 * terms[1] + .5 * terms[2])
            torch.testing.assert_close(loss, expected, rtol=0, atol=0)

    def test_normalization_cli_defaults_validation_and_shell_forwarding(self):
        self.assertEqual(train.parse_args([]).excess_event_normalization, 'none')
        self.assertEqual(LossConfig().excess_event_normalization, 'none')
        for mode in ('none', 'train_prior'):
            args = train.parse_args(['--model', 'perceiver3', '--excess_event_normalization', mode])
            self.assertEqual(args.excess_event_normalization, mode)
            with tempfile.TemporaryDirectory() as temporary:
                config = Path(temporary) / 'config.sh'
                config.write_text(f'MODEL=perceiver3\nEXCESS_EVENT_NORMALIZATION={mode}\n')
                self.assertEqual(train.parse_args(dry_commands(config)[0]).excess_event_normalization, mode)
        for options in (['--excess_event_normalization', 'batch'],
                        ['--excess_event_normalization', 'train_prior'],
                        ['--model', 'perceiver3', '--excess_formulation', 'severity_shape',
                         '--excess_event_normalization', 'train_prior']):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                train.parse_args(options)
        output, target, std = fixture()
        with self.assertRaisesRegex(ValueError, 'excess_event_normalization'):
            dual_loss_terms(output, target, std, excess_event_normalization='batch')

    def test_fitted_prior_and_checkpoint_provenance_and_old_loading(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            torch.manual_seed(83)
            graphs, stations = make_fixture(root)
            store = ForcingGraphStore(graphs, 'Battery')
            fitted = fit_loss_thresholds(store, store.split()['train'], exceedance_percentile=75)
            metrics = dict(rmse_all=1., mae_all=1., rmse_peak5=1., mae_peak5=1., peak_magnitude_rmse_top5=1.)
            # Exercise setup and persistence, with all epochs mocked: no training.
            epochs = [EpochResult(metrics), EpochResult(dict(metrics, rmse_all=.5, rmse_peak5=2.)),
                      EpochResult(metrics), EpochResult(dict(metrics, rmse_all=.7, rmse_peak5=.1)),
                      EpochResult(metrics), EpochResult(metrics, dict(y_true=np.zeros((0, 4)),
                          y_pred=np.zeros((0, 4)), tags=np.array([], dtype=str)))]
            settings = dict(excess_event_normalization='train_prior', excess_horizon_weighting='relative_magnitude',
                            excess_magnitude_alpha=1.0)
            options = [part for key, value in settings.items() for part in ('--' + key, str(value))]
            destination = root / 'run'
            with patch.object(train, 'ForecastLoss', wraps=ForecastLoss) as constructor, \
                 patch.object(train, 'run_epoch', side_effect=epochs), \
                 patch.object(train.torch.optim.lr_scheduler.LambdaLR, 'step'), \
                 contextlib.redirect_stdout(io.StringIO()):
                train.main(['--root_dir', str(graphs), '--station', 'Battery', '--station_json_dir', str(stations),
                    '--output_dir', str(destination), '--model', 'perceiver3', '--epochs', '2', '--device', 'cpu',
                    '--num_workers', '0', '--hidden_channels', '16', '--exceedance_percentile', '75', *options])
            self.assertEqual(constructor.call_args.kwargs['event_prior'], fitted['event_prior'])
            checkpoint = torch.load(next(destination.glob('best_*.pth')), weights_only=False)
            self.assertEqual(checkpoint['loss_thresholds'], fitted)
            self.assertEqual(checkpoint['epoch'], 1)  # Overall VAL wins despite worse peak RMSE.
            snapshot = json.loads(next(destination.glob('config_*.json')).read_text())
            summary = json.loads(next(destination.glob('summary_*.json')).read_text())
            shell_snapshot = (destination / 'config_used.sh').read_text()
            for key, value in settings.items():
                self.assertEqual(checkpoint['training_config'][key], value)
                self.assertEqual(checkpoint['dual_metadata'][key], value)
                self.assertEqual(summary['dual_metadata'][key], value)
                self.assertEqual(snapshot[key], value)
                self.assertIn(f'{key.upper()}={value}', shell_snapshot)
                self.assertNotIn(key, checkpoint['model_config'])
            for legacy in (False, True):
                if legacy:
                    for key in settings:
                        checkpoint['training_config'].pop(key)
                        checkpoint['dual_metadata'].pop(key)
                    restored = LossConfig(**{f.name: checkpoint['training_config'][f.name]
                        for f in fields(LossConfig) if f.name in checkpoint['training_config']})
                    self.assertEqual(restored.excess_event_normalization, 'none')
                    self.assertEqual(restored.excess_horizon_weighting, 'uniform')
                    self.assertEqual(restored.excess_magnitude_alpha, 1.0)
                path = root / f'checkpoint_{legacy}.pth'
                torch.save(checkpoint, path)
                with contextlib.redirect_stdout(io.StringIO()):
                    infer.main(['--ckpt', str(path), '--root_dir', str(graphs), '--device', 'cpu',
                        '--num_workers', '0', '--out_dir', str(root / f'infer_{legacy}'), '--save_npz'])
            with np.load(root / 'infer_False/predictions.npz') as current, np.load(root / 'infer_True/predictions.npz') as old:
                np.testing.assert_array_equal(current['y_pred'], old['y_pred'])

    def test_relative_weights_use_physical_ordering_and_mean_one(self):
        # Normalized ordering [0, 4, 1] differs from physical meters [0, 2, 4].
        target = torch.tensor([[0., 4., 1.], [0., 0., 0.]], dtype=torch.float64)
        std = torch.tensor([1., .5, 4.], dtype=torch.float64)
        weights = relative_excess_weights(target, std)
        raw = torch.tensor([[1., 1.5, 2.], [1., 1., 1.]], dtype=torch.float64)
        torch.testing.assert_close(weights, raw / raw.mean(dim=1, keepdim=True), rtol=0, atol=0)
        self.assertTrue((weights >= 0).all())
        torch.testing.assert_close(weights.mean(dim=1), torch.ones(2, dtype=weights.dtype))
        self.assertGreater(weights[0, 2], weights[0, 1])
        self.assertGreater(weights[0, 1], weights[0, 0])
        torch.testing.assert_close(relative_excess_weights(target, std * 10), weights)

    def test_alpha_zero_is_exactly_uniform(self):
        output, target, std = fixture()
        for mode in ('none', 'train_prior'):
            options = dict(excess_event_normalization=mode, event_prior=.17)
            old = dual_loss_terms(output, target, std, **options)[1]
            new = dual_loss_terms(output, target, std, excess_horizon_weighting='relative_magnitude',
                                  excess_magnitude_alpha=0., **options)[1]
            torch.testing.assert_close(new, old, rtol=0, atol=0)
            torch.testing.assert_close(torch.autograd.grad(new, output.excess, retain_graph=True)[0],
                                       torch.autograd.grad(old, output.excess, retain_graph=True)[0], rtol=0, atol=0)

    def test_all_four_formulas_and_gradient_isolation(self):
        output, target, std = fixture(torch.float64)
        old = dual_loss_terms(output, target, std)
        event, excess_target = dual_excess_target(target, output.threshold)
        weights = relative_excess_weights(excess_target, std)
        squared = ((output.excess - excess_target) * std).square()
        old_gradient = torch.autograd.grad(old[1], output.excess, retain_graph=True)[0]
        for normalization in ('none', 'train_prior'):
            for weighting in ('uniform', 'relative_magnitude'):
                with self.subTest(normalization=normalization, weighting=weighting):
                    options = dict(excess_event_normalization=normalization, excess_horizon_weighting=weighting)
                    terms = dual_loss_terms(output, target, std, event_prior=.17, **options)
                    scale = .17 if normalization == 'train_prior' else 1.
                    horizon_weights = weights if weighting == 'relative_magnitude' else torch.ones_like(weights)
                    expected = (event * horizon_weights * squared).mean() / scale
                    torch.testing.assert_close(terms[1], expected)
                    for index in (0, 2):
                        torch.testing.assert_close(terms[index], old[index], rtol=0, atol=0)
                    body_grad, excess_grad, gate_grad = torch.autograd.grad(terms[1],
                        (output.body, output.excess, output.gate_logits), allow_unused=True, retain_graph=True)
                    self.assertIsNone(body_grad)
                    self.assertIsNone(gate_grad)
                    torch.testing.assert_close(excess_grad, old_gradient * horizon_weights / scale)
                    torch.testing.assert_close(excess_grad[0], torch.zeros_like(std), rtol=0, atol=0)
                    stats = dict(y_mean=torch.zeros_like(std), y_std=std)
                    prediction, truth = output.prediction * std, target * std
                    loss = ForecastLoss(LossConfig(**options), stats, 1., 1., event_prior=.17)(output, prediction, truth)
                    # Optional terms stay off; the final physical MSE is identical.
                    reference = (prediction - truth).square().mean() + sum(terms)
                    torch.testing.assert_close(loss, reference)
        e2 = dual_loss_terms(output, target, std, excess_horizon_weighting='relative_magnitude')[1]
        e3 = dual_loss_terms(output, target, std, excess_horizon_weighting='relative_magnitude',
                            excess_event_normalization='train_prior', event_prior=.17)[1]
        torch.testing.assert_close(e3, e2 / .17, rtol=0, atol=0)

    def test_event_free_batch_has_finite_zero_weighted_loss_and_gradient(self):
        output, target, std = fixture()
        for mode in ('none', 'train_prior'):
            loss = dual_loss_terms(output, torch.ones_like(target), std,
                excess_horizon_weighting='relative_magnitude', excess_event_normalization=mode, event_prior=.17)[1]
            self.assertEqual(loss.item(), 0.)
            gradient = torch.autograd.grad(loss, output.excess, retain_graph=True)[0]
            torch.testing.assert_close(gradient, torch.zeros_like(gradient), rtol=0, atol=0)

    def test_cpu_bfloat16_weights_and_loss_are_finite(self):
        for dtype in (torch.float16, torch.bfloat16):
            target = torch.tensor([[0., 4., 1.], [0., 0., 0.]], dtype=dtype)
            std = torch.tensor([1., .5, 4.], dtype=dtype)
            weights = relative_excess_weights(target, std)
            self.assertEqual(weights.dtype, torch.float32)
            self.assertTrue(torch.isfinite(weights).all())
            torch.testing.assert_close(weights.mean(dim=1), torch.ones(2))
        output, target, std = fixture()
        with torch.autocast('cpu', dtype=torch.bfloat16):
            loss = dual_loss_terms(output, target, std, excess_horizon_weighting='relative_magnitude',
                                  excess_event_normalization='train_prior', event_prior=.17)[1]
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(torch.autograd.grad(loss, output.excess)[0]).all())

    def test_weighting_cli_api_validation_and_shell_forwarding(self):
        defaults = train.parse_args([])
        self.assertEqual((defaults.excess_horizon_weighting, defaults.excess_magnitude_alpha), ('uniform', 1.0))
        for value in (-1., float('nan'), float('inf'), -float('inf')):
            for weighting in ('uniform', 'relative_magnitude'):
                with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    train.parse_args(['--model', 'perceiver3', '--excess_horizon_weighting', weighting,
                                      f'--excess_magnitude_alpha={value}'])
                with self.assertRaisesRegex(ValueError, 'finite and nonnegative'):
                    ForecastLoss(LossConfig(excess_horizon_weighting=weighting, excess_magnitude_alpha=value),
                                 dict(y_mean=torch.zeros(3), y_std=torch.ones(3)), 1., 1.)
        for weighting in ('uniform', 'relative_magnitude'):
            with tempfile.TemporaryDirectory() as temporary:
                path = Path(temporary) / 'config.sh'
                path.write_text(f'MODEL=perceiver3\nEXCESS_HORIZON_WEIGHTING={weighting}\nEXCESS_MAGNITUDE_ALPHA=0.5\n')
                args = train.parse_args(dry_commands(path)[0])
                self.assertEqual((args.excess_horizon_weighting, args.excess_magnitude_alpha), (weighting, .5))
        for options in (['--excess_horizon_weighting', 'peak'],
                        ['--excess_horizon_weighting', 'relative_magnitude'],
                        ['--model', 'perceiver3', '--excess_formulation', 'severity_shape',
                         '--excess_horizon_weighting', 'relative_magnitude']):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                train.parse_args(options)


if __name__ == '__main__':
    unittest.main()
