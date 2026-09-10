"""TRAIN event fitting, explicit supervision ablations and diagnostic exports."""

import contextlib
from dataclasses import asdict
import io
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import torch
from torch_geometric.data import Data

import infer
import train
from emulator.common.dual import DUAL_ABLATIONS
from emulator.data import ForcingGraphStore, fit_loss_thresholds
from emulator.inference.dual_diagnostics import summarize_dual
from emulator.models import ForecastOutput, ModelConfig, build_model
from emulator.models.heads import ExceedanceHead
from emulator.training import ForecastLoss, LossConfig, dual_loss_terms
from test_config_interfaces import dry_commands
from test_models import graph_batch
from test_pipeline import make_fixture


class DualExperimentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_train_prevalence_uses_strict_events_ties_and_independent_thresholds(self):
        store = SimpleNamespace(graphs=[Data(y=torch.tensor([float(peak), -1.]))
                                        for peak in (0, 1, 2, 2, 2, 5, 10000)])
        indices = list(range(6))
        fitted = fit_loss_thresholds(store, indices, tail_frac=.5, exceedance_percentile=75)
        self.assertEqual(fitted['tau_phys'], 2.)
        self.assertEqual(fitted['event_count'], 1)
        self.assertEqual(fitted['event_prior'], 1 / 6)
        other_tail = fit_loss_thresholds(store, indices, tail_frac=.05, exceedance_percentile=75)
        self.assertNotEqual(fitted['tail_threshold'], other_tail['tail_threshold'])
        self.assertEqual(fitted['tau_phys'], other_tail['tau_phys'])
        self.assertEqual(fitted['event_prior'], other_tail['event_prior'])
        other_event = fit_loss_thresholds(store, indices, tail_frac=.5, exceedance_percentile=20)
        self.assertEqual(fitted['tail_threshold'], other_event['tail_threshold'])
        self.assertEqual(other_event['tau_phys'], 1.)
        self.assertEqual(other_event['event_prior'], 4 / 6)
        store.graphs[-1].y.fill_(-1e6)
        self.assertEqual(fitted, fit_loss_thresholds(store, indices, tail_frac=.5, exceedance_percentile=75))

    def test_zero_prevalence_is_saved_exactly_and_learned_logits_stay_finite(self):
        store = SimpleNamespace(graphs=[Data(y=torch.ones(2)) for _ in range(4)])
        fitted = fit_loss_thresholds(store, list(range(4)))
        self.assertEqual((fitted['tau_phys'], fitted['event_prior'], fitted['event_count']), (1., 0., 0))
        for prior in (0., 1 / 6, 1.):
            learned = ExceedanceHead(8, 0, [1., 1.], prior)
            result = learned(torch.randn(3, 2, 8))
            self.assertTrue(torch.isfinite(result.gate_logits).all())
            torch.testing.assert_close(result.gate_probability,
                                       torch.full((3, 1), min(max(prior, 1e-6), 1 - 1e-6)))
            fixed = ExceedanceHead(8, 0, [1., 1.], prior, fixed_gate=True)
            first, second = fixed(torch.randn(3, 2, 8)), fixed(torch.randn(3, 2, 8) * 100)
            self.assertIsNone(first.gate_logits)
            self.assertFalse(any('gate' in name for name, _ in fixed.named_parameters()))
            torch.testing.assert_close(first.gate_probability, torch.full((3, 1), prior), rtol=0, atol=0)
            torch.testing.assert_close(first.gate_probability, second.gate_probability, rtol=0, atol=0)

    def test_named_ablations_remove_only_the_requested_supervision(self):
        stats = dict(y_mean=torch.tensor([.3, -.2]), y_std=torch.tensor([2., 3.]))
        norm_target = torch.tensor([[2., 0.], [0., 0.]])
        target = norm_target * stats['y_std'] + stats['y_mean']
        for mode, disabled in DUAL_ABLATIONS.items():
            with self.subTest(mode=mode):
                output = ForecastOutput(torch.zeros(2, 2, requires_grad=True),
                                        torch.zeros(2, 2, requires_grad=True),
                                        torch.ones(2, 2, requires_grad=True),
                                        None if mode == 'fixed_gate' else torch.zeros(2, 1, requires_grad=True),
                                        torch.ones(1, 2))
                config = LossConfig(dual_ablation=mode)
                prediction = output.prediction * stats['y_std'] + stats['y_mean']
                loss = ForecastLoss(config, stats, 3., 2.)(output, prediction, target)
                terms = dual_loss_terms(output, norm_target, stats['y_std'])
                expected = (prediction - target).square().mean()
                for name, term in zip(('body_loss_weight', 'excess_loss_weight', 'gate_loss_weight'), terms):
                    self.assertEqual(getattr(config, name), 0 if name in disabled else 1)
                    if name not in disabled:
                        expected = expected + term
                torch.testing.assert_close(loss, expected)
                loss.backward()
                for name, tensor in zip(('body_loss_weight', 'excess_loss_weight', 'gate_loss_weight'),
                                        (output.body, output.excess, output.gate_logits)):
                    if tensor is None:
                        continue
                    norm = float(tensor.grad.abs().sum()) if tensor.grad is not None else 0.
                    self.assertEqual(norm == 0, name in disabled)
                self.assertGreater(float(output.prediction.grad.abs().sum()), 0)

    def test_fixed_gate_model_and_loss_settings_must_agree(self):
        model = build_model(ModelConfig(3, 4, hidden_channels=16, node_read_heads=2, time_read_heads=2,
                                        peak_threshold_norm=[1.] * 4, peak_prior=.2, dual_ablation='fixed_gate'))
        batch = graph_batch()
        model.eval()
        output = model(batch)
        del batch.y
        torch.testing.assert_close(output.prediction, model(batch).prediction)
        stats = dict(y_mean=torch.zeros(4), y_std=torch.ones(4))
        with self.assertRaisesRegex(ValueError, 'same fixed_gate'):
            ForecastLoss(LossConfig(), stats, 1., 1.)(output, output.prediction, torch.zeros(2, 4))
        restored = build_model(ModelConfig(**asdict(ModelConfig(3, 4, hidden_channels=16,
            node_read_heads=2, time_read_heads=2, peak_threshold_norm=[1.] * 4,
            peak_prior=.2, dual_ablation='fixed_gate')))).eval()
        restored.load_state_dict(model.state_dict(), strict=True)
        torch.testing.assert_close(output.prediction, restored(batch).prediction, rtol=0, atol=0)

    def test_single_and_baseline_reject_ablation_and_invalid_percentiles(self):
        for args in (['--model', 'baseline', '--dual_ablation', 'fixed_gate'],
                     ['--model', 'perceiver3', '--head_type', 'single', '--dual_ablation', 'no_gate_bce'],
                     *(['--exceedance_percentile', value] for value in ('0', '100', 'nan', 'inf'))):
            with self.subTest(args=args), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                train.parse_args(args)

    def test_ablation_profiles_reach_cli_and_keep_the_matched_protocol(self):
        root = Path(__file__).resolve().parents[1]
        modes = set()
        full = train.parse_args(dry_commands(root / 'configs/train_config_NCEP_Battery_Stable_Dual.sh')[0])
        for path in sorted((root / 'configs/dual_ablations').glob('*.sh')):
            args = train.parse_args(dry_commands(path)[0])
            modes.add(args.dual_ablation)
            for key in ('history_hours', 'lr', 'batch_size', 'grad_accum_steps', 'epochs',
                        'encoder_type', 'temporal_block', 'exceedance_percentile', 'tail_frac', 'seed'):
                self.assertEqual(getattr(full, key), getattr(args, key), key)
            for key in DUAL_ABLATIONS[args.dual_ablation]:
                self.assertEqual(getattr(args, key), 0)
        self.assertEqual(modes, set(DUAL_ABLATIONS) - {'none'})

    def test_every_mode_trains_reloads_and_exports_aligned_physical_diagnostics(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            torch.manual_seed(83)
            graphs, stations = make_fixture(root)
            store = ForcingGraphStore(graphs, 'Battery')
            reference = fit_loss_thresholds(store, store.split()['train'], tail_frac=.4, exceedance_percentile=75)
            for mode in DUAL_ABLATIONS:
                with self.subTest(mode=mode):
                    output = root / mode
                    with contextlib.redirect_stdout(io.StringIO()):
                        train.main(['--root_dir', str(graphs), '--station', 'Battery',
                                    '--station_json_dir', str(stations), '--output_dir', str(output),
                                    '--model', 'perceiver3', '--head_type', 'dual', '--dual_ablation', mode,
                                    '--exceedance_percentile', '75', '--tail_frac', '.4', '--loss_mode', 'mse_tail',
                                    '--epochs', '1', '--warmup_epochs', '0', '--device', 'cpu', '--num_workers', '0',
                                    '--batch_size', '4', '--hidden_channels', '16', '--history_hours', '12'])
                    checkpoint_path = next(output.glob('best_*.pth'))
                    checkpoint = torch.load(checkpoint_path, weights_only=False)
                    meta = checkpoint['dual_metadata']
                    self.assertEqual(meta['tau_phys'], reference['tau_phys'])
                    self.assertEqual(meta['event_prior'], reference['event_prior'])
                    self.assertEqual(checkpoint['model_config']['peak_prior'], reference['event_prior'])
                    self.assertEqual(checkpoint['loss_thresholds'], reference)
                    self.assertEqual(checkpoint['model_config']['dual_ablation'], mode)
                    logs = [json.loads(line) for line in next(output.glob('metrics_*.jsonl')).read_text().splitlines()]
                    self.assertEqual(set(logs[0]), {'epoch', 'train', 'val'})
                    for diagnostics in (False, True):
                        destination = output / ('diagnostics' if diagnostics else 'plain')
                        with contextlib.redirect_stdout(io.StringIO()):
                            infer.main(['--ckpt', str(checkpoint_path), '--root_dir', str(graphs),
                                        '--out_dir', str(destination), '--device', 'cpu', '--num_workers', '0',
                                        '--batch_size', '2', '--save_npz',
                                        *(['--dual_diagnostics'] if diagnostics else [])])
                        if not diagnostics:
                            self.assertFalse((destination / 'dual_diagnostics.npz').exists())
                    with np.load(output / 'diagnostics/dual_diagnostics.npz') as arrays:
                        np.testing.assert_allclose(arrays['y_pred'], arrays['body_phys'] + arrays['contribution_phys'],
                                                   rtol=1e-5, atol=1e-6)
                        np.testing.assert_array_equal(arrays['event'], (arrays['y_true'].astype(float) > meta['tau_phys']).any(axis=1))
                        np.testing.assert_array_equal(arrays['tags'], checkpoint['split_tags']['test'])
                        if mode == 'fixed_gate':
                            np.testing.assert_allclose(arrays['gate_probability'], meta['event_prior'])
                        with np.load(output / 'plain/predictions.npz') as plain:
                            np.testing.assert_array_equal(arrays['y_pred'], plain['y_pred'])
                        report = json.loads((output / 'diagnostics/dual_diagnostics.json').read_text())
                        self.assertAlmostEqual(report['overall']['brier'],
                                               float(np.mean((arrays['gate_probability'].astype(float) - arrays['event']) ** 2)))
                    if mode == 'fixed_gate':
                        repo = Path(__file__).resolve().parents[1]
                        for launcher in ('infer.sh', 'infer_multi.sh'):
                            destination = output / launcher
                            settings = dict(ROOT_DIR=graphs, STATION_JSON_DIR=stations, STATION='Battery',
                                            CKPT_PATH=checkpoint_path, PYTHON_BIN=sys.executable,
                                            INFERENCE_RESULTS_ROOT=destination, MODEL='perceiver3', HEAD_TYPE='dual',
                                            ENCODER_TYPE='GraphSAGE', TEMPORAL_BLOCK='Transformer', HISTORY_HOURS=12,
                                            USE_TMUX=0, DO_CONDA=0, TORCH_GPU_PROBE=0, NUM_WORKERS=0,
                                            BATCH_SIZE=2, DUAL_DIAGNOSTICS=1)
                            config_path = output / (launcher + '.config')
                            config_path.write_text(''.join(f'{key}={shlex.quote(str(value))}\n' for key, value in settings.items())
                                                   + 'RUNS=(' + shlex.quote('Fixture|' + str(graphs)) + ')\n')
                            result = subprocess.run(['bash', launcher, str(config_path)], cwd=repo,
                                env=dict(os.environ, CUDA_VISIBLE_DEVICES=''), capture_output=True, text=True)
                            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                            self.assertEqual(len(list(destination.rglob('dual_diagnostics.npz'))), 1)
                            snapshot = next(destination.rglob('infer_config_used.sh')).read_text()
                            self.assertIn('DUAL_DIAGNOSTICS=1', snapshot)


class DualDiagnosticMetricsTests(unittest.TestCase):
    def arrays(self, labels, probabilities):
        truth = np.asarray(labels, dtype=float)[:, None]
        body = np.minimum(truth, .5)
        excess = np.maximum(truth - .5, 0)
        return dict(y_true=truth, y_pred=truth, gate_probability=np.asarray(probabilities),
                    body_phys=body, excess_phys=excess)

    def test_known_brier_pr_ties_and_reliability(self):
        arrays = self.arrays([1, 0, 1, 0], [.9, .8, .8, .1])
        result = summarize_dual(arrays, .5, bins=2)
        self.assertAlmostEqual(result['brier'], (.01 + .64 + .04 + .01) / 4)
        self.assertAlmostEqual(result['average_precision'], 5 / 6)
        self.assertAlmostEqual(result['pr_auc_trapezoid'], 11 / 12)
        self.assertEqual([item['count'] for item in result['reliability']], [1, 3])
        self.assertEqual(result['reliability'][1]['event_rate'], 2 / 3)
        self.assertEqual(result['excess_event_rmse'], 0.)
        order = [2, 3, 0, 1]
        shuffled = summarize_dual({key: value[order] for key, value in arrays.items()}, .5, bins=2)
        self.assertAlmostEqual(result['brier'], shuffled['brier'])
        for key in ('average_precision', 'pr_auc_trapezoid', 'precision', 'recall'):
            self.assertEqual(result[key], shuffled[key])

    def test_no_events_all_events_empty_sets_and_probability_endpoints(self):
        for labels, probabilities in (([0, 0], [0., 1.]), ([1, 1], [0., 1.]), ([], [])):
            result = summarize_dual(self.arrays(labels, probabilities), .5)
            json.dumps(result, allow_nan=False)
            self.assertEqual(sum(item['count'] for item in result['reliability']), len(labels))
            if not any(labels):
                self.assertIsNone(result['average_precision'])
                self.assertIsNone(result['event_window_rmse'])
            else:
                self.assertEqual(result['average_precision'], 1.)
        for probability in (-.1, 1.1, float('nan')):
            with self.assertRaises(ValueError):
                summarize_dual(self.arrays([1], [probability]), .5)
