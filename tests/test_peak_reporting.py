"""Peak metrics in per-epoch logs, best-checkpoint summaries and ordinary inference."""

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

import infer
import train
from emulator.data import ForcingGraphStore, fit_loss_thresholds
from emulator.training import EpochResult, ForecastLoss
from emulator.training.metrics import METRIC_NAMES, PEAK_METRIC_NAMES, PEAK_METRIC_STEMS, summarize_windows
import test_inference_reporting as reporting_tests
from test_peak_metrics import make_records, numpy_oracle
from test_pipeline import make_fixture


class PeakReportingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_single_direct_severity_metadata_epoch_jsonl_summary_and_rmse_checkpoint_selection(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            torch.manual_seed(812)
            graphs, stations = make_fixture(root)
            store = ForcingGraphStore(graphs, 'Battery')
            fitted = fit_loss_thresholds(store, store.split()['train'], tail_frac=.4, exceedance_percentile=75)
            truth = np.array([[3., 1., 0., 2.], [2., 3., 1., 0.]], dtype=np.float32)
            prediction = truth / 2
            metrics = summarize_windows(make_records(prediction, truth), event_threshold=1.)
            # Contradictory synthetic VAL rankings explicitly guard the production selection rule.
            first = dict(metrics, rmse_all=2., peak_magnitude_rmse_top5=.1)
            second = dict(metrics, rmse_all=1., peak_magnitude_rmse_top5=.9)
            arrays = dict(y_true=truth, y_pred=prediction, tags=np.array(['a', 'b']))
            for formulation in ('single', 'direct', 'severity_shape'):
                with self.subTest(formulation=formulation):
                    destination = root / formulation
                    epochs = [EpochResult(metrics), EpochResult(first), EpochResult(metrics),
                              EpochResult(second), EpochResult(metrics, arrays)]
                    # Checkpoint orchestration only: no optimizer step or training epoch runs.
                    with patch.object(train, 'ForecastLoss', wraps=ForecastLoss) as criterion, \
                         patch.object(train, 'run_epoch', side_effect=epochs) as evaluated, \
                         patch.object(train.torch.optim.Adam, 'step', side_effect=AssertionError('no training')), \
                         patch.object(train.torch.optim.lr_scheduler.LambdaLR, 'step'), \
                         contextlib.redirect_stdout(io.StringIO()):
                        train.main(['--root_dir', str(graphs), '--station', 'Battery', '--station_json_dir', str(stations),
                            '--output_dir', str(destination), '--model', 'perceiver3',
                            '--head_type', 'single' if formulation == 'single' else 'dual',
                            '--excess_formulation', 'direct' if formulation == 'single' else formulation,
                            '--excess_amp_loss_weight', '0' if formulation == 'single' else '.7',
                            '--shape_loss_weight', '.3' if formulation == 'severity_shape' else '0',
                            '--peak_loss_weight', '.4', '--peak_pool', 'smoothmax', '--peak_pool_beta', '2.5',
                            '--tail_frac', '.4', '--exceedance_percentile', '75', '--epochs', '2', '--device', 'cpu',
                            '--num_workers', '0', '--hidden_channels', '16', '--history_hours', '12'])
                    self.assertEqual(criterion.call_args.kwargs,
                                     dict(event_prior=fitted['event_prior'], event_threshold=fitted['tau_phys']))
                    self.assertEqual(evaluated.call_count, 5)
                    self.assertTrue(all(c.kwargs['event_threshold'] == fitted['tau_phys'] for c in evaluated.call_args_list))
                    self.assertTrue(all(not c.kwargs.get('save_predictions', False) for c in evaluated.call_args_list[:4]))
                    checkpoint = next(destination.glob('best_*.pth'))
                    saved = torch.load(checkpoint, weights_only=False)
                    self.assertEqual(saved['epoch'], 2)
                    self.assertEqual(saved['val'], second)
                    self.assertEqual(saved['loss_thresholds'], fitted)
                    if formulation == 'single':
                        self.assertIsNone(saved['dual_metadata'])
                    config = json.loads(next(destination.glob('config_*.json')).read_text())
                    shell = (destination / 'config_used.sh').read_text()
                    for key, value in dict(peak_loss_weight=.4, peak_pool='smoothmax', peak_pool_beta=2.5).items():
                        self.assertEqual(saved['training_config'][key], value)
                        self.assertEqual(config[key], value)
                        self.assertIn(f'{key.upper()}={value}', shell)
                        self.assertNotIn(key, saved['model_config'])
                    logs = [json.loads(line) for line in next(destination.glob('metrics_*.jsonl')).read_text().splitlines()]
                    self.assertEqual([r['val'] for r in logs], [first, second])
                    summary = json.loads(next(destination.glob('summary_*.json')).read_text())
                    self.assertEqual(summary['val'], second)
                    self.assertEqual(summary['test'], metrics)
                    self.assertEqual(summary['best_epoch'], 2)
                    for key in ('val', 'test'):
                        self.assertEqual(tuple(summary[key]), METRIC_NAMES + PEAK_METRIC_NAMES)
                        json.dumps(summary[key], allow_nan=False)
                    # Normal inference, without branch diagnostics, must report the same peak metric family.
                    output = destination / 'inference'
                    with contextlib.redirect_stdout(io.StringIO()), patch.object(infer, 'run_epoch', wraps=infer.run_epoch) as ordinary:
                        infer.main(['--ckpt', str(checkpoint), '--root_dir', str(graphs), '--out_dir', str(output),
                                    '--device', 'cpu', '--num_workers', '0', '--batch_size', '2', '--save_npz'])
                    report = json.loads((output / 'metrics.json').read_text())
                    self.assertEqual(report['event_threshold_phys'], fitted['tau_phys'])
                    self.assertEqual(report['event_threshold_source'],
                                     'loss_thresholds.tau_phys' if formulation == 'single' else 'dual_metadata.tau_phys')
                    self.assertTrue(all(not call.kwargs['save_dual_diagnostics'] for call in ordinary.call_args_list))
                    self.assertEqual(ordinary.call_count, len(report['years']))
                    with np.load(output / 'predictions.npz') as exported:
                        expected = numpy_oracle(exported['y_pred'].astype(np.float64), exported['y_true'].astype(np.float64))
                        event = exported['y_true'].astype(np.float64).max(axis=1) > fitted['tau_phys']
                        expected_event = numpy_oracle(exported['y_pred'][event].astype(np.float64),
                                                     exported['y_true'][event].astype(np.float64))
                    for key in PEAK_METRIC_STEMS:
                        self.assertEqual(report['metrics'][f'{key}_all'], expected[key])
                        self.assertEqual(report['metrics'][f'{key}_event'], expected_event[key])
                    for year in report['years']:
                        self.assertTrue(all(key in report['results'][year] for key in PEAK_METRIC_NAMES))

    def test_legacy_single_inference_uses_saved_train_threshold_or_reports_unavailable(self):
        fixture = reporting_tests.InferenceReportingTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        saved = torch.load(fixture.ckpt, weights_only=False)
        # A genuinely old checkpoint omits all three new controls and dual metadata.
        for key in ('peak_loss_weight', 'peak_pool', 'peak_pool_beta'):
            saved['training_config'].pop(key)
        before = None
        for threshold in (None, 3., 100.):
            with self.subTest(threshold=threshold):
                if threshold is None:
                    saved.pop('loss_thresholds', None)
                else:
                    saved['loss_thresholds'] = dict(tau_phys=threshold, tail_threshold=-999.)
                torch.save(saved, fixture.ckpt)
                _, directory = fixture.evaluate(f'legacy_{threshold}', ['--scope', 'all'])
                report = json.loads((directory / 'metrics.json').read_text())
                yearly = json.loads(next(directory.glob('metrics_per_year_*.json')).read_text())
                self.assertEqual(yearly['metrics'], report['metrics'])
                self.assertEqual(report['event_threshold_phys'], threshold)
                self.assertEqual(report['event_threshold_source'], None if threshold is None else 'loss_thresholds.tau_phys')
                if threshold in (None, 100.):
                    self.assertTrue(all(report['metrics'][f'{k}_event'] is None for k in PEAK_METRIC_STEMS))
                else:
                    self.assertIsNotNone(report['metrics']['peak_magnitude_rmse_event'])
                unchanged = {k: v for k, v in report['metrics'].items() if not k.endswith('_event')}
                if before is not None:
                    self.assertEqual(unchanged, before)
                before = unchanged
                self.assertFalse(list(directory.glob('*.npz')))
                json.dumps(report['metrics'], allow_nan=False)
