"""Independent NumPy peak metrics, legacy exactness and lightweight DDP evaluation."""

from datetime import timedelta
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler

from emulator.data import build_loader
from emulator.training import run_epoch
from emulator.training import engine
from emulator.training.metrics import (METRIC_NAMES, PEAK_METRIC_NAMES, PEAK_METRIC_STEMS, format_metrics,
    physical_peak_columns, summarize_windows, unique_rows_and_top5_indices)
import test_training as training_tests


def legacy_summary(records, validation=False):
    """Frozen four-metric implementation from post-#3 revision 50ec23f."""
    _, first = np.unique(records[:, 0], return_index=True)
    rows = records[first]
    all_rows = records if validation else rows
    peak_rows = rows[:, 1:].astype(np.float32) if validation else rows[:, 1:]
    top = np.argsort(peak_rows[:, 0], kind='quicksort' if validation else 'stable')[-max(1, math.ceil(.05 * len(rows))):]
    return dict(zip(METRIC_NAMES, (float(np.sqrt(all_rows[:, 2].mean())), float(all_rows[:, 3].mean()),
        float(np.sqrt(peak_rows[top, 1].mean())), float(peak_rows[top, 2].mean())))), rows[top, 0]


def make_records(prediction, truth):
    prediction, truth = np.asarray(prediction), np.asarray(truth)
    error = prediction.astype(np.float64) - truth.astype(np.float64)
    peaks = physical_peak_columns(torch.from_numpy(prediction), torch.from_numpy(truth)).numpy()
    return np.column_stack((np.arange(len(truth)), peaks[:, 0], (error ** 2).mean(1), abs(error).mean(1), peaks[:, 1:]))


def numpy_oracle(prediction, truth):
    """Derive every metric directly from arrays without production selectors or reductions."""
    if not len(truth):
        return dict.fromkeys(PEAK_METRIC_STEMS)
    true_index, pred_index = np.argmax(truth, 1), np.argmax(prediction, 1)
    error = prediction[np.arange(len(truth)), true_index] - truth[np.arange(len(truth)), true_index]
    return dict(true_peak_rmse=float(np.sqrt(np.average(error ** 2))),
        true_peak_mae=float(np.average(abs(error))), true_peak_bias=float(np.average(error)),
        true_peak_underprediction_fraction=float(np.count_nonzero(error < 0) / len(truth)),
        peak_timing_mae_steps=float(np.average(abs(pred_index - true_index))))


def metric_ddp_worker(rank, rendezvous, destination):
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method=Path(rendezvous).as_uri(), rank=rank,
                            world_size=2, timeout=timedelta(seconds=30))
    try:
        fixture = training_tests.TrainingTests()
        fixture.setUp()
        sampler = DistributedSampler(fixture.data, num_replicas=2, rank=rank, shuffle=False, drop_last=False)
        loader = build_loader(fixture.data, sampler, 7, 0, False, False, 0, 'fork')
        model = training_tests.CountingModel()
        result = run_epoch(model, loader, torch.device('cpu'), fixture.stats, distributed=True, event_threshold=30.)
        assert model.calls == 3 and result.predictions is None
        if rank == 0:
            Path(destination).write_text(json.dumps(result.metrics, allow_nan=False))
    finally:
        dist.destroy_process_group()


class PeakMetricTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_numpy_reproduces_every_metric_for_all_three_populations(self):
        truth = np.array([[3., 3., 0.], [1., 4., 2.], [6., 0., 1.], [1., 2., 3.],
                          [1., 0., 2.], [2., 1., 0.], [-3., -1., -2.]])
        pred = np.array([[1., 4., 4.], [2., 3., 1.], [4., 2., 1.], [1., 3., 2.],
                         [1., 3., 2.], [3., 1., 0.], [-4., -1., -1.]])
        records = make_records(pred, truth)
        for validation in (False, True):
            metrics = summarize_windows(records, validation=validation, event_threshold=3.)
            self.assertEqual(tuple(metrics), METRIC_NAMES + PEAK_METRIC_NAMES)
            old, selected_ids = legacy_summary(records[:, :4], validation)
            self.assertEqual({k: metrics[k] for k in METRIC_NAMES}, old)
            populations = dict(all=np.arange(len(truth)), top5=selected_ids.astype(int), event=np.flatnonzero(truth.max(1) > 3.))
            for name, indices in populations.items():
                expected = numpy_oracle(pred[indices], truth[indices])
                for key, value in expected.items():
                    self.assertEqual(metrics[f'{key}_{name}'], value, (name, key))
            json.dumps(metrics, allow_nan=False)

    def test_ties_choose_first_occurrence_and_timing_is_in_forecast_steps(self):
        truth = np.array([[3., 3., 0.]])
        pred = np.array([[1., 4., 4.]])
        columns = physical_peak_columns(torch.from_numpy(pred), torch.from_numpy(truth))
        self.assertEqual(columns.tolist(), [[3., -2., 1.]])
        metrics = summarize_windows(make_records(pred, truth), event_threshold=2.)
        self.assertEqual(metrics['peak_timing_mae_steps_all'], 1.)
        self.assertEqual(metrics['true_peak_bias_all'], -2.)

    def test_equal_maxima_still_penalize_amplitude_at_the_wrong_time(self):
        truth = np.array([[.1, .2, .5, .3]])
        pred = np.array([[.1, .5, .4, .3]])
        self.assertEqual(pred.max(), truth.max())
        metrics = summarize_windows(make_records(pred, truth), event_threshold=.45)
        for population in ('all', 'top5', 'event'):
            for name, expected in dict(true_peak_rmse=.1, true_peak_mae=.1, true_peak_bias=-.1,
                                       true_peak_underprediction_fraction=1., peak_timing_mae_steps=1.).items():
                self.assertAlmostEqual(metrics[f'{name}_{population}'], expected)

    def test_true_peak_underprediction_ignores_larger_predicted_max_elsewhere(self):
        truth = np.array([[.1, .2, .5, .3]])
        pred = np.array([[.1, .8, .4, .3]])
        self.assertGreater(pred.max(), truth.max())
        metrics = summarize_windows(make_records(pred, truth), event_threshold=.45)
        for population in ('all', 'top5', 'event'):
            self.assertEqual(metrics[f'true_peak_underprediction_fraction_{population}'], 1.)
            self.assertAlmostEqual(metrics[f'true_peak_bias_{population}'], -.1)

    def test_canonical_peak_family_contains_exactly_five_stems(self):
        self.assertEqual(PEAK_METRIC_STEMS, ('true_peak_rmse', 'true_peak_mae', 'true_peak_bias',
                         'true_peak_underprediction_fraction', 'peak_timing_mae_steps'))
        self.assertEqual(len(PEAK_METRIC_NAMES), 15)

    def test_exact_top5_membership_and_legacy_values_with_ties_precision_and_padding(self):
        rng = np.random.default_rng(72)
        records = rng.uniform(size=(83, 6))
        records[:, 0] = np.arange(len(records))
        records[:, 1] = 1. + rng.uniform(0, 1e-9, len(records))  # FP32 tie, FP64 distinct
        records = np.concatenate((records[::2], records[1::2], records[[0, 82, 82]]))
        for validation in (False, True):
            old, selected = legacy_summary(records[:, :4], validation)
            rows, top = unique_rows_and_top5_indices(records, validation=validation)
            np.testing.assert_array_equal(rows[top, 0], selected)
            with patch('emulator.training.metrics.unique_rows_and_top5_indices',
                       wraps=unique_rows_and_top5_indices) as selector:
                new = summarize_windows(records, validation=validation, event_threshold=1.)
            self.assertEqual(selector.call_count, 1)
            self.assertEqual({k: new[k] for k in METRIC_NAMES}, old)
            self.assertEqual(new['true_peak_bias_top5'], rows[top, 4].mean())

    def test_saved_fixture_preserves_trajectory_metrics_and_reports_true_peaks(self):
        path = Path(__file__).parent / 'fixtures/post3_peak_metrics.json'
        saved = json.loads(path.read_text())
        self.assertEqual(saved['base'], '50ec23fc23866cb225c594bb403e1505704cda79')
        pred, truth = np.array(saved['y_pred'], dtype=np.float32), np.array(saved['y_true'], dtype=np.float32)
        # The same saved prediction schema used by train.py/infer.py, without a live dataset.
        with tempfile.TemporaryDirectory() as temporary:
            archive = Path(temporary) / 'predictions.npz'
            np.savez_compressed(archive, y_true=truth, y_pred=pred, tags=np.array(saved['tags']))
            with np.load(archive) as arrays:
                rows = make_records(arrays['y_pred'], arrays['y_true'])
        padded = np.concatenate((rows[::2], rows[1::2], rows[[0, 42, 42]]))
        for records, validation, key in ((rows, False, 'inference_metrics'), (padded, True, 'validation_metrics')):
            actual = summarize_windows(records, validation=validation, event_threshold=2.)
            self.assertEqual({k: actual[k] for k in METRIC_NAMES}, saved[key])
            self.assertEqual({k: actual[k] for k in PEAK_METRIC_NAMES}, saved[f'{key}_true_peak'])

    def test_new_populations_deduplicate_all_metrics_while_legacy_val_all_keeps_padding(self):
        truth = np.array([[5., 0.], [4., 1.], [2., 0.]])
        pred = np.array([[2., 1.], [6., 1.], [1., 0.]])
        rows = make_records(pred, truth)
        repeated = np.concatenate((rows[[1, 2]], rows[[0]], rows[[0, 0, 0]]))
        metrics = summarize_windows(rows, validation=True, event_threshold=3.)
        padded = summarize_windows(repeated, validation=True, event_threshold=3.)
        self.assertNotEqual(metrics['rmse_all'], padded['rmse_all'])
        self.assertNotEqual(metrics['mae_all'], padded['mae_all'])
        self.assertEqual({k: metrics[k] for k in PEAK_METRIC_NAMES}, {k: padded[k] for k in PEAK_METRIC_NAMES})
        self.assertEqual(padded['true_peak_underprediction_fraction_all'], 2 / 3)
        self.assertEqual(padded['true_peak_underprediction_fraction_event'], .5)

    def test_event_metrics_use_strict_exact_train_threshold_not_top5_or_test_fit(self):
        truth = np.array([[2., 0.], [3., 0.], [5., 0.]])
        pred = np.array([[100., 0.], [2., 0.], [3., 0.]])
        rows = make_records(pred, truth)
        metrics = summarize_windows(rows, event_threshold=3.)
        self.assertEqual(metrics['true_peak_bias_event'], -2.)
        self.assertEqual(summarize_windows(rows, event_threshold=3. - 1e-8)['true_peak_bias_event'], -1.5)
        for threshold in (None, 5.):
            absent = summarize_windows(rows, event_threshold=threshold)
            self.assertTrue(all(absent[f'{k}_event'] is None for k in PEAK_METRIC_STEMS))
            self.assertIsNotNone(absent['true_peak_rmse_all'])

    def test_empty_and_undefined_metrics_are_json_null_not_nan(self):
        empty = summarize_windows(np.empty((0, 6)), event_threshold=1.)
        self.assertEqual(empty, dict.fromkeys(METRIC_NAMES + PEAK_METRIC_NAMES))
        rows = make_records(np.array([[1., 2.]]), np.array([[1., 2.]]))
        exact = summarize_windows(rows, event_threshold=1.)
        for pop in ('all', 'top5', 'event'):
            self.assertEqual(exact[f'true_peak_underprediction_fraction_{pop}'], 0.)
        json.dumps([empty, exact], allow_nan=False)

    def test_top5_membership_depends_only_on_true_peaks(self):
        truth = np.column_stack((np.arange(41.), np.zeros(41)))
        pred = truth.copy()
        changed = truth.copy()
        changed[:39, 1] = 1000.  # Huge peaks elsewhere, including just one selected window.
        for validation in (False, True):
            original = make_records(pred, truth)
            modified = make_records(changed, truth)
            rows, selected = unique_rows_and_top5_indices(modified, validation=validation)
            np.testing.assert_array_equal(rows[selected, 0], [38, 39, 40])
            old, expected_ids = legacy_summary(original[:, :4], validation)
            np.testing.assert_array_equal(rows[selected, 0], expected_ids)
            expected, _ = legacy_summary(modified[:, :4], validation)
            actual = summarize_windows(modified, validation=validation)
            self.assertEqual({key: actual[key] for key in METRIC_NAMES}, expected)
            self.assertNotEqual(actual['rmse_peak5'], old['rmse_peak5'])

    def test_validation_retains_six_scalars_and_only_one_forward_per_batch(self):
        fixture = training_tests.TrainingTests()
        fixture.setUp()
        model = training_tests.CountingModel()
        with patch.object(engine, 'summarize_windows', wraps=summarize_windows) as summary:
            result = run_epoch(model, fixture.loader(), torch.device('cpu'), fixture.stats, event_threshold=30.)
        self.assertEqual(model.calls, 6)
        self.assertIsNone(result.predictions)
        self.assertEqual(summary.call_args.args[0].shape, (41, 6))
        self.assertFalse(torch.is_tensor(summary.call_args.args[0]))
        self.assertEqual(tuple(result.metrics), METRIC_NAMES + PEAK_METRIC_NAMES)
        self.assertIsNotNone(result.metrics['true_peak_rmse_top5'])

    def test_console_keeps_four_core_metrics_and_one_peak_metric_with_extended_mode(self):
        metrics = dict.fromkeys(METRIC_NAMES + PEAK_METRIC_NAMES, 1.)
        text = format_metrics('Val', metrics)
        self.assertEqual(len(text.split()), 6)
        for name in (*METRIC_NAMES, 'true_peak_rmse_top5'):
            self.assertIn(f'{name}=1.000000', text)
        extended = format_metrics('Val', metrics, extended=True)
        for name in metrics:
            self.assertIn(f'{name}=1.000000', extended)
        self.assertIn('true_peak_rmse_top5=NA', format_metrics('Val', dict.fromkeys(PEAK_METRIC_NAMES)))

    def test_ddp_sampler_padding_counts_each_peak_window_once(self):
        with tempfile.TemporaryDirectory() as temporary:
            init, output = Path(temporary) / 'init', Path(temporary) / 'metrics.json'
            mp.spawn(metric_ddp_worker, args=(str(init), str(output)), nprocs=2, join=True)
            actual = json.loads(output.read_text())
        fixture = training_tests.TrainingTests()
        fixture.setUp()
        reference = run_epoch(training_tests.CountingModel(), fixture.loader(), torch.device('cpu'), fixture.stats,
                              event_threshold=30.)
        self.assertEqual({k: actual[k] for k in PEAK_METRIC_NAMES}, {k: reference.metrics[k] for k in PEAK_METRIC_NAMES})
        self.assertNotEqual(actual['rmse_all'], reference.metrics['rmse_all'])
