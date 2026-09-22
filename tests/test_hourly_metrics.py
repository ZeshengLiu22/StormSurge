"""Numerical population, unit, timestamp, and episode contracts."""

import json
import math
import unittest
from unittest.mock import patch

import numpy as np
import torch

from emulator.training.metrics import (EPOCH_METRIC_KEYS, METRIC_GROUPS, METRIC_KEYS,
    METRIC_LABELS, evaluate_metrics, format_metrics, gt_event_episodes)


class HourlyMetricTests(unittest.TestCase):
    def setUp(self):
        self.truth = np.array([[0., 3., 5.], [4., 0., 0.], [0., 0., 6.]])
        self.pred = np.array([[2., 6., 2.], [3., 2., 1.], [0., 2., 4.]])
        self.times = np.arange(9).reshape(3, 3) * 3600

    def test_five_distinct_populations_and_exact_episode_values(self):
        metrics = evaluate_metrics(self.pred, self.truth, 2., target_timestamps=self.times)
        expected = dict(all_rmse=2., all_mae=16/9,
            extreme_hour_n=4, extreme_hour_rate=4/9,
            exceedance_rmse=math.sqrt(23/4), exceedance_mae=9/4, exceedance_bias=-3/4,
            exceedance_under_pct=75., exceedance_under_5mm_pct=75.,
            exceedance_under_10mm_pct=75., exceedance_under_20mm_pct=75.,
            exceedance_precision=1., exceedance_recall=3/4, exceedance_f1=6/7,
            event_window_n=3, event_window_rate=1., event_window_rmse=2., event_window_mae=16/9,
            window_peak_rmse=math.sqrt(2.), window_peak_mae=4/3, window_peak_bias=-2/3,
            gt_aligned_peak_rmse=math.sqrt(14/3), gt_aligned_peak_mae=2., gt_aligned_peak_bias=-2.,
            gt_aligned_peak_under_pct=100., gt_aligned_peak_under_5mm_pct=100.,
            gt_aligned_peak_under_10mm_pct=100., gt_aligned_peak_under_20mm_pct=100.,
            peak_timing_mae_steps=1/3, peak_timing_mae_hours=1/3,
            episode_n=2, episode_peak_rmse=math.sqrt(5/2), episode_peak_mae=1.5,
            episode_peak_bias=-.5, episode_gt_aligned_peak_rmse=math.sqrt(13/2),
            episode_gt_aligned_peak_mae=2.5, episode_gt_aligned_peak_bias=-2.5,
            episode_gt_aligned_peak_under_pct=100., episode_peak_timing_mae_hours=.5,
            episode_detection_recall=1., episode_excess_area_mae=1.5, episode_excess_area_bias=-1.5)
        self.assertEqual(set(expected), set(METRIC_KEYS))
        for key, value in expected.items():
            self.assertAlmostEqual(metrics[key], value, msg=key)
        self.assertEqual(len(set(metrics[key] for key in ("exceedance_rmse", "event_window_rmse",
            "window_peak_rmse", "gt_aligned_peak_rmse", "episode_peak_rmse"))), 5)
        self.assertEqual([metrics[f"extreme_hour_n_lead_{h}"] for h in range(3)], [1, 1, 2])
        for lead, expected in enumerate((1., 3., math.sqrt(13/2))):
            self.assertAlmostEqual(metrics[f"exceedance_rmse_lead_{lead}"], expected)

    def test_window_qualifies_by_truth_and_scores_all_horizons(self):
        truth, pred = np.array([[5., 0.], [2., 2.]]), np.array([[4., 10.], [20., 20.]])
        metrics = evaluate_metrics(pred, truth, 2.)
        self.assertEqual(metrics['extreme_hour_n'], 1)
        self.assertEqual(metrics['event_window_n'], 1)
        self.assertEqual(metrics['exceedance_rmse'], 1.)
        self.assertEqual(metrics['event_window_rmse'], math.sqrt(101/2))
        self.assertEqual(metrics['window_peak_bias'], 5.)
        self.assertEqual(metrics['gt_aligned_peak_bias'], -1.)
        self.assertEqual(metrics['exceedance_precision'], .25)
        self.assertEqual(metrics['exceedance_recall'], 1.)

    def test_strict_threshold_and_tolerance_percentages(self):
        pred = np.array([[0., -.005, -.010, -.020, -.021, .1]])
        truth = np.zeros_like(pred)
        metrics = evaluate_metrics(pred, truth, -1.)
        for suffix, expected in (("", 4/6*100), ("_5mm", 50.), ("_10mm", 2/6*100), ("_20mm", 1/6*100)):
            self.assertAlmostEqual(metrics[f'exceedance_under{suffix}_pct'], expected)
        exact = evaluate_metrics(np.array([[1., 2.]]), np.array([[1., 2.]]), 2.)
        self.assertEqual(exact['extreme_hour_n'], 0)
        self.assertIsNone(exact['exceedance_rmse'])
        self.assertIsNone(exact['exceedance_precision'])

    def test_first_argmax_ties_and_hours_conversion(self):
        truth, pred = np.array([[3., 3., 0.]]), np.array([[1., 4., 4.]])
        metrics = evaluate_metrics(pred, truth, 2., target_interval_hours=2.)
        self.assertEqual(metrics['gt_aligned_peak_bias'], -2.)
        self.assertEqual(metrics['window_peak_bias'], 1.)
        self.assertEqual(metrics['peak_timing_mae_steps'], 1.)
        self.assertEqual(metrics['peak_timing_mae_hours'], 2.)

    def test_episode_crosses_windows_but_never_gaps_or_splits(self):
        truth = np.array([[3., 3.], [3., 3.], [0., 3.]])
        times = np.array([[0, 3600], [7200, 14400], [18000, 21600]])
        episodes = gt_event_episodes(truth, times, 2.)
        self.assertEqual([e.tolist() for e in episodes], [[0, 1, 2], [3], [5]])
        episodes = gt_event_episodes(truth, times, 2., split_ids=['train', 'val', 'test'])
        self.assertEqual([e.tolist() for e in episodes], [[5], [0, 1], [2], [3]])
        order = [2, 0, 1]
        shuffled = evaluate_metrics(truth[order], truth[order], 2., target_timestamps=times[order])
        self.assertEqual(shuffled['episode_n'], 3)
        self.assertEqual(shuffled['episode_peak_rmse'], 0.)

    def test_episode_population_independent_of_prediction_and_detection_uses_exact_interval(self):
        truth = np.array([[3., 0., 3.]])
        times = np.array([[0, 3600, 7200]])
        metrics = evaluate_metrics(np.array([[0., 100., 3.]]), truth, 2., target_timestamps=times)
        self.assertEqual(metrics['episode_n'], 2)
        self.assertEqual(metrics['episode_detection_recall'], .5)
        self.assertEqual(metrics['episode_peak_rmse'], math.sqrt(9/2))
        self.assertEqual(metrics['episode_excess_area_bias'], -.5)
        self.assertEqual(metrics['episode_peak_timing_mae_hours'], 0.)

    def test_timestamp_duplicates_fail_before_any_silent_deduplication(self):
        times = self.times.copy(); times[1, 0] = times[0, 0]
        with self.assertRaisesRegex(ValueError, 'Duplicate target timestamp.*exactly once'):
            evaluate_metrics(self.pred, self.truth, 2., target_timestamps=times)
        # Identical timestamps belong to different split series and cannot join episodes.
        metrics = evaluate_metrics(np.ones((2, 1)), np.ones((2, 1)), 0.,
                                   target_timestamps=np.zeros((2, 1), dtype=int), split_ids=['train', 'val'])
        self.assertEqual(metrics['episode_n'], 2)

    def test_numpy_torch_iso_and_datetime_inputs_agree_without_mutation(self):
        reference = evaluate_metrics(self.pred, self.truth, 2., target_timestamps=self.times)
        times = (self.times.astype('datetime64[s]') + np.timedelta64(20000, 'D')).astype('datetime64[ns]')
        for value in (times, times.astype(str)):
            actual = evaluate_metrics(torch.tensor(self.pred, requires_grad=True), torch.tensor(self.truth), 2., target_timestamps=value)
            self.assertEqual(actual, reference)
        np.testing.assert_array_equal(self.truth, [[0., 3., 5.], [4., 0., 0.], [0., 0., 6.]])

    def test_empty_and_undefined_populations_serialize_null(self):
        metrics = evaluate_metrics(np.empty((0, 3)), np.empty((0, 3)), 2., target_timestamps=np.empty((0, 3), dtype=int))
        self.assertEqual(metrics['episode_n'], 0)
        self.assertEqual(metrics['extreme_hour_n'], 0)
        self.assertIsNone(metrics['all_rmse'])
        self.assertIsNone(metrics['extreme_hour_rate'])
        self.assertIsNone(metrics['episode_detection_recall'])
        json.dumps(metrics, allow_nan=False)
        missed = evaluate_metrics(np.zeros((1, 1)), np.ones((1, 1)), .5)
        self.assertIsNone(missed['exceedance_precision'])
        self.assertEqual(missed['exceedance_recall'], 0.)
        self.assertEqual(missed['exceedance_f1'], 0.)
        false_alarm = evaluate_metrics(np.ones((1, 1)), np.zeros((1, 1)), .5)
        self.assertEqual(false_alarm['exceedance_precision'], 0.)
        self.assertIsNone(false_alarm['exceedance_recall'])
        self.assertEqual(false_alarm['exceedance_f1'], 0.)

    def test_invalid_inputs_fail_explicitly(self):
        for tau in (float('nan'), float('inf')):
            with self.assertRaisesRegex(ValueError, 'TRAIN'):
                evaluate_metrics(self.pred, self.truth, tau)
        with self.assertRaisesRegex(ValueError, 'nonfinite'):
            evaluate_metrics(self.pred * np.nan, self.truth, 2.)
        with self.assertRaisesRegex(ValueError, 'identical'):
            evaluate_metrics(self.pred[:1], self.truth, 2.)
        with self.assertRaisesRegex(ValueError, 'shape'):
            evaluate_metrics(self.pred.ravel(), self.truth, 2.)
        with self.assertRaisesRegex(ValueError, 'shape'):
            evaluate_metrics(self.pred, self.truth, 2., target_timestamps=[0])
        with self.assertRaisesRegex(ValueError, 'NaT'):
            evaluate_metrics(self.pred, self.truth, 2., target_timestamps=np.full(self.truth.shape, np.datetime64('NaT')))
        with self.assertRaisesRegex(ValueError, 'integer UNIX'):
            evaluate_metrics(self.pred, self.truth, 2., target_timestamps=self.times + .5)
        with self.assertRaisesRegex(ValueError, 'positive'):
            evaluate_metrics(self.pred, self.truth, 2., target_interval_hours=0.)

    def test_labels_groups_epoch_availability_and_console(self):
        metrics = evaluate_metrics(self.pred, self.truth, 2., include_leadwise=False)
        self.assertEqual(set(metrics), set(EPOCH_METRIC_KEYS))
        self.assertEqual(set(METRIC_LABELS), set(METRIC_KEYS))
        self.assertEqual(len(METRIC_KEYS), len(set(METRIC_KEYS)))
        self.assertEqual(set(key for group in METRIC_GROUPS.values() for key in group), set(METRIC_KEYS))
        self.assertIn('ExceedanceRMSE=', format_metrics('Val', metrics))
        self.assertIn('EventWindowRMSE=', format_metrics('Val', metrics))
        self.assertIn('ExtremeHourN=4', format_metrics('Val', metrics))
        self.assertIn('EventWindowN=3', format_metrics('Val', metrics))
        self.assertIn('AllRMSE=NA', format_metrics('Val', {'all_rmse': None}))
        self.assertIn('ExceedanceRecall=', format_metrics('Val', metrics, extended=True))

    def test_distributed_sampler_padding_never_changes_evaluation_populations(self):
        from emulator.training import run_epoch
        from test_training import CountingModel, TrainingTests
        fixture = TrainingTests()
        fixture.setUp()
        reference = run_epoch(CountingModel(), fixture.loader(), torch.device('cpu'), fixture.stats,
                              tau_physical=20., save_predictions=True)
        # Two simulated sampler ranks: even IDs and odd IDs plus padded ID 0.
        # Gathered payloads are the actual physical predictions from the engine.
        remote = np.r_[np.arange(1, 41, 2), 0]
        def gather(destination, local):
            destination[0] = local
            destination[1] = {key: reference.predictions[key][remote] for key in local}
        model = CountingModel()
        with patch('emulator.training.engine.dist.get_world_size', return_value=2), \
                patch('emulator.training.engine.dist.all_gather_object', side_effect=gather):
            actual = run_epoch(model, fixture.loader(fixture.data[::2]), torch.device('cpu'), fixture.stats,
                               tau_physical=20., distributed=True)
        self.assertEqual(model.calls, 3)
        self.assertIsNone(actual.predictions)
        for key, value in actual.metrics.items():
            self.assertEqual(value, reference.metrics[key], key)


if __name__ == '__main__':
    unittest.main()
