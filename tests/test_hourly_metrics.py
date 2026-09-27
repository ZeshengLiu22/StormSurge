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

    def test_exact_eleven_metrics_and_known_episode_values(self):
        metrics = evaluate_metrics(self.pred, self.truth, 2., target_timestamps=self.times)
        expected = dict(all_rmse=2., all_mae=16/9,
            exceedance_rmse=math.sqrt(23/4), exceedance_mae=9/4,
            episode_peak_rmse=math.sqrt(5/2), episode_peak_mae=1.5, episode_peak_bias=-.5,
            episode_gt_aligned_peak_rmse=math.sqrt(13/2), episode_gt_aligned_peak_mae=2.5,
            episode_gt_aligned_peak_bias=-2.5, episode_peak_timing_mae_hours=.5)
        self.assertEqual(set(metrics), set(expected))
        self.assertEqual(len(metrics), 11)
        for key, value in expected.items():
            self.assertAlmostEqual(metrics[key], value, msg=key)

    def test_strict_threshold_and_first_chronological_argmax(self):
        truth, pred = np.array([[3., 3., 2.]]), np.array([[1., 4., 100.]])
        metrics = evaluate_metrics(pred, truth, 2., target_timestamps=np.array([[0, 3600, 7200]]))
        self.assertEqual(metrics['episode_gt_aligned_peak_bias'], -2.)
        self.assertEqual(metrics['episode_peak_bias'], 1.)
        self.assertEqual(metrics['episode_peak_timing_mae_hours'], 1.)
        exact = evaluate_metrics(np.array([[1., 2.]]), np.array([[1., 2.]]), 2.)
        self.assertIsNone(exact['exceedance_rmse'])

    def test_episode_crosses_windows_but_never_gaps_or_splits(self):
        truth = np.array([[3., 3.], [3., 3.], [0., 3.]])
        times = np.array([[0, 3600], [7200, 14400], [18000, 21600]])
        episodes = gt_event_episodes(truth, times, 2.)
        self.assertEqual([e.tolist() for e in episodes], [[0, 1, 2], [3], [5]])
        episodes = gt_event_episodes(truth, times, 2., split_ids=['train', 'val', 'test'])
        self.assertEqual([e.tolist() for e in episodes], [[5], [0, 1], [2], [3]])
        order = [2, 0, 1]
        shuffled = evaluate_metrics(truth[order], truth[order], 2., target_timestamps=times[order])
        self.assertEqual(len(gt_event_episodes(truth[order], times[order], 2.)), 3)
        self.assertEqual(shuffled['episode_peak_rmse'], 0.)

    def test_episode_population_independent_of_prediction_uses_exact_interval(self):
        truth = np.array([[3., 0., 3.]])
        times = np.array([[0, 3600, 7200]])
        metrics = evaluate_metrics(np.array([[0., 100., 3.]]), truth, 2., target_timestamps=times)
        self.assertEqual(metrics['episode_peak_rmse'], math.sqrt(9/2))
        self.assertEqual(metrics['episode_peak_timing_mae_hours'], 0.)

    def test_timestamp_duplicates_fail_before_any_silent_deduplication(self):
        times = self.times.copy(); times[1, 0] = times[0, 0]
        with self.assertRaisesRegex(ValueError, 'Duplicate target timestamp.*exactly once'):
            evaluate_metrics(self.pred, self.truth, 2., target_timestamps=times)
        # Identical timestamps belong to different split series and cannot join episodes.
        episodes = gt_event_episodes(np.ones((2, 1)), np.zeros((2, 1), dtype=int), 0.,
                                     split_ids=['train', 'val'])
        self.assertEqual([indices.tolist() for indices in episodes], [[0], [1]])

    def test_numpy_torch_iso_and_datetime_inputs_agree_without_mutation(self):
        reference = evaluate_metrics(self.pred, self.truth, 2., target_timestamps=self.times)
        times = (self.times.astype('datetime64[s]') + np.timedelta64(20000, 'D')).astype('datetime64[ns]')
        for value in (times, times.astype(str)):
            actual = evaluate_metrics(torch.tensor(self.pred, requires_grad=True), torch.tensor(self.truth), 2., target_timestamps=value)
            self.assertEqual(actual, reference)
        np.testing.assert_array_equal(self.truth, [[0., 3., 5.], [4., 0., 0.], [0., 0., 6.]])

    def test_empty_and_undefined_populations_serialize_null(self):
        metrics = evaluate_metrics(np.empty((0, 3)), np.empty((0, 3)), 2., target_timestamps=np.empty((0, 3), dtype=int))
        self.assertIsNone(metrics['all_rmse'])
        self.assertIsNone(metrics['episode_peak_rmse'])
        self.assertIsNone(metrics['exceedance_rmse'])
        json.dumps(metrics, allow_nan=False)

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

    def test_labels_groups_epoch_availability_and_console(self):
        metrics = evaluate_metrics(self.pred, self.truth, 2.)
        self.assertEqual(set(metrics), set(EPOCH_METRIC_KEYS))
        self.assertEqual(set(METRIC_LABELS), set(METRIC_KEYS))
        self.assertEqual(len(METRIC_KEYS), len(set(METRIC_KEYS)))
        self.assertEqual(set(key for group in METRIC_GROUPS.values() for key in group), set(METRIC_KEYS))
        self.assertIn('ExceedanceRMSE=', format_metrics('Val', metrics))
        self.assertIn('AllRMSE=NA', format_metrics('Val', {'all_rmse': None}))
        self.assertIn('ExceedanceMAE=', format_metrics('Val', metrics, extended=True))

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
