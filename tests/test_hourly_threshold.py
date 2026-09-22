"""TRAIN physical-hour threshold, target chronology and leakage regression tests."""

from types import SimpleNamespace
import unittest

import numpy as np
import torch
from torch_geometric.data import Data

from emulator.data import (ForcingGraphView, fit_loss_thresholds, supervised_targets,
                           target_timestamps_from_graph, threshold_population, validate_target_timestamps)


def fixture():
    labels = [[0., 1., 2., 3., 4., 5.], [6., 7., 8., 9., 10., 11.],
              [100., 200., 300., 400., 500., 600.], [1000., 2000., 3000., 4000., 5000., 6000.]]
    graphs = [Data(y=torch.tensor(row), center_time=f"2001-01-01T{index * 6:02}:00:00",
                   x=torch.full((2, 1), 1e8), x_hist=torch.full((9, 2, 1), 1e9),
                   edge_index=torch.tensor([[0, 1], [1, 0]])) for index, row in enumerate(labels)]
    return SimpleNamespace(graphs=graphs, graph_tags=[f"g{index}" for index in range(len(graphs))])


class HourlyThresholdTests(unittest.TestCase):
    def test_physical_target_hours_only_and_explicit_linear_quantile(self):
        store = fixture()
        fitted = fit_loss_thresholds(store, [0, 1], 75., stats=dict(y_mean=torch.tensor([2.] * 6), y_std=torch.tensor([4.] * 6)))
        self.assertEqual(fitted["tau_physical"], 8.25)
        self.assertEqual(fitted["quantile_method"], "linear")
        self.assertEqual(fitted["tau_normalized"], [1.5625] * 6)
        self.assertEqual(fitted["train_target_hour_count"], 12)
        self.assertEqual(fitted["train_extreme_hour_count"], 3)
        self.assertEqual(fitted["train_event_window_count"], 1)
        self.assertEqual(fitted["train_episode_count"], 1)
        self.assertEqual(fitted["threshold_schema"], "train_hourly_q95_v1")

    def test_val_test_and_history_changes_do_not_change_train_threshold(self):
        store = fixture()
        before = fit_loss_thresholds(store, [0, 1])
        for graph in store.graphs:
            graph.x.fill_(-1e8)
            graph.x_hist.fill_(-1e9)
        store.graphs[2].y.fill_(-1e6)
        store.graphs[3].y.fill_(1e6)
        self.assertEqual(fit_loss_thresholds(store, [0, 1]), before)

    def test_duplicate_target_timestamps_raise(self):
        store = fixture()
        store.graphs[1].center_time = "2001-01-01T05:00:00"
        with self.assertRaisesRegex(ValueError, "Duplicate supervised target timestamps.*do not silently deduplicate"):
            fit_loss_thresholds(store, [0, 1])

    def test_missing_timestamp_metadata_raises(self):
        with self.assertRaisesRegex(ValueError, "no center_time or target_timestamps"):
            target_timestamps_from_graph(Data(y=torch.arange(6)))

    def test_lead_zero_six_hour_disjoint_blocks_and_view_history(self):
        store = fixture()
        labels, timestamps = supervised_targets(store, [0, 1])
        self.assertEqual(labels.shape, (2, 6))
        np.testing.assert_array_equal(np.diff(timestamps.reshape(-1)), 3600)
        self.assertEqual(timestamps[0, 0], np.datetime64("2001-01-01T00:00:00", "s").astype(np.int64))
        store.target_timestamps = lambda indices: supervised_targets(store, indices)[1]
        graph = ForcingGraphView(store, [0, 1], history_steps=2)[0]
        self.assertEqual(tuple(graph.x_hist.shape), (2, 3, 1))
        np.testing.assert_array_equal(graph.target_timestamps.numpy(), timestamps[:1])

    def test_strict_extremes_event_windows_and_cross_window_episode(self):
        labels = np.array([[0., 2., 2.], [2., 1., 2.]])
        timestamps = np.arange(6).reshape(2, 3) * 3600
        values = threshold_population(labels, timestamps, 1.)
        self.assertEqual(values["extreme_hour_count"], 4)
        self.assertEqual(values["event_window_count"], 2)
        self.assertEqual(values["episode_count"], 2)
        # Hour 2 and hour 3 form a single episode across the block boundary.
        separate = sum(threshold_population(y[None, :], t[None, :], 1.)["episode_count"] for y, t in zip(labels, timestamps))
        self.assertEqual(separate, 3)
        self.assertEqual(threshold_population(np.ones((1, 3)), timestamps[:1], 1.)["extreme_hour_count"], 0)

    def test_missing_hour_breaks_episode(self):
        population = threshold_population(np.array([[2., 2.], [2., 2.]]), np.array([[0, 3600], [10800, 14400]]), 1.)
        self.assertEqual(population["episode_count"], 2)

    def test_iso_and_datetime64_timestamps_share_utc_seconds(self):
        actual = validate_target_timestamps(["2001-01-01T00:00:00Z", "2001-01-01T02:00:00+01:00"])
        expected = validate_target_timestamps(np.array(["2001-01-01T00", "2001-01-01T01"], dtype="datetime64[h]"))
        np.testing.assert_array_equal(actual, expected)

    def test_invalid_or_subsecond_timestamp_rejected(self):
        for value in (np.array(["NaT"], dtype="datetime64[s]"), ["2001-01-01T00:00:00.1"], ["bad"]):
            with self.assertRaises(ValueError):
                validate_target_timestamps(value)

    def test_explicit_timestamps_are_checked_against_center_and_interval(self):
        graph = Data(y=torch.tensor([1., 2.]), center_time="1970-01-01T00:00:00", target_timestamps=torch.tensor([0, 7200]))
        with self.assertRaisesRegex(ValueError, "consecutive hourly"):
            target_timestamps_from_graph(graph)
        graph.target_timestamps = torch.tensor([3600, 7200])
        with self.assertRaisesRegex(ValueError, "lead zero"):
            target_timestamps_from_graph(graph)


if __name__ == "__main__":
    unittest.main()
