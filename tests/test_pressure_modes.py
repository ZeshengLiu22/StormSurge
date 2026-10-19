"""CPU pressure reconstruction, TRAIN statistics, and unchanged target contracts."""

from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
from torch_geometric.data import Data

from emulator.data import (
    ForcingGraphStore,
    ForcingGraphView,
    fit_loss_thresholds,
    fit_statistics,
    normalize_inputs,
    supervised_targets,
)
from emulator.data.graph_store import transform_pressure_features
from emulator.training.episode_peaks import build_episode_peak_targets


MODES = ("anomaly", "raw", "anomaly_plus_mean")


def make_pressure_graph(index=0, *, metadata=True, dtype=torch.float32):
    """Nine six-hour forcing steps; x deliberately aliases the final history."""
    steps = torch.arange(9, dtype=dtype).view(9, 1)
    nodes = torch.arange(4, dtype=dtype).view(1, 4)
    history = torch.empty(9, 4, 5, dtype=dtype)
    history[:, :, 0] = steps + nodes + index * 10
    history[:, :, 1] = -steps + nodes - index * 3
    history[:, :, 2] = (nodes - 1.5) * (steps + 1)
    history[:, :, 3] = -74 + nodes * 0.1
    history[:, :, 4] = 40 + nodes * 0.1
    graph = Data(
        x=history[-1],
        x_hist=history,
        edge_index=torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]]),
        y=torch.tensor([0.0, 3.0, 5.0 + index, 0.0]),
        center_time=f"{2000 + index}-01-01 00:00:00",
        grid_H=2,
        grid_W=2,
        station="Battery",
        station_metadata={"lat": 40.0, "lon": -74.0},
        split_tag=f"original-{index}",
        time_index=index,
        hour_start=0,
        max_history_hours=48,
        nc="unchanged-forcing-source.nc",
        nc_tide="unchanged-target-source.nc",
    )
    center = int(np.datetime64(graph.center_time, "s").astype(np.int64))
    graph.target_timestamps = torch.arange(4) * 3600 + center
    if metadata:
        graph.p_mean_hist = torch.arange(9, dtype=dtype) * 20 + 1000 + index * 200
        graph.p_mean_curr = graph.p_mean_hist[-1]
    return graph


def save_pressure_graphs(root, count=5, *, metadata=True):
    root = Path(root)
    paths = []
    for index in range(count):
        path = root / f"{2000 + index}_{2001 + index}_Battery_fixture_graphs.pt"
        torch.save([make_pressure_graph(index, metadata=metadata)], path)
        paths.append(path)
    return paths


class PressureModeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def assertExact(self, actual, expected):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_default_and_explicit_anomaly_preserve_original_object_and_tensors(self):
        for dtype in (torch.float32, torch.float64):
            graph = make_pressure_graph(metadata=False, dtype=dtype)
            original_x, original_history = graph.x, graph.x_hist
            for mode_args in ({}, {"pressure_mode": "anomaly"}):
                transformed = transform_pressure_features(graph, **mode_args)
                self.assertIs(transformed, graph)
                self.assertIs(transformed.x, original_x)
                self.assertIs(transformed.x_hist, original_history)
                self.assertEqual(transformed.x.shape, (4, 5))
                self.assertEqual(transformed.x.dtype, dtype)
        with tempfile.TemporaryDirectory() as temporary:
            paths = save_pressure_graphs(temporary, count=1, metadata=False)
            original = torch.load(paths[0], weights_only=False)[0]
            default = ForcingGraphStore(temporary, "Battery")
            explicit = ForcingGraphStore(temporary, "Battery", pressure_mode="anomaly")
            for store in (default, explicit):
                self.assertExact(store.graphs[0].x, original.x)
                self.assertExact(store.graphs[0].x_hist, original.x_hist)
                self.assertEqual(store.graphs[0].x.size(-1), 5)
            self.assertEqual(default.graph_tags, explicit.graph_tags)

    def test_raw_reconstructs_all_timesteps_without_changing_other_features(self):
        graph = make_pressure_graph()
        transformed = transform_pressure_features(graph, "raw")
        self.assertIsNot(transformed, graph)
        self.assertEqual(transformed.x_hist.shape, (9, 4, 5))
        self.assertExact(
            transformed.x_hist[:, :, 2], graph.x_hist[:, :, 2] + graph.p_mean_hist[:, None]
        )
        self.assertExact(transformed.x_hist[:, :, [0, 1, 3, 4]], graph.x_hist[:, :, [0, 1, 3, 4]])
        self.assertExact(transformed.x, transformed.x_hist[-1])
        self.assertEqual(transformed.x.data_ptr(), transformed.x_hist[-1].data_ptr())

    def test_anomaly_plus_mean_appends_spatially_constant_temporally_aligned_feature(self):
        graph = make_pressure_graph()
        transformed = transform_pressure_features(graph, "anomaly_plus_mean")
        self.assertEqual(transformed.x.shape, (4, 6))
        self.assertEqual(transformed.x_hist.shape, (9, 4, 6))
        self.assertExact(transformed.x_hist[:, :, :5], graph.x_hist)
        self.assertExact(transformed.x_hist[:, :, 5], graph.p_mean_hist[:, None].expand(9, 4))
        self.assertExact(transformed.x[:, :5], graph.x)
        self.assertExact(transformed.x, transformed.x_hist[-1])
        self.assertEqual(transformed.x.data_ptr(), transformed.x_hist[-1].data_ptr())

    def test_every_supported_history_window_selects_the_correct_pressure_sequence(self):
        with tempfile.TemporaryDirectory() as temporary:
            save_pressure_graphs(temporary, count=1)
            original = make_pressure_graph()
            for mode in MODES:
                store = ForcingGraphStore(temporary, "Battery", pressure_mode=mode)
                for history_steps in range(9):
                    with self.subTest(mode=mode, history_hours=history_steps * 6):
                        batch = ForcingGraphView(store, [0], history_steps)[0]
                        window = history_steps + 1
                        self.assertEqual(batch.x_hist.shape, (4, window, 6 if mode == "anomaly_plus_mean" else 5))
                        if mode == "raw":
                            expected = original.x_hist[-window:, :, 2] + original.p_mean_hist[-window:, None]
                            self.assertExact(batch.x_hist[:, :, 2], expected.T)
                        elif mode == "anomaly_plus_mean":
                            expected = original.p_mean_hist[-window:].expand(4, window)
                            self.assertExact(batch.x_hist[:, :, 5], expected)
                            self.assertExact(batch.x_hist[:, :, :5], original.x_hist[-window:].permute(1, 0, 2))
                        else:
                            self.assertExact(batch.x_hist, original.x_hist[-window:].permute(1, 0, 2))
                        self.assertExact(batch.x, batch.x_hist[:, -1, :])

    def test_missing_metadata_names_mode_graph_and_field(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = save_pressure_graphs(temporary, count=1, metadata=False)[0]
            ForcingGraphStore(temporary, "Battery")
            for mode in MODES[1:]:
                with self.subTest(mode=mode), self.assertRaises(ValueError) as error:
                    ForcingGraphStore(temporary, "Battery", pressure_mode=mode)
                message = str(error.exception)
                self.assertIn(mode, message)
                self.assertIn("p_mean_hist", message)
                self.assertIn(path.stem.removesuffix("_graphs"), message)

    def test_bad_mean_metadata_is_rejected(self):
        variants = [
            ("length", "p_mean_hist", torch.ones(8)),
            ("rank", "p_mean_hist", torch.ones(9, 1)),
            ("nan", "p_mean_hist", torch.tensor([1000.0] * 8 + [float("nan")])),
            ("inf", "p_mean_hist", torch.tensor([float("inf")] + [1000.0] * 8)),
            ("float32 overflow", "p_mean_hist", torch.full((9,), 1e100, dtype=torch.float64)),
            ("current mismatch", "p_mean_curr", torch.tensor(50.0)),
            ("current nan", "p_mean_curr", torch.tensor(float("nan"))),
            ("current inf", "p_mean_curr", torch.tensor(float("inf"))),
            ("current shape", "p_mean_curr", torch.tensor([1160.0, 1160.0])),
        ]
        for mode in MODES[1:]:
            for name, field, value in variants:
                graph = make_pressure_graph()
                graph[field] = value
                with self.subTest(mode=mode, case=name), self.assertRaisesRegex(ValueError, field):
                    transform_pressure_features(graph, mode, graph_tag="bad-pressure-fixture")

    def test_current_mean_is_optional_but_history_is_required(self):
        graph = make_pressure_graph()
        del graph.p_mean_curr
        for mode in MODES[1:]:
            transformed = transform_pressure_features(graph, mode)
            if mode == "raw":
                self.assertExact(transformed.x[:, 2], graph.x[:, 2] + graph.p_mean_hist[-1])
            else:
                self.assertExact(transformed.x[:, 5], graph.p_mean_hist[-1].expand(4))

    def test_invalid_original_feature_shapes_and_dtype_fail_clearly(self):
        variants = [
            ("x channels", "x", torch.zeros(4, 6)),
            ("history channels", "x_hist", torch.zeros(9, 4, 6)),
            ("x rank", "x", torch.zeros(4, 1, 5)),
            ("history rank", "x_hist", torch.zeros(4, 5)),
            ("node counts", "x", torch.zeros(3, 5)),
            ("empty history", "x_hist", torch.zeros(0, 4, 5)),
            ("integer current", "x", torch.zeros(4, 5, dtype=torch.int64)),
            ("integer history", "x_hist", torch.zeros(9, 4, 5, dtype=torch.int64)),
            ("device mismatch", "x", torch.empty(4, 5, device="meta")),
        ]
        for mode in MODES[1:]:
            for name, field, value in variants:
                graph = make_pressure_graph()
                graph[field] = value
                with self.subTest(mode=mode, case=name), self.assertRaises(ValueError) as error:
                    transform_pressure_features(graph, mode, graph_tag="bad-shape-fixture")
                self.assertIn(mode, str(error.exception))
                self.assertIn("bad-shape-fixture", str(error.exception))
            graph = make_pressure_graph()
            del graph.x_hist
            with self.assertRaisesRegex(ValueError, "x_hist"):
                transform_pressure_features(graph, mode)

    def test_nondefault_outputs_are_float32_on_input_device_without_mutating_metadata(self):
        for mode in MODES[1:]:
            for mean_dtype in (torch.float64, torch.int64):
                graph = make_pressure_graph(dtype=torch.float64)
                graph.p_mean_hist = graph.p_mean_hist.to(mean_dtype)
                graph.p_mean_curr = graph.p_mean_hist[-1]
                original_history = graph.x_hist.clone()
                original_mean = graph.p_mean_hist.clone()
                transformed = transform_pressure_features(graph, mode)
                self.assertEqual(transformed.x.dtype, torch.float32)
                self.assertEqual(transformed.x_hist.dtype, torch.float32)
                self.assertEqual(transformed.x.device, graph.x.device)
                self.assertEqual(transformed.x_hist.device, graph.x_hist.device)
                self.assertExact(graph.x_hist, original_history)
                self.assertExact(graph.p_mean_hist, original_mean)
                self.assertIs(transformed.p_mean_hist, graph.p_mean_hist)

    def test_transform_does_not_mutate_shared_input_storage_or_saved_files(self):
        for mode in MODES[1:]:
            graph = make_pressure_graph()
            history = graph.x_hist.clone()
            current = graph.x.clone()
            mean = graph.p_mean_hist.clone()
            self.assertEqual(graph.x.data_ptr(), graph.x_hist[-1].data_ptr())
            first = transform_pressure_features(graph, mode)
            second = transform_pressure_features(graph, mode)
            self.assertExact(first.x_hist, second.x_hist)
            self.assertNotEqual(first.x_hist.data_ptr(), graph.x_hist.data_ptr())
            first.x_hist[0, 0, 0] += 100
            self.assertExact(graph.x_hist, history)
            self.assertExact(graph.x, current)
            self.assertExact(graph.p_mean_hist, mean)
        with tempfile.TemporaryDirectory() as temporary:
            path = save_pressure_graphs(temporary, count=1)[0]
            serialized = path.read_bytes()
            for mode in MODES:
                first = ForcingGraphStore(temporary, "Battery", pressure_mode=mode)
                second = ForcingGraphStore(temporary, "Battery", pressure_mode=mode)
                self.assertExact(first.graphs[0].x_hist, second.graphs[0].x_hist)
                self.assertEqual(path.read_bytes(), serialized)
            reloaded = torch.load(path, weights_only=False)[0]
            self.assertExact(reloaded.x_hist, make_pressure_graph().x_hist)
            self.assertExact(reloaded.x, make_pressure_graph().x)

    def test_statistics_use_transformed_train_features_for_all_normalizations(self):
        with tempfile.TemporaryDirectory() as temporary:
            save_pressure_graphs(temporary)
            for mode in MODES:
                for x_norm in ("zscore", "robust", "mag"):
                    with self.subTest(mode=mode, normalization=x_norm):
                        store = ForcingGraphStore(temporary, "Battery", pressure_mode=mode)
                        splits = store.split()
                        train = splits["train"]
                        stats = fit_statistics(store, train, x_norm=x_norm)
                        features = 6 if mode == "anomaly_plus_mean" else 5
                        self.assertEqual(stats["x_center"].shape, (features,))
                        self.assertEqual(stats["x_scale"].shape, (features,))
                        values = torch.cat([store.graphs[index].x for index in train])
                        if x_norm == "zscore":
                            expected_center = values.sum(0, dtype=torch.float64).div(values.size(0)).float()
                            self.assertExact(stats["x_center"], expected_center)
                        elif x_norm == "robust":
                            low, high = np.percentile(values.numpy(), [1.0, 99.0], axis=0).astype(np.float32)
                            self.assertExact(stats["x_center"], torch.from_numpy((low + high) * 0.5))
                            self.assertExact(stats["x_scale"], torch.from_numpy(np.maximum((high - low) * 0.5, 1e-6)))
                        else:
                            expected_scale = np.percentile(np.abs(values.numpy()), 99.0, axis=0).astype(np.float32)
                            self.assertExact(stats["x_center"], torch.zeros(features))
                            self.assertExact(stats["x_scale"], torch.from_numpy(np.maximum(expected_scale, 1e-6)))
                        # A changed validation/test population must never affect TRAIN fitting.
                        for index in splits["val"] + splits["test"]:
                            store.graphs[index].x = torch.full_like(store.graphs[index].x, 1e8)
                            store.graphs[index].y = torch.full_like(store.graphs[index].y, -1e8)
                        refitted = fit_statistics(store, train, x_norm=x_norm)
                        for key in stats:
                            self.assertExact(refitted[key], stats[key])

    def test_raw_reconstruction_precedes_existing_normalization(self):
        with tempfile.TemporaryDirectory() as temporary:
            save_pressure_graphs(temporary)
            store = ForcingGraphStore(temporary, "Battery", pressure_mode="raw")
            stats = fit_statistics(store, store.split()["train"], x_norm="robust")
            batch = ForcingGraphView(store, [0], 8)[0]
            original = make_pressure_graph()
            raw_history = original.x_hist.clone()
            raw_history[:, :, 2] += original.p_mean_hist[:, None]
            expected = (raw_history.permute(1, 0, 2) - stats["x_center"]) / stats["x_scale"]
            normalize_inputs(batch, stats)
            self.assertExact(batch.x_hist, expected)
            self.assertExact(batch.x, expected[:, -1])
            self.assertExact(store.graphs[0].x_hist, raw_history)

    def test_targets_metadata_splits_thresholds_and_episode_supervision_are_unchanged(self):
        with tempfile.TemporaryDirectory() as temporary:
            save_pressure_graphs(temporary)
            original = ForcingGraphStore(temporary, "Battery")
            splits = original.split()
            labels, times = supervised_targets(original, splits["train"])
            threshold = fit_loss_thresholds(original, splits["train"])
            episodes = build_episode_peak_targets(labels, times, threshold["tau_physical"])
            baseline_view = ForcingGraphView(original, splits["train"], 2)
            baseline_view.episode_peak_targets = episodes
            baseline_stats = fit_statistics(original, splits["train"])
            for mode in MODES[1:]:
                store = ForcingGraphStore(temporary, "Battery", pressure_mode=mode)
                self.assertEqual(store.graph_tags, original.graph_tags)
                self.assertEqual(store.year_to_indices, original.year_to_indices)
                self.assertEqual(store.split(), splits)
                self.assertEqual(fit_loss_thresholds(store, splits["train"]), threshold)
                actual_labels, actual_times = supervised_targets(store, splits["train"])
                np.testing.assert_array_equal(actual_labels, labels)
                np.testing.assert_array_equal(actual_times, times)
                stats = fit_statistics(store, splits["train"])
                self.assertExact(stats["y_mean"], baseline_stats["y_mean"])
                self.assertExact(stats["y_std"], baseline_stats["y_std"])
                actual_episodes = build_episode_peak_targets(actual_labels, actual_times, threshold["tau_physical"])
                np.testing.assert_array_equal(actual_episodes.episode_id, episodes.episode_id)
                np.testing.assert_array_equal(actual_episodes.is_episode_gt_peak, episodes.is_episode_gt_peak)
                self.assertEqual(actual_episodes.metadata, episodes.metadata)
                view = ForcingGraphView(store, splits["train"], 2)
                view.episode_peak_targets = actual_episodes
                for actual, expected in zip(store.graphs, original.graphs):
                    for key in expected.keys():
                        if key in ("x", "x_hist"):
                            continue
                        if isinstance(expected[key], torch.Tensor):
                            self.assertExact(actual[key], expected[key])
                        else:
                            self.assertEqual(actual[key], expected[key])
                for index in range(len(view)):
                    actual, expected = view[index], baseline_view[index]
                    for key in ("edge_index", "y", "target_timestamps", "sample_id", "episode_id", "is_episode_gt_peak"):
                        self.assertExact(actual[key], expected[key])
                    for key in ("tag", "grid_H", "grid_W"):
                        self.assertEqual(actual[key], expected[key])

    def test_targets_only_accepts_missing_pressure_metadata_and_preserves_labels(self):
        with tempfile.TemporaryDirectory() as temporary:
            save_pressure_graphs(temporary, metadata=False)
            baseline = ForcingGraphStore(temporary, "Battery", targets_only=True)
            for mode in MODES:
                selected = ForcingGraphStore(temporary, "Battery", pressure_mode=mode, targets_only=True)
                self.assertEqual(selected.graph_tags, baseline.graph_tags)
                self.assertEqual(selected.split(), baseline.split())
                for actual, expected in zip(selected.graphs, baseline.graphs):
                    self.assertIsNone(getattr(actual, "x", None))
                    self.assertIsNone(getattr(actual, "x_hist", None))
                    self.assertEqual(actual.source_history_steps, 9)
                    for key in expected.keys():
                        if isinstance(expected[key], torch.Tensor):
                            self.assertExact(actual[key], expected[key])
                        else:
                            self.assertEqual(actual[key], expected[key])
                self.assertEqual(
                    fit_loss_thresholds(selected, selected.split()["train"]),
                    fit_loss_thresholds(baseline, baseline.split()["train"]),
                )

    def test_pattern_station_and_year_filters_apply_before_pressure_validation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            save_pressure_graphs(root, count=1)
            # Every excluded file lacks pressure metadata, so loading it would fail.
            excluded = make_pressure_graph(metadata=False)
            for filename in (
                "2000_2001_Battery_other_graphs.pt",
                "2000_2001_Other_fixture_graphs.pt",
                "2002_2003_Battery_fixture_graphs.pt",
            ):
                torch.save([excluded], root / filename)
            for mode in MODES:
                store = ForcingGraphStore(
                    root, "Battery", pressure_mode=mode, pattern="*_fixture_graphs.pt",
                    year_filter=["2000_2001"],
                )
                self.assertEqual(store.graph_tags, ["2000_2001_Battery_fixture_0"])
                self.assertEqual(dict(store.year_to_indices), {"2000_2001": [0]})

    def test_unknown_pressure_mode_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "pressure_mode"):
            transform_pressure_features(make_pressure_graph(), "absolute")
        with tempfile.TemporaryDirectory() as temporary:
            save_pressure_graphs(temporary, count=1)
            with self.assertRaisesRegex(ValueError, "pressure_mode"):
                ForcingGraphStore(temporary, "Battery", pressure_mode="absolute")


if __name__ == "__main__":
    unittest.main()
