"""Numerical stability and legacy compatibility of TRAIN pressure statistics."""

from datetime import timedelta
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch_geometric.data import Data

from emulator.data import ForcingGraphStore, fit_statistics


PRESSURE_CHANNELS = {"raw": 2, "anomaly_plus_mean": 5}


def make_store(mode, pressure_groups):
    """Build already-transformed current features, with different graph sizes."""
    graphs = []
    for i, pressures in enumerate(pressure_groups):
        nodes = torch.arange(len(pressures), dtype=torch.float32)
        x = torch.stack((nodes + i, 2 * nodes - i, nodes - 1,
                         -74 + 2 * nodes, 40 + 3 * nodes), dim=1)
        if mode == "anomaly_plus_mean":
            x = torch.cat((x, torch.zeros(len(nodes), 1)), dim=1)
        x[:, PRESSURE_CHANNELS.get(mode, 2)] = torch.tensor(pressures)
        graphs.append(SimpleNamespace(
            x=x, x_hist=x.unsqueeze(0).repeat(3, 1, 1),
            y=torch.tensor([i - 1.0, float("nan") if i == 0 else i + 2.0, 3 - i / 4]),
        ))
    return SimpleNamespace(graphs=graphs, pressure_mode=mode)


def legacy_statistics(store, indices):
    """Frozen pre-fix z-score arithmetic, including FP32 feature multiplication."""
    total = torch.zeros(store.graphs[indices[0]].x.size(-1), dtype=torch.float64)
    square = torch.zeros_like(total)
    count = torch.zeros((), dtype=torch.float64)
    for i in indices:
        x = store.graphs[i].x.float()
        total += x.sum(dim=0, dtype=torch.float64)
        square += (x * x).sum(dim=0, dtype=torch.float64)
        count += x.size(0)
    center = (total / count).float()
    variance = (square / count).float() - center**2
    result = dict(x_center=center, x_scale=torch.sqrt(variance + 1e-6))
    total = torch.zeros(store.graphs[indices[0]].y.numel(), dtype=torch.float64)
    square, count = torch.zeros_like(total), torch.zeros_like(total)
    for i in indices:
        y = store.graphs[i].y.reshape(-1).double()
        finite = torch.isfinite(y)
        y = torch.where(finite, y, 0.0)
        total += y
        square += y**2
        count += finite
    center = (total / count).float()
    variance = (square / count).float() - center**2
    result.update(y_mean=center, y_std=torch.sqrt(variance + 1e-6))
    return result


def pressure_scale(store, indices):
    """Independent population-variance reference on the effective FP32 samples."""
    channel = PRESSURE_CHANNELS[store.pressure_mode]
    pressure = torch.cat([store.graphs[i].x[:, channel].float() for i in indices]).double()
    return torch.sqrt(pressure.var(correction=0).float() + 1e-6)


def distributed_statistics_worker(rank, root):
    """Exercise real collectives with rank two receiving no TRAIN graphs."""
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo", rank=rank, world_size=3,
        init_method=f"file://{root}/rendezvous", timeout=timedelta(seconds=45),
    )
    try:
        results = {}
        for mode in PRESSURE_CHANNELS:
            store = make_store(mode, [[101325.0], [101325.0078125] * 4])
            results[mode] = fit_statistics(store, [0, 1], device="cpu")
        torch.save(results, Path(root) / f"rank-{rank}.pt")
    finally:
        dist.destroy_process_group()


class PressureStatisticsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def assertExact(self, actual, expected):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)

    def assertUnchangedStatistics(self, actual, expected, pressure_channel=None):
        for key in expected:
            if key == "x_scale" and pressure_channel is not None:
                keep = torch.arange(len(expected[key])) != pressure_channel
                self.assertExact(actual[key][keep], expected[key][keep])
            else:
                self.assertExact(actual[key], expected[key])

    def test_small_variations_survive_negative_and_zero_legacy_variance(self):
        for mode, channel in PRESSURE_CHANNELS.items():
            for delta in (5, 10, 20, 50, 100):
                with self.subTest(mode=mode, delta=delta):
                    store = make_store(mode, [[101325 - delta] * 2, [101325 + delta] * 2])
                    legacy = legacy_statistics(store, [0, 1])
                    fitted = fit_statistics(store, [0, 1])
                    self.assertExact(fitted["x_scale"][channel], torch.sqrt(torch.tensor(delta**2 + 1e-6)))
                    self.assertTrue(torch.isfinite(fitted["x_scale"][channel]))
                    self.assertUnchangedStatistics(fitted, legacy, channel)
                    if delta == 5:
                        self.assertTrue(torch.isnan(legacy["x_scale"][channel]))
                    elif delta in (10, 20):
                        self.assertExact(legacy["x_scale"][channel], torch.sqrt(torch.tensor(1e-6)))

    def test_constant_absolute_pressure_retains_epsilon_scale(self):
        for mode, channel in PRESSURE_CHANNELS.items():
            with self.subTest(mode=mode):
                store = make_store(mode, [[101325.0], [101325.0] * 4])
                fitted = fit_statistics(store, [0, 1])
                self.assertExact(fitted["x_center"][channel], torch.tensor(101325.0))
                self.assertExact(fitted["x_scale"][channel], torch.sqrt(torch.tensor(1e-6)))

    def test_unequal_node_counts_keep_the_mean_in_fp64_for_centering(self):
        for mode, channel in PRESSURE_CHANNELS.items():
            with self.subTest(mode=mode):
                store = make_store(mode, [[101325.0], [101325.0078125] * 4])
                fitted = fit_statistics(store, [0, 1])
                self.assertExact(fitted["x_scale"][channel], pressure_scale(store, [0, 1]))
                self.assertUnchangedStatistics(fitted, legacy_statistics(store, [0, 1]), channel)
                values = torch.cat([g.x[:, channel] for g in store.graphs]).double()
                rounded_mean_variance = ((values - values.mean().float()) ** 2).mean()
                self.assertGreater(float(rounded_mean_variance), float(values.var(correction=0)) * 1.2)
                # Both passes must see the same FP32-rounded samples even for
                # custom stores whose source tensors contain extra FP64 precision.
                for graph in store.graphs:
                    graph.x = graph.x.double() + 0.001
                fitted = fit_statistics(store, [0, 1])
                self.assertExact(fitted["x_scale"][channel], pressure_scale(store, [0, 1]))
                self.assertUnchangedStatistics(fitted, legacy_statistics(store, [0, 1]), channel)

    def test_default_and_explicit_anomaly_exactly_match_legacy(self):
        for pressures in (([-5.0, 0.0], [2.0, 3.0, 7.0]),
                          ([101320.0], [101330.0])):
            for explicit_mode in (False, True):
                with self.subTest(pressures=pressures, explicit_mode=explicit_mode):
                    store = make_store("anomaly", pressures)
                    # Also preserve the established cast of FP64 source features.
                    for graph in store.graphs:
                        graph.x = graph.x.double() + 1e-8
                    if not explicit_mode:
                        del store.pressure_mode
                    self.assertUnchangedStatistics(
                        fit_statistics(store, [0, 1]), legacy_statistics(store, [0, 1])
                    )

    def test_held_out_graphs_and_history_do_not_affect_train_fitting(self):
        for mode in PRESSURE_CHANNELS:
            with self.subTest(mode=mode):
                store = make_store(mode, [[101320.0], [101330.0] * 3, [1e8] * 5])
                original = [graph.x.clone() for graph in store.graphs]
                expected = fit_statistics(store, [0, 1])
                for graph, before in zip(store.graphs, original):
                    self.assertExact(graph.x, before)
                    graph.x_hist.fill_(-1e9)
                store.graphs[2].x.fill_(-1e8)
                store.graphs[2].y.fill_(1e9)
                self.assertUnchangedStatistics(fit_statistics(store, [0, 1]), expected)

    def test_percentile_modes_are_independent_of_pressure_mode(self):
        for mode in PRESSURE_CHANNELS:
            for x_norm in ("robust", "mag"):
                with self.subTest(mode=mode, x_norm=x_norm):
                    store = make_store(mode, [[101320.0], [101330.0] * 4])
                    legacy_store = SimpleNamespace(graphs=store.graphs)
                    options = dict(x_norm=x_norm, nodes_per_graph=2, seed=17)
                    self.assertUnchangedStatistics(
                        fit_statistics(store, [0, 1], **options),
                        fit_statistics(legacy_store, [0, 1], **options),
                    )

    def test_loaded_store_selects_stability_from_its_pressure_mode(self):
        with tempfile.TemporaryDirectory() as root:
            graphs = []
            for i, mean in enumerate((101320.0, 101330.0)):
                x = make_store("anomaly", [[0.0, 0.0]]).graphs[0].x
                graphs.append(Data(x=x, x_hist=x.unsqueeze(0), y=torch.tensor([float(i)]),
                                   p_mean_hist=torch.tensor([mean]), p_mean_curr=torch.tensor(mean)))
            torch.save(graphs, Path(root) / "2000_2001_Battery_graphs.pt")
            for mode in (None, "anomaly", "raw", "anomaly_plus_mean"):
                with self.subTest(mode=mode):
                    options = {} if mode is None else {"pressure_mode": mode}
                    store = ForcingGraphStore(root, "Battery", **options)
                    self.assertEqual(store.pressure_mode, mode or "anomaly")
                    fitted = fit_statistics(store, [0, 1])
                    channel = PRESSURE_CHANNELS.get(mode)
                    self.assertUnchangedStatistics(fitted, legacy_statistics(store, [0, 1]), channel)
                    if channel is not None:
                        self.assertExact(fitted["x_scale"][channel], torch.sqrt(torch.tensor(25.0 + 1e-6)))

    @unittest.skipUnless(dist.is_available() and dist.is_gloo_available(), "CPU Gloo is unavailable")
    def test_distributed_global_variance_with_an_empty_rank(self):
        with tempfile.TemporaryDirectory() as root:
            mp.spawn(distributed_statistics_worker, args=(root,), nprocs=3, join=True)
            results = [torch.load(Path(root) / f"rank-{rank}.pt", weights_only=True) for rank in range(3)]
            for mode, channel in PRESSURE_CHANNELS.items():
                with self.subTest(mode=mode):
                    store = make_store(mode, [[101325.0], [101325.0078125] * 4])
                    expected = legacy_statistics(store, [0, 1])
                    expected["x_scale"][channel] = pressure_scale(store, [0, 1])
                    for result in results:
                        self.assertUnchangedStatistics(result[mode], expected)


if __name__ == "__main__":
    unittest.main()
