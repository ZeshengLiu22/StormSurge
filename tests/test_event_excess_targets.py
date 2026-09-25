"""Event-conditional target diagnostics on the saved sampled population."""

import csv
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch_geometric.data import Data

from emulator.data import ForcingGraphStore, fit_loss_thresholds, supervised_targets
from emulator.training.excess_amplitude import physical_excess_target
from tools.audit_event_excess_targets import audit_targets, indices_from_tags, load_export, main


def audit(labels, tau=0.):
    labels = np.asarray(labels, dtype=np.float64)
    return audit_targets(labels, np.arange(labels.size).reshape(labels.shape) * 3600,
                         [f"sample-{i}" for i in range(len(labels))], np.arange(len(labels)), tau, "X", "train")


class EventExcessTargetTests(unittest.TestCase):
    def test_event_conditioned_denominators_horizons_and_window_rows(self):
        with patch("tools.audit_event_excess_targets.physical_excess_target", wraps=physical_excess_target) as shared:
            actual = audit([[0., 0., 0.], [0., .005, .010], [.020, .040, 0.], [.004, .009, .019]])
        shared.assert_called_once()
        s = actual["summary"]
        self.assertEqual((s["event_window_count"], s["target_hour_count"], s["zero_excess_count"], s["positive_excess_count"]), (3, 9, 2, 7))
        self.assertAlmostEqual(s["zero_excess_pct"], 200 / 9)
        self.assertAlmostEqual(s["positive_excess_pct"], 700 / 9)
        self.assertEqual(s["mixed_window_count"], 2)
        self.assertAlmostEqual(s["mixed_window_pct"], 200 / 3)
        self.assertEqual([h["zero_excess_count"] for h in actual["by_horizon"]], [1, 0, 1])
        self.assertEqual([h["target_hour_count"] for h in actual["by_horizon"]], [3, 3, 3])
        self.assertEqual([r["sample_id"] for r in actual["per_event_window"]], [1, 2, 3])
        np.testing.assert_allclose([r["zero_excess_fraction"] for r in actual["per_event_window"]], [1/3, 1/3, 0.])
        self.assertEqual([h["event_window_count"] for h in actual["positive_hours_per_window"]["histogram"]], [0, 0, 2, 1])
        expected = dict(mean=107/7, median=10, q25=7, q75=19.5, q90=28, q95=34, max=40)
        for name, value in expected.items():
            self.assertAlmostEqual(s[f"positive_excess_{name}_mm"], value)
        for limit, count in ((5, 1), (10, 3), (20, 5)):
            self.assertEqual(s[f"positive_excess_lt{limit}mm_count"], count)
            self.assertAlmostEqual(s[f"positive_excess_lt{limit}mm_pct"], 100 * count / 7)

    def test_strict_threshold_and_unrounded_physical_tau(self):
        actual = audit([[.125, .1], [.125, .25]], tau=.125)
        self.assertEqual(actual["summary"]["event_window_count"], 1)
        self.assertEqual(actual["summary"]["zero_excess_count"], 1)
        # Rounding tau to the label's FP32 dtype would incorrectly remove this event.
        tau = np.nextafter(.25, 0.)
        actual = audit([[.25, tau]], tau=tau)
        self.assertEqual(actual["summary"]["positive_excess_count"], 1)
        self.assertEqual(actual["summary"]["zero_excess_count"], 1)

    def test_empty_event_population_is_undefined_not_all_window_zero_pct(self):
        for labels in (np.zeros((2, 3)), np.empty((0, 3))):
            actual = audit(labels)
            s = actual["summary"]
            self.assertEqual(s["target_hour_count"], 0)
            self.assertEqual(s["event_window_count"], 0)
            self.assertIsNone(s["zero_excess_pct"])
            self.assertIsNone(s["mixed_window_pct"])
            self.assertIsNone(s["positive_excess_mean_mm"])
            self.assertEqual(actual["per_event_window"], [])
            json.dumps(actual, allow_nan=False)

    def test_saved_sample_selection_and_overlap_or_missing_rejection(self):
        store = SimpleNamespace(graph_tags=["unused", "train", "test", "val"])
        self.assertEqual(indices_from_tags(store, dict(train=["train"], val=["val"], test=["test"])),
                         dict(train=[1], val=[3], test=[2]))
        with self.assertRaisesRegex(ValueError, "cross-split"):
            indices_from_tags(store, dict(train=["train"], val=["train"], test=["test"]))
        with self.assertRaisesRegex(ValueError, "missing"):
            indices_from_tags(store, dict(train=["missing"], val=["val"], test=["test"]))

    def test_duplicate_target_hours_rejected(self):
        with self.assertRaisesRegex(ValueError, "Duplicate supervised target timestamps"):
            audit_targets([[0., 1.], [1., 0.]], [[0, 3600], [3600, 7200]], ["a", "b"], [0, 1], 0., "X", "val")

    def test_saved_graph_and_export_round_trip_with_fixed_train_threshold(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            graphs, run, output = root / "graphs", root / "run", root / "output"
            graphs.mkdir()
            run.mkdir()
            for year in range(2001, 2006):
                rows = [Data(y=torch.tensor([float(year - 2001), .5, 1.]),
                             x_hist=torch.zeros((2, 1, 1)), center_time=f"{year}-01-01T{6 * i:02}:00:00")
                        for i in range(2)]
                torch.save(rows, graphs / f"{year}_{year+1}_X_graphs.pt")
            store = ForcingGraphStore(graphs, "X", targets_only=True)
            split_config = dict(train_ratio=.6, val_ratio=.2, shuffle_years=False, seed=42)
            splits = store.split(**split_config)
            threshold = fit_loss_thresholds(store, splits["train"], 75.)
            torch.save(dict(station="X", training_config=dict(root_dir=str(graphs), test_root_dir="", history_hours=6),
                            threshold_metadata=threshold, split_config=split_config,
                            split_tags={split: [store.graph_tags[i] for i in indices] for split, indices in splits.items()}),
                       run / "best_overall.pt")
            for split in ("val", "test"):
                labels, times = supervised_targets(store, splits[split])
                arrays = dict(y_true=labels, target_timestamps=times, sample_id=np.asarray(splits[split]),
                              tags=np.asarray([store.graph_tags[i] for i in splits[split]]))
                path = run / f"{split}_predictions_overall.npz"
                np.savez(path, **arrays, station="X", split=split, tau_physical=threshold["tau_physical"])
                wrong = dict(arrays, y_true=labels + 1.)
                with self.assertRaisesRegex(ValueError, "y_true differs"):
                    load_export(path, "X", split, threshold["tau_physical"], wrong)
            main(["--run-dir", str(run), "--output-dir", str(output)])
            report = json.loads((output / "audit.json").read_text())
            audits = report["stations"]["X"]["splits"]
            # TRAIN tau=1; two of its six saved windows are events. VAL/TEST reuse it.
            self.assertEqual(threshold["tau_physical"], 1.)
            for split in ("train", "val", "test"):
                self.assertEqual(audits[split]["summary"]["target_hour_count"], 6)
                self.assertEqual(audits[split]["summary"]["zero_excess_count"], 4)
                self.assertEqual(audits[split]["summary"]["mixed_window_pct"], 100.)
            with (output / "per_event_window.csv").open() as handle:
                self.assertEqual(len(list(csv.DictReader(handle))), 6)
            self.assertIn("zero % / positive %", (output / "REPORT.md").read_text())


if __name__ == "__main__":
    unittest.main()
