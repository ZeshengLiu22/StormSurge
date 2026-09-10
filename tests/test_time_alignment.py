"""Regression checks for timestamp corruption in station preprocessing."""

from contextlib import redirect_stdout
from datetime import date
from io import StringIO
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
import torch

from preprocessing.time_align_unified import (
    build_forcing_time_index,
    compute_last_full_day_for_year,
    process_one_pair,
)


class StationTimeAlignmentTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="pact-time-alignment-")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        (self.root / "dicts").mkdir()
        (self.root / "graphs").mkdir()
        self.csv_path = self.root / "2000_2001_Battery.csv"
        times = pd.date_range("2000-10-25 01:00", "2000-11-01 23:00", freq="h")
        self.frame = pd.DataFrame({
            "time": times,
            "nc": np.arange(len(times), dtype=float),
            "nc_tide": np.zeros(len(times)),
        })

    def build_graphs(self):
        process_one_pair(
            year=2000,
            station="Battery",
            forcing=np.zeros((32, 2, 2, 5), dtype=np.float32),
            t_forcing=build_forcing_time_index(2000, 32),
            edge_index=torch.empty((2, 0), dtype=torch.long),
            H=2,
            W=2,
            version="peryear",
            out_root_fixed=self.root,
            out_root_peryear=self.root,
            csv_dir=self.root,
            last_full_day=date(2000, 11, 1),
        )

    def test_valid_timestamps_preserve_label_alignment(self):
        self.frame.to_csv(self.csv_path, index=False)
        with redirect_stdout(StringIO()):
            cutoff = compute_last_full_day_for_year(2000, ["Battery"], self.root)
            self.build_graphs()
        self.assertEqual(cutoff, date(2000, 11, 1))
        graphs = torch.load(
            next((self.root / "graphs").glob("*.pt")),
            map_location="cpu", weights_only=False,
        )
        self.assertEqual(len(graphs), 4)
        self.assertEqual(graphs[0].center_time, "2000-11-01 00:00:00")
        expected = self.frame.loc[
            self.frame["time"].between("2000-11-01 00:00", "2000-11-01 05:00"), "nc"
        ].to_numpy(dtype=np.float32)
        np.testing.assert_array_equal(graphs[0].y.numpy(), expected)

    def test_shifted_missing_and_duplicate_hours_fail_in_both_entrypoints(self):
        shifted = self.frame.copy()
        shifted["time"] += pd.Timedelta(hours=1)
        missing = self.frame.drop(index=20)
        duplicate = self.frame.copy()
        duplicate.loc[20, "time"] = duplicate.loc[19, "time"]
        invalid = self.frame.copy()
        invalid.loc[20, "time"] = pd.NaT
        for name, frame in [("shifted", shifted), ("missing", missing), ("duplicate", duplicate), ("invalid", invalid)]:
            with self.subTest(name=name):
                frame.to_csv(self.csv_path, index=False)
                with self.assertRaisesRegex(ValueError, "Station timestamp mismatch"):
                    compute_last_full_day_for_year(2000, ["Battery"], self.root)
                with self.assertRaisesRegex(ValueError, "Station timestamp mismatch"):
                    self.build_graphs()
        self.assertEqual(list((self.root / "graphs").iterdir()), [])

    def test_legacy_csv_without_times_keeps_explicit_fallback(self):
        self.frame.drop(columns="time").to_csv(self.csv_path, index=False)
        output = StringIO()
        with redirect_stdout(output):
            self.assertEqual(
                compute_last_full_day_for_year(2000, ["Battery"], self.root),
                date(2000, 11, 1),
            )
            self.build_graphs()
        self.assertIn("legacy assumption", output.getvalue())
        self.assertEqual(len(list((self.root / "graphs").glob("*.pt"))), 1)

    def test_fixed_march15_path_preserves_history_and_output_names(self):
        times = pd.date_range('2000-10-25 01:00', '2001-03-16 23:00', freq='h')
        frame = pd.DataFrame(dict(time=times, nc=np.arange(len(times)), nc_tide=np.zeros(len(times))))
        frame.to_csv(self.csv_path, index=False)
        forcing_times = pd.date_range('2000-10-25', '2001-03-16 18:00', freq='6h')
        forcing = np.arange(len(forcing_times) * 2 * 2 * 5, dtype=np.float32).reshape(-1, 2, 2, 5)
        with redirect_stdout(StringIO()):
            process_one_pair(year=2000, station='Battery', forcing=forcing,
                             t_forcing=build_forcing_time_index(2000, len(forcing_times)),
                             edge_index=torch.empty((2, 0), dtype=torch.long), H=2, W=2, version='fixed',
                             out_root_fixed=self.root, out_root_peryear=self.root / 'unused', csv_dir=self.root)
        stem = '2000_2001_Battery_fixed315_hist48'
        self.assertTrue((self.root / 'dicts' / f'{stem}_dict.npz').exists())
        graphs = torch.load(self.root / 'graphs' / f'{stem}_graphs.pt', weights_only=False)
        self.assertEqual(graphs[0].center_time, '2000-11-01 00:00:00')
        self.assertEqual(graphs[-1].center_time, '2001-03-15 18:00:00')
        self.assertEqual(tuple(graphs[0].x_hist.shape), (9, 4, 5))
        np.testing.assert_array_equal(graphs[0].x_hist, forcing[20:29].reshape(9, 4, 5))
        self.assertEqual(set(graphs[0].keys()), {'x', 'x_hist', 'edge_index', 'y', 'nc', 'nc_tide',
                                                'time_index', 'hour_start', 'max_history_hours',
                                                'grid_H', 'grid_W', 'center_time'})
        expected = frame.loc[frame.time.between('2001-03-15 18:00', '2001-03-15 23:00'), 'nc'].to_numpy(np.float32)
        np.testing.assert_array_equal(graphs[-1].y, expected)


if __name__ == "__main__":
    unittest.main()
