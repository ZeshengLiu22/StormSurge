"""Fresh checkpoint train/infer round trips and exact split populations."""

import contextlib
import io
import itertools
import json
import math
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch_geometric.data import Data

import infer
import train
from emulator.data import ForcingGraphStore, ForcingGraphView
from emulator.training.metrics import METRIC_NAMES, PEAK_METRIC_NAMES


def make_fixture(root, years=5):
    graphs = root / "graphs"
    graphs.mkdir()
    for year in range(2000, 2000 + years):
        items = []
        for index in range(1 + (year - 2000) % 3):
            history = torch.randn(9, 6, 3)
            items.append(Data(x=history[-1], x_hist=history, y=torch.randn(4) + index * .3,
                edge_index=torch.tensor([[0, 1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 0]]), grid_H=2, grid_W=3))
        torch.save(items, graphs / f"{year}_{year + 1}_Battery_fixture_graphs.pt")
    stations = root / "stations"
    stations.mkdir()
    (stations / "Battery.json").write_text(json.dumps(dict(lat=40, lon=-74, elevation_m=2, bathymetry_m=8)))
    return graphs, stations


class PipelineTests(unittest.TestCase):
    def test_custom_graph_pattern_filters_before_loading(self):
        with tempfile.TemporaryDirectory() as temporary:
            graphs, _ = make_fixture(Path(temporary))
            expected = ForcingGraphStore(graphs, "Battery")
            (graphs / "2000_2001_Battery_other_graphs.pt").write_text("excluded by pattern")
            (graphs / "2000_2001_Other_fixture_graphs.pt").write_text("excluded by station")
            selected = ForcingGraphStore(graphs, "Battery", pattern="*_fixture_graphs.pt")
            self.assertEqual(selected.graph_tags, expected.graph_tags)
            self.assertEqual(selected.year_to_indices, expected.year_to_indices)
            for actual, reference in zip(selected.graphs, expected.graphs):
                for key in ("x", "x_hist", "y", "edge_index"):
                    torch.testing.assert_close(actual[key], reference[key], rtol=0, atol=0)
            with self.assertRaisesRegex(ValueError, "No graphs"):
                ForcingGraphStore(graphs, "Battery", pattern="*_missing_graphs.pt")

    def test_original_default_glob_and_custom_suffix_preserve_standard_tags(self):
        with tempfile.TemporaryDirectory() as temporary:
            graphs, _ = make_fixture(Path(temporary))
            standard = ForcingGraphStore(graphs, "Battery")
            graph = standard.graphs[0]
            torch.save([graph], graphs / "1999_2000_Battery_legacygraphs.pt")
            torch.save([graph], graphs / "2005_2006_Battery_manual.pt")
            broad = ForcingGraphStore(graphs, "Battery")
            self.assertEqual(broad.graph_tags, ["1999_2000_Battery_legacygraphs_0", *standard.graph_tags])
            narrow = ForcingGraphStore(graphs, "Battery", pattern="*_graphs.pt")
            self.assertEqual(narrow.graph_tags, standard.graph_tags)
            custom = ForcingGraphStore(graphs, "Battery", pattern="*_manual.pt")
            self.assertEqual(custom.graph_tags, ["2005_2006_Battery_manual_0"])
            torch.testing.assert_close(custom.graphs[0].x, graph.x, rtol=0, atol=0)
            torch.save([graph], graphs / "2006_2007_Battery_graphs.pt")
            no_version = ForcingGraphStore(graphs, "Battery", pattern="2006_*.pt")
            self.assertEqual(no_version.graph_tags, ["2006_2007_Battery_0"])

    def test_model_dimensions_follow_train_when_earliest_year_is_excluded(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            graphs, stations = make_fixture(root)
            excluded_path = graphs / "1979_1980_Battery_excluded_graphs.pt"
            history = torch.randn(9, 6, 2)
            excluded = Data(x=history[-1], x_hist=history, y=torch.randn(2),
                            edge_index=torch.empty(2, 0, dtype=torch.long), grid_H=2, grid_W=3)
            variants = (("baseline", "single", "CNN", 12), ("perceiver3", "dual", "GraphSAGE", 48))
            for model, head, encoder, hours in variants:
                checkpoints, prediction_paths = [], []
                for include_excluded in (False, True):
                    if include_excluded:
                        torch.save([excluded], excluded_path)
                        store = ForcingGraphStore(graphs, "Battery")
                        splits = store.split(future_only=True, future_year_threshold=1999)
                        self.assertNotIn(0, splits["train"])
                        self.assertEqual((store.graphs[0].x.size(-1), store.graphs[0].y.numel()), (2, 2))
                    else:
                        excluded_path.unlink(missing_ok=True)
                    output = root / f"{model}_{include_excluded}"
                    args = ["--root_dir", str(graphs), "--station", "Battery", "--station_json_dir", str(stations),
                            "--output_dir", str(output), "--device", "cpu", "--model", model, "--head_type", head,
                            "--encoder_type", encoder, "--temporal_block", "MLP", "--history_hours", str(hours),
                            "--future_only", "1", "--future_year_threshold", "1999", "--hidden_channels", "16",
                            "--node_read_heads", "2", "--time_read_heads", "2", "--epochs", "1", "--warmup_epochs", "0",
                            "--batch_size", "2", "--num_workers", "0", "--x_aug", "0"]
                    with contextlib.redirect_stdout(io.StringIO()):
                        train.main(args)
                    checkpoint_path = next(output.glob("best_*.pth"))
                    checkpoint = torch.load(checkpoint_path, weights_only=False)
                    self.assertEqual((checkpoint["model_config"]["in_channels"], checkpoint["model_config"]["out_channels"]), (3, 4))
                    self.assertTrue(all(not tag.startswith("1979_") for tags in checkpoint["split_tags"].values() for tag in tags))
                    checkpoints.append(checkpoint)
                    prediction_paths.append(next(output.glob("test_preds_*.npz")))
                    if include_excluded:
                        inferred = root / f"infer_{model}"
                        with contextlib.redirect_stdout(io.StringIO()):
                            infer.main(["--ckpt", str(checkpoint_path), "--root_dir", str(graphs), "--out_dir", str(inferred),
                                        "--device", "cpu", "--batch_size", "2", "--num_workers", "0", "--save_npz"])
                        with np.load(prediction_paths[-1]) as expected, np.load(inferred / "predictions.npz") as actual:
                            np.testing.assert_array_equal(actual["tags"], expected["tags"])
                            np.testing.assert_array_equal(actual["y_pred"], expected["y_pred"])
                for field in ("model_state", "normalization"):
                    for key in checkpoints[0][field]:
                        torch.testing.assert_close(checkpoints[0][field][key], checkpoints[1][field][key], rtol=0, atol=0)
                with np.load(prediction_paths[0]) as before, np.load(prediction_paths[1]) as after:
                    np.testing.assert_array_equal(before["tags"], after["tags"])
                    np.testing.assert_array_equal(before["y_pred"], after["y_pred"])

    def test_metadata_aliases_and_independent_switches_survive_train_and_infer(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            graphs, stations = make_fixture(root)
            canonical = json.loads((stations / "Battery.json").read_text())
            (stations / "Battery.json").unlink()
            external = root / "external"
            external.mkdir()
            for graph in graphs.glob("*.pt"):
                shutil.copyfile(graph, external / graph.name.replace("Battery", "Lewes"))
            for elevation, bathymetry in itertools.product((0, 1), repeat=2):
                with self.subTest(elevation=elevation, bathymetry=bathymetry):
                    aliases = dict(Latitude=40, Longitude=-74, elev_m=2 if elevation else "invalid",
                                   bathymetry_m=8 if bathymetry else "invalid")
                    (stations / "battery.json").write_text(json.dumps(aliases))
                    output = root / f"train_{elevation}_{bathymetry}"
                    with contextlib.redirect_stdout(io.StringIO()):
                        train.main(["--root_dir", str(graphs), "--station", "Battery", "--station_json_dir", str(stations),
                                    "--output_dir", str(output), "--device", "cpu", "--model", "perceiver3", "--head_type", "single",
                                    "--temporal_block", "MLP", "--history_hours", "0", "--hidden_channels", "16",
                                    "--epochs", "1", "--warmup_epochs", "0", "--batch_size", "2", "--num_workers", "0",
                                    "--x_aug", "0", "--use_site_elevation", str(elevation), "--use_bathymetry", str(bathymetry)])
                    checkpoint_path = next(output.glob("best_*.pth"))
                    checkpoint = torch.load(checkpoint_path, weights_only=False)
                    self.assertEqual(checkpoint["training_config"]["use_site_elevation"], elevation)
                    self.assertEqual(checkpoint["training_config"]["use_bathymetry"], bathymetry)
                    lat, lon = math.radians(40), math.radians(-74)
                    expected = ([40 / 90, -74 / 180] + ([.2] if elevation else [])
                                + [math.sin(lat), math.cos(lat), math.sin(lon), math.cos(lon)] + ([.8] if bathymetry else []))
                    self.assertEqual(checkpoint["model_config"]["station_feat_dim"], 6 + elevation + bathymetry)
                    torch.testing.assert_close(checkpoint["station_feat"], torch.tensor(expected), rtol=0, atol=0)
                    (stations / "LEWES.json").write_text(json.dumps(aliases))
                    infer_args = ["--ckpt", str(checkpoint_path), "--root_dir", str(graphs), "--test_root_dir", str(external), "--station", "Lewes",
                                  "--station_json_dir", str(stations), "--device", "cpu", "--batch_size", "2", "--num_workers", "0", "--save_npz"]
                    alias_output = root / f"alias_{elevation}_{bathymetry}"
                    canonical_output = root / f"canonical_{elevation}_{bathymetry}"
                    with contextlib.redirect_stdout(io.StringIO()):
                        infer.main([*infer_args, "--out_dir", str(alias_output)])
                        (stations / "Lewes.json").write_text(json.dumps(canonical))
                        infer.main([*infer_args, "--out_dir", str(canonical_output)])
                    (stations / "Lewes.json").unlink()
                    with np.load(alias_output / "predictions.npz") as alias, np.load(canonical_output / "predictions.npz") as reference:
                        np.testing.assert_array_equal(alias["tags"], reference["tags"])
                        np.testing.assert_array_equal(alias["y_pred"], reference["y_pred"])
                    for flag, saved in (("use_site_elevation", elevation), ("use_bathymetry", bathymetry)):
                        with self.assertRaisesRegex(ValueError, f"--{flag}=.*does not match checkpoint"):
                            infer.main([*infer_args, "--out_dir", str(root / "mismatch"), f"--{flag}", str(1 - saved)])

    def test_fresh_train_checkpoint_and_inference_roundtrip(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            graphs, stations = make_fixture(root)
            variants = [("baseline", "single", "CNN", "MLP", 0), ("baseline", "single", "GraphSAGE", "LSTM", 12),
                        ("perceiver3", "single", "CNN", "MLP", 0), ("perceiver3", "dual", "GraphSAGE", "Transformer", 48)]
            for index, (model, head, encoder, temporal, history) in enumerate(variants):
                output = root / f"train_{index}"
                args = ["--root_dir", str(graphs), "--station", "Battery", "--station_json_dir", str(stations),
                        "--output_dir", str(output), "--device", "cpu", "--model", model, "--head_type", f" {head.upper()} ",
                        "--encoder_type", encoder, "--temporal_block", f" {temporal.lower()} ", "--history_hours", str(history),
                        "--hidden_channels", "16", "--node_read_heads", "2", "--time_read_heads", "2",
                        "--dropout", ".13", "--head_dropout", ".27", "--transformer_dropout", ".19",
                        "--transformer_layers", "1", "--batch_size", "2", "--grad_accum_steps", "2",
                        "--epochs", "2", "--warmup_epochs", "0", "--use_bathymetry", "1",
                        "--num_workers", "0", "--run_tag", "roundtrip", "--x_aug", "0", "--x_norm", "zscore"]
                console = io.StringIO()
                with contextlib.redirect_stdout(console), patch.object(train, "run_epoch", wraps=train.run_epoch) as observed:
                    train.main(args)
                # Two epochs of train/Val plus one final Test; reporting adds no evaluation pass.
                self.assertEqual(observed.call_count, 5)
                lines = console.getvalue().splitlines()
                # Best-checkpoint messages were added to production after this test.
                best_lines = [line.split('] ', 1)[1] for line in lines if '] [Best] ' in line]
                self.assertEqual(len(lines) - len(best_lines), 4)
                for line in console.getvalue().splitlines():
                    self.assertRegex(line, r"^\[\d{4}-\d{2}-\d{2}\|\d{2}:\d{2}:\d{2}\]")
                self.assertIn("Wall time:", console.getvalue().splitlines()[-1])
                logs = [json.loads(line) for line in next(output.glob("metrics_*.jsonl")).read_text().splitlines()]
                running_best, expected_best_lines = float('inf'), []
                for record in logs:
                    self.assertEqual(set(record), {"epoch", "train", "val"})
                    for part in ("train", "val"):
                        self.assertEqual(tuple(record[part]), METRIC_NAMES + PEAK_METRIC_NAMES)
                    if record['val']['rmse_all'] < running_best:
                        running_best = record['val']['rmse_all']
                        expected_best_lines.append(f"[Best] Epoch {record['epoch']:03d}/2 | "
                                                   f"{train.format_metrics('Val', record['val'])}")
                self.assertEqual(best_lines, expected_best_lines)
                checkpoint = torch.load(next(output.glob("best_*.pth")), weights_only=False)
                summary = json.loads(next(output.glob("summary_*.json")).read_text())
                self.assertEqual(summary["val"], checkpoint["val"])
                self.assertEqual(summary["val"], logs[checkpoint["epoch"] - 1]["val"])
                self.assertEqual(summary["best_val_rmse"], summary["val"]["rmse_all"])
                self.assertEqual(summary["best_epoch"], checkpoint["epoch"])
                self.assertEqual(tuple(summary["test"]), METRIC_NAMES + PEAK_METRIC_NAMES)
                final_metrics = console.getvalue().splitlines()[-2]
                self.assertIn(f'Best epoch {checkpoint["epoch"]:03d}', final_metrics)
                self.assertIn(train.format_metrics("Val", summary["val"]), final_metrics)
                self.assertIn(train.format_metrics("Test", summary["test"]), final_metrics)
                self.assertNotIn("stable_arch", checkpoint["training_config"])
                self.assertNotIn("STABLE_ARCH=", (output / "config_used.sh").read_text())
                self.assertEqual(checkpoint["model_config"]["history_steps"], history // 6)
                self.assertEqual(checkpoint["model_config"]["station_feat_dim"], 8 if model == "perceiver3" else 0)
                self.assertEqual(checkpoint["model_config"]["hidden_channels"], 16)
                self.assertEqual(checkpoint["model_config"]["dropout"], .13)
                self.assertEqual(checkpoint["model_config"]["head_dropout"], .27)
                self.assertEqual(checkpoint["model_config"]["temporal_dropout"], .19)
                # Inference uses saved features; station JSON is not an inference dependency.
                inferred = root / f"infer_{index}"
                with contextlib.redirect_stdout(io.StringIO()):
                    infer.main(["--ckpt", str(next(output.glob("best_*.pth"))), "--root_dir", str(graphs),
                                "--head_type", f"\t{head}\t", "--temporal_block", f"\t{temporal}\t",
                                "--out_dir", str(inferred), "--device", "cpu", "--batch_size", "2", "--save_npz", "--num_workers", "0"])
                with np.load(next(output.glob("test_preds_*.npz"))) as trained, np.load(inferred / "predictions.npz") as restored:
                    np.testing.assert_array_equal(trained["tags"], restored["tags"])
                    np.testing.assert_allclose(trained["y_pred"], restored["y_pred"], rtol=1e-6, atol=1e-6)
                with self.assertRaises(FileExistsError), contextlib.redirect_stdout(io.StringIO()):
                    train.main(args)

    def test_station_filter_precedes_loading_and_split_is_by_year(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            graphs, _ = make_fixture(root, 8)
            (graphs / "2000_2001_Other_fixture_graphs.pt").write_text("invalid unrelated file")
            store = ForcingGraphStore(graphs, "Battery")
            splits = store.split(train_ratio=.5, val_ratio=.25)
            self.assertEqual(len(splits["train"]) + len(splits["val"]) + len(splits["test"]), len(store.graphs))
            years = [{store.graph_tags[i].split("_")[0] for i in indices} for indices in splits.values()]
            self.assertFalse(years[0] & years[1] or years[0] & years[2] or years[1] & years[2])
            with self.assertRaisesRegex(ValueError, "No graphs"):
                ForcingGraphStore(graphs, "Missing")
            with self.assertRaisesRegex(ValueError, "need 10"):
                ForcingGraphView(store, splits["train"], 9)

    def test_original_flags_and_defaults(self):
        args = train.parse_args([])
        self.assertEqual(args.model, "baseline")
        self.assertEqual(args.loss_mode, "mse")
        self.assertEqual(args.lr, .003)
        self.assertEqual(args.hidden_channels, 64)
        self.assertEqual((args.dropout, args.head_dropout, args.transformer_dropout), (.05, 0., .05))
        self.assertEqual((args.use_site_elevation, args.use_bathymetry), (1, 0))
        self.assertFalse(args.amp or args.tf32 or args.pin_memory or args.persistent_workers)
        args = train.parse_args(["--model", "perceiver3", "--filter", "Battery", "--temporal_block", "attn",
                                 "--amp", "--tf32", "--pin_memory", "--persistent_workers",
                                 "--shuffle_split_years", "true", "--future_only_years", "yes"])
        self.assertEqual(args.station, "Battery")
        self.assertEqual(args.temporal_block, "Transformer")
        self.assertEqual(args.dual_mode, "exceedance")
        self.assertTrue(args.amp and args.tf32 and args.pin_memory and args.persistent_workers)
        self.assertTrue(args.shuffle_years and args.future_only)

    def test_warmup_cosine_matches_previous_schedule_without_deprecated_calls(self):
        import warnings
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            graphs, stations = make_fixture(root)
            for total, warm in ((300, 5), (3, 1), (2, 0)):
                args = ['--root_dir', str(graphs), '--station', 'Battery', '--station_json_dir', str(stations),
                        '--output_dir', str(root / f'schedule_{total}'), '--device', 'cpu', '--head_type', 'single',
                        '--epochs', str(total), '--warmup_epochs', str(warm), '--hidden_channels', '16', '--lr', '.005', '--num_workers', '0']
                factory = torch.optim.lr_scheduler.LambdaLR
                with patch.object(train.torch.optim.lr_scheduler, 'LambdaLR', wraps=factory) as observed:
                    with patch.object(train, 'run_epoch', side_effect=StopIteration), self.assertRaises(StopIteration):
                        train.main(args)
                    factor = observed.call_args.args[1]
                optimizer = torch.optim.Adam([torch.nn.Parameter(torch.ones(1))], lr=.005)
                if warm:
                    first = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=.1, total_iters=warm)
                    second = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total - warm, eta_min=1e-6)
                    reference = torch.optim.lr_scheduler.SequentialLR(optimizer, [first, second], milestones=[warm])
                else:
                    reference = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total, eta_min=1e-6)
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore', UserWarning)
                    for epoch in range(total + 1):
                        self.assertAlmostEqual(.005 * factor(epoch), optimizer.param_groups[0]['lr'], places=14)
                        optimizer.step()
                        reference.step()
