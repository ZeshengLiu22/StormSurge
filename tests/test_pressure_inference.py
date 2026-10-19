"""CPU checkpoint round trips for pressure representations and legacy Single heads."""

import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch_geometric.data import Data

import infer
import train
from emulator.models import build_model, model_config_from_checkpoint


MODES = ("anomaly", "raw", "anomaly_plus_mean")


def write_graphs(root, mean_offset=0):
    root.mkdir()
    generator = torch.Generator().manual_seed(715)
    for year in range(2000, 2003):
        graphs = []
        for index in range(2):
            history = torch.randn(9, 6, 5, generator=generator)
            mean = torch.arange(9, dtype=torch.float32) * 2 + 1000 + (year - 2000) * 20 + mean_offset
            graphs.append(Data(
                x=history[-1], x_hist=history, p_mean_hist=mean, p_mean_curr=mean[-1],
                edge_index=torch.tensor([[0, 1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 0]]),
                y=torch.tensor([-0.2, 0.1, 0.3, 1.0 + (year - 2000) * 0.1 + index * 0.05]),
                center_time=f"{year}-11-01 {index * 6:02d}:00:00", grid_H=2, grid_W=3,
            ))
        torch.save(graphs, root / f"{year}_{year + 1}_Battery_fixture_graphs.pt")
    return root


class PressureInferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_checkpoint_mode_and_cli_assertion(self):
        self.assertEqual(infer.resolve_pressure_mode({"training_config": {}}), "anomaly")
        self.assertIsNone(infer.parse_args(["--ckpt", "legacy.pt", "--root_dir", "."]).pressure_mode)
        self.assertEqual(train.parse_args([]).pressure_mode, "anomaly")
        for mode in MODES:
            checkpoint = {"training_config": {"pressure_mode": mode}}
            self.assertEqual(infer.resolve_pressure_mode(checkpoint), mode)
            self.assertEqual(infer.resolve_pressure_mode(checkpoint, mode), mode)
            for other in set(MODES) - {mode}:
                with self.assertRaisesRegex(ValueError, "does not match checkpoint"):
                    infer.resolve_pressure_mode(checkpoint, other)
        with self.assertRaisesRegex(ValueError, "Unsupported checkpoint pressure_mode"):
            infer.resolve_pressure_mode({"training_config": {"pressure_mode": "typo"}})

    def test_graph_and_normalization_dimension_checks(self):
        graph = Data(x=torch.zeros(6, 6), x_hist=torch.zeros(9, 6, 6))
        store = SimpleNamespace(graphs=[graph], graph_tags=["fixture"])
        config = SimpleNamespace(in_channels=6)
        stats = {"x_center": torch.zeros(6), "x_scale": torch.ones(6)}
        infer._validate_input_dimensions(store, [0], config, stats, "anomaly_plus_mean")
        for name in stats:
            for bad in (torch.ones(5), torch.ones(1, 6), None):
                with self.subTest(name=name, bad=bad), self.assertRaisesRegex(ValueError, name):
                    infer._validate_input_dimensions(store, [0], config, {**stats, name: bad}, "anomaly_plus_mean")
        for name in ("x", "x_hist"):
            malformed = graph.clone()
            malformed[name] = malformed[name][..., :5]
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "fixture.*" + name):
                infer._validate_input_dimensions(SimpleNamespace(graphs=[malformed], graph_tags=["fixture"]),
                                                 [0], config, stats, "anomaly_plus_mean")

    def test_train_and_infer_cpu_roundtrip_including_external_graphs(self):
        for mode in MODES:
            for encoder in ("GraphSAGE", "CNN"):
                with self.subTest(mode=mode, encoder=encoder), tempfile.TemporaryDirectory() as temporary:
                    root = Path(temporary)
                    graphs = write_graphs(root / "source")
                    external = write_graphs(root / "target", mean_offset=150)
                    output = root / "run"
                    args = [
                        "--root_dir", str(graphs), "--station", "Battery", "--output_dir", str(output),
                        "--model", "perceiver3", "--encoder_type", encoder, "--temporal_block", "MLP",
                        "--history_hours", "12", "--hidden_channels", "8", "--node_read_heads", "2",
                        "--time_read_heads", "2", "--cnn_intermediate_channel", "4", "--epochs", "1",
                        "--warmup_epochs", "0", "--batch_size", "2", "--num_workers", "0",
                        "--device", "cpu", "--use_station_meta", "0", "--x_aug", "0",
                    ]
                    # Omission exercises the actual backward-compatible training default.
                    if mode != "anomaly":
                        args += ["--pressure_mode", mode, "--test_root_dir", str(external)]
                    with contextlib.redirect_stdout(io.StringIO()):
                        train.main(args)
                    checkpoint_path = output / "best.pt"
                    checkpoint = torch.load(checkpoint_path, weights_only=False)
                    channels = 6 if mode == "anomaly_plus_mean" else 5
                    self.assertEqual(checkpoint["training_config"]["pressure_mode"], mode)
                    self.assertEqual(checkpoint["model_config"]["in_channels"], channels)
                    self.assertNotIn("pressure_mode", checkpoint["model_config"])
                    for name in ("x_center", "x_scale"):
                        self.assertEqual(tuple(checkpoint["normalization"][name].shape), (channels,))
                    original_config = model_config_from_checkpoint(checkpoint)
                    if mode == "anomaly":
                        # Representative old Single-head format: no preprocessing mode,
                        # with architecture metadata handled by the existing contract.
                        del checkpoint["training_config"]["pressure_mode"]
                        checkpoint["model_config"]["head_type"] = "single"
                        checkpoint_path = root / "legacy_single.pt"
                        torch.save(checkpoint, checkpoint_path)
                    restored_config = model_config_from_checkpoint(checkpoint)
                    self.assertEqual(restored_config, original_config)
                    restored = build_model(restored_config)
                    loaded = restored.load_state_dict(checkpoint["model_state"], strict=True)
                    self.assertEqual(loaded.missing_keys, [])
                    self.assertEqual(loaded.unexpected_keys, [])
                    dual = copy.deepcopy(checkpoint)
                    dual["model_config"]["head_type"] = "dual"
                    with self.assertRaisesRegex(ValueError, "[Dd]ual"):
                        model_config_from_checkpoint(dual)
                    inferred = root / "inference"
                    inference_args = [
                        "--ckpt", str(checkpoint_path), "--root_dir", str(graphs), "--out_dir", str(inferred),
                        "--device", "cpu", "--batch_size", "2", "--num_workers", "0", "--save_npz",
                    ]
                    if mode != "anomaly":
                        inference_args += ["--test_root_dir", str(external)]
                    # Any accidental target-statistics fit is forbidden, even with
                    # a target pressure distribution different from source TRAIN.
                    with contextlib.redirect_stdout(io.StringIO()), patch(
                        "emulator.data.stats.fit_statistics", side_effect=AssertionError("target statistics refit")
                    ), patch("emulator.data.fit_statistics", side_effect=AssertionError("target statistics refit")):
                        infer.main(inference_args)
                    report = json.loads((inferred / "metrics.json").read_text())
                    self.assertEqual(report["pressure_mode"], mode)
                    self.assertEqual(report["normalization_origin"], "source_checkpoint_train")
                    with np.load(next(output.glob("test_preds_*.npz"))) as expected, np.load(inferred / "predictions.npz") as actual:
                        for name in ("y_pred", "y_true", "tags", "target_timestamps"):
                            np.testing.assert_array_equal(actual[name], expected[name])
                        self.assertEqual(str(actual["pressure_mode"]), mode)
                    wrong = "raw" if mode == "anomaly" else "anomaly"
                    with self.assertRaisesRegex(ValueError, "does not match checkpoint"), contextlib.redirect_stdout(io.StringIO()):
                        infer.main(inference_args + ["--pressure_mode", wrong])
                    if encoder == "GraphSAGE":
                        malformed = copy.deepcopy(checkpoint)
                        malformed["normalization"]["x_center"] = None
                        bad_path = root / "invalid_statistics.pt"
                        torch.save(malformed, bad_path)
                        bad_args = inference_args.copy()
                        bad_args[bad_args.index("--ckpt") + 1] = str(bad_path)
                        with self.assertRaisesRegex(ValueError, "normalization x_center"), contextlib.redirect_stdout(io.StringIO()):
                            infer.main(bad_args)


if __name__ == "__main__":
    unittest.main()
