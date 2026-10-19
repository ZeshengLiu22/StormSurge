"""Synthetic source/transfer inference and replay for the seven shallow baselines."""

import contextlib
import copy
from dataclasses import asdict
import io
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch_geometric.data import Batch, Data

import infer
from emulator.data import (
    ForcingGraphStore,
    ForcingGraphView,
    fit_loss_thresholds,
    fit_statistics,
    normalize_inputs,
)
from emulator.models import (
    BASELINE_ARCHITECTURES,
    NEW_BASELINE_MODELS,
    STATIC_BASELINE_MODELS,
    ModelConfig,
    build_model,
    count_model_parameters,
)


REPO = Path(__file__).resolve().parents[1]
MODES = ("anomaly", "raw", "anomaly_plus_mean")


def write_graphs(root, height, width, *, mean_offset=0):
    """Write two years of chronological 48-hour forcing and hourly targets."""
    root.mkdir()
    generator = torch.Generator().manual_seed(715)
    for year in (2000, 2001):
        graphs = []
        for index in range(2):
            history = torch.randn(9, height * width, 5, generator=generator)
            mean = torch.arange(9, dtype=torch.float32) * 2 + 1000 + mean_offset
            graphs.append(Data(
                x=history[-1], x_hist=history, p_mean_hist=mean, p_mean_curr=mean[-1],
                y=torch.tensor([-0.2, 0.1, 0.3, 1.0 + index * 0.1]),
                center_time=f"{year}-11-01 {index * 6:02d}:00:00",
                grid_H=height, grid_W=width,
            ))
        torch.save(graphs, root / f"{year}_{year + 1}_Battery_fixture_graphs.pt")
    return root


def checkpoint_fixture(source, name, mode="anomaly", pooling="adaptive"):
    """Construct a small checkpoint using actual source-TRAIN statistics only."""
    store = ForcingGraphStore(source, "Battery", pressure_mode=mode)
    training_indices = store.year_to_indices["2000_2001"]
    history_steps = 0 if name in STATIC_BASELINE_MODELS else 4
    config = ModelConfig(
        in_channels=store.graphs[0].x.size(-1), out_channels=4, hidden_channels=8,
        model=name, history_steps=history_steps, baseline_stem_channels=4,
        baseline_pooling=pooling, dropout=0.05,
    )
    model = build_model(config).eval()
    checkpoint = dict(
        model_config=asdict(config), model_state=model.state_dict(),
        training_config=dict(
            model=name, seed=42, deterministic=False, pressure_mode=mode,
            use_site_elevation=0, use_bathymetry=0, x_clip=4.0, x_norm="zscore",
            train_ratio=0.5, val_ratio=0.25, shuffle_years=False, future_only=False,
            future_year_threshold=2030,
        ),
        station="Battery", station_feat=None,
        split_tags={
            "train": [store.graph_tags[i] for i in training_indices],
            "val": [],
            "test": [store.graph_tags[i] for i in store.year_to_indices["2001_2002"]],
        },
        normalization=fit_statistics(store, training_indices),
        threshold_metadata=fit_loss_thresholds(store, training_indices),
    )
    return checkpoint, config, model


class BaselineInferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Keep the complete synthetic inference matrix inexpensive on CPU."""
        torch.set_num_threads(1)

    def test_exact_model_assertions_ignore_unrelated_legacy_options(self):
        """Model keys determine new architectures; legacy assertions still apply."""
        for name in NEW_BASELINE_MODELS:
            config = ModelConfig(
                5, 4, 8, model=name,
                history_steps=0 if name in STATIC_BASELINE_MODELS else 4,
            )
            args = infer.parse_args([
                "--ckpt", "fixture.pt", "--root_dir", ".", "--model", name,
                "--encoder_type", "CNN", "--temporal_block", "Transformer",
                "--cnn_intermediate_channel", "999", "--use_site_elevation", "1",
            ])
            self.assertEqual(infer._validate_checkpoint_options(args, config, {}), name)
            args.model = "baseline"
            with self.assertRaisesRegex(ValueError, "--model"):
                infer._validate_checkpoint_options(args, config, {})
            args.model = name
            args.history_hours = config.history_steps * 6 + 6
            with self.assertRaisesRegex(ValueError, "history_hours"):
                infer._validate_checkpoint_options(args, config, {})

        training = dict(use_site_elevation=1, use_bathymetry=0)
        for name, public_name in (("pact", "perceiver3"), ("baseline", "baseline")):
            config = ModelConfig(5, 4, 8, model=name, encoder_type="CNN")
            args = infer.parse_args(["--ckpt", "fixture.pt", "--root_dir", "."])
            self.assertEqual(infer._validate_checkpoint_options(args, config, training), public_name)
            args.encoder_type = "GraphSAGE"
            with self.assertRaisesRegex(ValueError, "encoder_type"):
                infer._validate_checkpoint_options(args, config, training)
            args.encoder_type = "CNN"
            args.cnn_intermediate_channel = config.cnn_intermediate_channel + 1
            with self.assertRaisesRegex(ValueError, "cnn_intermediate_channel"):
                infer._validate_checkpoint_options(args, config, training)
            args.cnn_intermediate_channel = None
            args.temporal_block = "GRU"
            if name == "pact":
                with self.assertRaisesRegex(ValueError, "temporal_block"):
                    infer._validate_checkpoint_options(args, config, training)
            else:
                self.assertEqual(infer._validate_checkpoint_options(args, config, training), public_name)

    def test_all_baselines_source_and_cross_resolution_pressure_inference(self):
        """Verify predictions, metadata, year filters, and frozen source statistics."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = write_graphs(root / "source", 2, 3)
            target = write_graphs(root / "target", 4, 5, mean_offset=150)
            # Selective external inference must not deserialize unrelated years.
            (target / "1999_2000_Battery_unused_graphs.pt").write_bytes(b"not a graph")
            for name in NEW_BASELINE_MODELS:
                for mode in MODES:
                    for pooling in ("adaptive", "global_mean"):
                        with self.subTest(model=name, pressure=mode, pooling=pooling):
                            checkpoint, config, model = checkpoint_fixture(source, name, mode, pooling)
                            checkpoint_path = root / "checkpoint.pt"
                            torch.save(checkpoint, checkpoint_path)
                            for external in (False, True):
                                out = root / f"{name}_{mode}_{pooling}_{external}"
                                evaluation_root = target if external else source
                                options = [
                                    "--ckpt", str(checkpoint_path), "--root_dir", str(source),
                                    "--out_dir", str(out), "--device", "cpu",
                                    "--batch_size", "2", "--num_workers", "0", "--save_npz",
                                    "--years", "2001_2002", "--strict_years",
                                    "--encoder_type", "GraphSAGE", "--temporal_block", "Transformer",
                                    "--source_name", "SourceFixture", "--target_name", "TargetFixture",
                                ]
                                if external:
                                    options += ["--test_root_dir", str(target)]
                                else:
                                    options += ["--model", name]
                                with contextlib.redirect_stdout(io.StringIO()), patch(
                                    "emulator.data.fit_statistics", side_effect=AssertionError("target refit")
                                ), patch(
                                    "emulator.data.stats.fit_statistics", side_effect=AssertionError("target refit")
                                ), patch(
                                    "emulator.data.fit_loss_thresholds", side_effect=AssertionError("target threshold")
                                ), patch.object(infer, "run_epoch", wraps=infer.run_epoch) as run:
                                    infer.main(options)
                                self.assertEqual(run.call_count, 1)
                                for key, value in checkpoint["normalization"].items():
                                    torch.testing.assert_close(run.call_args.args[3][key], value, rtol=0, atol=0)
                                self.assertEqual(
                                    run.call_args.kwargs["tau_physical"],
                                    checkpoint["threshold_metadata"]["tau_physical"],
                                )
                                store = ForcingGraphStore(
                                    evaluation_root, "Battery", pressure_mode=mode, year_filter=["2001_2002"]
                                )
                                view = ForcingGraphView(store, list(range(len(store.graphs))), config.history_steps)
                                batch = Batch.from_data_list([view[i] for i in range(len(view))])
                                normalize_inputs(batch, checkpoint["normalization"], 4.0)
                                with torch.no_grad():
                                    expected = model(batch).prediction * checkpoint["normalization"]["y_std"]
                                    expected += checkpoint["normalization"]["y_mean"]
                                metrics = json.loads((out / "metrics.json").read_text())
                                report_path = out / f"metrics_per_year_TargetFixture_Battery_{name}.json"
                                report = json.loads(report_path.read_text())
                                spatial, temporal = BASELINE_ARCHITECTURES[name]
                                saved_args = json.loads((out / "config.json").read_text())
                                self.assertEqual(saved_args["model"], name)
                                self.assertEqual(saved_args["encoder_type"], spatial)
                                self.assertEqual(saved_args["temporal_block"], temporal)
                                self.assertIsNone(saved_args["cnn_intermediate_channel"])
                                self.assertEqual(saved_args["history_hours"], config.history_steps * 6)
                                self.assertEqual(saved_args["effective_spatial_encoder"], spatial)
                                self.assertEqual(saved_args["effective_temporal_module"], temporal)
                                self.assertEqual(saved_args["supplied_architecture_assertions"]["encoder_type"], "GraphSAGE")
                                self.assertEqual(saved_args["supplied_architecture_assertions"]["temporal_block"], "Transformer")
                                self.assertEqual(report["inference_args"], saved_args)
                                expected_metadata = dict(
                                    model=name, encoder_type=spatial, temporal_block=temporal,
                                    spatial_encoder=spatial, temporal_module=temporal,
                                    baseline_pooling=pooling, baseline_stem_channels=4,
                                    history_hours=config.history_steps * 6, in_channels=config.in_channels,
                                    pressure_mode=mode, parameter_count=count_model_parameters(model)["total"],
                                    source_name="SourceFixture", target_name="TargetFixture",
                                    source_root=str(source), target_root=str(evaluation_root),
                                    checkpoint=str(checkpoint_path), strict_years=True,
                                    requested_years=["2001_2002"], evaluated_years=["2001_2002"],
                                    normalization_origin="source_checkpoint_train",
                                    evaluation_threshold_origin="source_checkpoint_train",
                                )
                                if pooling == "adaptive":
                                    expected_metadata["baseline_pool_grid"] = 4
                                with np.load(out / "predictions.npz", allow_pickle=False) as arrays:
                                    np.testing.assert_allclose(arrays["y_pred"], expected.numpy(), rtol=1e-6, atol=1e-7)
                                    self.assertEqual(arrays["y_pred"].shape, (2, 4))
                                    for document in (metrics, report, arrays):
                                        for key, value in expected_metadata.items():
                                            np.testing.assert_equal(document[key], value, err_msg=key)
                                        if pooling == "global_mean":
                                            self.assertNotIn("baseline_pool_grid", document)
                                self.assertIsNone(report["cnn_intermediate_channel"])

    def test_strict_weights_pressure_and_year_failures(self):
        """Invalid overrides and incomplete weights fail before writing predictions."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = write_graphs(root / "source", 2, 3)
            checkpoint, _, _ = checkpoint_fixture(source, "baseline_cnn_gru", "anomaly_plus_mean")
            path = root / "model.pt"
            torch.save(checkpoint, path)
            options = [
                "--ckpt", str(path), "--root_dir", str(source),
                "--out_dir", str(root / "output"), "--device", "cpu",
            ]
            for extra, message in (
                (["--pressure_mode", "raw"], "does not match checkpoint"),
                (["--model", "baseline"], "--model"),
                (["--history_hours", "12"], "history_hours"),
                (["--years", "2099_2100", "--strict_years"], "missing from target"),
            ):
                with self.subTest(options=extra), contextlib.redirect_stdout(io.StringIO()):
                    with self.assertRaisesRegex(ValueError, message):
                        infer.main(options + extra)
            self.assertFalse((root / "output").exists())
            malformed = copy.deepcopy(checkpoint)
            del malformed["model_state"][next(iter(malformed["model_state"]))]
            torch.save(malformed, path)
            with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(RuntimeError, "Missing key"):
                infer.main(options)
            self.assertFalse((root / "output").exists())

    def test_launcher_exact_identities_effective_labels_and_replay(self):
        """All baseline launchers omit irrelevant assertions and replay saved history."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root / "placeholder.pt"
            checkpoint.touch()
            capture = root / "capture.py"
            capture.write_text(
                "import json, pathlib, sys\n"
                "out = pathlib.Path(sys.argv[sys.argv.index('--out_dir') + 1])\n"
                "(out / 'argv.json').write_text(json.dumps(sys.argv[1:]))\n"
            )
            environment = dict(os.environ, DRY_RUN="1", USE_TMUX="0")
            environment.pop("PRESSURE_MODE", None)
            environment.pop("HISTORY_HOURS", None)
            for name in NEW_BASELINE_MODELS:
                result_root = root / name
                config = root / f"{name}.sh"
                values = dict(
                    CKPT_PATH=checkpoint, PYTHON_BIN=sys.executable, INFER_PY=capture,
                    INFERENCE_RESULTS_ROOT=result_root, ROOT_DIR="./source/graphs",
                    MODEL=name, USE_TMUX=0, TORCH_GPU_PROBE=0,
                    ENCODER_TYPE="GraphSAGE", TEMPORAL_BLOCK="Transformer",
                    CNN_INTERMEDIATE_CHANNEL=999,
                )
                config.write_text("".join(f"{key}={shlex.quote(str(value))}\n" for key, value in values.items()))
                command = ["bash", str(REPO / "infer.sh"), str(config)]
                dry = subprocess.run(command, cwd=REPO, env=environment, capture_output=True, text=True)
                self.assertEqual(dry.returncode, 0, dry.stdout + dry.stderr)
                spatial, temporal = BASELINE_ARCHITECTURES[name]
                self.assertIn(f"Encoder:       {spatial}", dry.stdout)
                self.assertIn(f"Temporal:      {temporal}", dry.stdout)
                self.assertIn("History hours: <checkpoint>", dry.stdout)
                self.assertFalse(result_root.exists())
                result = subprocess.run(
                    command, cwd=REPO, env=environment | {"DRY_RUN": "0"}, capture_output=True, text=True
                )
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                captured = next(result_root.rglob("argv.json"))
                original = json.loads(captured.read_text())
                args = infer.parse_args(original)
                self.assertEqual(args.model, name)
                self.assertEqual(args.history_hours, -1)
                for option in ("--encoder_type", "--temporal_block", "--cnn_intermediate_channel"):
                    self.assertNotIn(option, original)
                run = captured.parent.parent
                for script in ("run_infer.sh", "command_used.sh"):
                    replay = subprocess.run(
                        ["bash", str(run / script)], cwd=REPO,
                        env=environment | {"HISTORY_HOURS": "48"}, capture_output=True, text=True,
                    )
                    self.assertEqual(replay.returncode, 0, replay.stdout + replay.stderr)
                    self.assertEqual(json.loads(captured.read_text()), original)


if __name__ == "__main__":
    unittest.main()
