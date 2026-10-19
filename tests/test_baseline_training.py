"""CLI, launcher provenance, and assignment-only config coverage for new baselines."""

import contextlib
import io
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import unittest

import train
from emulator.models import BASELINE_ARCHITECTURES, NEW_BASELINE_MODELS, STATIC_BASELINE_MODELS


REPO = Path(__file__).resolve().parents[1]
STAMP = "BASELINE_DRY_RUN_TEST"


class BaselineTrainingTests(unittest.TestCase):
    """Check launcher behavior without executing a training process or submitting jobs."""

    def setUp(self):
        """Create isolated temporary configs and artifact paths for each test."""
        temporary = tempfile.TemporaryDirectory(prefix="stormsurge-baseline-cli-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.environment = dict(os.environ)
        for key in (
            "PACT_RUN_NAME", "SESSION_NAME", "PYTHON_RUN_TAG_BASE", "PRESSURE_MODE",
            "MODEL", "BASELINE_POOLING", "BASELINE_POOL_GRID", "BASELINE_STEM_CHANNELS",
            "SLURM_NTASKS_PER_NODE", "SLURM_JOB_ID",
        ):
            self.environment.pop(key, None)
        # A dry run must never execute even this harmless interpreter stand-in.
        interpreter = self.root / "capture-python"
        interpreter.write_text(f"#!{sys.executable}\nfrom pathlib import Path\nPath({str(self.root / 'executed')!r}).touch()\n")
        interpreter.chmod(0o755)
        self.environment.update(
            DRY_RUN="1", USE_TMUX="0", PYTHON_BIN=str(interpreter),
            S0_PYTHON_BIN=str(interpreter), PACT_RUNSTAMP=STAMP,
            CUDA_VISIBLE_DEVICES="", ALL_RESULTS_ROOT=str(self.root / "results"),
        )

    def config(self, text):
        """Save assignment-only shell text and return its isolated path."""
        path = self.root / "config.sh"
        path.write_text(text + f"\nALL_RESULTS_ROOT={shlex.quote(str(self.root / 'results'))}\n")
        return path

    def launch(self, config, success=True):
        """Run train.sh only in dry-run mode and return stdout and parsed commands."""
        result = subprocess.run(
            ["bash", str(REPO / "train.sh"), str(config)], cwd=REPO,
            env=self.environment, capture_output=True, text=True,
        )
        if success:
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        else:
            self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertFalse((self.root / "results").exists())
        self.assertFalse((self.root / "executed").exists())
        commands = [shlex.split(line[5:])[1:] for line in result.stdout.splitlines() if line.startswith("CMD: ")]
        return result, commands

    def test_all_model_choices_and_defaults(self):
        """New keys select exact identities and defaults while legacy defaults persist."""
        for model in NEW_BASELINE_MODELS:
            history = 0 if model in STATIC_BASELINE_MODELS else 48
            args = train.parse_args(["--model", model, "--history_hours", str(history)])
            self.assertEqual(args.model, model)
            self.assertEqual((args.baseline_pooling, args.baseline_pool_grid, args.baseline_stem_channels), ("adaptive", 4, 32))
            self.assertEqual((args.effective_spatial_encoder, args.effective_temporal_module), BASELINE_ARCHITECTURES[model])
        for model in ("baseline", "perceiver3"):
            args = train.parse_args(["--model", model])
            self.assertEqual(args.hidden_channels, 64)
            self.assertEqual(args.history_hours, 24)
            self.assertFalse(hasattr(args, "effective_spatial_encoder"))

    def test_validation_and_irrelevant_legacy_options(self):
        """Reject invalid baseline/history settings without applying legacy depth constraints."""
        for model in NEW_BASELINE_MODELS:
            prefix = ["--model", model, "--history_hours", "0"]
            args = train.parse_args(prefix + ["--num_layers", "0", "--transformer_layers", "0", "--cnn_intermediate_channel", "0"])
            self.assertEqual(args.model, model)
            for option, value in (
                ("baseline_pooling", "attention"), ("baseline_pool_grid", "0"),
                ("baseline_stem_channels", "-1"), ("hidden_channels", "0"),
                ("history_hours", "5"), ("history_hours", "-6"), ("dropout", "nan"),
            ):
                with self.subTest(model=model, option=option), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    train.parse_args(prefix + [f"--{option}", value])
        for model in STATIC_BASELINE_MODELS:
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                train.parse_args(["--model", model, "--history_hours", "12"])
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            train.parse_args(["--model", "baseline_unknown"])

    def test_seven_example_configs_are_assignment_only_and_dry_run(self):
        """Every supplied example previews one valid command using common settings."""
        configs = sorted((REPO / "configs/baselines").glob("*.sh"))
        self.assertEqual(len(configs), 7)
        seen = set()
        for config in configs:
            with self.subTest(config=config.name):
                for line in config.read_text().splitlines():
                    if line.strip() and not line.lstrip().startswith("#"):
                        self.assertRegex(line, r"^[A-Za-z_][A-Za-z0-9_]*=")
                        self.assertNotIn("$(", line)
                syntax = subprocess.run(["bash", "-n", str(config)], capture_output=True, text=True)
                self.assertEqual(syntax.returncode, 0, syntax.stderr)
                result, commands = self.launch(config)
                self.assertEqual(len(commands), 1)
                args = train.parse_args(commands[0])
                seen.add(args.model)
                self.assertEqual((args.hidden_channels, args.dropout), (128, 0.05))
                self.assertEqual((args.exceedance_loss_weight, args.episode_gt_aligned_peak_weight), (0.0, 0.0))
                self.assertEqual(args.history_hours, 0 if args.model in STATIC_BASELINE_MODELS else 24)
                self.assertNotIn("--encoder_type", commands[0])
                self.assertNotIn("--temporal_block", commands[0])
                self.assertIn(f"Temporal:      {BASELINE_ARCHITECTURES[args.model][1]}", result.stdout)
                for value in (args.model, args.station, "NCEP", f"hist{args.history_hours}h", "adaptive4x4", "panomaly", "seed42"):
                    self.assertIn(value, args.run_tag)
                    self.assertIn(value, args.output_dir)
        self.assertEqual(seen, set(NEW_BASELINE_MODELS))

    def test_pooling_pressure_names_and_explicit_names(self):
        """Automatically generated names preserve experimental identity; explicit names persist."""
        for pooling, grid, label in (("adaptive", 4, "adaptive4x4"), ("adaptive", 2, "adaptive2x2"), ("global_mean", 4, "global_mean")):
            for pressure in ("anomaly", "raw", "anomaly_plus_mean"):
                settings = (
                    "MODEL=baseline_convlstm\nSTATION=Battery\nROOT_DIR=/source/CMIP6/graphs\n"
                    f"BASELINE_POOLING={pooling}\nBASELINE_POOL_GRID={grid}\nPRESSURE_MODE={pressure}\n"
                    "SEED=123\nHISTORY_HOURS_LIST=(12)\n"
                    "EXCEEDANCE_LOSS_WEIGHT=.2\nEPISODE_GT_ALIGNED_PEAK_WEIGHT=.1\n"
                )
                _, commands = self.launch(self.config(settings))
                args = train.parse_args(commands[0])
                self.assertEqual((args.baseline_pooling, args.baseline_pool_grid, args.pressure_mode), (pooling, grid, pressure))
                self.assertEqual((args.exceedance_loss_weight, args.episode_gt_aligned_peak_weight), (.2, .1))
                for value in ("baseline_convlstm", "Battery", "CMIP6", "hist12h", label, f"p{pressure}", "seed123"):
                    self.assertIn(value, args.run_tag)
                    self.assertIn(value, args.output_dir)
                _, commands = self.launch(self.config(settings + "PACT_RUN_NAME=chosen\nPYTHON_RUN_TAG_BASE=chosen-tag\n"))
                explicit = train.parse_args(commands[0])
                self.assertEqual(explicit.run_tag, f"chosen-tag_{STAMP}")
                self.assertEqual(Path(explicit.output_dir).name, f"{STAMP}_chosen")

    def test_history_sweep_has_separate_output_directories(self):
        """Each baseline history/rate combination has a distinct checkpoint destination."""
        _, commands = self.launch(self.config(
            "MODEL=baseline_mlp_gru\nHISTORY_HOURS_LIST=(0 12 24 48)\nLR_LIST=(.001 .005)\n"
        ))
        args = [train.parse_args(command) for command in commands]
        self.assertEqual(len(args), 8)
        self.assertEqual(len({item.output_dir for item in args}), 8)
        self.assertEqual({item.history_hours for item in args}, {0, 12, 24, 48})

    def test_shell_rejects_invalid_baseline_settings_before_artifacts(self):
        """Shell validation rejects incompatible static history and invalid pooling settings."""
        for settings in (
            "MODEL=baseline_mlp\nHISTORY_HOURS_LIST=(12)",
            "MODEL=baseline_cnn\nHISTORY_HOURS_LIST=(0 6)",
            "MODEL=baseline_cnn_gru\nBASELINE_POOL_GRID=0",
            "MODEL=baseline_cnn_lstm\nBASELINE_POOLING=attention",
            "MODEL=baseline_convlstm\nBASELINE_STEM_CHANNELS=-1",
            "MODEL=baseline_mlp_lstm\nHISTORY_HOURS_LIST=(7)",
            "MODEL=baseline_mlp_gru\nHISTORY_HOURS_LIST=()",
            "MODEL=baseline_unknown",
        ):
            with self.subTest(settings=settings):
                _, commands = self.launch(self.config(settings), success=False)
                self.assertEqual(commands, [])

    def test_legacy_launchers_do_not_receive_baseline_options(self):
        """Existing public model commands retain their architecture arguments."""
        for model in ("baseline", "perceiver3"):
            _, commands = self.launch(self.config(f"MODEL={model}\n"))
            args = train.parse_args(commands[0])
            self.assertEqual(args.model, model)
            self.assertEqual(args.hidden_channels, 64)
            self.assertIn("--encoder_type", commands[0])
            self.assertFalse(any(option.startswith("--baseline_") for option in commands[0]))

    def test_resolved_snapshot_includes_baseline_and_effective_options(self):
        """The launcher snapshot writer persists exact baseline values without launching it."""
        shell = (REPO / "train.sh").read_text()
        body = shell.split("write_resolved_config() {", 1)[1].split("\n}", 1)[0]
        destination = self.root / "snapshot.sh"
        script = (
            "CONFIG_PATH=example.sh\nMODEL=baseline_cnn_gru\n"
            "BASELINE_POOLING=global_mean\nBASELINE_POOL_GRID=3\nBASELINE_STEM_CHANNELS=12\n"
            "EFFECTIVE_SPATIAL_ENCODER=CNN\nEFFECTIVE_TEMPORAL_MODULE=GRU\n"
            "HISTORY_HOURS_LIST=(12 48)\n"
            "write_resolved_config() {" + body + "\n}\n"
            + f"write_resolved_config {shlex.quote(str(destination))}\n"
        )
        result = subprocess.run(["bash", "-c", script], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        text = destination.read_text()
        for value in ('BASELINE_POOLING="global_mean"', 'BASELINE_POOL_GRID="3"', 'BASELINE_STEM_CHANNELS="12"', 'EFFECTIVE_SPATIAL_ENCODER="CNN"', 'EFFECTIVE_TEMPORAL_MODULE="GRU"'):
            self.assertIn(value, text)
        self.assertIn('HISTORY_HOURS_LIST=([0]="12" [1]="48")', text)

    def test_collision_preserves_existing_baseline_results(self):
        """A stand-in interpreter verifies collision rejection without running training."""
        config = self.config("MODEL=baseline_cnn_gru\nHISTORY_HOURS_LIST=(12)\n")
        _, commands = self.launch(config)
        output = Path(train.parse_args(commands[0]).output_dir)
        environment = self.environment | {"DRY_RUN": "0"}
        first = subprocess.run(
            ["bash", str(REPO / "train.sh"), str(config)], cwd=REPO,
            env=environment, capture_output=True, text=True,
        )
        self.assertEqual(first.returncode, 0, first.stdout + first.stderr)
        marker = self.root / "executed"
        self.assertTrue(marker.exists())
        marker.unlink()
        sentinel = output / "best_overall.pt"
        sentinel.write_text("existing experiment")
        before = (output / "config_used.sh").read_bytes()
        second = subprocess.run(
            ["bash", str(REPO / "train.sh"), str(config)], cwd=REPO,
            env=environment, capture_output=True, text=True,
        )
        self.assertNotEqual(second.returncode, 0)
        self.assertIn("fresh run directory", second.stderr)
        self.assertFalse(marker.exists())
        self.assertEqual(sentinel.read_text(), "existing experiment")
        self.assertEqual((output / "config_used.sh").read_bytes(), before)

    def test_direct_output_rejects_existing_result_markers_before_writing(self):
        """Interrupted or historical baseline outputs are protected even without role checkpoints."""
        markers = (
            "config_previous.json", "best.pt", "best_model.pth", "threshold_metadata.json",
            "episode_peak_metadata.json", "metrics_previous.jsonl", "summary_previous.json",
        )
        for index, marker_name in enumerate(markers):
            with self.subTest(marker=marker_name):
                destination = self.root / f"interrupted-{index}"
                destination.mkdir()
                marker = destination / marker_name
                marker.write_text("previous results")
                args = train.parse_args([
                    "--model", "baseline_mlp", "--history_hours", "0",
                    "--output_dir", str(destination), "--seed", "123",
                    "--baseline_pooling", "global_mean",
                ])
                with self.assertRaisesRegex(FileExistsError, "existing baseline results"):
                    train._prepare_output_paths(args, False, 0)
                self.assertEqual(list(destination.iterdir()), [marker])
                self.assertEqual(marker.read_text(), "previous results")

    def test_direct_output_allows_launcher_scaffolding_then_rejects_changed_configuration(self):
        """Fresh shell scaffolding is allowed, but changing seed/pooling cannot reuse completed setup."""
        destination = self.root / "fresh-launcher-output"
        destination.mkdir()
        for name in ("config_used.sh", "config_original.sh", "launcher.log"):
            (destination / name).write_text("launcher scaffolding")
        arguments = ["--model", "baseline_cnn", "--history_hours", "0", "--output_dir", str(destination)]
        train._prepare_output_paths(train.parse_args(arguments), False, 0)
        before = {path.name: path.read_bytes() for path in destination.iterdir()}
        changed = train.parse_args(arguments + ["--seed", "123", "--baseline_pooling", "global_mean"])
        with self.assertRaisesRegex(FileExistsError, "existing baseline results"):
            train._prepare_output_paths(changed, False, 0)
        self.assertEqual({path.name: path.read_bytes() for path in destination.iterdir()}, before)

    def test_direct_python_names_and_config_snapshots(self):
        """Direct train.py setup records the baseline identity before loading any data."""
        previous = Path.cwd()
        try:
            os.chdir(self.root)
            args = train.parse_args([
                "--model", "baseline_cnn_gru", "--root_dir", "/datasets/NCEP/graphs",
                "--station", "Battery", "--history_hours", "48", "--pressure_mode", "raw",
                "--baseline_pooling", "adaptive", "--baseline_pool_grid", "2", "--seed", "123",
            ])
            paths = train._prepare_output_paths(args, False, 0)
        finally:
            os.chdir(previous)
        for value in ("baseline_cnn_gru", "Battery", "NCEP", "hist48h", "adaptive2x2", "praw", "seed123"):
            self.assertIn(value, args.run_tag)
            self.assertIn(value, paths.directory.name)
        config = json.loads(next(paths.directory.glob("config_*.json")).read_text())
        self.assertEqual(config["baseline_pool_grid"], 2)
        self.assertEqual(config["effective_spatial_encoder"], "CNN")
        self.assertEqual(config["effective_temporal_module"], "GRU")


if __name__ == "__main__":
    unittest.main()
