"""Pressure-mode launch arguments and replay contracts, without model execution."""

import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import unittest


REPO = Path(__file__).resolve().parents[1]
RUNSTAMP = "PRESSURE_SHELL_TEST"


class PressureShellTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="stormsurge-pressure-shell-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.environment = dict(os.environ)
        for key in (
            "PRESSURE_MODE", "PACT_RUN_NAME", "SESSION_NAME", "PYTHON_RUN_TAG_BASE",
            "SLURM_NTASKS_PER_NODE", "SLURM_JOB_ID", "SLURM_STEP_GPUS", "SLURM_JOB_GPUS",
        ):
            self.environment.pop(key, None)
        self.environment.update(
            DRY_RUN="1", USE_TMUX="0", PYTHON_BIN=sys.executable,
            S0_PYTHON_BIN=sys.executable, PACT_RUNSTAMP=RUNSTAMP,
            CUDA_VISIBLE_DEVICES="", CAPTURE_FILE=str(self.root / "argv.json"),
        )
        # This interpreter records arguments without importing or running train/infer.py.
        self.interpreter = self.root / "capture-python"
        self.interpreter.write_text(
            f"#!{sys.executable}\n"
            "import json, os, sys\n"
            "with open(os.environ['CAPTURE_FILE'], 'w') as output:\n"
            "    json.dump(sys.argv[1:], output)\n"
        )
        self.interpreter.chmod(0o755)

    def config(self, name="auto", **settings):
        path = self.root / f"{name}.sh"
        values = dict(
            ALL_RESULTS_ROOT=self.root / "training-results",
            INFERENCE_RESULTS_ROOT=self.root / f"inference-{name}",
            USE_TMUX=0, num_gpus=1, TORCH_GPU_PROBE=0,
        )
        values.update(settings)
        path.write_text("".join(f"{key}={shlex.quote(str(value))}\n" for key, value in values.items()))
        return path

    def launch(self, script, config, *, success=True, **environment):
        result = subprocess.run(
            ["bash", str(REPO / script), str(config)], cwd=REPO,
            env=self.environment | environment, capture_output=True, text=True,
        )
        if success:
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        else:
            self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        return result

    def command(self, result):
        commands = [shlex.split(line[5:]) for line in result.stdout.splitlines() if line.startswith("CMD: ")]
        self.assertEqual(len(commands), 1)
        return commands[0]

    def captured(self):
        return json.loads((self.root / "argv.json").read_text())

    def test_legacy_training_config_keeps_default_arguments_and_names(self):
        config = REPO / "configs_old/single_tail_episodepeak_4x4/train_config_NCEP_Boston_G0_T0250_EP0050.sh"
        default = self.command(self.launch("train.sh", config))
        explicit = self.command(self.launch("train.sh", config, PRESSURE_MODE="anomaly"))
        self.assertEqual(default, explicit)
        expected = {
            "--pressure_mode": "anomaly", "--root_dir": "./Data/Grid4_New/NCEP/graphs",
            "--station": "Boston", "--history_hours": "24", "--lr": "5e-3",
            "--grad_accum_steps": "4", "--exceedance_loss_weight": "0.025",
            "--episode_gt_aligned_peak_weight": "0.005",
            "--run_tag": f"NCEP_Boston_G0_T0250_EP0050_{RUNSTAMP}",
        }
        for option, value in expected.items():
            self.assertEqual(default[default.index(option) + 1], value)
        output = Path(default[default.index("--output_dir") + 1])
        self.assertEqual(output.name, f"NCEP_Boston_G0_T0250_EP0050__{RUNSTAMP}")
        self.assertFalse(output.exists())

    def test_training_suffixes_only_automatic_names(self):
        automatic = self.config()
        explicit = self.config("explicit", PACT_RUN_NAME="chosen", PYTHON_RUN_TAG_BASE="chosen-tag")
        for mode, suffix in (("anomaly", ""), ("raw", "_praw"), ("anomaly_plus_mean", "_panomaly_mean")):
            with self.subTest(mode=mode):
                command = self.command(self.launch("train.sh", automatic, PRESSURE_MODE=mode))
                self.assertEqual(command[command.index("--pressure_mode") + 1], mode)
                self.assertEqual(
                    Path(command[command.index("--output_dir") + 1]).name,
                    f"{RUNSTAMP}_auto{suffix}",
                )
                self.assertTrue(command[command.index("--run_tag") + 1].endswith(f"{suffix}_{RUNSTAMP}"))
                command = self.command(self.launch("train.sh", explicit, PRESSURE_MODE=mode))
                self.assertEqual(command[command.index("--run_tag") + 1], f"chosen-tag_{RUNSTAMP}")
                self.assertEqual(Path(command[command.index("--output_dir") + 1]).name, f"{RUNSTAMP}_chosen")
        self.assertFalse((self.root / "training-results").exists())

    def test_training_snapshot_replays_mode_and_rejects_nondefault_collision(self):
        config = self.config(PYTHON_BIN=self.interpreter)
        self.launch("train.sh", config, DRY_RUN="0", PRESSURE_MODE="raw")
        original = self.captured()
        output = Path(original[original.index("--output_dir") + 1])
        snapshot = output / "config_used.sh"
        replay = self.command(self.launch("train.sh", snapshot, PRESSURE_MODE="anomaly"))
        self.assertEqual(replay[replay.index("--pressure_mode") + 1], "raw")
        sentinel = output / "do-not-overwrite"
        sentinel.write_text("existing experiment")
        failed = self.launch("train.sh", config, success=False, DRY_RUN="0", PRESSURE_MODE="raw")
        self.assertIn("fresh run directory", failed.stderr)
        self.assertEqual(sentinel.read_text(), "existing experiment")
        self.assertEqual(self.captured(), original)

    def test_inference_optional_assertions_survive_snapshot_and_command_replay(self):
        checkpoint = self.root / "placeholder.pt"
        checkpoint.touch()
        for mode in (None, "anomaly", "raw", "anomaly_plus_mean"):
            with self.subTest(mode=mode):
                label = mode or "checkpoint"
                config = self.config(label, CKPT_PATH=checkpoint, PYTHON_BIN=self.interpreter)
                overrides = {} if mode is None else {"PRESSURE_MODE": mode}
                self.launch("infer.sh", config, **overrides)
                results = self.root / f"inference-{label}"
                self.assertFalse(results.exists())
                self.launch("infer.sh", config, DRY_RUN="0", **overrides)
                original = self.captured()
                if mode is None:
                    self.assertNotIn("--pressure_mode", original)
                else:
                    self.assertEqual(original[original.index("--pressure_mode") + 1], mode)
                run, = results.iterdir()
                snapshot = (run / "infer_config_used.sh").read_text()
                self.assertIn("unset PRESSURE_MODE" if mode is None else f"PRESSURE_MODE={mode}", snapshot)
                # Replay must retain the original assertion (including its absence).
                conflicting = "anomaly" if mode == "raw" else "raw"
                for script in ("run_infer.sh", "command_used.sh"):
                    result = subprocess.run(
                        ["bash", str(run / script)], cwd=REPO,
                        env=self.environment | {"PRESSURE_MODE": conflicting},
                        capture_output=True, text=True,
                    )
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                    self.assertEqual(self.captured(), original)


if __name__ == "__main__":
    unittest.main()
