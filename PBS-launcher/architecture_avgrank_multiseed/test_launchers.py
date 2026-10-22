#!/usr/bin/env python3
"""CPU-only launcher behavior tests; training, CUDA probes and modules are stubbed."""

import csv
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import unittest

from experiments import EXPERIMENTS, SEEDS, STATIONS, config_path, family, source_path


SUPPORT = Path(__file__).resolve().parent
REPO = SUPPORT.parents[1]


class LauncherTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="avgrank-launcher-test-")
        self.addCleanup(self.temp.cleanup)
        self.repo = Path(self.temp.name)
        self.experiment = EXPERIMENTS[0]
        self.group_root = self.repo / "PBS-launcher" / family(self.experiment)
        self.arch_root = self.group_root / self.experiment.architecture
        (self.repo / "PBS-launcher/architecture_avgrank_multiseed").mkdir(parents=True)
        self.group_root.mkdir()
        for filename in ("run_architecture.sh", "experiments.py"):
            shutil.copy2(SUPPORT / filename, self.repo / "PBS-launcher/architecture_avgrank_multiseed" / filename)
        shutil.copy2(REPO / "PBS-launcher" / family(self.experiment) / "run_architecture.sh", self.group_root)
        for station in STATIONS:
            for relative in [source_path(self.experiment, station)] + [config_path(self.experiment, station, seed) for seed in SEEDS]:
                target = self.repo / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(REPO / relative, target)
        # This fixture has no model implementation and cannot start training.
        (self.repo / "train.py").write_text("raise RuntimeError('The CPU-only fixture must never execute train.py')\n")
        (self.repo / "train.sh").write_text('''#!/usr/bin/env bash
set -euo pipefail
[[ "$PWD" == "$TEST_REPO" && "$USE_TMUX" == 0 ]]
[[ -z "${SLURM_NTASKS_PER_NODE+x}" && -z "${SLURM_JOB_ID+x}" ]]
[[ "$S0_PYTHON_BIN" == "$PYTHON_BIN" ]]
mkdir "$TEST_REPO/active"  # Fails if another config is still active.
trap 'rmdir "$TEST_REPO/active"' EXIT
config="$1"
printf 'START %s\\n' "$config" >> "$TEST_REPO/calls"
if [[ "${TEST_MODE:-}" == signal ]]; then
  bash -c 'trap "" TERM; exec sleep 300' &
  child=$!
  printf '%s\\n' "$child" > "$TEST_REPO/worker_pid"
  trap 'kill -KILL "$child" 2>/dev/null || true; wait "$child" 2>/dev/null || true; exit 143' TERM INT
  wait "$child"
fi
sleep 0.02
if [[ "${TEST_MODE:-}" == failure && "$config" == */Boston/seed123/* ]]; then
  printf 'END %s 17\\n' "$config" >> "$TEST_REPO/calls"
  exit 17
fi
printf 'END %s 0\\n' "$config" >> "$TEST_REPO/calls"
''')
        self.bin_dir = self.repo / "bin"
        self.bin_dir.mkdir()
        python = self.bin_dir / "python"
        python.write_text(f'''#!{sys.executable}
import os, pathlib, sys
if sys.argv[1:] == ["-"]:
    pathlib.Path(os.environ["TEST_REPO"], "cuda_probe_stub").touch()
    sys.exit(0)
os.execv({sys.executable!r}, [{sys.executable!r}, *sys.argv[1:]])
''')
        python.chmod(0o755)
        module = self.bin_dir / "module"
        module.write_text('#!/usr/bin/env bash\n[[ "$*" == "load cuda/12.9.0" ]] || exit 9\ntouch "$TEST_REPO/module_stub"\n')
        module.chmod(0o755)
        self.job_id = self.repo.name
        self.scratch = Path("/glade/derecho/scratch/zel/tmp") / f"stormsurge-{self.job_id}"
        self.addCleanup(lambda: self.scratch.rmdir() if self.scratch.exists() else None)
        self.env = {**os.environ, "PYTHON_BIN": str(python), "DRY_RUN": "0",
                    "PYTHONDONTWRITEBYTECODE": "1", "PBS_JOBID": self.job_id,
                    "TEST_REPO": str(self.repo), "PATH": f"{self.bin_dir}:{os.environ['PATH']}",
                    "SLURM_NTASKS_PER_NODE": "999", "SLURM_JOB_ID": "not-a-number",
                    "PACT_RUNSTAMP": "must-not-be-inherited"}
        # Do not inherit a site module function into the mock subprocess.
        self.env = {k: v for k, v in self.env.items() if not k.startswith("BASH_FUNC_module")}
        self.command = ["bash", str(self.group_root / "run_architecture.sh"), self.experiment.architecture]

    def run_launcher(self, **env):
        return subprocess.run(self.command, env={**self.env, **env}, cwd="/tmp",
                              text=True, capture_output=True, timeout=45)

    def manifest(self):
        manifests = list((self.arch_root / "manifests").glob("*.tsv"))
        self.assertEqual(len(manifests), 1)
        with manifests[0].open() as stream:
            return list(csv.DictReader(stream, delimiter="\t"))

    def assert_no_execution(self):
        for name in ("calls", "module_stub", "cuda_probe_stub"):
            self.assertFalse((self.repo / name).exists(), name)
        self.assertFalse(self.arch_root.exists())
        self.assertFalse(self.scratch.exists())

    def test_dry_run_is_read_only_and_lists_exact_sequence(self):
        before = {p.relative_to(self.repo) for p in self.repo.rglob("*")}
        result = self.run_launcher(DRY_RUN="1")
        self.assertEqual(result.returncode, 0, result.stderr)
        paths = [line.split("\t")[4] for line in result.stdout.splitlines() if line.startswith("CONFIG\t")]
        self.assertEqual(paths, [str(self.repo / config_path(self.experiment, station, seed))
                                 for station in STATIONS for seed in SEEDS])
        self.assertEqual(before, {p.relative_to(self.repo) for p in self.repo.rglob("*")})
        self.assert_no_execution()

    def test_missing_last_config_prevents_first_run(self):
        (self.repo / config_path(self.experiment, "Lewes", 2026)).unlink()
        result = self.run_launcher()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Missing config", result.stderr)
        self.assert_no_execution()

    def test_changed_training_setting_fails_preflight(self):
        config = self.repo / config_path(self.experiment, "Lewes", 2026)
        config.write_text(config.read_text().replace("WARMUP_EPOCHS=5", "WARMUP_EPOCHS=6"))
        result = self.run_launcher()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Reference settings changed", result.stderr)
        self.assert_no_execution()

    def test_missing_python_prevents_execution(self):
        result = self.run_launcher(PYTHON_BIN=str(self.repo / "does-not-exist"))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Python is not executable", result.stderr)
        self.assert_no_execution()

    def test_pbs_allocation_is_required(self):
        result = self.run_launcher(PBS_JOBID="")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("PBS allocation is required", result.stderr)
        self.assert_no_execution()

    def test_failure_continues_sequentially_and_preserves_status(self):
        result = self.run_launcher(TEST_MODE="failure")
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        calls = (self.repo / "calls").read_text().splitlines()
        self.assertEqual(len(calls), 24)
        for index in range(0, len(calls), 2):
            self.assertTrue(calls[index].startswith("START "))
            self.assertTrue(calls[index + 1].startswith("END " + calls[index][6:] + " "))
        terminal = [row for row in self.manifest() if row["status"] in ("FAILED", "SUCCEEDED")]
        self.assertEqual(len(terminal), 12)
        failed = [row for row in terminal if row["status"] == "FAILED"]
        self.assertEqual([(r["station"], r["seed"], r["exit_code"]) for r in failed], [("Boston", "123", "17")])
        self.assertEqual(sum(r["status"] == "SUCCEEDED" for r in terminal), 11)
        self.assertEqual(len(list((self.arch_root / "execution_logs").rglob("*.log"))), 12)
        self.assertTrue(all(row["start_utc"] != "-" and row["end_utc"] != "-" for row in terminal))

    def test_success_returns_zero_for_twelve_fresh_runs(self):
        result = self.run_launcher()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        terminal = [row for row in self.manifest() if row["status"] == "SUCCEEDED"]
        self.assertEqual(len(terminal), 12)
        self.assertEqual(sum(row["seed"] == "42" for row in terminal), 4)

    def test_sigterm_cleans_worker_and_marks_remaining_runs(self):
        process = subprocess.Popen(self.command, env={**self.env, "TEST_MODE": "signal"},
                                   cwd="/tmp", text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        worker = None
        try:
            deadline = time.monotonic() + 15
            while not (self.repo / "worker_pid").exists() and time.monotonic() < deadline:
                if process.poll() is not None:
                    self.fail("Launcher exited before the CPU-only worker started: " + str(process.communicate()))
                time.sleep(0.05)
            self.assertTrue((self.repo / "worker_pid").exists())
            worker = int((self.repo / "worker_pid").read_text())
            process.send_signal(signal.SIGTERM)
            stdout, stderr = process.communicate(timeout=35)
            self.assertEqual(process.returncode, 143, stdout + stderr)
            with self.assertRaises(ProcessLookupError):
                os.kill(worker, 0)
            rows = self.manifest()
            self.assertEqual(sum(row["status"] == "INTERRUPTED" for row in rows), 1)
            self.assertEqual(sum(row["status"] == "NOT_RUN" for row in rows), 11)
            self.assertFalse(any(row["status"] == "SUCCEEDED" for row in rows))
        finally:
            if process.poll() is None:
                process.kill()
                process.communicate()
            if worker is not None:
                try:
                    os.kill(worker, signal.SIGKILL)
                except ProcessLookupError:
                    pass


if __name__ == "__main__":
    unittest.main(verbosity=2)
