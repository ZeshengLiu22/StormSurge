"""CPU-only tests: synthetic artifacts, simulated PBS records, bounded fake children.

Never invokes qsub, torch, CUDA, train.py, or the real preflight.
"""
import argparse
import contextlib
import getpass
import io
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch
import zipfile

import experiments
from experiments import Experiment, digest, verify_result
import scheduler
import state
import sweep


def fixture_config(root, group="LR3e-3", encoder="CNN", index=0):
    exp = Experiment(root, group, encoder, index)
    exp.config.parent.mkdir(parents=True, exist_ok=True)
    values = {"STATION": exp.station, "ENCODER_TYPE": "CNN" if encoder == "CNN" else "GraphSAGE",
              "TEMPORAL_BLOCK": "Transformer" if exp.temporal == "Attn" else exp.temporal,
              "HISTORY_HOURS_LIST": f"({exp.history[:-1]})", "LR_LIST": f'("{experiments.GROUPS[group][0]}")',
              "MAX_GRAD_NORM": experiments.GROUPS[group][1], "EPOCHS": "300", "num_gpus": "4",
              "RUN_DIR_NAME_STYLE": "runname_timestamp", "PACT_RUN_NAME": exp.name, "PYTHON_RUN_TAG_BASE": exp.name,
              "ALL_RESULTS_ROOT": str(root.parent / f"PACT_StormSurge_FormalRuns_1021_{group}" / "architecture_ablation" / exp.history / exp.station)}
    exp.config.write_text("\n".join(f"{k}={v}" for k, v in values.items()) + "\n")
    return exp


def expected_args(exp):
    return {"epochs": "300", "lr": experiments.GROUPS[exp.group][0], "max_grad_norm": experiments.GROUPS[exp.group][1],
            "station": exp.station, "history_hours": exp.history[:-1], "encoder_type": exp.settings["ENCODER_TYPE"],
            "temporal_block": exp.settings["TEMPORAL_BLOCK"], "seed": "42"}


def zip_fixture(path, names):
    with zipfile.ZipFile(path, "w") as archive:
        for name in names:
            archive.writestr(name, b"synthetic fixture; no trained model")


def complete_artifacts(exp, output, epochs=300):
    output.mkdir(parents=True)
    stamp = output.name.split("__")[1]
    config = dict(expected_args(exp), epochs=300, lr=float(experiments.GROUPS[exp.group][0]),
                  max_grad_norm=float(experiments.GROUPS[exp.group][1]), output_dir=str(output), run_tag=exp.name + "_" + stamp)
    (output / "config_fixture.json").write_text(json.dumps(config))
    (output / "exit_status").write_text("0\n")
    (output / "config_used.sh").write_text("# synthetic resolved configuration\n")
    (output / "launcher.log").write_text("Epoch 300/300\nFINAL MULTI-CHECKPOINT RE-EVALUATION\nWall time (all combinations): 1 s\n")
    (output / f"train_{exp.name}.log").write_text("fixture\n")
    metrics = [{"epoch": i, "train": {"all_rmse": 1.0}, "val": {"all_rmse": 1.0}} for i in range(1, epochs + 1)]
    (output / "metrics_fixture.jsonl").write_text("\n".join(json.dumps(x) for x in metrics) + "\n")
    comparison = {}
    for role in ("overall", "exceedance"):
        comparison[role] = {"epoch": 100, "selection_split": "val", "path": str(output / f"best_{role}.pt"),
                            "val": {"all_rmse": 1.0}, "test": {"all_rmse": 1.0}}
        for split in ("val", "test"):
            zip_fixture(output / f"{split}_predictions_{role}.npz", ["y_true.npy", "y_pred.npy"])
    for name in ("best.pt", "best_overall.pt", "best_exceedance.pt", "best_fixture.pth"):
        zip_fixture(output / name, ["archive/data.pkl"])
    zip_fixture(output / "test_preds_fixture.npz", ["y_true.npy", "y_pred.npy"])
    for name in ("threshold_metadata.json", "episode_peak_metadata.json"):
        (output / name).write_text("{}")
    (output / "checkpoint_comparison.json").write_text(json.dumps(comparison))
    (output / "checkpoint_comparison.md").write_text("fixture report")
    (output / "summary_fixture.json").write_text(json.dumps({"run_tag": config["run_tag"], "best_epoch": 100, "wall_seconds": 1.0,
        "val": {"all_rmse": 1.0}, "test": {"all_rmse": 1.0}, "checkpoint_comparison": comparison,
        "checkpoint_selection": {"checkpoint_selection_mode": "overall"}}))


class LauncherTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "StormSurge"
        self.exp = fixture_config(self.root)
        self.env = patch.dict(os.environ, {"PBS_JOBID": "123[0].desched1", "PBS_ARRAY_INDEX": "0", "STORMSURGE_TMP_ROOT": str(Path(self.temp.name) / "tmp")})
        self.env.start()
        self.addCleanup(self.env.stop)
        self.preview = patch("experiments.preview", side_effect=lambda exp, *a, **kw: ("", expected_args(exp)))
        self.preview.start()
        self.addCleanup(self.preview.stop)
        self.preflight = patch("state.preflight", side_effect=lambda env: env)
        self.preflight.start()
        self.addCleanup(self.preflight.stop)

    def fake_runner(self, rc=143, success=False, numerical=False):
        def run(command, env, cwd, console, lock_fd, signal_path, *args):
            Path(console).write_text("ValueError: Cannot evaluate nonfinite y_pred\n" if numerical else "fake child\n")
            if success:
                complete_artifacts(self.exp, self.exp.results / f"{self.exp.name}__{env['PACT_RUNSTAMP']}")
            return rc, signal.SIGTERM if rc == 143 else None
        return run

    def execute(self, rc=143, success=False, numerical=False, count=1):
        with patch("state.supervise", side_effect=self.fake_runner(rc, success, numerical)) as runner, contextlib.redirect_stdout(io.StringIO()):
            values = [state.run_experiment(self.exp) for _ in range(count)]
            return values, runner.call_count

    def test_mapping_is_unique_and_complete(self):
        for group in experiments.GROUPS:
            for encoder in experiments.ENCODERS:
                exps = [Experiment(self.root, group, encoder, i) for i in range(144)]
                self.assertEqual(len({e.identity for e in exps}), 144)
                self.assertEqual(len({e.config for e in exps}), 144)
                self.assertEqual({e.station for e in exps}, set(experiments.STATIONS))
                self.assertEqual({e.history for e in exps}, set(experiments.HISTORIES))
                self.assertEqual({e.temporal for e in exps}, set(experiments.TEMPORALS))
                self.assertEqual((exps[0].station, exps[0].history, exps[0].temporal), ("Boston", "0H", "MLP"))
                self.assertEqual((exps[-1].station, exps[-1].history, exps[-1].temporal), ("Battery", "48H", "Attn"))
        for i in (-1, 144):
            with self.assertRaises(ValueError):
                Experiment(self.root, "LR3e-3", "CNN", i)

    def test_three_attempts_never_start_fourth(self):
        values, calls = self.execute(count=5)
        self.assertEqual(calls, 3)
        self.assertEqual(values, [1, 1, 78, 78, 78])
        rows = state.attempts(self.exp)
        self.assertEqual([r["attempt"] for r in rows], [1, 2, 3])
        self.assertEqual(len({r["output_dir"] for r in rows}), 3)
        self.assertEqual(len({r["record_dir"] for r in rows}), 3)
        self.assertEqual(state.read_json(self.exp.state / "state.json")["status"], "EXHAUSTED")
        self.assertEqual(sweep.report(self.exp, {})["status"], "EXHAUSTED")

    def test_configurable_initial_budget(self):
        with patch("state.supervise", side_effect=self.fake_runner()) as runner, contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(state.run_experiment(self.exp, max_attempts=1), 78)
            self.assertEqual(state.run_experiment(self.exp, max_attempts=99), 78)
        self.assertEqual(runner.call_count, 1)  # persisted budget cannot silently increase

    def test_preflight_failure_does_not_consume_or_loop_training(self):
        with patch("state.preflight", side_effect=subprocess.CalledProcessError(1, "fake_preflight")), patch("state.supervise") as runner, contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(subprocess.CalledProcessError):
                state.run_experiment(self.exp)
            self.assertEqual(state.run_experiment(self.exp), 65)
        runner.assert_not_called()
        self.assertEqual(state.attempts(self.exp), [])
        self.assertEqual(sweep.report(self.exp, {})["status"], "FAILED")

    def test_application_failure_does_not_retry(self):
        values, calls = self.execute(rc=2, count=4)
        self.assertEqual(calls, 1)
        self.assertEqual(values, [1, 65, 65, 65])
        self.assertEqual(sweep.report(self.exp, {})["reason"], "APPLICATION_EXIT")

    def test_numerical_failure_does_not_retry(self):
        values, calls = self.execute(rc=1, numerical=True, count=3)
        self.assertEqual(calls, 1)
        self.assertEqual(sweep.report(self.exp, {})["reason"], "NUMERICAL_FAILURE")

    def test_success_is_revalidated_and_never_rerun(self):
        values, calls = self.execute(rc=0, success=True, count=3)
        self.assertEqual((values, calls), ([0, 0, 0], 1))
        self.assertEqual(sweep.report(self.exp, {})["status"], "SUCCESS")
        output = Path(state.attempts(self.exp)[0]["output_dir"])
        (output / "best_overall.pt").write_text("broken")
        values, calls = self.execute(rc=0, success=True)
        self.assertEqual((values, calls), ([65], 0))
        self.assertEqual(sweep.report(self.exp, {})["status"], "FAILED")

    def test_partial_artifacts_never_prove_success(self):
        output = self.exp.results / f"{self.exp.name}__fixture"
        complete_artifacts(self.exp, output, epochs=299)
        self.assertFalse(verify_result(self.exp, output, expected_args(self.exp))[0])
        with (output / "metrics_fixture.jsonl").open("a") as handle:
            handle.write(json.dumps({"epoch": 300, "train": {"all_rmse": 1.0}, "val": {"all_rmse": 1.0}}) + "\n")
        self.assertTrue(verify_result(self.exp, output, expected_args(self.exp))[0])
        config = json.loads((output / "config_fixture.json").read_text())
        config["lr"] = .001
        (output / "config_fixture.json").write_text(json.dumps(config))
        self.assertFalse(verify_result(self.exp, output, expected_args(self.exp))[0])

    def test_nan_and_nonzero_exit_are_not_success(self):
        output = self.exp.results / f"{self.exp.name}__fixture"
        complete_artifacts(self.exp, output)
        metrics = output / "metrics_fixture.jsonl"
        metrics.write_text(metrics.read_text().replace('"all_rmse": 1.0', '"all_rmse": NaN', 1))
        self.assertFalse(verify_result(self.exp, output, expected_args(self.exp))[0])
        metrics.write_text(metrics.read_text().replace('NaN', '1.0'))
        (output / "exit_status").write_text("1")
        self.assertFalse(verify_result(self.exp, output, expected_args(self.exp))[0])

    def test_dead_owner_recovery_and_stored_application_exit(self):
        self.execute()
        row = state.attempts(self.exp)[0]
        (Path(row["record_dir"]) / "end.json").unlink()  # simulate supervisor SIGKILL before final record
        values, calls = self.execute()
        self.assertEqual(calls, 1)
        self.assertEqual(len(state.attempts(self.exp)), 2)
        row = state.attempts(self.exp)[-1]
        (Path(row["record_dir"]) / "end.json").unlink()
        output = Path(row["output_dir"])
        output.mkdir(parents=True)
        (output / "exit_status").write_text("2")
        values, calls = self.execute()
        self.assertEqual((values, calls), ([65], 0))

    def test_explicit_extension_and_reset_preserve_records(self):
        self.execute(count=3)
        args = argparse.Namespace(root=self.root, group="LR3e-3", encoder="CNN", index=0, offline=False,
                                  apply=True, max_attempts=5, reset=False, retry_failed=False)
        with patch("scheduler.snapshot", return_value={}), contextlib.redirect_stdout(io.StringIO()):
            sweep.update_budget(args)
        self.assertEqual(self.execute(count=3)[1], 2)
        before = {str(p): p.read_bytes() for p in self.exp.state.glob("attempts/*/*.json")}
        args.max_attempts, args.reset = 2, True
        with patch("scheduler.snapshot", return_value={}), contextlib.redirect_stdout(io.StringIO()):
            sweep.update_budget(args)
        self.assertEqual(self.execute(count=3)[1], 2)
        self.assertEqual(len(state.attempts(self.exp)), 7)
        for p, contents in before.items():
            self.assertEqual(Path(p).read_bytes(), contents)

    def test_exclusive_atomic_records_and_execution_lock(self):
        path = self.exp.state / "immutable.json"
        state.atomic_json(path, {"a": 1}, exclusive=True)
        with self.assertRaises(FileExistsError):
            state.atomic_json(path, {"a": 2}, exclusive=True)
        self.assertEqual(state.read_json(path), {"a": 1})
        with state.lock(self.exp.state / "execution.lock"):
            self.assertTrue(state.busy(self.exp))
            values, calls = self.execute()
            self.assertEqual((values, calls), ([75], 0))

    def test_failure_categories(self):
        self.assertEqual(state.classify(143, "/missing", {"Exit_status": -29})["reason"], "WALLTIME")
        self.assertEqual(state.classify(143, "/missing", {"comment": "Job preempted by scheduler"})["status"], "PREEMPTED")
        self.assertEqual(state.classify(2, "/missing")["reason"], "APPLICATION_EXIT")
        self.assertEqual(state.classify(None, "/missing", incomplete=True)["status"], "INTERRUPTED")
        console = Path(self.temp.name) / "numeric.log"
        console.write_text("ValueError: Cannot evaluate nonfinite y_pred")
        self.assertEqual(state.classify(1, console, received=signal.SIGTERM)["reason"], "NUMERICAL_FAILURE")

    def test_queue_comments_and_live_snapshots_do_not_prove_termination_cause(self):
        for comment in ("Queue preempt", "Not Running: insufficient resources in preempt queue",
                        "Job was not preempted", "Cannot preempt job", "Requested walltime limit: 1800"):
            with self.subTest(comment=comment):
                outcome = state.classify(143, "/missing", {"job_state": "F", "comment": comment})
                self.assertEqual((outcome["status"], outcome["reason"]), ("INTERRUPTED", "SIGNAL_OR_PBS_RERUN"))
        live = {"job_state": "R", "comment": "Job preempted by scheduler", "Exit_status": -29}
        self.assertEqual(state.classify(2, "/missing", live)["reason"], "APPLICATION_EXIT")
        for comment in ("walltime limit exceeded", "Job killed due to exceeding walltime"):
            self.assertEqual(state.classify(143, "/missing", {"job_state": "F", "comment": comment})["reason"], "WALLTIME")

    def test_terminal_diagnosis_matches_exact_attempt_and_run_count(self):
        with patch("state.supervise", side_effect=self.fake_runner()), contextlib.redirect_stdout(io.StringIO()):
            state.run_experiment(self.exp, scheduler={"job_state": "R", "run_count": 4})
        row = state.attempts(self.exp)[0]
        info = {"job_id": row["job_id"], "job_state": "F", "run_count": 4,
                "Exit_status": -29, "Job_Name": "ss3_CNN"}
        self.assertEqual(state.terminal_result(self.exp, row, info)["reason"], "WALLTIME")
        for changes in ({"job_id": "456[0].desched1"}, {"run_count": 5}, {"run_count": None}, {"job_state": "R"}):
            self.assertEqual(state.terminal_result(self.exp, row, info | changes)["reason"], "SIGNAL_OR_PBS_RERUN")
        original = info | {"Exit_status": 143, "comment": "Job preempted by scheduler"}
        jobs = {row["job_id"]: original, "456[0].desched1": info | {"job_id": "456[0].desched1"}}
        reported = sweep.report(self.exp, jobs)
        self.assertEqual(reported["reason"], "PBS_PREEMPTION")
        self.assertEqual(reported["job_id"], row["job_id"])
        self.assertEqual(reported["pbs_run_count"], 4)
        self.assertEqual(reported["pbs_exit_status"], 143)

    def test_elapsed_walltime_estimate_is_not_a_confirmed_cause(self):
        self.execute()
        row = state.attempts(self.exp)[0]
        directory = Path(row["record_dir"])
        (directory / "end.json").unlink()
        # Both old and new estimate fields remain diagnostic; a TERM can also
        # be preemption or cancellation, even when delivered near the limit.
        state.atomic_json(directory / "signal.json", {"signal": signal.SIGTERM,
                          "near_walltime_estimate": True, "walltime_reached": True})
        self.assertEqual(state.terminal_result(self.exp, row)["reason"], "SIGNAL_OR_PBS_RERUN")

    def test_parent_array_and_subjob_status(self):
        parent = {"Job_Name": "ss3_CNN", "job_state": "B", "array_indices_remaining": "0-143"}
        jobs = {"123[].desched1": parent}
        self.assertEqual(scheduler.job_for(self.exp, jobs)["job_state"], "Q")
        jobs["123[0].desched1"] = {"Job_Name": "ss3_CNN", "job_state": "R"}
        self.assertEqual(scheduler.job_for(self.exp, jobs)["job_state"], "R")
        jobs["123[0].desched1"]["job_state"] = "F"
        parent["array_indices_remaining"] = "1-143"
        self.assertEqual(scheduler.job_for(self.exp, jobs)["job_state"], "F")
        self.assertEqual(scheduler.contiguous([0, 1, 3, 4, 7]), [[0, 1], [3, 4], [7]])

    def test_json_query_does_not_use_text_format_selector(self):
        payload = {"pbs_server": "desched1", "Jobs": {"123[].desched1": {"Job_Owner": getpass.getuser() + "@host"},
                                                      "456[].desched1": {"Job_Owner": "someone_else@host"}}}
        result = subprocess.CompletedProcess([], 0, json.dumps(payload), "")
        with patch("scheduler.subprocess.run", return_value=result) as run:
            jobs = scheduler.snapshot()
        self.assertEqual(list(jobs), ["123[].desched1"])
        self.assertNotIn("-u", run.call_args.args[0])
        self.assertEqual(run.call_args.kwargs["env"]["PBS_DEFAULT"], "desched1")
        payload["pbs_server"] = "casper-pbs"
        with patch("scheduler.subprocess.run", return_value=subprocess.CompletedProcess([], 0, json.dumps(payload), "")):
            with self.assertRaises(RuntimeError):
                scheduler.snapshot()
        with patch("scheduler.subprocess.run", return_value=subprocess.CompletedProcess([], 1, "", "connection failure")):
            with self.assertRaises(RuntimeError):
                scheduler.snapshot()

    def test_preview_never_submits_and_filters_terminal_states(self):
        args = argparse.Namespace(root=self.root, groups=["LR3e-3"], encoders=["CNN"], indices="0-6",
                                  max_attempts=3, max_concurrent="unlimited", cnn_walltime=None, gnn_walltime=None,
                                  submit=False, offline=True)
        statuses = ["SUCCESS", "FAILED", "EXHAUSTED", "RUNNING", "QUEUED", "PENDING", "INTERRUPTED"]
        out = io.StringIO()
        with patch.object(Experiment, "validate", return_value=None), patch("sweep.report", side_effect=lambda exp, *a: {"status": statuses[exp.index]}), patch("subprocess.run") as run, contextlib.redirect_stdout(out):
            sweep.submit_selected(args)
        run.assert_not_called()
        self.assertIn("-J 5-6", out.getvalue())
        self.assertNotIn("%", out.getvalue())
        self.assertFalse(self.exp.state_base.exists())
        args.max_concurrent = "48"
        cmd = sweep.command_for(self.exp, list(range(144)), args, "PREVIEW")
        self.assertEqual(cmd[2], "0-143%48")
        self.assertNotIn("depend", " ".join(cmd))

    def test_default_cli_previews_two_independent_arrays_with_24_each(self):
        for command in ("submit", "resubmit"):
            out = io.StringIO()
            with patch.object(Experiment, "validate"), patch("sweep.report", return_value={"status": "PENDING"}), patch("subprocess.run") as run, contextlib.redirect_stdout(out):
                self.assertEqual(sweep.main(["--root", str(self.root), command, "--offline"]), 0)
            run.assert_not_called()
            commands = [shlex.split(line) for line in out.getvalue().splitlines() if line.startswith("/opt/pbs/bin/qsub ")]
            self.assertEqual(len(commands), 2)
            for cmd, group in zip(commands, ("LR3e-3", "LR5e-3_Clip0.25")):
                self.assertEqual(cmd[cmd.index("-J") + 1], "0-143%24")
                self.assertIn(f"SS_GROUP={group},SS_ENCODER=CNN,MAX_ATTEMPTS=3,", cmd[cmd.index("-v") + 1])
                self.assertNotIn("depend", " ".join(cmd))
                self.assertNotIn("-h", cmd)
            self.assertFalse(self.exp.state_base.exists())

    def test_all_templates_default_to_24_with_one_four_gpu_node(self):
        launchers = Path(__file__).resolve().parent.parent
        for group in experiments.GROUPS:
            for encoder in experiments.ENCODERS:
                template = (launchers / f"architecture_ablation_{group}" / f"run_{encoder}_array.pbs").read_text()
                self.assertIn("#PBS -J 0-143%24\n", template)
                self.assertIn("#PBS -l select=1:ncpus=64:mem=487GB:mpiprocs=4:ompthreads=1:ngpus=4\n", template)
                self.assertEqual(template.count("exec bash "), 1)
                self.assertNotIn("depend=", template)

    def test_ambiguous_submission_reservation(self):
        token = "a" * 32
        path = self.exp.state_base / "submissions" / token
        state.atomic_json(path / "request.json", {"token": token, "group": self.exp.group, "encoder": self.exp.encoder, "indices": [0]})
        self.assertEqual(sweep.report(self.exp, {})["status"], "QUEUED")
        state.atomic_json(path / "released.json", {"time": state.now()})
        self.assertEqual(sweep.report(self.exp, {})["status"], "PENDING")

    def test_sigterm_kills_detached_worker_and_releases_lock(self):
        directory = Path(self.temp.name)
        pid_file = directory / "worker.pid"
        worker = "import os,signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); open(" + repr(str(pid_file)) + ", 'w').write(str(os.getpid())); time.sleep(60)"
        child = "import subprocess,sys,signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); subprocess.Popen([sys.executable,'-c'," + repr(worker) + "],start_new_session=True); time.sleep(60)"
        script = directory / "supervisor.py"
        script.write_text("import sys,os\nfrom pathlib import Path\nsys.path.insert(0," + repr(str(Path(__file__).parent)) + ")\nimport state\nwith state.lock(Path(" + repr(str(directory / "lock")) + ")) as fd:\n result=state.supervise([sys.executable,'-c'," + repr(child) + "],os.environ.copy(),Path(" + repr(str(directory)) + "),Path(" + repr(str(directory / "console")) + "),fd,Path(" + repr(str(directory / "signal.json")) + "),terminate_grace=0.3)\n print(result)\n")
        supervisor = subprocess.Popen([sys.executable, str(script)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, start_new_session=True)
        try:
            deadline = time.monotonic() + 8
            while not pid_file.exists() and time.monotonic() < deadline:
                time.sleep(.02)
            self.assertTrue(pid_file.exists())
            worker_pid = int(pid_file.read_text())
            supervisor.send_signal(signal.SIGTERM)
            stdout, stderr = supervisor.communicate(timeout=8)
            self.assertEqual(supervisor.returncode, 0, stderr)
            self.assertEqual(state.read_json(directory / "signal.json")["signal"], signal.SIGTERM)
            with self.assertRaises(ProcessLookupError):
                os.kill(worker_pid, 0)
            with state.lock(directory / "lock"):
                pass
        finally:
            if supervisor.poll() is None:
                supervisor.kill()
                supervisor.wait()


if __name__ == "__main__":
    unittest.main()
