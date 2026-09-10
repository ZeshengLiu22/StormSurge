#!/usr/bin/env python3
"""Exercise the real local queue/tmux with DRY_RUN and a stdlib-only process stub.

Uses private task-spooler and tmux sockets, temporary queue metadata and results.
Never executes train.py. Does not modify or submit jobs to the normal local queue.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
CONFIG = HERE / "train_config_NCEP_Battery_24h_dual_tail_slope.sh"


def run(command, *, cwd, env, check=True, timeout=25):
    result = subprocess.run(command, cwd=cwd, env=env, capture_output=True, text=True, timeout=timeout)
    if check and result.returncode:
        raise AssertionError(f"Failed {command}: {result.returncode}\n{result.stdout}\n{result.stderr}")
    return result


def until(predicate, description, timeout=20):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(.1)
    raise AssertionError(f"Timed out: {description}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", type=Path, help="Persist evidence here; default is a temporary directory")
    options = parser.parse_args()
    evidence = options.report_dir or Path(tempfile.mkdtemp(prefix="p0-queue-evidence-"))
    evidence.mkdir(parents=True, exist_ok=True)
    tmux_binary, tsp_binary = shutil.which("tmux"), shutil.which("tsp")
    assert tmux_binary and tsp_binary, "tmux and task-spooler are required for this compatibility check"
    bashrc = Path.home() / ".bashrc"
    worker = Path.home() / ".local/bin/qsub_local_worker"
    assert worker.is_file()
    text = bashrc.read_text()
    # Exact current function definitions, without interactive module/Conda startup.
    functions = text[text.index("_qsub_local_state_dir() {"):]
    assert all(f"{name}() {{" in functions for name in ("qsub_local", "qstat_local", "qlog_local", "qattach_local", "qdel_local"))
    events = []
    with tempfile.TemporaryDirectory(prefix="p0-queue-probe-") as temporary:
        root = Path(temporary)
        work = root / "probe work"
        work.mkdir()
        (work / "train.sh").symlink_to(REPO / "train.sh")
        bins = root / "bin"
        bins.mkdir()
        spool = root / "spool"
        spool.mkdir()
        queue_functions = root / "queue_functions.sh"
        queue_functions.write_text(functions)
        tmux_socket, trace = root / "tmux.sock", root / "tmux_calls.txt"
        wrapper = bins / "tmux"
        wrapper.write_text(f'''#!/usr/bin/env bash
if [[ "${{1:-}}" == new-session ]]; then
    printf '%q ' "$@" >> {shlex.quote(str(trace))}
    printf '\\n' >> {shlex.quote(str(trace))}
fi
exec {shlex.quote(tmux_binary)} -S {shlex.quote(str(tmux_socket))} "$@"
''')
        wrapper.chmod(0o755)
        environment = {key: value for key, value in os.environ.items()
                       if not key.startswith(("SLURM_", "BASH_FUNC_")) and key not in (
                           "TMUX", "TS_ONFINISH", "TS_ENV", "TS_SAVELIST", "QSUB_LOCAL_SESSION_NAME",
                           "RUN_NAME", "LOG_FILE", "LOG_DIR", "CONFIG_PATH", "PACT_RUN_NAME")}
        environment.update(PATH=f"{bins}:{os.environ['PATH']}", TS_SOCKET=str(root / "tsp.sock"),
                           TMPDIR=str(spool), XDG_STATE_HOME=str(root / "state"),
                           QSUB_LOCAL_SLOTS="1", TS_SLOTS="1", QSUB_LOCAL_TMUX_POLL_SECONDS="0.1",
                           PYTHONDONTWRITEBYTECODE="1", CUDA_VISIBLE_DEVICES="", USE_TMUX="1")
        jobs = []

        def command(argv, *, env=None, cwd=work, check=True, timeout=25):
            return run(argv, cwd=cwd, env=env or environment, check=check, timeout=timeout)

        def queue(function, *arguments, cwd=work, env=None):
            return command(["bash", "--noprofile", "--norc", "-c",
                            'source "$1"; shift; "$@"', "bash", str(queue_functions), function, *arguments],
                           cwd=cwd, env=env)

        def submit(label, config, *, cwd=work, dry=False, stamp="20000101_000000"):
            env = dict(environment, DRY_RUN=str(int(dry)), PACT_RUNSTAMP=stamp)
            result = queue("qsub_local", "train.sh", label, str(config), cwd=cwd, env=env)
            ids = re.findall(r"(?m)^\s*(\d+)\s*$", result.stdout)
            assert len(ids) == 1, result.stdout + result.stderr
            jobs.append(ids[0])
            return ids[0]

        def state(job):
            return command([tsp_binary, "-s", job]).stdout.strip()

        def log(job):
            # tsp -c successfully reads a failed job's log but returns its exit status.
            return command([tsp_binary, "-c", job], check=False).stdout

        def record(label):
            path = root / "state/qsub_local" / f"{label}.env"
            script = 'source "$1"; printf "%s\\n" "$script" "$session_name" "$config_path" "$script_args"'
            return command(["bash", "-c", script, "bash", str(path)]).stdout.splitlines()

        def live(label):
            return command([str(wrapper), "has-session", "-t", label], check=False).returncode == 0

        try:
            command(["bash", "-n", str(queue_functions)])
            command(["bash", "-n", str(wrapper)])
            relative = str(CONFIG.relative_to(REPO))
            original_root = REPO / "All_Results/P0_QuickRun"
            original_entries = sorted(str(p) for p in original_root.rglob("*")) if original_root.exists() else None
            dry_id = submit("P0_Test", relative, cwd=REPO, dry=True)
            assert command([tsp_binary, "-w", dry_id], check=False).returncode == 0
            dry_log = log(dry_id)
            assert dry_log.count("CMD: train.py ") == 1
            assert "USE_TMUX=1 SESSION_NAME=P0_Test DRY_RUN=1" in dry_log
            assert "Run name:      NCEP_Battery_24h_dual_tail_slope" in dry_log
            assert record("P0_Test")[:3] == ["train.sh", "P0_Test", relative]
            assert shlex.split(record("P0_Test")[3]) == [relative]
            assert not trace.exists(), "DRY_RUN started tmux"
            current_entries = sorted(str(p) for p in original_root.rglob("*")) if original_root.exists() else None
            assert current_entries == original_entries
            events.append("Actual qsub_local train.sh P0_Test <selected P0 config>: one dry command; correct script, argument, session and semantic name; no tmux session or result folder.")

            # Only this temporary config changes TRAIN_PY; it cannot execute training.
            stub = root / "process_stub.py"
            stub.write_text(f'''import json, os, pathlib, sys, time
assert "train.py" not in sys.argv[0]
output = pathlib.Path(sys.argv[sys.argv.index("--output_dir") + 1]).resolve()
label = os.environ["SESSION_NAME"]
payload = dict(argv=sys.argv, output_dir=str(output), session=label,
               run_name=os.environ["PACT_RUN_NAME"], stamp=os.environ["PACT_RUNSTAMP"],
               python=sys.executable, tmux=bool(os.environ.get("TMUX")))
(output / "stub_invocation.json").write_text(json.dumps(payload, indent=2))
print("P0 PROCESS STUB: no torch import, model, optimizer, data load or training", flush=True)
release = pathlib.Path({str(root)!r}) / (label + ".release")
deadline = time.monotonic() + 20
while not release.exists():
    if time.monotonic() > deadline:
        raise SystemExit(99)
    time.sleep(0.05)
raise SystemExit(23 if label.endswith("_Fail") else 0)
''')
            compile(stub.read_text(), str(stub), "exec")
            probe_config = work / "probe config.sh"
            probe_config.write_text(CONFIG.read_text().replace('TRAIN_PY="train.py"', f"TRAIN_PY={shlex.quote(str(stub))}"))
            command(["bash", "-n", str(probe_config)])
            output_root = work / "All_Results/P0_QuickRun"
            name = "NCEP_Battery_24h_dual_tail_slope"
            failed_dir = output_root / f"{name}__20000101_000001"
            passed_dir = output_root / f"{name}__20000101_000002"
            fail_id = submit("P0_Test_Fail", probe_config, stamp="20000101_000001")
            until(lambda: (failed_dir / "stub_invocation.json").exists(), "failure stub started")
            assert live("P0_Test_Fail") and state(fail_id) == "running"
            assert list(output_root.iterdir()) == [failed_dir]
            first = json.loads((failed_dir / "stub_invocation.json").read_text())
            assert first["session"] == "P0_Test_Fail" and first["run_name"] == name
            assert first["stamp"] == "20000101_000001" and first["tmux"]
            assert first["python"] == "/media/volume/PACT-Data/conda_envs/torchpyg-cu124/bin/python"
            pass_id = submit("P0_Test_Pass", probe_config, stamp="20000101_000002")
            until(lambda: state(pass_id) == "queued", "second task stays queued")
            assert state(fail_id) == "running" and live("P0_Test_Fail")
            assert not passed_dir.exists()
            active_state = queue("qstat_local", "--raw").stdout
            assert "P0_Test_Fail" in active_state and "P0_Test_Pass" in active_state
            events.append("While the first detached tmux stub was active, its worker retained the single queue slot and the second task stayed queued.")
            (root / "P0_Test_Fail.release").touch()
            fail_status = command([tsp_binary, "-w", fail_id], check=False).returncode
            assert fail_status == 23, (fail_status, log(fail_id))
            until(lambda: (passed_dir / "stub_invocation.json").exists(), "next queued stub started after failure")
            assert not live("P0_Test_Fail") and live("P0_Test_Pass")
            assert state(pass_id) == "running"
            (root / "P0_Test_Pass.release").touch()
            pass_status = command([tsp_binary, "-w", pass_id], check=False).returncode
            assert pass_status == 0, log(pass_id)
            until(lambda: not live("P0_Test_Pass"), "successful tmux session closed")
            assert set(output_root.iterdir()) == {failed_dir, passed_dir}
            assert len(trace.read_text().splitlines()) == 2, trace.read_text()
            for folder, label, status in ((failed_dir, "P0_Test_Fail", 23), (passed_dir, "P0_Test_Pass", 0)):
                assert (folder / "exit_status").read_text().strip() == str(status)
                config_used = (folder / "config_used.sh").read_text()
                assert f'PACT_RUN_NAME="{name}"' in config_used
                assert f'SESSION_NAME="{label}"' in config_used
                assert 'USE_TMUX="1"' in config_used
                assert len(list(folder.glob("train_*.log"))) == 1
                assert "P0 PROCESS STUB" in (folder / "launcher.log").read_text()
                assert not list(folder.rglob("*.pth")), "Probe must not create real checkpoints"
            assert not list((root / "state/qsub_local/status").glob("*.status"))
            events.append("Exactly one tmux session and one semantic/timestamp result folder per stub invocation; exit statuses 23 and 0 propagated to task-spooler; failed job released the slot.")

            # Duplicate timestamp/name fails before touching existing artifacts.
            collision = work / "collision.sh"
            collision.write_text(probe_config.read_text().replace("USE_TMUX=1", "USE_TMUX=0"))
            command(["bash", "-n", str(collision)])
            hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in failed_dir.iterdir()}
            result = command(["bash", "train.sh", str(collision)],
                             env=dict(environment, DRY_RUN="0", PACT_RUNSTAMP="20000101_000001"), check=False)
            assert result.returncode == 2 and "Cannot create a fresh run directory" in result.stdout + result.stderr, (result.returncode, result.stdout, result.stderr)
            assert hashes == {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in failed_dir.iterdir()}
            events.append("A duplicate semantic name/timestamp is rejected without overwriting the earlier run.")
            (evidence / "queue_dry_run.txt").write_text(dry_log)
            (evidence / "queue_failure.txt").write_text(log(fail_id))
            (evidence / "queue_success.txt").write_text(log(pass_id))
            (evidence / "queue_active.txt").write_text(active_state)
            (evidence / "tmux_calls.txt").write_text(trace.read_text())
            for folder, prefix in ((failed_dir, "failure"), (passed_dir, "success")):
                for filename in ("config_used.sh", "launcher.log", "exit_status", "stub_invocation.json"):
                    (evidence / f"probe_{prefix}_{filename}.txt").write_bytes((folder / filename).read_bytes())
        finally:
            # Only private sockets and our own temporary tasks are touched.
            command([str(wrapper), "kill-server"], check=False)
            for job in jobs:
                result = command([tsp_binary, "-s", job], check=False)
                if result.stdout.strip() == "queued":
                    command([tsp_binary, "-r", job], check=False)
                elif result.stdout.strip() == "running":
                    command([tsp_binary, "-k", job], check=False)
            command([tsp_binary, "-K"], check=False)
            assert not tmux_socket.exists() or command([str(wrapper), "list-sessions"], check=False).returncode != 0
    report = dict(status="PASS", production_queue_functions_sha256=hashlib.sha256(functions.encode()).hexdigest(),
                  bashrc_sha256=hashlib.sha256(bashrc.read_bytes()).hexdigest(),
                  worker_sha256=hashlib.sha256(worker.read_bytes()).hexdigest(), checks=events,
                  dry_queue_jobs=1, stub_queue_jobs=2, tmux_sessions_started=2, result_directories=2,
                  failure_status=23, success_status=0, single_slot_wait_verified=True,
                  stale_test_sessions=False, stale_test_queue=False, normal_queue_modified=False,
                  real_training_started=False, gpu_training_seconds=0)
    (evidence / "queue_validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"PASS: real qsub_local/worker/tmux, isolated sockets, one dry job + two process stubs, exits 23/0, no stale jobs or training. Evidence: {evidence}")


if __name__ == "__main__":
    main()
