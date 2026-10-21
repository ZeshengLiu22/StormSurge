"""Durable experiment-level attempt records and bounded process supervision."""
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
import ctypes
import fcntl
import json
import os
import re
import signal
import socket
import subprocess
import tempfile
import time
import uuid

from experiments import PYTHON, digest, runtime_env, successful_result, verify_result


def now():
    return datetime.now(timezone.utc).isoformat()


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except FileNotFoundError:
        return default


def atomic_json(path, value, exclusive=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".write-", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(value, handle, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        if exclusive:
            os.link(temporary, path)  # atomic publication, never replace a record
        else:
            os.replace(temporary, path)
        dirfd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(dirfd)
        finally:
            os.close(dirfd)
    finally:
        Path(temporary).unlink(missing_ok=True)


@contextmanager
def lock(path, create=True):
    path = Path(path)
    if create:
        path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(path, os.O_RDWR | (os.O_CREAT if create else 0), 0o600)
    except FileNotFoundError:
        yield None
        return
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield fd
    finally:
        os.close(fd)


def busy(exp):
    try:
        with lock(exp.state / "execution.lock", create=False):
            return False
    except BlockingIOError:
        return True


def attempts(exp):
    return [read_json(p) | {"record_dir": str(p.parent)}
            for p in sorted((exp.state / "attempts").glob("*/start.json"))]


def policy(exp, max_attempts=3):
    return read_json(exp.state / "policy.json", {"max_attempts": max_attempts, "base_attempt": 0,
                                               "allow_failed_through": 0, "config_sha256": digest(exp.config)})


def budget_used(records, rules):
    return sum(row["attempt"] > rules["base_attempt"] for row in records)


def state_marker(exp, status, **details):
    atomic_json(exp.state / "state.json", {"identity": exp.identity, "status": status, "updated": now(), **details})


def terminal_result(exp, start, scheduler=None):
    """Infer a dead attempt's outcome without confusing a PBS rerun with success."""
    # PBS reuses a subjob ID on rerun. Only a finished record for this exact
    # execution can diagnose its termination; running snapshots can carry stale
    # comments/exit codes from previous runs. Missing evidence stays unknown.
    scheduler = scheduler or {}
    if (scheduler.get("job_id") != start.get("job_id")
            or scheduler.get("job_state") != "F"
            or start.get("pbs_run_count") is None
            or str(scheduler.get("run_count")) != str(start["pbs_run_count"])):
        scheduler = {}
    directory = Path(start["record_dir"])
    recorded = read_json(directory / "end.json")
    if recorded:
        if recorded["status"] in {"INTERRUPTED", "PREEMPTED"} and scheduler:
            refined = classify(recorded.get("exit_status"), directory / "console.log", scheduler)
            if refined["reason"] in {"WALLTIME", "PBS_PREEMPTION"}:
                return recorded | refined
        return recorded
    output = Path(start["output_dir"])
    valid, evidence = verify_result(exp, output)
    if valid and start["config_sha256"] == digest(exp.config):
        return {"status": "SUCCESS", "reason": evidence, "end": None, "observed_finished_at": now(),
                "pbs_obittime": (scheduler or {}).get("obittime"), "exit_status": 0, "inferred": True}
    try:
        rc = int((output / "exit_status").read_text().strip())
    except (OSError, ValueError):
        rc = None
    received = read_json(directory / "signal.json", {})
    return classify(rc, directory / "console.log", scheduler, received.get("signal"), incomplete=True) | {
                        "end": None, "observed_finished_at": now(), "pbs_obittime": (scheduler or {}).get("obittime"), "inferred": True}


def classify(rc, console, scheduler=None, received=None, incomplete=False):
    scheduler = scheduler or {}
    if scheduler.get("job_state") not in (None, "F"):
        scheduler = {}  # live PBS comments are scheduling information, not an exit diagnosis
    comment = str(scheduler.get("comment", "")).lower()
    pbs_exit = scheduler.get("Exit_status")
    try:
        log = Path(console).read_text(errors="replace")
    except OSError:
        log = ""
    numeric = re.search(r"(?:ValueError|RuntimeError|FloatingPointError):[^\n]*(?:nonfinite|non-finite|\bnan\b|\binf\b|requires finite)|(?:loss|rmse|mae)\s*[=:]\s*(?:nan|[+-]?inf)\b", log, re.I)
    if numeric:
        status, reason = "FAILED", "NUMERICAL_FAILURE"
    elif pbs_exit == -29 or re.search(r"\bwalltime(?:\s+limit)?\s+(?:was\s+)?exceeded\b|\bexceed(?:ed|ing)\s+(?:requested\s+)?walltime\b", comment):
        status, reason = "INTERRUPTED", "WALLTIME"
    elif re.search(r"^\s*(?:job\s+(?:was\s+|has\s+been\s+)?)?preempted\b", comment):
        status, reason = "PREEMPTED", "PBS_PREEMPTION"
    elif received or rc in (-signal.SIGTERM, -signal.SIGKILL, -signal.SIGINT, 130, 137, 143) or pbs_exit in (-11, -14, -20, -4, -5, -6):
        status, reason = "INTERRUPTED", "SIGNAL_OR_PBS_RERUN"
    else:
        if rc is None and incomplete:
            status, reason = "INTERRUPTED", "OWNER_LOST_NO_EXIT_RECORD"
        else:
            status, reason = "FAILED", "APPLICATION_EXIT" if rc else "INCOMPLETE_ARTIFACTS"
    return {"status": status, "reason": reason, "exit_status": rc, "pbs_exit_status": pbs_exit}


def decision(exp, records, rules, last=None):
    if rules["config_sha256"] != digest(exp.config):
        return "FAILED", "CONFIG_CHANGED_REQUIRES_MANUAL_RESET"
    if budget_used(records, rules) >= rules["max_attempts"]:
        return "EXHAUSTED", "ATTEMPT_BUDGET"
    if records and records[-1]["attempt"] > max(rules["base_attempt"], rules.get("allow_failed_through", 0)):
        last = last or terminal_result(exp, records[-1])
        if last["status"] == "FAILED":
            return "FAILED", last["reason"]
        if last["status"] == "SUCCESS":
            return "FAILED", "PREVIOUS_SUCCESS_NO_LONGER_VERIFIES"
    return "PENDING", "ELIGIBLE"


def preflight(env):
    """Load the existing CUDA environment before charging any training attempt."""
    result = subprocess.run(["bash", "-l", "-c", "module load cuda/12.9.0 >&2 && env -0"],
                            env=env, capture_output=True, check=True, timeout=60)
    merged = env.copy()
    for item in result.stdout.split(b"\0"):
        if b"=" in item:
            key, value = item.split(b"=", 1)
            if re.fullmatch(rb"[A-Za-z_][A-Za-z_0-9]*", key):
                merged[key.decode()] = value.decode()
    # Retain CUDA's exported paths while preventing login profiles from changing
    # this launcher's interpreter, dry-run mode, output stamp or DDP controls.
    for key in ("PYTHON_BIN", "S0_PYTHON_BIN", "USE_TMUX", "DRY_RUN", "OMP_NUM_THREADS",
                "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "PYTHONUNBUFFERED", "PACT_RUNSTAMP", "TMPDIR"):
        if key in env:
            merged[key] = env[key]
    for key in ("SLURM_NTASKS_PER_NODE", "SLURM_JOB_ID", "SESSION_NAME"):
        merged.pop(key, None)
    check = """import torch, torch.distributed as dist, torch_geometric
assert torch.cuda.is_available() and torch.cuda.device_count() == 4, 'Exactly four visible GPUs required'
assert dist.is_available() and dist.is_nccl_available(), 'NCCL required'
assert all(torch.cuda.get_device_capability(i)[0] >= 8 for i in range(4)), 'Ampere or newer required'
print('Runtime:', torch.__version__, 'CUDA:', torch.version.cuda, 'GPUs:', torch.cuda.device_count())
"""
    subprocess.run([PYTHON, "-c", check], env=merged, check=True, timeout=90)
    return merged


def supervise(command, env, cwd, console, lock_fd, signal_path, wall_seconds=None, job_elapsed=0,
              terminate_grace=20):
    """One child process group; TERM then bounded KILL, with orphan reaping on Linux."""
    received = []
    old = {}
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        old[sig] = signal.signal(sig, lambda signum, frame: received.append(signum))
    # Adopt grandchildren if bash/torchrun exits before its workers.
    if ctypes.CDLL(None, use_errno=True).prctl(36, 1, 0, 0, 0) != 0:  # PR_SET_CHILD_SUBREAPER
        raise OSError(ctypes.get_errno(), "Cannot enable child subreaper")
    process = None
    beginning = time.monotonic()
    stopping = None
    def send(sig):
        if process is not None:
            # torchrun workers may create their own sessions. Include the full
            # descendant tree and adopted orphans, not just bash's process group.
            parents = {}
            for entry in Path("/proc").glob("[0-9]*/stat"):
                try:
                    tail = entry.read_text().rsplit(")", 1)[1].split()
                    parents[int(entry.parent.name)] = int(tail[1])
                except (OSError, ValueError, IndexError):
                    pass
            descendants = {os.getpid()}
            while True:
                expanded = descendants | {pid for pid, parent in parents.items() if parent in descendants}
                if expanded == descendants:
                    break
                descendants = expanded
            try:
                os.killpg(process.pid, sig)
            except ProcessLookupError:
                pass
            for pid in descendants - {os.getpid()}:
                try:
                    os.kill(pid, sig)
                except ProcessLookupError:
                    pass
    try:
        with Path(console).open("x") as out:
            process = subprocess.Popen(command, env=env, cwd=cwd, stdout=out, stderr=subprocess.STDOUT,
                                       start_new_session=True, pass_fds=(lock_fd,))
            while process.poll() is None:
                if received and stopping is None:
                    stopping = time.monotonic()
                    atomic_json(signal_path, {"time": now(), "signal": received[0],
                        "near_walltime_estimate": bool(wall_seconds and job_elapsed + stopping - beginning >= wall_seconds - 2)}, exclusive=True)
                    send(signal.SIGTERM)
                if stopping is not None and time.monotonic() - stopping >= terminate_grace:
                    send(signal.SIGKILL)
                time.sleep(0.1)
            rc = process.wait()
            # Flush normal tee children, then terminate any surviving DDP workers.
            time.sleep(0.2)
            send(signal.SIGTERM)
            deadline = time.monotonic() + terminate_grace
            while True:
                try:
                    child, _ = os.waitpid(-1, os.WNOHANG)
                    if child == 0:
                        if time.monotonic() >= deadline:
                            send(signal.SIGKILL)
                        time.sleep(0.05)
                        continue
                except ChildProcessError:
                    break
            return rc, received[0] if received else None
    finally:
        send(signal.SIGKILL)
        if process is not None:
            process.wait()
        for sig, handler in old.items():
            signal.signal(sig, handler)


def run_experiment(exp, max_attempts=3, scheduler=None):
    invocation_started = time.monotonic()
    if max_attempts < 1:
        raise ValueError("MAX_ATTEMPTS must be positive")
    if not os.environ.get("PBS_JOBID"):
        raise ValueError("Actual execution requires PBS_JOBID; use preview on login nodes")
    exp.validate()
    try:
        with lock(exp.state / "execution.lock") as lock_fd:
            success = successful_result(exp)
            if success:
                state_marker(exp, "SUCCESS", output_dir=success, reason="VERIFIED_EXISTING_RESULT")
                print(f"SKIP SUCCESS {exp.identity}: {success}", flush=True)
                return 0
            rules = policy(exp, max_attempts)
            if not (exp.state / "policy.json").exists():
                atomic_json(exp.state / "policy.json", rules, exclusive=True)
            rows = attempts(exp)
            for row in rows:
                end = Path(row["record_dir"]) / "end.json"
                if not end.exists():
                    atomic_json(end, terminal_result(exp, row, scheduler), exclusive=True)
            status, reason = decision(exp, rows, rules)
            if status != "PENDING":
                state_marker(exp, status, reason=reason, attempts=len(rows), max_attempts=rules["max_attempts"])
                print(f"SKIP {status} {exp.identity}: {reason}", flush=True)
                return 78 if status == "EXHAUSTED" else 65
            marker = read_json(exp.state / "state.json", {})
            if marker.get("reason") == "PREFLIGHT_FAILED" and not rules.get("retry_preflight"):
                print(f"SKIP FAILED {exp.identity}: PREFLIGHT_FAILED (manual reset required)", flush=True)
                return 65
            if rules.get("retry_preflight"):
                rules["retry_preflight"] = False
                atomic_json(exp.state / "policy.json", rules)
            attempt = max((row["attempt"] for row in rows), default=0) + 1
            stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f") + f"_a{attempt:04d}_{uuid.uuid4().hex[:8]}"
            env = runtime_env(stamp)
            env["TMPDIR"] = str(Path(os.environ.get("STORMSURGE_TMP_ROOT", "/glade/derecho/scratch/zel/tmp")) /
                                f"stormsurge-{os.environ['PBS_JOBID']}-{stamp}")
            try:
                Path(env["TMPDIR"]).mkdir(parents=True, exist_ok=False)
                env = preflight(env)
            except (OSError, subprocess.SubprocessError) as exc:
                state_marker(exp, "FAILED", reason="PREFLIGHT_FAILED", error=str(exc))
                raise
            if digest(exp.config) != rules["config_sha256"]:
                state_marker(exp, "FAILED", reason="CONFIG_CHANGED_REQUIRES_MANUAL_RESET")
                return 65
            # No loop: PBS may rerun the job, but this invocation can start training only once.
            record_dir = exp.state / "attempts" / f"{attempt:06d}-{uuid.uuid4().hex[:12]}"
            output = exp.results / f"{exp.name}__{stamp}"
            start = {"identity": exp.identity, "group": exp.group, "encoder": exp.encoder,
                     "station": exp.station, "history": exp.history, "temporal_block": exp.temporal,
                     "index": exp.index, "attempt": attempt, "budget_attempt": budget_used(rows, rules) + 1,
                     "max_attempts": rules["max_attempts"], "start": now(), "job_id": os.environ["PBS_JOBID"],
                     "pbs_run_count": (scheduler or {}).get("run_count"), "hostname": socket.gethostname(),
                     "supervisor_pid": os.getpid(), "config": str(exp.config), "config_sha256": digest(exp.config),
                     "output_dir": str(output), "record_dir": str(record_dir)}
            print(f"START {exp.identity} attempt={attempt} output={output}", flush=True)
            state_marker(exp, "RUNNING", **{k: v for k, v in start.items() if k != "identity"})
            # Durable and atomic immediately before Popen. A crash in the reservation/exec
            # window conservatively consumes an attempt rather than undercounting it.
            atomic_json(record_dir / "start.json", start, exclusive=True)
            try:
                wall_seconds = parse_walltime((scheduler or {}).get("Resource_List", {}).get("walltime"))
                elapsed = parse_walltime((scheduler or {}).get("resources_used", {}).get("walltime")) or 0
                elapsed += time.monotonic() - invocation_started
                rc, received = supervise(["bash", str(exp.root / "train.sh"), str(exp.config)], env, exp.root,
                                         record_dir / "console.log", lock_fd, record_dir / "signal.json",
                                         wall_seconds, elapsed)
                valid, evidence = verify_result(exp, output)
                if rc == 0 and valid:
                    outcome = {"status": "SUCCESS", "reason": evidence, "exit_status": 0}
                else:
                    outcome = classify(rc, record_dir / "console.log", scheduler, received)
                    outcome["artifact_check"] = evidence
            except Exception as exc:
                outcome = {"status": "FAILED", "reason": "LAUNCHER_ERROR", "exit_status": None, "error": str(exc)}
            outcome["end"] = now()
            atomic_json(record_dir / "end.json", outcome, exclusive=True)
            final_status = outcome["status"]
            if final_status != "SUCCESS" and start["budget_attempt"] >= rules["max_attempts"]:
                final_status = "EXHAUSTED"
            state_marker(exp, final_status, attempt=attempt, output_dir=str(output), last_outcome=outcome)
            print(f"END {exp.identity} attempt={attempt} status={final_status} reason={outcome['reason']}", flush=True)
            return 0 if final_status == "SUCCESS" else 78 if final_status == "EXHAUSTED" else 1
    except BlockingIOError:
        print(f"SKIP RUNNING {exp.identity}: execution lock is held", flush=True)
        return 75


def parse_walltime(value):
    if not value:
        return None
    hours, minutes, seconds = map(int, value.split(":"))
    return hours * 3600 + minutes * 60 + seconds
