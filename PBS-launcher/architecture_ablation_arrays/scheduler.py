"""Read-only PBS snapshots; array parents include not-yet-instantiated subjobs."""
import getpass
import json
import os
import re
import subprocess

from experiments import JOB_NAMES

ACTIVE = {"Q", "H", "W", "T", "R", "E", "S", "B"}


def indices(spec):
    if not spec or spec == "-":
        return set()
    answer = set()
    for part in str(spec).split("%", 1)[0].split(","):
        match = re.fullmatch(r"(\d+)(?:-(\d+))?(?::(\d+))?", part.strip())
        if not match:
            raise ValueError(f"Invalid index range: {part}")
        lo, hi, step = match.groups()
        lo, hi, step = int(lo), int(hi or lo), int(step or 1)
        if hi < lo or step < 1 or hi > 100000:
            raise ValueError(f"Invalid index range: {part}")
        answer.update(range(lo, hi + 1, step))
    return answer


def contiguous(values):
    chunks = []
    for value in sorted(set(values)):
        if chunks and chunks[-1][-1] + 1 == value:
            chunks[-1].append(value)
        else:
            chunks.append([value])
    return chunks


def snapshot(job_id=None):
    # On the installed PBS release, -u selects the text table even alongside
    # -f -F json. Read JSON without -u, then filter ownership locally.
    selection = [job_id] if job_id else []
    proc = subprocess.run(["/opt/pbs/bin/qstat", "-x", "-t", "-f", "-F", "json", *selection],
                          env={**os.environ, "PBS_DEFAULT": "desched1"}, capture_output=True, text=True, timeout=30)
    if proc.returncode:
        raise RuntimeError(f"PBS query failed; refusing to assume jobs are absent: {proc.stderr.strip()}")
    data = json.loads(proc.stdout)
    if data.get("pbs_server", "").split(".")[0] != "desched1":
        raise RuntimeError(f"Wrong PBS server: {data.get('pbs_server')}")
    return {job_id: info for job_id, info in data.get("Jobs", {}).items()
            if info.get("Job_Owner", "").split("@", 1)[0] == getpass.getuser()}


def relevant(exp, job):
    variables = job.get("Variable_List", {})
    return (variables.get("SS_GROUP") == exp.group and variables.get("SS_ENCODER") == exp.encoder
            or job.get("Job_Name") == f"{JOB_NAMES[exp.group]}_{exp.encoder}")


def job_for(exp, jobs):
    candidates = []
    for job_id, info in jobs.items():
        if not relevant(exp, info):
            continue
        match = re.search(r"\[(\d+)\]", job_id)
        if match:
            if int(match[1]) != exp.index:
                continue
        elif "[]" in job_id:
            if info.get("job_state") not in ACTIVE:
                continue
            remaining = info.get("array_indices_remaining", info.get("array_indices_submitted", ""))
            if exp.index not in indices(remaining):
                continue
            # A begun parent does not mean all remaining indices are running.
            info = {**info, "job_state": "Q" if info.get("job_state") == "B" else info.get("job_state")}
        else:
            continue
        candidates.append({**info, "job_id": job_id})
    # Prefer a live child to its queued parent, then live parents, then history.
    return max(candidates, key=lambda j: (j.get("job_state") in ACTIVE,
        j.get("job_state") in {"R", "E", "S"}, "[]" not in j["job_id"],
        int(j["job_id"].split("[", 1)[0].split(".", 1)[0])), default=None)


def reservation(exp, jobs):
    """An ambiguous qsub outcome stays reserved until explicitly reconciled."""
    base = exp.state_base / "submissions"
    for path in sorted(base.glob("*/request.json")):
        directory = path.parent
        if (directory / "released.json").exists() or (directory / "rejected.json").exists():
            continue
        item = json.loads(path.read_text())
        if item["group"] != exp.group or item["encoder"] != exp.encoder or exp.index not in item["indices"]:
            continue
        if not (directory / "submitted.json").exists():
            return item["token"]
        # A confirmed submission is covered by the live scheduler snapshot. In
        # offline mode the caller must conservatively retain its reservation.
        if jobs is None:
            return item["token"]
    return None
