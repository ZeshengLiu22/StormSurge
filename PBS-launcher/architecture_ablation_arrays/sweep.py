#!/usr/bin/env python3
"""Preview, run, inspect, and selectively submit the architecture ablation arrays."""
import argparse
from collections import Counter
from contextlib import nullcontext
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import uuid

from experiments import (ENCODERS, GROUPS, JOB_NAMES, Experiment, digest, preview, successful_result)
import scheduler
from state import (attempts, atomic_json, budget_used, busy, classify, decision, lock, now,
                   policy, read_json, run_experiment, state_marker, terminal_result)


def all_experiments(root, groups, encoders, selected=None):
    for group in groups:
        for encoder in encoders:
            for index in sorted(selected if selected is not None else range(144)):
                yield Experiment(root, group, encoder, index)


def report(exp, jobs, max_attempts=3):
    records = attempts(exp)
    rules = policy(exp, max_attempts)
    job = scheduler.job_for(exp, jobs or {})
    last_start = records[-1] if records else {}
    row = {"identity": exp.identity, "group": exp.group, "encoder": exp.encoder, "station": exp.station,
           "history": exp.history, "temporal_block": exp.temporal, "index": exp.index,
           "attempt": last_start.get("attempt", 0), "budget_used": budget_used(records, rules),
           "max_attempts": rules["max_attempts"], "job_id": (job or {}).get("job_id", last_start.get("job_id")),
           "start": last_start.get("start"), "end": None, "exit_status": None,
           "pbs_state": (job or {}).get("job_state"), "pbs_exit_status": (job or {}).get("Exit_status"),
           "pbs_run_count": (job or {}).get("run_count"), "output_dir": last_start.get("output_dir")}
    def finish(status, reason, **extra):
        return row | {"status": status, "reason": reason} | extra
    if busy(exp):
        return finish("RUNNING", "EXECUTION_LOCK_HELD")
    success = successful_result(exp)
    if success:
        end = read_json(Path(last_start.get("record_dir", "/nonexistent")) / "end.json", {})
        return finish("SUCCESS", "VERIFIED_RESULT", output_dir=success, exit_status=0, end=end.get("end"))
    if job and job.get("job_state") in scheduler.ACTIVE:
        return finish("RUNNING" if job["job_state"] in {"R", "E", "S"} else "QUEUED", "PBS_" + job["job_state"])
    token = scheduler.reservation(exp, jobs)
    if token:
        return finish("QUEUED", "SUBMISSION_RESERVED:" + token)
    # A later submission of the same index must not supply the cause of an
    # earlier attempt's termination. Use that attempt's exact PBS subjob.
    last_job_id = last_start.get("job_id")
    last_job = ({**jobs[last_job_id], "job_id": last_job_id}
                if jobs and last_job_id in jobs else None)
    last = terminal_result(exp, records[-1], last_job) if records else None
    if last:
        row.update(end=last.get("end"), exit_status=last.get("exit_status"),
                   job_id=last_job_id, pbs_run_count=last_start.get("pbs_run_count"),
                   pbs_state=(last_job or {}).get("job_state"), pbs_exit_status=last.get("pbs_exit_status"))
    status, reason = decision(exp, records, rules, last)
    if status != "PENDING":
        return finish(status, reason)
    marker = read_json(exp.state / "state.json", {})
    if marker.get("reason") == "PREFLIGHT_FAILED" and not rules.get("retry_preflight"):
        return finish("FAILED", "PREFLIGHT_FAILED")
    if records and records[-1]["attempt"] > max(rules["base_attempt"], rules.get("allow_failed_through", 0)):
        return finish(last["status"], last["reason"])
    if not records and job and job.get("job_state") == "F" and not rules.get("manual_update"):
        result = classify(job.get("Exit_status"), exp.state / "absent.log", job)
        return finish(result["status"], "PBS_WITHOUT_ATTEMPT:" + result["reason"])
    return finish("PENDING", "ELIGIBLE")


def aggregates(rows):
    return {field: {value: dict(Counter(r["status"] for r in rows if r[field] == value))
                    for value in sorted({r[field] for r in rows})}
            for field in ("group", "encoder", "station", "history")}


def query(args):
    if getattr(args, "offline", False):
        if getattr(args, "submit", False) or getattr(args, "apply", False):
            raise ValueError("Offline mode cannot submit jobs or alter retry controls")
        return None
    return scheduler.snapshot()


def command_for(exp, values, args, token):
    span = f"{values[0]}-{values[-1]}"
    if args.max_concurrent != "unlimited":
        span += f"%{min(int(args.max_concurrent), len(values))}"
    variables = f"SS_GROUP={exp.group},SS_ENCODER={exp.encoder},MAX_ATTEMPTS={args.max_attempts},SS_SUBMIT_TOKEN={token}"
    cmd = ["/opt/pbs/bin/qsub", "-J", span, "-v", variables]
    walltime = args.cnn_walltime if exp.encoder == "CNN" else args.gnn_walltime
    if walltime:
        cmd += ["-l", f"walltime={walltime}"]
    return cmd + [str(exp.launcher)]


def submit_selected(args):
    if args.max_attempts < 1 or args.max_concurrent != "unlimited" and int(args.max_concurrent) < 1:
        raise ValueError("Attempt and concurrency limits must be positive; or use unlimited concurrency")
    for value in (args.cnn_walltime, args.gnn_walltime):
        if value and (not re.fullmatch(r"\d+:[0-5]\d:[0-5]\d", value) or not any(int(x) for x in value.split(":"))):
            raise ValueError("Walltime must be a positive HH:MM:SS value")
    state_base = Experiment(args.root, args.groups[0], args.encoders[0], 0).state_base
    context = lock(state_base / "submission.lock") if args.submit else nullcontext()
    with context:
        jobs = query(args)
        selected = scheduler.indices(args.indices) if args.indices else None
        total = 0
        for group in args.groups:
            for encoder in args.encoders:
                experiments = list(all_experiments(args.root, [group], [encoder], selected))
                chosen = []
                excluded = Counter()
                for exp in experiments:
                    exp.validate()
                    row = report(exp, jobs, args.max_attempts)
                    if row["status"] in {"PENDING", "INTERRUPTED", "PREEMPTED"}:
                        chosen.append(exp.index)
                    else:
                        excluded[row["status"]] += 1
                print(f"# {group}/{encoder}: eligible={len(chosen)}, excluded={dict(excluded)}", flush=True)
                # PBS documents X-Y[:Z], not comma-separated arbitrary subsets.
                # Contiguous blocks preserve the original deterministic indices.
                for block in scheduler.contiguous(chosen):
                    exp = Experiment(args.root, group, encoder, block[0])
                    token = uuid.uuid4().hex if args.submit else "PREVIEW"
                    command = command_for(exp, block, args, token)
                    print(shlex.join(command), flush=True)
                    if args.submit:
                        directory = state_base / "submissions" / token
                        atomic_json(directory / "request.json", {"token": token, "time": now(), "group": group,
                            "encoder": encoder, "indices": block, "max_attempts": args.max_attempts,
                            "command": command}, exclusive=True)
                        # A timeout or unparseable response remains reserved. Never
                        # silently retry an ambiguously accepted qsub request.
                        proc = subprocess.run(command, cwd=args.root, capture_output=True, text=True, timeout=60)
                        if proc.returncode or not re.fullmatch(r"\d+\[\]\.desched1(?:\.[A-Za-z0-9_.-]+)?", proc.stdout.strip()):
                            raise RuntimeError(f"Unconfirmed qsub ({token}); inspect PBS then use release-reservation. {proc.stderr.strip()} {proc.stdout.strip()}")
                        atomic_json(directory / "submitted.json", {"job_id": proc.stdout.strip(), "time": now()}, exclusive=True)
                        print(f"Submitted {proc.stdout.strip()}", flush=True)
                    total += len(block)
        print(f"# {'Submitted' if args.submit else 'Preview only:'} {total} experiments; no inter-array dependencies.")


def update_budget(args):
    exp = Experiment(args.root, args.group, args.encoder, args.index).validate()
    jobs = query(args)
    job = scheduler.job_for(exp, jobs or {})
    if job and job.get("job_state") in scheduler.ACTIVE or scheduler.reservation(exp, jobs):
        raise ValueError("Experiment is queued/running/reserved; finish or cancel it before changing its budget")
    if successful_result(exp):
        raise ValueError("A verified success is never reset or resubmitted by this tool")
    context = lock(exp.state / "execution.lock") if args.apply else nullcontext()
    with context:
        if busy(exp) and not args.apply:
            raise ValueError("Experiment is currently executing")
        records = attempts(exp)
        previous = policy(exp)
        updated = dict(previous, manual_update=now())
        if args.max_attempts is not None:
            if args.max_attempts < 1:
                raise ValueError("max-attempts must be positive")
            updated["max_attempts"] = args.max_attempts
        if args.reset:
            updated.update(base_attempt=max((r["attempt"] for r in records), default=0),
                           config_sha256=digest(exp.config), retry_preflight=True)
        if args.retry_failed:
            updated.update(allow_failed_through=max((r["attempt"] for r in records), default=0), retry_preflight=True)
        print(json.dumps({"identity": exp.identity, "before": previous, "after": updated, "apply": args.apply}, indent=2))
        if args.apply:
            atomic_json(exp.state / "control_events" / f"{uuid.uuid4().hex}.json",
                        {"time": now(), "before": previous, "after": updated}, exclusive=True)
            atomic_json(exp.state / "policy.json", updated)
            state_marker(exp, "PENDING", reason="MANUAL_BUDGET_UPDATE")


def release_reservation(args):
    base = Experiment(args.root, "LR3e-3", "CNN", 0).state_base
    if not re.fullmatch(r"[0-9a-f]{32}", args.token):
        raise ValueError("Invalid submission token")
    context = lock(base / "submission.lock") if args.apply else nullcontext()
    with context:
        jobs = query(args)
        directory = base / "submissions" / args.token
        item = read_json(directory / "request.json")
        if not item:
            raise ValueError("Unknown submission token")
        for exp in all_experiments(args.root, [item["group"]], [item["encoder"]], item["indices"]):
            job = scheduler.job_for(exp, jobs or {})
            if busy(exp) or job and job.get("job_state") in scheduler.ACTIVE:
                raise ValueError(f"Still queued/running: {exp.identity}")
        print(f"{'Release' if args.apply else 'Would release'} reservation {args.token}; preserve all attempt/result records")
        if args.apply:
            atomic_json(directory / "released.json", {"time": now(), "reason": "MANUAL_PBS_CHECK"}, exclusive=True)


def validate(args):
    rows = list(all_experiments(args.root, args.groups, args.encoders))
    for group in args.groups:
        for encoder in args.encoders:
            configs = [e.config for e in rows if e.group == group and e.encoder == encoder]
            actual = set((args.root / "configs" / f"architecture_ablation_{group}").glob(f"*H/*/train_config_*_{encoder}_*.sh"))
            if len(configs) != 144 or len(set(configs)) != 144 or set(configs) != actual:
                raise ValueError(f"Expected exactly 144 unique, exhaustive configs: {group}/{encoder}")
    for exp in rows:
        exp.validate()
        if args.commands:
            _, command = preview(exp)
            if command["epochs"] != "300" or command["lr"] != GROUPS[exp.group][0] or command["max_grad_norm"] != GROUPS[exp.group][1]:
                raise ValueError(f"Effective arguments mismatch: {exp.identity}")
        if args.mapping:
            print(f"{exp.group}\t{exp.encoder}\t{exp.index}\t{exp.station}\t{exp.history}\t{exp.temporal}\t{exp.config}")
    print(f"Validated {len(rows)} unique configurations ({144} per selected group/encoder); no training or submission.")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    sub = parser.add_subparsers(dest="command", required=True)
    def selection(p, overnight=False):
        p.add_argument("--groups", nargs="+", choices=GROUPS, default=["LR3e-3", "LR5e-3_Clip0.25"] if overnight else list(GROUPS))
        p.add_argument("--encoders", nargs="+", choices=ENCODERS, default=["CNN"] if overnight else list(ENCODERS))
    def one(p):
        p.add_argument("--group", required=True, choices=GROUPS)
        p.add_argument("--encoder", required=True, choices=ENCODERS)
        p.add_argument("--index", type=int, required=True)
    p = sub.add_parser("validate")
    selection(p)
    p.add_argument("--commands", action="store_true", help="also run train.sh DRY_RUN for every config")
    p.add_argument("--mapping", action="store_true")
    p = sub.add_parser("preview")
    one(p)
    p = sub.add_parser("run", help="PBS-only execution; DRY_RUN=1 is always honored")
    one(p)
    p.add_argument("--max-attempts", type=int, default=int(os.environ.get("MAX_ATTEMPTS", "3")))
    for name in ("submit", "resubmit"):
        p = sub.add_parser(name, help="preview by default; only pending/interrupted experiments are eligible")
        selection(p, True)
        p.add_argument("--indices", help="filter original indices, e.g. 0-3,8,12-15")
        p.add_argument("--max-concurrent", default="24",
                       help="maximum running subjobs per array (default: 24), or unlimited")
        p.add_argument("--max-attempts", type=int, default=3)
        p.add_argument("--cnn-walltime")
        p.add_argument("--gnn-walltime")
        action = p.add_mutually_exclusive_group()
        action.add_argument("--submit", action="store_true")
        action.add_argument("--dry-run", action="store_true")
        p.add_argument("--offline", action="store_true", help="preview only, no PBS query")
    p = sub.add_parser("status")
    selection(p)
    p.add_argument("--states", nargs="+", choices=["PENDING", "QUEUED", "RUNNING", "SUCCESS", "PREEMPTED", "INTERRUPTED", "FAILED", "EXHAUSTED"])
    p.add_argument("--json", action="store_true")
    p.add_argument("--counts-only", action="store_true")
    p.add_argument("--offline", action="store_true")
    p = sub.add_parser("history")
    one(p)
    p = sub.add_parser("budget", help="explicit extension/reset; keeps all previous attempts and outputs")
    one(p)
    p.add_argument("--max-attempts", type=int)
    p.add_argument("--reset", action="store_true")
    p.add_argument("--retry-failed", action="store_true")
    p.add_argument("--apply", action="store_true")
    p.add_argument("--offline", action="store_true")
    p = sub.add_parser("release-reservation")
    p.add_argument("--token", required=True)
    p.add_argument("--apply", action="store_true")
    p.add_argument("--offline", action="store_true")
    args = parser.parse_args(argv)
    args.root = args.root.absolute()
    if hasattr(args, "groups"):
        args.groups = list(dict.fromkeys(args.groups))
        args.encoders = list(dict.fromkeys(args.encoders))
    if args.command == "validate":
        validate(args)
    elif args.command in {"preview", "run"}:
        exp = Experiment(args.root, args.group, args.encoder, args.index).validate()
        if args.command == "preview" or os.environ.get("DRY_RUN") == "1":
            print(f"# {exp.identity}, array index={exp.index}")
            print(preview(exp)[0], end="")
        else:
            if not os.environ.get("PBS_JOBID") or os.environ.get("PBS_ARRAY_INDEX") != str(exp.index):
                raise ValueError("Execution requires matching PBS_JOBID and PBS_ARRAY_INDEX")
            try:
                jobs = scheduler.snapshot(os.environ["PBS_JOBID"])
                info = jobs.get(os.environ["PBS_JOBID"], {})
            except (RuntimeError, OSError, subprocess.SubprocessError) as exc:
                print(f"PBS diagnostics unavailable: {exc}; persistent attempt budget and execution lock still apply", file=sys.stderr)
                info = {}
            return run_experiment(exp, args.max_attempts, info)
    elif args.command in {"submit", "resubmit"}:
        submit_selected(args)
    elif args.command == "status":
        jobs = query(args)
        rows = [report(exp, jobs) for exp in all_experiments(args.root, args.groups, args.encoders)]
        counts = aggregates(rows)
        visible = [row for row in rows if not args.states or row["status"] in args.states]
        if args.json:
            print(json.dumps({"scheduler_queried": jobs is not None, "experiments": visible, "counts": counts}, indent=2))
        else:
            if jobs is None:
                print("# OFFLINE: queued/running PBS state is unavailable; submission is disabled")
            if not args.counts_only:
                fields = ("identity", "index", "status", "reason", "attempt", "budget_used", "max_attempts", "job_id", "start", "end", "exit_status", "output_dir")
                print("\t".join(fields))
                for row in visible:
                    print("\t".join(str(row.get(key) if row.get(key) is not None else "-") for key in fields))
            print("# Aggregates: " + json.dumps(counts, sort_keys=True))
    elif args.command == "history":
        exp = Experiment(args.root, args.group, args.encoder, args.index)
        print(json.dumps([{**row, "end_record": read_json(Path(row["record_dir"]) / "end.json")} for row in attempts(exp)], indent=2))
    elif args.command == "budget":
        if not (args.reset or args.retry_failed or args.max_attempts is not None):
            raise ValueError("Choose --max-attempts, --reset, or --retry-failed")
        update_budget(args)
    elif args.command == "release-reservation":
        release_reservation(args)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(2)
