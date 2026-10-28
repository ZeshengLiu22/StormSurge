#!/usr/bin/env python3
"""Supervise station workers; CUDA contexts are created only in child processes."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import uuid

from common import LAUNCHER, SOURCES, STATIONS, atomic_json, require


def derecho_devices(visible):
    # Preserve PBS order and physical/UUID tokens; never remap a PBS token to ordinal 0.
    tokens = ['0', '1', '2', '3'] if visible is None else [v.strip() for v in visible.split(',')]
    require(len(tokens) == 4 and len(set(tokens)) == 4 and all(tokens) and '-1' not in tokens,
            'Derecho requires four distinct PBS-visible GPUs (or an unset CUDA_VISIBLE_DEVICES on its exclusive node)')
    return tokens


def supervise(specifications, status_path=None):
    """Poll every child, collect every failure, and forward termination to all workers."""
    children, logs, results = [], [], {}
    stopping = None
    deadline = None

    def stop(signum, frame):
        nonlocal stopping, deadline
        stopping = signum
        deadline = time.monotonic() + 30
        for name, child in children:
            if child.poll() is None:
                child.send_signal(signal.SIGTERM)

    previous = {sig: signal.signal(sig, stop) for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP)}
    try:
        for name, command, env, log_path in specifications:
            if stopping:
                break
            log = log_path.open('w') if log_path else None
            if log:
                logs.append(log)
            child = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT,
                                     start_new_session=True)
            children.append((name, child))
            if stopping:
                child.terminate()
        while len(results) < len(children):
            for name, child in children:
                code = child.poll()
                if code is not None and name not in results:
                    results[name] = code
                    print(f'[STATION EXIT] {name}: {code}', flush=True)
            if stopping and time.monotonic() > deadline:
                for name, child in children:
                    if child.poll() is None:
                        os.killpg(child.pid, signal.SIGKILL)
            if len(results) < len(children):
                time.sleep(0.2)
    finally:
        # Also clean up workers if spawning one of the other stations fails.
        for name, child in children:
            if child.poll() is None:
                child.terminate()
        for name, child in children:
            if child.poll() is None:
                try:
                    child.wait(timeout=25)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
        for log in logs:
            log.close()
        for sig, handler in previous.items():
            signal.signal(sig, handler)
    if status_path:
        atomic_json(status_path, dict(stations=results, signal=stopping))
    return 128 + stopping if stopping else int(any(results.values()))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('cluster', choices=('derecho', 'casper'))
    parser.add_argument('source', choices=SOURCES)
    parser.add_argument('station', choices=STATIONS, nargs='?')
    parser.add_argument('--manifest', type=Path, default=LAUNCHER / 'manifest.json')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)
    require((args.cluster == 'casper') == bool(args.station), 'Casper requires one station; Derecho owns all four')
    stations = (args.station,) if args.station else STATIONS
    # CPU preview intentionally does not depend on scheduler GPU variables.
    tokens = None if args.dry_run or args.cluster == 'casper' else derecho_devices(os.environ.get('CUDA_VISIBLE_DEVICES'))
    if not args.dry_run:
        require(os.environ.get('PBS_JOBID'), 'PBS allocation required; use --dry-run')
    directory = Path(__file__).resolve().parent
    log_dir = directory / 'runlogs' / (os.environ.get('PBS_JOBID', 'preview') + '_' + uuid.uuid4().hex)
    if not args.dry_run:
        log_dir.mkdir(parents=True)
    specs = []
    for index, station in enumerate(stations):
        env = os.environ.copy()
        if tokens:
            env['CUDA_VISIBLE_DEVICES'] = tokens[index]
        # Casper's scheduler mask, including an unset mask under cgroup isolation, is untouched.
        for key in ('RANK', 'WORLD_SIZE', 'LOCAL_RANK', 'LOCAL_WORLD_SIZE', 'MASTER_ADDR', 'MASTER_PORT'):
            env.pop(key, None)
        command = [sys.executable, '-u', str(directory / 'station_runner.py'), args.source, station,
                   '--cluster', args.cluster, '--manifest', str(args.manifest)]
        if args.dry_run:
            code = subprocess.call(command + ['--dry-run'], env=env)
            if code:
                return code
        else:
            specs.append((station, command, env, log_dir / f'{station}.log'))
    if args.dry_run:
        return 0
    print(f'[JOB LOGS] {log_dir}', flush=True)
    return supervise(specs, log_dir / 'status.json')


if __name__ == '__main__':
    try:
        sys.exit(main())
    except Exception as error:
        print(f'[FATAL] {error}', file=sys.stderr)
        sys.exit(2)
