#!/usr/bin/env python3
"""Print submission commands by default. Only --submit actually calls qsub."""
import argparse
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys

from common import LAUNCHER, SOURCES, STATIONS, atomic_json, load_manifest, require, run_lock


def active_job(jobid):
    result = subprocess.run(['qstat', '-f', '-F', 'json', jobid], capture_output=True, text=True)
    if result.returncode:
        require('Unknown Job Id' in result.stderr or 'Unknown Job' in result.stderr,
                f'Cannot establish status of {jobid}: {result.stderr}')
        return False
    jobs = json.loads(result.stdout)['Jobs']
    return any(job['job_state'] not in ('F', 'X') for job in jobs.values())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('cluster', choices=('derecho', 'casper'))
    parser.add_argument('--manifest', type=Path, default=LAUNCHER / 'manifest.json')
    parser.add_argument('--submit', action='store_true')
    parser.add_argument('--source', choices=SOURCES, help='Optional single-source pilot')
    parser.add_argument('--station', choices=STATIONS, help='Optional Casper pilot station')
    parser.add_argument('--walltime', help='Override PBS walltime, HH:MM:SS')
    args = parser.parse_args(argv)
    require(not args.station or args.cluster == 'casper', '--station applies only to Casper')
    if args.walltime:
        require(re.fullmatch(r'\d{2,3}:[0-5]\d:[0-5]\d', args.walltime), 'Invalid walltime')
    manifest = load_manifest(args.manifest)
    rows = [r for r in manifest['runs'] if (not args.source or r['source'] == args.source) and
            (not args.station or r['station'] == args.station)]
    scripts = sorted({r[args.cluster + '_pbs'] for r in rows})
    commands = []
    exports = {k: v for k, v in os.environ.items() if k.startswith('INFER_') or k in ('PYTHON_BIN', 'CUDA_MODULE')}
    require(all(',' not in v and '\n' not in v for v in exports.values()), 'PBS export values must not contain commas/newlines')
    for script in scripts:
        command = ['qsub']
        if args.walltime:
            command += ['-l', 'walltime=' + args.walltime]
        if exports:
            command += ['-v', ','.join(f'{k}={v}' for k, v in sorted(exports.items()))]
        command.append(script)
        commands.append(command)
    if not args.submit:
        for command in commands:
            print(shlex.join(command))
        print(f'# {len(commands)} jobs; preview only. Choose one cluster, then add --submit.')
        return 0
    from preflight import validate
    validate(args.manifest)
    # Serialize submissions and refuse overlap with either strategy already recorded here.
    root = Path(manifest['result_root']) / '.submissions'
    with run_lock(root):
        ledger = root / 'jobs.json'
        records = json.loads(ledger.read_text()) if ledger.exists() else []
        active = [record for record in records if active_job(record['jobid'])]
        require(not active, f'Previous inference jobs are still active: {active}. Do not submit both strategies simultaneously.')
        for command in commands:
            result = subprocess.run(command, capture_output=True, text=True, check=True)
            jobid = result.stdout.strip()
            require(bool(re.fullmatch(r'\d+(?:\.[\w.-]+)?', jobid)), f'Unexpected qsub response: {jobid!r}; inspect queue before retrying')
            records.append(dict(cluster=args.cluster, jobid=jobid, command=command))
            atomic_json(ledger, records)
            print(jobid, flush=True)
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except Exception as error:
        print(f'[SUBMISSION FAILED] {error}', file=sys.stderr)
        sys.exit(1)
