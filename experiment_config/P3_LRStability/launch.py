#!/usr/bin/env python3
"""Validate all 31 treatments, then select P3 commands. This Python script never submits jobs."""

import argparse
from pathlib import Path
import shlex
import sys

sys.dont_write_bytecode = True
from p3_common import LRS, STATIONS, require
from validate_configs import validate


def learning_rate(value):
    token = value.removeprefix('LR')
    try:
        numeric = float(token)
    except ValueError:
        raise argparse.ArgumentTypeError('LR must be 5e-3, 2e-3 or 1e-3') from None
    for label, decimal in LRS.items():
        if numeric == float(decimal):
            return decimal
    raise argparse.ArgumentTypeError('Only 0.005, 0.002 and 0.001 are permitted')


def station_name(value):
    for name in STATIONS:
        if name.lower() == value.lower():
            return name
    raise argparse.ArgumentTypeError('Station must be Lewes, Battery or Boston')


def semantic_name(value):
    import re
    token = value.upper().replace('_', '')
    if not re.fullmatch(r'T[01]A[01]S[01]P[01]', token):
        raise argparse.ArgumentTypeError('Use a semantic config such as T1A1S0P0')
    return token


def select(rows, args):
    def reason_matches(row):
        if not args.reason:
            return True
        return any(reason == row['reason'] or reason in row['reason'].split('+') for reason in args.reason)
    return [row for row in rows
            if (not args.station or row['station'] in args.station)
            and (not args.lr or row['learning_rate'] in args.lr)
            and (not args.semantic_config or row['semantic_config'] in args.semantic_config)
            and reason_matches(row)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument('--all', action='store_true', help='Select all 31 runs (also the default).')
    parser.add_argument('--station', type=station_name, action='append')
    parser.add_argument('--lr', type=learning_rate, action='append', help='e.g. 2e-3, LR2e-3 or 0.002')
    parser.add_argument('--reason', action='append', choices=(
        'core_probe', 'p2_degradation_alert', 'core_probe+p2_degradation_alert'),
        help='A single reason includes treatments shared with the other part.')
    parser.add_argument('--semantic-config', type=semantic_name, action='append')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--submit', action='store_true', help='Submit via the sourced launch.sh wrapper.')
    mode.add_argument('--dry-run', action='store_true', help='Print commands only (default).')
    parser.add_argument('--queue-records', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.all and any((args.station, args.lr, args.reason, args.semantic_config)):
        parser.error('--all cannot be combined with subset filters')
    if args.submit and not args.queue_records:
        parser.error('For submission use: source experiment_config/P3_LRStability/launch.sh --submit [filters]')
    repo = args.repo.resolve()
    report = validate(repo, Path(__file__).resolve().parent)
    rows = select(report['rows'], args)
    require(rows, 'No treatments match the requested filters; nothing submitted')
    print(f'P3 validation PASS: 31 unique configs, zero unexpected P2 differences. Selected {len(rows)} runs.',
          file=sys.stderr)
    if args.queue_records:
        print('MODE\t' + ('SUBMIT' if args.submit else 'PREVIEW'))
        for row in rows:
            print(f'RUN\t{row["run_name"]}\t{row["config_path"]}')
    else:
        print('cd ' + shlex.quote(str(repo)))
        for row in rows:
            print(shlex.join(['qsub_local', 'train.sh', row['run_name'], row['config_path']]))
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (ValueError, OSError, KeyError) as error:
        print(f'P3 launch validation FAILED — nothing submitted: {error}', file=sys.stderr)
        sys.exit(1)
