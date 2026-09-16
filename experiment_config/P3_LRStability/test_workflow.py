#!/usr/bin/env python3
"""Exercise P3 selection and pre-submission rejection with a mock queue only."""

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

sys.dont_write_bytecode = True
from p3_common import read_manifest, require, results_snapshot, sha256

MOCK = r'''set -eu
qsub_local() {
    printf '%s\t%s\t%s\n' "$@" >> "$P3_TEST_QUEUE_LOG"
}
p3_test_cwd="$PWD"
source "$1" "${@:2}"
[[ "$PWD" == "$p3_test_cwd" ]]
'''


def invoke(directory, repo, flags, scratch, mock=True):
    log = scratch / 'mock_queue.tsv'
    log.write_text('')
    env = dict(PATH='/usr/bin:/bin', PYTHONDONTWRITEBYTECODE='1',
               P3_REPO_ROOT=str(repo), P3_TEST_QUEUE_LOG=str(log))
    script = MOCK if mock else 'source "$1" "${@:2}"'
    completed = subprocess.run(['bash', '--noprofile', '--norc', '-c', script, 'bash',
                                str(directory / 'launch.sh'), *flags],
                               cwd=scratch, env=env, text=True, capture_output=True, timeout=60)
    submissions = [line.split('\t') for line in log.read_text().splitlines()]
    return completed, submissions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    repo, directory = args.repo.resolve(), Path(__file__).resolve().parent
    before_results = results_snapshot(repo)
    manifest_hash = sha256(directory / 'manifest.csv')
    rows = read_manifest(directory / 'manifest.csv')
    by_name = {row['run_name']: row for row in rows}
    checks = []
    # These cardinalities are the independent user-requested acceptance criteria.
    cases = [
        ('all', ['--all'], 31),
        ('Battery', ['--station', 'Battery'], 15),
        ('Boston', ['--station', 'Boston'], 13),
        ('Lewes', ['--station', 'Lewes'], 3),
        ('LR2e-3', ['--lr', 'LR2e-3'], 12),
        ('LR5e-3 fresh reruns', ['--lr', '0.005'], 7),
        ('degradation alerts including overlaps', ['--reason', 'p2_degradation_alert'], 21),
        ('core including overlaps', ['--reason', 'core_probe'], 16),
        ('overlaps only', ['--reason', 'core_probe+p2_degradation_alert'], 6),
        ('semantic config', ['--semantic-config', 'T1_A1_S0_P0'], 6),
        ('filter intersection', ['--station', 'Boston', '--lr', '0.001',
                                 '--reason', 'p2_degradation_alert'], 3),
    ]
    with tempfile.TemporaryDirectory(prefix='p3-workflow-') as temp:
        scratch = Path(temp)
        for name, flags, count in cases:
            result, queued = invoke(directory, repo, ['--submit', *flags], scratch)
            require(result.returncode == 0, f'{name}: {result.stderr}')
            require(len(queued) == count and len({q[1] for q in queued}) == count,
                    f'{name}: wrong/duplicate selection: {queued}')
            for entry, label, config in queued:
                require(entry == 'train.sh' and config == by_name[label]['config_path'],
                        f'{name}: wrong local queue calling convention')
                require(config.startswith('experiment_config/P3_LRStability/'), 'Selected a non-P3 config')
            checks.append(dict(check=name, status='PASS', mocked_submissions=count,
                               run_names=[q[1] for q in queued]))
            print(f'PASS {name}: {count} mock queue calls', flush=True)
        result, queued = invoke(directory, repo, ['--station', 'Battery'], scratch)
        require(result.returncode == 0 and not queued and result.stdout.count('qsub_local train.sh') == 15,
                'Default preview must never submit')
        checks.append(dict(check='default preview never submits', status='PASS'))
        result, queued = invoke(directory, repo, ['--help'], scratch)
        require(result.returncode == 0 and not queued and '--semantic-config' in result.stdout, 'Help failed')
        checks.append(dict(check='help never submits', status='PASS'))
        for name, flags in [
            ('invalid LR', ['--lr', '0.003']),
            ('empty selection', ['--station', 'Lewes', '--semantic-config', 'T1A1S0P0']),
            ('all mixed with a filter', ['--all', '--station', 'Boston']),
        ]:
            result, queued = invoke(directory, repo, ['--submit', *flags], scratch)
            require(result.returncode != 0 and not queued, f'{name}: failed to reject before submission')
            checks.append(dict(check=name, status='PASS', rejected_before_submission=True))
        result, queued = invoke(directory, repo, ['--submit', '--station', 'Lewes'], scratch, mock=False)
        require(result.returncode == 2 and not queued and 'qsub_local is unavailable' in result.stderr,
                'Missing qsub_local did not produce actionable failure')
        checks.append(dict(check='missing queue helper', status='PASS'))
        for mutation in ('batch size', 'extra LR sweep', 'output root', 'missing row', 'duplicate row'):
            copied = scratch / mutation.replace(' ', '_')
            shutil.copytree(directory, copied)
            config = copied / Path(rows[0]['config_path']).name
            if mutation in ('missing row', 'duplicate row'):
                changed = list(rows)
                if mutation == 'missing row':
                    changed.pop()
                else:
                    changed[-1] = changed[0]
                with (copied / 'manifest.csv').open('w', newline='') as handle:
                    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                    writer.writeheader()
                    writer.writerows(changed)
            else:
                replacements = {
                    'batch size': ('BATCH_SIZE=256', 'BATCH_SIZE=128'),
                    'extra LR sweep': ('LR_LIST=("5e-3")', 'LR_LIST=("5e-3" "2e-3")'),
                    'output root': ('./All_Results/P3_LRStability', './All_Results/P2_PeakAwareFactorial'),
                }
                old, new = replacements[mutation]
                require(old in config.read_text(), f'Invalid rejection fixture: {mutation}')
                config.write_text(config.read_text().replace(old, new))
            # Even selecting Boston must reject a bad unselected Lewes treatment.
            result, queued = invoke(copied, repo, ['--submit', '--station', 'Boston'], scratch)
            require(result.returncode != 0 and not queued and 'nothing submitted' in result.stderr,
                    f'{mutation}: validation failed to block the queue')
            checks.append(dict(check='reject ' + mutation, status='PASS', rejected_before_submission=True))
            print(f'PASS reject {mutation} before any mock submission', flush=True)
        # Generation must be idempotent without erasing manual changes.
        reproduced = scratch / 'reproduced'
        result = subprocess.run([sys.executable, '-B', str(directory / 'generate_configs.py'),
                                 '--repo', str(repo), '--config-dir', str(reproduced)],
                                text=True, capture_output=True, timeout=30)
        require(result.returncode == 0, result.stderr)
        for path in reproduced.iterdir():
            require(path.read_bytes() == (directory / path.name).read_bytes(), f'Non-reproducible file: {path.name}')
        config = next(reproduced.glob('train_config_*.sh'))
        config.write_text(config.read_text() + '# locally edited\n')
        digest = sha256(config)
        result = subprocess.run([sys.executable, '-B', str(directory / 'generate_configs.py'),
                                 '--repo', str(repo), '--config-dir', str(reproduced)],
                                text=True, capture_output=True, timeout=30)
        require(result.returncode != 0 and sha256(config) == digest, 'Regeneration overwrote manual edits')
        checks.append(dict(check='reproducible generation; edited files preserved', status='PASS'))
    require(results_snapshot(repo) == before_results, 'Workflow checks changed real result artifacts')
    require(sha256(directory / 'manifest.csv') == manifest_hash, 'Workflow checks changed the real manifest')
    report = dict(status='PASS', checked_utc=datetime.now(timezone.utc).isoformat(),
                  checks=checks, real_queue_submissions=0, training_started=False,
                  results_unchanged=True)
    if args.report:
        args.report.write_text(json.dumps(report, indent=2) + '\n')
    print(f'PASS: {len(checks)} workflow checks; no real queue submissions or training.')


if __name__ == '__main__':
    main()
