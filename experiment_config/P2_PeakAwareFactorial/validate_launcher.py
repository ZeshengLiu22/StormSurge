#!/usr/bin/env python3
"""Regress the opt-in run tag using DRY_RUN only; never execute train.py."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import tempfile
from types import SimpleNamespace

sys.dont_write_bytecode = True
BASELINE_COMMIT = '3bfe08c0978a4b9278e5536debf1852e9fceb54f'
STAMP = '20000101_000000'


def run(command, repo, environment=None):
    completed = subprocess.run(command, cwd=repo, env=environment, text=True,
                               capture_output=True, timeout=30)
    assert completed.returncode == 0, (command, completed.stdout, completed.stderr)
    return completed.stdout


def dry_run(config, repo, launcher, parse_args, extra_env=None):
    environment = dict(PATH='/usr/bin:/bin', DRY_RUN='1', PACT_RUNSTAMP=STAMP,
                       SESSION_NAME='P2_Validation', PYTHONDONTWRITEBYTECODE='1')
    environment.update(extra_env or {})
    log = run(['bash', str(launcher), str(config)], repo, environment)
    lines = [line for line in log.splitlines() if line.startswith('CMD: ')]
    assert len(lines) == 1, (config, lines)
    command = shlex.split(lines[0][5:])
    assert command[0] == 'train.py', command
    assert 'DRY_RUN=1' in log and 'launching inside tmux' not in log, log
    args = vars(parse_args(command[1:]))
    args = {key: str(value) if isinstance(value, Path) else value for key, value in args.items()}
    return dict(exit_code=0, command=command, cmd_line=lines[0], args=args, stdout=log)


def tail_semantics():
    # A tiny CPU-only forward calculation; no model, dataset, backward pass,
    # optimizer, epoch, training entry point, queue or tmux invocation.
    import torch
    from emulator.training.losses import ForecastLoss, LossConfig

    torch.set_num_threads(1)
    prediction = torch.zeros((2, 2))
    target = torch.tensor([[0., 0.], [2., 2.]])
    stats = dict(y_mean=torch.zeros(2), y_std=torch.ones(2))
    output = SimpleNamespace(body=None)
    for irrelevant_lambda in (0., .025, .10, 1000.):
        config = LossConfig(loss_mode='mse', tail_lambda=irrelevant_lambda)
        # If mse enters the tail branch, comparing to this sentinel must fail.
        objective = ForecastLoss(config, stats, peak_threshold=object(), wmse_threshold=1.)
        assert objective(output, prediction, target).item() == 2.
    config = LossConfig(loss_mode='mse_tail', tail_lambda=.025, tail_frac=.05)
    objective = ForecastLoss(config, stats, peak_threshold=1., wmse_threshold=1.)
    assert objective(output, prediction, target).item() == 3.
    return dict(status='PASS', mse_tail_branch_unreachable=True,
                tested_inactive_lambdas=[0, .025, .10, 1000], mse_objective=2.,
                mse_tail_objective=3., active_tail_lambda=.025,
                training_started=False, device='cpu', backward_called=False)


def validate(repo, launcher, destination, baseline_launcher=None):
    sys.path.insert(0, str(repo))
    from emulator.training.arguments import parse_args

    with tempfile.TemporaryDirectory(prefix='p2-launcher-regression-') as temporary:
        temporary = Path(temporary)
        baseline = baseline_launcher or temporary / 'original_train.sh'
        if baseline_launcher is None:
            baseline.write_text(run(['git', 'show', f'{BASELINE_COMMIT}:train.sh'], repo))
        records = []
        configs = sorted([*repo.glob('experiment_config/P0_QuickRun/train_config_*.sh'),
                          *repo.glob('experiment_config/P1_WeightProbe/train_config_*.sh')])
        assert len(configs) == 64
        for config in configs:
            assert not re.search(r'^\s*(?:export )?PYTHON_RUN_TAG_BASE=', config.read_text(), re.M)
            before = dry_run(config, repo, baseline, parse_args)
            after = dry_run(config, repo, launcher, parse_args)
            assert before['cmd_line'] == after['cmd_line'], config
            assert before['args'] == after['args'], config
            records.append(dict(config_path=str(config.relative_to(repo)),
                                status='PASS', entire_cmd_line_identical=True,
                                baseline_cmd_line=before['cmd_line'],
                                updated_cmd_line=after['cmd_line'],
                                run_tag=after['args']['run_tag']))
        # Explicit empty and absent both select exactly the historical tag.
        empty_checks = []
        for group in ('P0_QuickRun', 'P1_WeightProbe'):
            config = next(path for path in configs if path.parent.name == group)
            absent = dry_run(config, repo, launcher, parse_args)
            empty = dry_run(config, repo, launcher, parse_args, {'PYTHON_RUN_TAG_BASE': ''})
            assert absent['cmd_line'] == empty['cmd_line']
            empty_checks.append(str(config.relative_to(repo)))

        probes = []
        p0 = repo / 'experiment_config/P0_QuickRun/train_config_NCEP_CBBT_24h_dual_mse.sh'
        for enabled in (False, True):
            name = f'NCEP_CBBT_24h_dual_direct_T{int(enabled)}_A0_S0_P1'
            text = p0.read_text().replace('LOSS_MODE_LIST=("mse")',
                                         'LOSS_MODE_LIST=("mse_tail")' if enabled else 'LOSS_MODE_LIST=("mse")')
            text = re.sub(r'^PACT_RUN_NAME=.*$', f'PACT_RUN_NAME="{name}"', text, flags=re.M)
            text = text.replace('./All_Results/P0_QuickRun', './All_Results/P2_PeakAwareFactorial')
            text += 'PYTHON_RUN_TAG_BASE="${PACT_RUN_NAME}"\nPEAK_LOSS_WEIGHT=0.0035\n'
            config = temporary / f'{name}.sh'
            config.write_text(text)
            probe = dry_run(config, repo, launcher, parse_args)
            args = probe['args']
            assert args['run_tag'] == f'{name}_{STAMP}'
            assert Path(args['output_dir']).resolve() == repo / 'All_Results/P2_PeakAwareFactorial' / f'{name}__{STAMP}'
            if enabled:
                assert args['loss_mode'] == 'mse_tail' and args['tail_lambda'] == .025
                assert '--tail_lambda' in probe['command']
            else:
                assert args['loss_mode'] == 'mse'
                assert '--tail_lambda' not in probe['command']
            probe.update(semantic_name=name, tail_enabled=enabled,
                         active_tail_lambda=.025 if enabled else 0,
                         configured_tail_lambda=.025)
            probes.append(probe)

        # Exercise only the actual snapshot function in isolation, never the
        # training launcher branch that creates run directories.
        source = launcher.read_text()
        snapshot_function = source.split('write_resolved_config() {', 1)[1].split('\n}\n', 1)[0]
        snapshot_script = 'set -eu\nwrite_resolved_config() {' + snapshot_function + '\n}\n'
        snapshot_script += 'CONFIG_PATH=probe.sh\nPYTHON_RUN_TAG_BASE=semantic_probe\nwrite_resolved_config "$1"\n'
        snapshot_path = temporary / 'resolved_snapshot.txt'
        run(['bash', '--noprofile', '--norc', '-c', snapshot_script, 'bash', str(snapshot_path)],
            repo, {'PATH': '/usr/bin:/bin'})
        assert 'declare -- PYTHON_RUN_TAG_BASE="semantic_probe"' in snapshot_path.read_text()
        semantics = tail_semantics()
        report = dict(status='PASS', generated_utc=datetime.now(timezone.utc).isoformat(),
                      baseline_commit=BASELINE_COMMIT, launcher_sha256=hashlib.sha256(launcher.read_bytes()).hexdigest(),
                      old_configs_checked=64, counts={'P0_QuickRun': 32, 'P1_WeightProbe': 32},
                      entire_commands_identical=True, explicit_empty_override_checks=empty_checks,
                      comparisons=records, semantic_override_probes=probes,
                      tail_semantics=semantics, snapshot_override_recorded=True,
                      training_started=False, qsub_local_called=False, tmux_called=False)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(report, indent=2) + '\n')
        print('PASS: all 64 P0/P1 commands byte-for-byte unchanged; empty/unset override; '
              'semantic tags; Tail OFF/ON semantics; resolved snapshot.', flush=True)
        return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument('--launcher', type=Path)
    parser.add_argument('--baseline-launcher', type=Path)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    repo = args.repo.resolve()
    validate(repo, (args.launcher or repo / 'train.sh').resolve(), args.report, args.baseline_launcher)


if __name__ == '__main__':
    main()
