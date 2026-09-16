#!/usr/bin/env python3
"""Compare every P3 config and resolved command against P2 using DRY_RUN only."""

import argparse
import ast
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

sys.dont_write_bytecode = True
from p3_common import (IDENTITY_CLI, IDENTITY_SHELL, P2_RELATIVE, RESULTS,
                       STAMP, audit, command, differences, dry_run, matrix,
                       matrix_text, read_manifest, render_config, require,
                       results_snapshot, sha256, source_values)


def verify_optimizer(repo):
    calls = [node for node in ast.walk(ast.parse((repo / 'train.py').read_text()))
             if isinstance(node, ast.Call) and ast.unparse(node.func) == 'torch.optim.Adam']
    require(len(calls) == 1, 'Expected unchanged production Adam optimizer')
    kwargs = {k.arg: ast.unparse(k.value) for k in calls[0].keywords}
    require(kwargs == {'lr': 'args.lr', 'weight_decay': '1e-05'},
            f'Unexpected optimizer settings: {kwargs}')
    return dict(type='Adam', weight_decay=1e-5, lr='args.lr', source='train.py')


def verify_semantics(row, shell, resolved):
    t, a, s, p = (int(row[k]) for k in ('tail', 'amp', 'shape', 'peak'))
    expected = dict(station=row['station'], head_type='dual', dual_mode='exceedance',
                    excess_formulation='severity_shape', loss_mode='mse_tail' if t else 'mse',
                    excess_amp_loss_weight=.0025 if a else 0.,
                    shape_loss_weight=.00003 if s else 0., peak_loss_weight=.0035 if p else 0.,
                    body_loss_weight=1., excess_loss_weight=2., gate_loss_weight=.5,
                    severity_shape_eps=1e-6, excess_amp_pool='max', peak_pool='max',
                    history_hours=24, model='perceiver3', encoder_type='GraphSAGE',
                    temporal_block='Transformer', hidden_channels=128, num_layers=2,
                    node_read_heads=8, time_read_heads=8, transformer_layers=2,
                    transformer_ff_mult=4., batch_size=256, grad_accum_steps=4, epochs=300,
                    scheduler='cosine', warmup_epochs=5, warmup_start_factor=.1, min_lr=1e-6,
                    seed=42, deterministic=0, max_grad_norm=0., amp=True, amp_dtype='bf16',
                    tf32=True, num_workers=0, checkpoint_selection='overall', save_aux_checkpoints=0,
                    x_norm='zscore', x_clip=0., x_aug=0, use_site_elevation=0, use_bathymetry=0)
    actual = resolved['args']
    for key, value in expected.items():
        require(actual.get(key) == value, f'{row["run_name"]}: {key}={actual.get(key)!r}, expected {value!r}')
    require(shell['TAIL_LAMBDA_LIST'] == ['0.025'], 'Configured P2 tail coefficient changed')
    require(shell['DISABLE_OOD'] == ['1'], 'OOD must stay disabled')
    if t:
        require(actual['tail_lambda'] == .025 and '--tail_lambda' in resolved['command'],
                'Active tail coefficient must be 0.025')
    else:
        require('--tail_lambda' not in resolved['command'], 'MSE must retain the inactive P2 tail semantics')


def validate(repo, directory):
    rows = read_manifest(directory / 'manifest.csv')
    expected = matrix()
    require(len(rows) == 31, f'Manifest has {len(rows)} rows, expected exactly 31; STOP before launch.')
    keys = [(r['station'], r['semantic_config'], r['learning_rate']) for r in rows]
    require(len(set(keys)) == 31, 'Duplicate station/config/LR treatments; STOP before launch.')
    require(len({r['run_name'] for r in rows}) == 31, 'Duplicate run names')
    require({p.name for p in directory.rglob('train_config_*.sh')} ==
            {Path(r['config_path']).name for r in rows}
            and len(list(directory.rglob('train_config_*.sh'))) == 31,
            'Expected exactly 31 matching standalone config scripts')
    provenance = json.loads((directory / 'provenance.json').read_text())
    for name, digest in {**provenance['shared_files_sha256'], **provenance['source_P2_sha256']}.items():
        require(sha256(repo / name) == digest, f'Frozen P2/shared source changed: {name}')
    require(set(provenance['source_P2_sha256']) == {r['source_P2_config_path'] for r in expected},
            'Provenance does not cover exactly the required P2 semantic sources')
    require(not any(p.is_symlink() for p in directory.rglob('*')), 'P3 files must not be source/result symlinks')
    for row, target in zip(rows, expected):
        target['source_P2_sha256'] = sha256(repo / target['source_P2_config_path'])
        require(row == target, f'Manifest mismatch for {target["run_name"]}: {differences(target, row)}')
    counts = audit(rows)
    require(counts['core_probe'] == 16 and counts['degradation_alert_treatments'] == 21
            and counts['overlap_removed'] == 6, f'Incorrect A/B coverage: {counts}')
    before_results = results_snapshot(repo)
    before_configs = {p.name: sha256(p) for p in directory.glob('train_config_*.sh')}
    for script in directory.glob('*.py'):
        compile(script.read_text(), str(script), 'exec')
    command(['bash', '-n', str(repo / 'train.sh')], repo)
    command(['bash', '-n', str(directory / 'launch.sh')], repo)
    sys.path.insert(0, str(repo))
    from emulator.training.arguments import parse_args
    optimizer = verify_optimizer(repo)
    p2_manifest = {r['run_name']: r for r in read_manifest(repo / P2_RELATIVE / 'manifest.csv')}
    references, records, logs = {}, [], []
    shared_shell = None
    for row in rows:
        config = directory / Path(row['config_path']).name
        source = repo / row['source_P2_config_path']
        # Full-file comparison permits only four assignment replacements and the group header.
        # This also guards against additional commands, inheritance, unset/default changes and sweeps.
        require(config.read_text() == render_config(source, row), f'Unexpected P2-vs-P3 file difference: {config}')
        require(os.access(config, os.X_OK), f'Config is not executable: {config}')
        command(['bash', '-n', str(config)], repo)
        if str(source) not in references:
            ref_shell = source_values(source, repo)
            ref_run = dry_run(source, repo, parse_args)
            verify_semantics(row, ref_shell, ref_run)
            require(ref_shell['LR_LIST'] == ['5e-3'] and ref_run['args']['lr'] == .005,
                    f'P2 base LR changed: {source}')
            require(p2_manifest[row['source_P2_semantic_config']]['config_path'] == row['source_P2_config_path'],
                    'P2 manifest/source mismatch')
            references[str(source)] = dict(shell=ref_shell, resolved=ref_run)
            logs.append(f'### P2 {source}\n{ref_run["stdout"]}')
        reference = references[str(source)]
        shell = source_values(config, repo)
        resolved = dry_run(config, repo, parse_args)
        verify_semantics(row, shell, resolved)
        shell_diff = differences(reference['shell'], shell)
        expected_shell_diff = set(IDENTITY_SHELL)
        if row['learning_rate'] != '0.005':
            expected_shell_diff.add('LR_LIST')
        require(set(shell_diff) == expected_shell_diff,
                f'{row["run_name"]}: unexpected configured differences {shell_diff}')
        require(shell['PACT_RUN_NAME'] == shell['SESSION_NAME'] == shell['PYTHON_RUN_TAG_BASE'] == [row['run_name']],
                'P3 semantic shell identity mismatch')
        require(shell['ALL_RESULTS_ROOT'] == ['./' + RESULTS] and shell['LR_LIST'] == [row['lr_label'][2:]],
                'P3 output root or LR mismatch')
        require(shell['PYTHON_BIN'] == [provenance['python_bin']], 'Interpreter differs from P2')
        actual = resolved['args']
        require(actual['lr'] == float(row['learning_rate']), 'Resolved LR does not match manifest')
        require(actual['run_tag'] == f'{row["run_name"]}_{STAMP}', 'Semantic name did not propagate to Python')
        output = Path(actual['output_dir']).resolve()
        require(output == repo / RESULTS / f'{row["run_name"]}__{STAMP}', 'Output escaped P3 or lost its timestamp')
        arg_diff = differences(reference['resolved']['args'], actual)
        expected_arg_diff = set(IDENTITY_CLI)
        if row['learning_rate'] != '0.005':
            expected_arg_diff.add('lr')
        require(set(arg_diff) == expected_arg_diff, f'Unexpected resolved argument differences: {arg_diff}')
        # Compare the actual argv too: parser equivalence cannot mask changed spelling/order/extra flags.
        normalized = list(resolved['command'])
        for key in ('lr', 'run_tag', 'output_dir'):
            i = normalized.index('--' + key)
            ref = reference['resolved']['command']
            normalized[i + 1] = ref[ref.index('--' + key) + 1]
        require(normalized == reference['resolved']['command'], 'Unexpected raw command change')
        varying = IDENTITY_SHELL | {'STATION', 'LOSS_MODE_LIST', 'EXCESS_AMP_LOSS_WEIGHT',
                                   'SHAPE_LOSS_WEIGHT', 'PEAK_LOSS_WEIGHT', 'LR_LIST'}
        fixed = {k: v for k, v in shell.items() if k not in varying}
        if shared_shell is None:
            shared_shell = fixed
        require(fixed == shared_shell, f'Non-factor settings vary across P2 sources: {row["run_name"]}')
        records.append(dict(run_name=row['run_name'], source_P2_config=row['source_P2_config_path'],
                            P2_sha256=sha256(source), P3_sha256=sha256(config),
                            configured_diff=shell_diff, resolved_diff=arg_diff,
                            configured_values=shell, resolved_args=actual,
                            command=resolved['command'], status='PASS'))
        logs.append(f'### P3 {row["run_name"]}\n{resolved["stdout"]}')
    require(len({r['resolved_args']['output_dir'] for r in records}) == 31, 'Duplicate output directories')
    require(len({r['resolved_args']['run_tag'] for r in records}) == 31, 'Duplicate Python run tags')
    require(results_snapshot(repo) == before_results, 'Result artifacts changed during dry-run validation')
    require({p.name: sha256(p) for p in directory.glob('train_config_*.sh')} == before_configs,
            'Configs changed during validation')
    for name, digest in {**provenance['shared_files_sha256'], **provenance['source_P2_sha256']}.items():
        require(sha256(repo / name) == digest, f'Protected source changed during validation: {name}')
    return dict(status='PASS', validated_utc=datetime.now(timezone.utc).isoformat(),
                repository=str(repo), manifest_sha256=sha256(directory / 'manifest.csv'),
                counts=counts, source_P2_configs=len(references),
                configured_fields_per_run=len(records[0]['configured_values']),
                resolved_fields_per_run=len(records[0]['resolved_args']),
                unexpected_differences=0, shared_configured_values=shared_shell,
                optimizer=optimizer, rows=rows, comparisons=records,
                p2_references=references, dry_run_log='\n'.join(logs),
                result_entries_unchanged=len(before_results),
                training_started=False, queue_submissions=0)


def summary(report):
    lines = [matrix_text(report['rows']).rstrip(), '', 'Config-vs-P2 validation: PASS',
             f'{len(report["comparisons"])} P3 configs vs {report["source_P2_configs"]} exact P2 semantic sources.',
             f'Per run: {report["configured_fields_per_run"]} shell fields and '
             f'{report["resolved_fields_per_run"]} parsed Python fields compared.',
             'Unexpected differences: 0. Shared train.sh and Python implementation unchanged.',
             'LR5e-3: identity/output only. Low LR: identity/output plus LR.',
             'Mode: DUAL_MODE=exceedance; EXCESS_FORMULATION=severity_shape (exact P2 interface).',
             '', 'Per-run diff summary (all PASS):']
    for item in report['comparisons']:
        changes = item['resolved_diff']
        lr = changes.get('lr')
        detail = f'lr {lr["P2"]} -> {lr["P3"]}; ' if lr else 'lr unchanged at 0.005; '
        lines.append(f'{item["run_name"]}: {detail}run_tag, output_dir')
    lines.append('\nNo training, queue submission or tmux session was started.')
    return '\n'.join(lines) + '\n'


def write_report(report, directory):
    directory.mkdir(parents=True, exist_ok=True)
    data = dict(report)
    log = data.pop('dry_run_log')
    (directory / 'validation_report.json').write_text(json.dumps(data, indent=2) + '\n')
    (directory / 'dry_runs.txt').write_text(log)
    text = '# P3 LR stability validation — PASS\n\n```text\n' + summary(report) + '```\n'
    text += '\n## Fixed controls read from P2\n\n| Config variable | Value |\n| --- | --- |\n'
    text += '\n'.join(f'| {k} | `{v[0]}` |' for k, v in sorted(report['shared_configured_values'].items()))
    text += ('\n\nAdam with weight decay 1e-5 is verified in `train.py`. All shared implementation files '
             'and station JSON files are checked against `provenance.json`. All actual argv tokens '
             'match P2 after replacing only LR, run tag and output directory. The saved dry-run '
             'timestamp is synthetic and is never used for real launches.\n')
    text += ('\nThese checks establish configuration equivalence and launcher command validity. '
             'No data loading, model execution, optimizer step, training or convergence check was performed.\n')
    (directory / 'validation_report.md').write_text(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument('--config-dir', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--report-dir', type=Path, help='Optional explicit evidence directory; default is read-only.')
    args = parser.parse_args()
    try:
        report = validate(args.repo.resolve(), args.config_dir.resolve())
        if args.report_dir:
            write_report(report, args.report_dir.resolve())
        print(summary(report), end='')
    except (ValueError, OSError, KeyError) as error:
        print(f'P3 VALIDATION FAILED — nothing launched: {error}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
