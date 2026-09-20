#!/usr/bin/env python3
"""Check the 16-entry matrix and dry-run every command without loading data or launching jobs."""

import argparse
from collections import Counter
import csv
from dataclasses import asdict, fields
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile

sys.dont_write_bytecode = True
from generate_configs import (FACTORS, FORMULATIONS, HERE, IMPLEMENTATION_COMMIT, PYTHON, REPO, RESULTS, STATIONS,
                              expected_files, reference_hashes, reference_path, run_name, source_hashes)
from run_f1 import source_values, validate_checkpoint, validate_config

sys.path.insert(0, str(REPO))
from emulator.models import ModelConfig
from emulator.training.arguments import parse_args as parse_train_args
from infer import parse_args as parse_infer_args


ENV = dict(PATH=os.environ.get('PATH', '/usr/bin:/bin'), LANG='C.UTF-8', PYTHONDONTWRITEBYTECODE='1')
STAMP = '20000101_000000'  # Command placeholder, never a created output directory.
IDENTITY = {'SESSION_NAME', 'PACT_RUN_NAME', 'PYTHON_RUN_TAG_BASE', 'ALL_RESULTS_ROOT'}


def run(command, **kwargs):
    result = subprocess.run(command, cwd=REPO, env=kwargs.pop('env', ENV), capture_output=True,
                            text=True, timeout=60, **kwargs)
    if result.returncode:
        raise RuntimeError(shlex.join(map(str, command)) + '\n' + result.stdout + result.stderr)
    return result.stdout


def checkpoint_guard_check(config, training):
    """Only save/load metadata fixtures; never construct or fit a model."""
    import torch

    values = {field.name: getattr(training, field.name) for field in fields(ModelConfig) if hasattr(training, field.name)}
    values.update(model='pact', in_channels=5, out_channels=6, history_steps=4, peak_threshold_norm=[1.] * 6)
    saved = dict(station=config['STATION'], model_config=asdict(ModelConfig(**values)),
                 training_config=vars(training).copy(), dual_metadata=dict(tau_phys=1., event_prior=.05))
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / 'best_f0_metadata_fixture.pth'
        torch.save(saved, path)
        validate_checkpoint(config, path)
        mismatches = [('model_config', 'direct_dual_reconstruction', 'additive'),
                      ('model_config', 'exceedance_head_experiment', 'legacy'),
                      ('model_config', 'excess_formulation', 'severity_shape'),
                      ('training_config', 'excess_supervision_scope', 'all'),
                      ('training_config', 'checkpoint_selection', 'peak'),
                      ('training_config', 'run_tag', '0919_other_run')]
        for section, key, value in mismatches:
            changed = {**saved, section: {**saved[section], key: value}}
            torch.save(changed, path)
            try:
                validate_checkpoint(config, path)
            except ValueError:
                pass
            else:
                raise AssertionError(f'Accepted invalid F0 metadata: {key}={value}')
        torch.save(dict(saved, station='AnotherStation'), path)
        try:
            validate_checkpoint(config, path)
        except ValueError:
            pass
        else:
            raise AssertionError('Accepted a checkpoint from another station')
    return dict(valid_F0_accepted=True, incompatible_checkpoints_rejected=7)


def validate():
    before_results = sorted(str(path) for path in RESULTS.glob('*'))
    expected = expected_files()
    actual = {str(path.relative_to(HERE)) for path in (HERE / 'configs').glob('*.sh')}
    assert actual == {name for name in expected if name.startswith('configs/')}
    assert len(actual) == 16
    for relative, content in expected.items():
        assert (HERE / relative).read_text() == content, relative
    provenance = json.loads((HERE / 'provenance.json').read_text())
    assert provenance['implementation_commit'] == IMPLEMENTATION_COMMIT
    assert provenance['source_sha256'] == source_hashes(), 'Model/training/inference source changed'
    assert provenance['reference_sha256'] == reference_hashes(), 'Frozen 0919 suite changed'
    assert Path(PYTHON).is_file()
    assert '/All_results_0920/' in (REPO / '.gitignore').read_text().splitlines()
    run(['git', 'check-ignore', '--no-index', 'All_results_0920/probe.txt'])
    for script in HERE.rglob('*.sh'):
        run(['bash', '-n', str(script)])
    rows = list(csv.DictReader((HERE / 'manifest.csv').open()))
    assert len(rows) == 16 and Counter(row['mode'] for row in rows) == {'train': 12, 'postprocess': 4}
    assert {(row['station'], row['formulation']) for row in rows} == {
        (station, formulation) for station in STATIONS for formulation in FORMULATIONS}
    commands, training, f1_configs, details = [], {}, {}, []
    for row in rows:
        path = REPO / row['config_path']
        config = source_values(path, env=ENV)
        assert config['STATION'] == row['station']
        assert config['PACT_RUN_NAME'] == row['run_name']
        assert config['ALL_RESULTS_ROOT'] == str(RESULTS)
        assert Path(config['ROOT_DIR']).is_dir() and Path(config['STATION_JSON_DIR']).is_dir()
        assert (Path(config['STATION_JSON_DIR']) / f'{row["station"]}.json').is_file()
        if row['mode'] == 'train':
            reference = source_values(reference_path(row['station']), env=ENV)
            reference.pop('EXCEEDANCE_HEAD_EXPERIMENT')
            inherited = {k: v for k, v in config.items() if k not in IDENTITY | {
                'DIRECT_DUAL_RECONSTRUCTION', 'EXCESS_SUPERVISION_SCOPE'}}
            assert inherited == {k: v for k, v in reference.items() if k not in IDENTITY}
            assert 'EXCEEDANCE_HEAD_EXPERIMENT' not in config
            assert (config['DIRECT_DUAL_RECONSTRUCTION'], config['EXCESS_SUPERVISION_SCOPE']) == FACTORS[row['formulation']]
            assert all(float(config[name]) == 0. for name in
                       ('TAIL_LAMBDA_LIST', 'SLOPE_LAMBDA_LIST', 'EXCESS_AMP_LOSS_WEIGHT', 'SHAPE_LOSS_WEIGHT', 'PEAK_LOSS_WEIGHT'))
            output = run(['bash', 'train.sh', row['config_path']], env=dict(ENV, DRY_RUN='1', PACT_RUNSTAMP=STAMP))
            command_lines = [line[5:] for line in output.splitlines() if line.startswith('CMD: ')]
            assert len(command_lines) == 1
            command = shlex.split(command_lines[0])
            assert command[0] == 'train.py'
            args = parse_train_args(command[1:])
            assert (args.direct_dual_reconstruction, args.excess_supervision_scope) == FACTORS[row['formulation']]
            assert (args.model, args.head_type, args.excess_formulation, args.exceedance_head_experiment,
                    args.exceedance_gate_pooling) == ('perceiver3', 'dual', 'direct', None, 'mean')
            assert (args.dual_loss, args.body_loss_weight, args.excess_loss_weight, args.gate_loss_weight) == (1, 1., 2., .5)
            assert (args.seed, args.deterministic, args.lr, args.epochs, args.history_hours, args.hidden_channels,
                    args.batch_size, args.grad_accum_steps) == (42, 0, .005, 300, 24, 128, 256, 4)
            assert (args.loss_mode, args.excess_amp_loss_weight, args.shape_loss_weight, args.peak_loss_weight) == ('mse', 0., 0., 0.)
            assert (args.checkpoint_selection, args.save_aux_checkpoints) == ('overall', 0)
            assert args.amp and args.amp_dtype == 'bf16' and args.tf32
            assert config['num_gpus'] == '1' and config['CUDA_VISIBLE_DEVICES'] == '0'
            training[row['station'], row['formulation']] = args
            resolved = vars(args)
        else:
            validate_config(config)
            assert row['source_checkpoint'] == run_name(row['station'], 'F0_Soft')
            assert row['source_config'] == config['SOURCE_CONFIG'] and row['source_checkpoint_pattern'] == config['SOURCE_CHECKPOINT']
            assert row['hard_gate_threshold'] == '0.5' and row['epochs'] == ''
            output = run([PYTHON, '-B', 'experiment_config_0920/run_f1.py', row['config_path'], '--dry-run'])
            command_lines = [line[5:] for line in output.splitlines() if line.startswith('CMD: ')]
            assert len(command_lines) == 1
            command = shlex.split(command_lines[0])
            assert command[:3] == [PYTHON, '-B', 'infer.py'] and 'train.py' not in command
            args = parse_infer_args(command[3:])
            assert args.ckpt == config['SOURCE_CHECKPOINT'] and args.scope == 'test'
            assert args.dual_diagnostics and args.save_npz and args.amp and args.amp_dtype == 'bf16' and args.tf32
            assert args.device == 'cuda' and args.station == row['station']
            f1_configs[row['station']] = config
            resolved = vars(args)
        commands.append(f'# {row["mode"]}: {row["run_name"]}\nCMD: {shlex.join(command)}\n')
        details.append(dict(station=row['station'], formulation=row['formulation'], mode=row['mode'],
                            config=row['config_path'], source_checkpoint=row['source_checkpoint'], resolved=resolved))
    for station in STATIONS:
        matched = []
        for formulation in FACTORS:
            args = training[station, formulation]
            matched.append({k: v for k, v in vars(args).items() if k not in {
                'direct_dual_reconstruction', 'excess_supervision_scope', 'run_tag', 'output_dir'}})
        assert matched[0] == matched[1] == matched[2], station
    guards = checkpoint_guard_check(f1_configs['CBBT'], training['CBBT', 'F0_Soft'])
    assert sorted(str(path) for path in RESULTS.glob('*')) == before_results, 'Dry-run created results'
    report = dict(status='passed', configs=16, stations=list(STATIONS), training_configs=12,
                  postprocess_configs=4, dry_run_commands=16, jobs_launched=0,
                  implementation_commit=IMPLEMENTATION_COMMIT, result_root=str(RESULTS),
                  inherited_0919_settings_equal=True, training_differences_only_reconstruction_scope_identity=True,
                  frozen_sources_and_0919_unchanged=True, results_ignored=True, result_tree_unchanged=True,
                  f1_checkpoint_guards=guards, configs_resolved=details,
                  notes=['F1 dry-runs show an unresolved F0 checkpoint template; actual execution requires an exact existing F0 checkpoint.',
                         'The inherited mse launcher omits inactive tail/slope flags. Config coefficients are zero; Python defaults 0.1/0.01 are inactive under mse.',
                         'Dry-runs validate GPU/BF16 flags but do not execute GPU inference or training.'])
    return report, ''.join(commands)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-only', action='store_true', help='Validate without rewriting evidence files')
    args = parser.parse_args()
    report, commands = validate()
    if not args.check_only:
        (HERE / 'validation_report.json').write_text(json.dumps(report, indent=2, default=str) + '\n')
        (HERE / 'dry_run_commands.txt').write_text(commands)
    print('PASS: 16 configs / 16 command dry-runs; 12 train + 4 F0-only postprocess; no jobs launched.')


if __name__ == '__main__':
    main()
