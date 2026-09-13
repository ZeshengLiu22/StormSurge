#!/usr/bin/env python3
"""Strict P2 factorial/config validation. Runs only train.sh with DRY_RUN=1."""

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import itertools
import json
import os
from pathlib import Path
import re
import shlex
import sys
import tempfile

sys.dont_write_bytecode = True
from validate_launcher import BASELINE_COMMIT, STAMP, dry_run, run

STATIONS = ('CBBT', 'Lewes', 'Battery', 'Boston')
RELATIVE = 'experiment_config/P2_PeakAwareFactorial'
RESULTS = 'All_Results/P2_PeakAwareFactorial'
FACTORS = {'loss_mode', 'tail_lambda', 'excess_formulation',
           'excess_amp_loss_weight', 'shape_loss_weight', 'peak_loss_weight'}
IDENTITY = {'station', 'run_tag', 'output_dir'}
FROZEN_SHELL = dict(
    MODEL='perceiver3', ENCODER_TYPE='GraphSAGE', TEMPORAL_BLOCK='Transformer',
    HISTORY_HOURS_LIST='24', HIDDEN_CHANNELS='128', NUM_LAYERS='2', DROPOUT='0.05',
    HEAD_DROPOUT='0.05', NODE_READ_HEADS='8', TIME_READ_HEADS='8', TRANSFORMER_LAYERS='2',
    TRANSFORMER_FF_MULT='4.0', TRANSFORMER_DROPOUT='0.0', MAX_TIME_STEPS='32',
    HEAD_TYPE='dual', DUAL_MODE='exceedance', GATE_MODE='window', DUAL_LOSS='1',
    DUAL_ABLATION='none', BODY_LOSS_WEIGHT='1', EXCESS_LOSS_WEIGHT='2', GATE_LOSS_WEIGHT='0.5',
    EXCEEDANCE_PERCENTILE='95', TRAIN_RATIO='0.6', VAL_RATIO='0.2', SHUFFLE_YEARS='0',
    FUTURE_ONLY='0', FUTURE_YEAR_THRESHOLD='2030', SEED='42', BATCH_SIZE='256',
    GRAD_ACCUM_STEPS='4', LR_LIST='5e-3', EPOCHS='300', SCHEDULER='cosine',
    WARMUP_EPOCHS='5', WARMUP_START_FACTOR='0.1', MIN_LR='1e-6', MAX_GRAD_NORM='0',
    DETERMINISTIC='0', num_gpus='1', CUDA_VISIBLE_DEVICES='0', USE_AMP='1', AMP_DTYPE='bf16',
    USE_TF32='1', TORCH_THREADS='1', NUM_WORKERS='0', PIN_MEMORY='0', PERSISTENT_WORKERS='0',
    PREFETCH_FACTOR='0', MP_CONTEXT='fork', DISABLE_OOD='1', X_NORM='zscore', X_CLIP='0',
    X_AUG='0', X_AUG_PROB='0', X_AUG_SCALE='0', X_AUG_BIAS='0', USE_SITE_ELEVATION='0',
    USE_BATHYMETRY='0', TAIL_FRAC='0.05', TAIL_LAMBDA_LIST='0.025',
    SLOPE_LAMBDA_LIST='0.01', DO_CONDA='0', PYTHONDONTWRITEBYTECODE='1',
    EXCESS_AMP_POOL='max', EXCESS_AMP_BETA='20', SEVERITY_SHAPE_EPS='1e-6',
    PEAK_POOL='max', PEAK_POOL_BETA='20', CHECKPOINT_SELECTION='overall',
    SAVE_AUX_CHECKPOINTS='0', CHECKPOINT_OVERALL_TOL='0.01', RUN_DIR_NAME_STYLE='runname_timestamp')


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def declarations(path):
    result = {}
    for line in path.read_text().splitlines():
        if not line.strip() or line.lstrip().startswith('#'):
            continue
        match = re.fullmatch(r'(?:export )?([A-Za-z_]\w*)=(.*)', line)
        assert match, (path, 'Non-assignment or config inheritance', line)
        name, value = match.groups()
        assert name not in result, (path, 'Duplicate declaration', name)
        code = ' '.join(shlex.split(value, comments=True))
        assert not any(token in code for token in ('$(', '`', ';')), (path, line)
        result[name] = value
    return result


def source_values(path, repo):
    names = list(declarations(path))
    script = r'''set -euo pipefail
source "$1"
shift
for config_key in "$@"; do
    declare -n config_value="$config_key"
    config_items=("${config_value[@]}")
    printf '%s\0' "$config_key" "${#config_items[@]}" "${config_items[@]}"
    unset -n config_value
done
'''
    output = run(['bash', '--noprofile', '--norc', '-c', script, 'bash', str(path), *names],
                 repo, {'PATH': '/usr/bin:/bin'})
    words = iter(output.removesuffix('\0').split('\0'))
    values = {}
    for name in words:
        count = int(next(words))
        values[name] = [next(words) for _ in range(count)]
    return values


def results_snapshot(repo):
    root = repo / 'All_Results'
    result = {}
    if root.exists():
        for directory, subdirs, files in os.walk(root, followlinks=False):
            for name in ['.', *subdirs, *files]:
                path = Path(directory) / name
                stat = path.lstat()
                result[str(path.relative_to(root))] = [stat.st_mode, stat.st_size, stat.st_mtime_ns]
    return result


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument('--config-dir', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--report-dir', type=Path, help='Defaults to a temporary directory; never writes run artifacts.')
    parser.add_argument('--original-state', type=Path, help='Optional original file/result integrity snapshot.')
    args = parser.parse_args()
    repo, configs_dir = args.repo.resolve(), args.config_dir.resolve()
    report_dir = args.report_dir or Path(tempfile.mkdtemp(prefix='p2-suite-validation-'))
    report_dir.mkdir(parents=True, exist_ok=True)
    evidence = report_dir / 'validation'
    evidence.mkdir(exist_ok=True)
    sys.path.insert(0, str(repo))
    from emulator.training.arguments import parse_args

    launcher = repo / 'train.sh'
    before_results = results_snapshot(repo)
    tracked = run(['git', 'ls-files', '-z'], repo).removesuffix('\0').split('\0')
    before_hashes = {name: sha256(repo / name) for name in tracked}
    regression = json.loads((configs_dir / 'validation/launcher_regression.json').read_text())
    assert regression['status'] == 'PASS' and regression['old_configs_checked'] == 64
    assert regression['entire_commands_identical'] and regression['tail_semantics']['status'] == 'PASS'
    assert regression['launcher_sha256'] == sha256(launcher), 'Launcher changed after regression'
    assert all(row['baseline_cmd_line'] == row['updated_cmd_line'] for row in regression['comparisons'])
    run(['bash', '-n', str(launcher)], repo)
    for script in configs_dir.glob('*.py'):
        compile(script.read_text(), str(script), 'exec')

    configs = sorted(configs_dir.glob('train_config_*.sh'))
    assert len(configs) == len(list(configs_dir.rglob('*.sh'))) == 96, 'P2 must contain exactly 96 configs'
    with (configs_dir / 'manifest.csv').open(newline='') as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 96
    assert [int(row['experiment_id']) for row in rows] == list(range(1, 97))
    assert len({row['config_path'] for row in rows}) == 96
    assert {Path(row['config_path']).name for row in rows} == {path.name for path in configs}
    assert len({row['run_tag'] for row in rows}) == 96

    references, p0_logs = {}, []
    for station in STATIONS:
        path = repo / 'experiment_config/P0_QuickRun' / f'train_config_NCEP_{station}_24h_dual_mse.sh'
        values = source_values(path, repo)
        resolved = dry_run(path, repo, launcher, parse_args)
        references[station] = dict(config_path=str(path.relative_to(repo)), sha256=sha256(path),
                                   configured_values=values, resolved_args=resolved['args'])
        p0_logs.append(f'### {path.relative_to(repo)}\n{resolved["stdout"]}')

    records, logs, coverage, directories, python_tags = [], [], {}, set(), set()
    frozen_resolved = None
    for row in rows:
        station = row['station']
        assert station in STATIONS
        formulation = row['formulation']
        assert formulation in ('direct', 'severity_shape')
        tail, amp, shape, peak = (int(row[name + '_enabled']) for name in ('tail', 'amp', 'shape', 'peak'))
        assert all(bit in (0, 1) for bit in (tail, amp, shape, peak))
        if formulation == 'direct':
            assert shape == 0
        form = 'direct' if formulation == 'direct' else 'severity'
        name = f'NCEP_{station}_24h_dual_{form}_T{tail}_A{amp}_S{shape}_P{peak}'
        label = f'P2_{station}_{form}_T{tail}_A{amp}_S{shape}_P{peak}'
        filename = f'train_config_{name}.sh'
        relative = f'{RELATIVE}/{filename}'
        assert row['config_path'] == relative
        config = configs_dir / filename
        assert os.access(config, os.X_OK)
        run(['bash', '-n', str(config)], repo)
        values = source_values(config, repo)
        assert all(len(value) == 1 for value in values.values()), (config, 'Nested sweep or empty setting')
        assert all(values[key] == [value] for key, value in FROZEN_SHELL.items()), config
        mode = 'mse_tail' if tail else 'mse'
        amp_weight = ('0.0007' if form == 'direct' else '0.0025') if amp else '0'
        shape_weight, peak_weight = ('0.00003' if shape else '0'), ('0.0035' if peak else '0')
        expected = dict(references[station]['configured_values'])
        overrides = dict(LOSS_MODE_LIST=mode, SESSION_NAME=label, PACT_RUN_NAME=name,
                         ALL_RESULTS_ROOT='./' + RESULTS, PYTHON_RUN_TAG_BASE=name,
                         EXCESS_FORMULATION=formulation, EXCESS_AMP_LOSS_WEIGHT=amp_weight,
                         SHAPE_LOSS_WEIGHT=shape_weight, PEAK_LOSS_WEIGHT=peak_weight,
                         EXCESS_AMP_POOL='max', EXCESS_AMP_BETA='20', SEVERITY_SHAPE_EPS='1e-6',
                         PEAK_POOL='max', PEAK_POOL_BETA='20', CHECKPOINT_SELECTION='overall',
                         SAVE_AUX_CHECKPOINTS='0', CHECKPOINT_OVERALL_TOL='0.01')
        expected.update({key: [value] for key, value in overrides.items()})
        assert values == expected, (config, {key: [values.get(key), expected.get(key)]
                                            for key in values.keys() | expected.keys()
                                            if values.get(key) != expected.get(key)})
        expected_row = dict(config_path=relative, station=station, formulation=formulation,
                            tail_enabled=str(tail), tail_lambda='0.025' if tail else '0',
                            configured_tail_lambda='0.025', amp_enabled=str(amp), excess_amp_loss_weight=amp_weight,
                            shape_enabled=str(shape), shape_loss_weight=shape_weight, peak_enabled=str(peak),
                            peak_loss_weight=peak_weight, loss_mode=mode, seed='42', checkpoint_selection='overall',
                            save_aux_checkpoints='0', checkpoint_overall_tol='0.01', run_tag=name, run_name=name,
                            python_run_tag_base=name, history_hours='24', excess_amp_pool='max', excess_amp_beta='20',
                            peak_pool='max', peak_pool_beta='20', severity_shape_eps='1e-6', qsub_label=label,
                            qsub_command=f'qsub_local train.sh {label} {relative}', results_root=RESULTS)
        assert set(row) == set(expected_row) | {'experiment_id'}
        assert all(row[key] == value for key, value in expected_row.items()), row

        resolved = dry_run(config, repo, launcher, parse_args)
        actual = resolved['args']
        assert actual['loss_mode'] == mode
        if tail:
            assert actual['tail_lambda'] == .025 and '--tail_lambda' in resolved['command']
        else:
            assert actual['loss_mode'] == 'mse' and '--tail_lambda' not in resolved['command']
            # Deliberately no assertion on inactive parsed tail_lambda.
        assert not any(word.startswith('--slope_') for word in resolved['command'])
        assert actual['excess_formulation'] == formulation
        assert actual['excess_amp_loss_weight'] == float(amp_weight)
        assert actual['shape_loss_weight'] == float(shape_weight)
        assert actual['peak_loss_weight'] == float(peak_weight)
        assert actual['run_tag'] == f'{name}_{STAMP}'
        output = Path(actual['output_dir']).resolve()
        assert output == repo / RESULTS / f'{name}__{STAMP}'
        assert output not in directories and actual['run_tag'] not in python_tags
        directories.add(output)
        python_tags.add(actual['run_tag'])
        reference = references[station]['resolved_args']
        differences = {key: dict(p0=reference.get(key), p2=actual.get(key))
                       for key in reference.keys() | actual.keys() if reference.get(key) != actual.get(key)}
        assert set(differences) <= FACTORS | {'run_tag', 'output_dir'}, (config, differences)
        common = {key: value for key, value in actual.items() if key not in FACTORS | IDENTITY}
        if frozen_resolved is None:
            frozen_resolved = common
        assert common == frozen_resolved, (config, 'Frozen resolved controls drift')
        for key in ('root_dir', 'station_json_dir'):
            assert Path(actual[key]).is_dir(), (config, key)
        assert (Path(actual['station_json_dir']) / f'{station}.json').is_file()
        assert actual['use_station_meta'] == 1 and actual['device'] == 'auto'
        assert f'LAUNCHER:      {values["PYTHON_BIN"][0]} -u' in resolved['stdout']
        assert 'num_gpus:      1' in resolved['stdout']
        assert 'workers=0 pin=0 pers=0 prefetch=0' in resolved['stdout']
        assert 'AMP=1 AMP_DTYPE=bf16 TF32=1' in resolved['stdout']
        assert 'DISABLE_OOD:   1 (x_norm=zscore, x_clip=0, x_aug=0)' in resolved['stdout']
        key = (tail, amp, peak) if formulation == 'direct' else (tail, amp, shape, peak)
        coverage.setdefault((station, formulation), Counter())[key] += 1
        records.append(dict(**row, status='PASS', sha256=sha256(config), standalone=True,
                            dry_run_exit_code=0, command_count=1, resolved_args=actual,
                            resolved_cli=resolved['command'], active_tail_lambda=.025 if tail else 0,
                            tail_active=bool(tail), slope_active=False,
                            p0_reference=references[station]['config_path'], p0_differences=differences,
                            unexpected_frozen_differences=[]))
        logs.append(f'### {relative}\n{resolved["stdout"]}')
        if len(records) % 24 == 0:
            print(f'PASS: {len(records)}/96 config dry runs and complete P0 comparisons', flush=True)

    counts, factorial = {}, []
    assert Counter(record['station'] for record in records) == Counter({station: 24 for station in STATIONS})
    for station in STATIONS:
        counts[station] = {'direct': 8, 'severity_shape': 16, 'total': 24}
        for formulation, dimension in (('direct', 3), ('severity_shape', 4)):
            expected = Counter(itertools.product((0, 1), repeat=dimension))
            actual = coverage[(station, formulation)]
            assert actual == expected, (station, formulation, actual, expected)
            factorial.append(dict(station=station, formulation=formulation, expected=len(expected),
                                  actual=sum(actual.values()), duplicates=[], missing=[], extra=[],
                                  combinations=[list(key) for key in sorted(actual)], status='PASS'))
    assert len(directories) == len(python_tags) == 96
    command_counts = {}
    for station in ('all', *STATIONS):
        path = configs_dir / f'commands_{station}.txt'
        lines = path.read_text().splitlines()
        expected = [row['qsub_command'] for row in rows if station == 'all' or row['station'] == station]
        assert lines == expected, path
        assert len(lines) == (96 if station == 'all' else 24), path
        command_counts[path.name] = len(lines)
    assert len({row['qsub_label'] for row in rows}) == 96
    shell_path = Path('/home/exouser/.bashrc')
    shell = shell_path.read_text()
    qsub_function = shell.split('\nqsub_local() {', 1)[1].split('\n}\n', 1)[0]
    assert 'local script="$1"' in qsub_function and 'script_args=("${@:3}")' in qsub_function
    assert 'label="${2:-' in qsub_function
    # Inspect the locally supported syntax as text; never invoke queue helpers.

    from emulator.data.station_metadata import load_station_json, station_features_from_json
    metadata = {}
    for station in STATIONS:
        directory = references[station]['resolved_args']['station_json_dir']
        features = station_features_from_json(load_station_json(directory, station),
                                              use_site_elevation=False, use_bathymetry=False)
        assert features.numel() == 6
        metadata[station] = dict(feature_count=6, features=features.tolist(),
                                 fields=['lat/90', 'lon/180', 'sin(lat)', 'cos(lat)', 'sin(lon)', 'cos(lon)'],
                                 use_site_elevation=0, use_bathymetry=0, status='PASS')
    train_source = (repo / 'train.py').read_text()
    assert 'torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)' in train_source
    after_hashes = {name: sha256(repo / name) for name in tracked}
    assert before_hashes == after_hashes, 'Validation changed a tracked file'
    after_results = results_snapshot(repo)
    assert before_results == after_results, 'Validation changed existing result/checkpoint metadata'
    assert not (repo / RESULTS).exists(), 'P2 dry runs must create no result directory'
    original_path = args.original_state or configs_dir / 'validation/original_state.json'
    original = json.loads(original_path.read_text())
    changed = sorted(name for name, digest in original['tracked_sha256'].items()
                     if sha256(repo / name) != digest)
    assert changed == ['train.sh'], changed
    assert original['results_metadata'] == after_results, 'Results changed since task start'
    assert original['commit'] == BASELINE_COMMIT
    commit = run(['git', 'rev-parse', 'HEAD'], repo).strip()
    assert 'train' not in sys.modules, 'The training entry point must not be imported'

    frozen_shell = {key: value for key, value in references['CBBT']['configured_values'].items()
                    if key not in {'STATION', 'LOSS_MODE_LIST', 'PACT_RUN_NAME', 'SESSION_NAME', 'ALL_RESULTS_ROOT'}}
    for key in ('EXCESS_AMP_POOL', 'EXCESS_AMP_BETA', 'SEVERITY_SHAPE_EPS', 'PEAK_POOL', 'PEAK_POOL_BETA',
                'CHECKPOINT_SELECTION', 'SAVE_AUX_CHECKPOINTS', 'CHECKPOINT_OVERALL_TOL'):
        frozen_shell[key] = [FROZEN_SHELL[key]]
    frozen_shell['ALL_RESULTS_ROOT'] = ['./' + RESULTS]
    weights = dict(body_loss_weight=1, excess_loss_weight=2, gate_loss_weight=.5,
                   tail_frac=.05, configured_tail_lambda=.025, active_tail_lambda=[0, .025],
                   direct_amplitude_weights=[0, .0007], severity_amplitude_weights=[0, .0025],
                   direct_shape_weights=[0], severity_shape_weights=[0, .00003], peak_weights=[0, .0035],
                   excess_amp_pool='max', excess_amp_beta=20, peak_pool='max', peak_pool_beta=20,
                   severity_shape_eps=1e-6, exceedance_percentile=95)
    report = dict(status='PASS', generated_utc=datetime.now(timezone.utc).isoformat(), repository=str(repo),
                  repository_commit=commit, baseline_commit=BASELINE_COMMIT, launcher_sha256=sha256(launcher),
                  launcher_change='Opt-in PYTHON_RUN_TAG_BASE enhancement; baseline production Python unchanged.',
                  configs=96, counts_by_station=counts, direct_total=32, severity_shape_total=64,
                  full_factorial=factorial, duplicate_count=0, missing_count=0,
                  weights=weights, frozen_shell_controls=frozen_shell, frozen_resolved_controls=frozen_resolved,
                  optimizer=dict(class_name='torch.optim.Adam', lr=.005, weight_decay=1e-5, source='train.py:136'),
                  station_features=metadata,
                  tail_factor_definition=dict(off='loss_mode=mse; tail branch inactive regardless of parsed tail_lambda',
                                              on='loss_mode=mse_tail; active tail_lambda=0.025',
                                              configured_tail_lambda=.025,
                                              manifest_tail_lambda='effective coefficient: 0 for inactive mse, 0.025 for mse_tail'),
                  checkpoint=dict(selection='overall', primary_metric='VAL rmse_all', auxiliary=0, tolerance=.01),
                  dry_runs=dict(requested=96, successful=96, failed=0, actual_launcher=str(launcher),
                                config_directory=str(configs_dir), single_command_per_config=True, dry_run='1'),
                  command_file_counts=command_counts, command_syntax='qsub_local train.sh <label> <config_path>',
                  command_syntax_source=str(shell_path), qsub_function_sha256=hashlib.sha256(qsub_function.encode()).hexdigest(),
                  semantic_names=dict(unique_result_directories=96, unique_python_tags=96, status='PASS',
                                      directory_format='<PACT_RUN_NAME>__<timestamp>',
                                      python_tag_format='<PYTHON_RUN_TAG_BASE>_<timestamp>'),
                  p0_cross_check=dict(configs_compared=96, stations=4, unexpected_differences=0,
                                      all_non_factor_cli_fields_identical=True, references=references),
                  launcher_regression=dict(status='PASS', p0_commands_identical=32, p1_commands_identical=32,
                                           evidence='validation/launcher_regression.json',
                                           empty_and_unset_preserve_legacy=True, snapshot_override_recorded=True,
                                           tail_semantics=regression['tail_semantics']),
                  protection=dict(tracked_files_checked=len(original['tracked_sha256']),
                                  authorized_changed_files=changed,
                                  unchanged_tracked_files=len(original['tracked_sha256']) - 1,
                                  production_python_modified=False,
                                  p0_modified=False, p1_modified=False, existing_results_unchanged=True,
                                  result_entries_checked=len(after_results), no_p2_result_directory_created=True,
                                  training_started=False, qsub_local_called=False, tmux_called=False,
                                  suite_directory=RELATIVE),
                  runs=records)
    write_json(report_dir / 'validation_report.json', report)
    write_json(evidence / 'frozen_controls.json', dict(shell=frozen_shell, resolved_cli=frozen_resolved,
                                                    weights=weights, optimizer=report['optimizer']))
    write_json(evidence / 'original_state.json', original)
    (evidence / 'dry_runs.txt').write_text('\n'.join(logs))
    (evidence / 'p0_reference_dry_runs.txt').write_text('\n'.join(p0_logs))
    station_table = '\n'.join(f'| {station} | 8 | 16 | 24 | PASS |' for station in STATIONS)
    (report_dir / 'validation_report.md').write_text(f'''# P2 Peak-Aware Factorial validation — PASS

Validated UTC: {report['generated_utc']}  
Repository: `{repo}`  
Baseline commit: `{BASELINE_COMMIT}`  
Repository commit at validation: `{commit}`  
Current `train.sh` SHA-256: `{report['launcher_sha256']}`

The validated source includes the authorized opt-in launcher enhancement.
The baseline commit identifies the unchanged production Python/model/loss code
and P0/P1 protocol; the launcher hash identifies the precise additional change.

## Experiment inventory and full factorial

| Station | Direct | Severity-shape | Total | Factorial coverage |
| --- | ---: | ---: | ---: | --- |
{station_table}
| Total | 32 | 64 | 96 | PASS |

All direct `(Tail, Amp, Peak)` and severity-shape `(Tail, Amp, Shape, Peak)`
combinations occur exactly once per station. Duplicate/missing/extra combinations:
**0/0/0**. Direct shape weights are always zero. All 96 configs are standalone,
contain singleton sweep arrays, pass `bash -n`, and have unique experiment IDs
1–96, semantic names, Python run tags, output directories and queue labels.

## Exact weights and active semantics

| Control | Direct OFF / ON | Severity-shape OFF / ON |
| --- | --- | --- |
| Active tail coefficient | 0 / 0.025 | 0 / 0.025 |
| Excess amplitude | 0 / 0.0007 | 0 / 0.0025 |
| Shape supervision | always 0 | 0 / 0.00003 |
| Final physical peak | 0 / 0.0035 | 0 / 0.0035 |

Every config declares `TAIL_LAMBDA_LIST=("0.025")`, preserving P0. Tail OFF
uses `mse`; no `--tail_lambda` is passed and the tail branch is inactive. The
inactive parser default is recorded but is not an experimental level. Tail ON
uses `mse_tail` and resolves the active coefficient to exactly 0.025. CPU-only
loss-forward checks certified that MSE never reads the tail threshold, even at
nonzero tail coefficients; no backward pass, optimizer or training loop ran.
No P2 mode enables slope supervision; P0's inactive slope controls are retained.

Frozen weights: body **1**, excess **2**, gate **0.5**; tail fraction **0.05**;
TRAIN exceedance percentile **95**. Amplitude and final-peak pools are hard
`max`; both inactive smoothmax betas are **20**; severity-shape epsilon **1e-6**.
Checkpoint selection is **overall**, primary minimum **VAL rmse_all**,
auxiliary checkpoints **0**, overall tolerance **0.01** for every config.

## Frozen architecture, data and training controls

- `perceiver3` / GraphSAGE / Transformer; history 24 h; hidden 128; graph layers 2;
  dropout/head dropout 0.05; node/time read heads 8; Transformer layers 2,
  FF multiplier 4.0, dropout 0; max time steps 32.
- Dual exceedance/window head; dual loss 1; ablation none; event definition unchanged.
- Chronological train/validation ratios 0.6/0.2; shuffle 0; future-only 0;
  inactive future threshold 2030; seed 42.
- Batch 256, accumulation 4; Adam, LR 0.005, weight decay 1e-5; 300 epochs;
  cosine; warmup 5, **start factor 0.1**; min LR 1e-6; max gradient norm 0;
  deterministic kernels 0. P0's inactive ROP/WMSE/slope settings are unchanged.
- One GPU/CUDA device 0; AMP bf16 and TF32 on; threads 1; workers, pin memory,
  persistent workers and prefetch all 0; multiprocessing fork.
- OOD disabled; zscore; clipping and augmentation/probability/scale/bias all 0.
- Site elevation and bathymetry off; existing six lat/lon-derived features
  validated for all four station JSON files; station metadata remains enabled.
- P0 data roots, station JSON path, verified Python interpreter and activation-off
  settings are preserved. No training data were loaded.

All configured fields and all resolved non-factor CLI fields were compared with
the corresponding P0 dual-MSE reference for **every config: 96 comparisons across
four stations**, with **zero unexpected differences**. Intentional changes are
the P2 factors, run identity and result root; amplitude/shape/peak/formulation and
checkpoint settings that were implicit P0 production defaults are now explicit.
The exact complete controls, including inactive values and production defaults
`device=auto` and `use_station_meta=1`, are in
[`validation/frozen_controls.json`](validation/frozen_controls.json).

## Dry runs, backward compatibility and command files

**96/96 current-repository P2 DRY_RUNs succeeded**, each resolving exactly one
command that passed the production argument parser. Semantic `T/A/S/P` names
appear in both the output directory and Python `run_tag`. The launcher preserves
timestamp naming and refuses to overwrite existing run directories.

All **32 P0 + 32 P1** full dry-run `CMD:` lines are **byte-for-byte unchanged**
against the baseline launcher when `PYTHON_RUN_TAG_BASE` is absent. Explicit
empty overrides were also checked. P2's nonempty override and resolved launcher
snapshot were verified before generation. Tail-launcher behavior was not changed.

| Command file | Commands |
| --- | ---: |
| commands_all.txt | 96 |
| commands_CBBT.txt | 24 |
| commands_Lewes.txt | 24 |
| commands_Battery.txt | 24 |
| commands_Boston.txt | 24 |

All command lines match the manifest and locally supported
`qsub_local train.sh <label> <config_path>` syntax, inspected from `.bashrc`.
The combined list uses CBBT, Lewes, Battery, Boston order. Command lists were
created as text and were never executed.

## Repository protection and evidence

Of **{len(original['tracked_sha256'])} baseline tracked files**, only the authorized
`train.sh` enhancement differs from task start; all **{len(original['tracked_sha256']) - 1} others** retain identical SHA-256
hashes. Production Python/model/loss/training mathematics, P0 and P1 are unchanged.
All **{len(after_results)}** existing result-tree entries retain their metadata;
no result/checkpoint was written and no P2 result directory was created.
**No training, qsub_local call, job submission or tmux execution occurred.**

- [`validation_report.json`](validation_report.json): all 96 resolved namespaces,
  CLI token lists, exact weights, config hashes and per-run P0 differences.
- [`validation/dry_runs.txt`](validation/dry_runs.txt): all 96 launcher outputs.
- [`validation/p0_reference_dry_runs.txt`](validation/p0_reference_dry_runs.txt):
  the four station references used for every non-factor comparison.
- [`validation/launcher_regression.json`](validation/launcher_regression.json):
  baseline/current P0/P1 command strings and semantic-tail/tag regression checks.
- [`validation/original_state.json`](validation/original_state.json): original
  tracked-file hashes and existing-results metadata for the integrity checks.

This certifies configuration and launcher behavior; training performance and
scientific outcomes have not been measured.
''')
    print(f'PASS: 96/96; complete factorial; commands 96 + 4×24; zero frozen-control drift. Reports: {report_dir}', flush=True)


if __name__ == '__main__':
    main()
