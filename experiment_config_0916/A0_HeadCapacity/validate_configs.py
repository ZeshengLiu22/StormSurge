#!/usr/bin/env python3
"""Validate A0 configs, dry-run the launcher, and count CPU models. Never train."""

from collections import Counter
import csv
from dataclasses import asdict, fields
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))

import torch
from emulator.data.graph_store import ForcingGraphStore
from emulator.data.station_metadata import load_station_json, station_features_from_json
from emulator.models import ModelConfig, build_model, count_model_parameters
from emulator.training.arguments import parse_args

STATIONS = ('CBBT', 'Lewes', 'Battery', 'Boston')
VARIANTS = ('single_base', 'dual_direct_base', 'dual_severity_base')
RESULTS = REPO / 'All_results_0916/A0_HeadCapacity'
STAMP = '20000101_000000'  # Reproducible dry-run identity; no directory is created.
COMMON = dict(
    TRAIN_PY='train.py', ROOT_DIR='/media/share/PACT/Data/Grid4_New/NCEP/graphs',
    STATION_JSON_DIR=str(REPO / 'station_json'), TEST_ROOT_DIR='',
    MODEL='perceiver3', ENCODER_TYPE='GraphSAGE', CNN_INTERMEDIATE_CHANNEL='29',
    TEMPORAL_BLOCK='Transformer', HISTORY_HOURS_LIST='24', HIDDEN_CHANNELS='128',
    NUM_LAYERS='2', DROPOUT='0.05', HEAD_DROPOUT='0.05', NODE_READ_HEADS='8',
    TIME_READ_HEADS='8', TRANSFORMER_LAYERS='2', TRANSFORMER_FF_MULT='4.0',
    TRANSFORMER_DROPOUT='0.0', MAX_TIME_STEPS='32', TRAIN_RATIO='0.6', VAL_RATIO='0.2',
    SHUFFLE_YEARS='0', FUTURE_ONLY='0', FUTURE_YEAR_THRESHOLD='2030', SEED='42',
    USE_SITE_ELEVATION='0', USE_BATHYMETRY='0', GATE_MODE='window', DUAL_MODE='exceedance',
    DUAL_ABLATION='none', EXCEEDANCE_PERCENTILE='95', LOSS_MODE_LIST='mse',
    TAIL_FRAC='0.05', TAIL_LAMBDA_LIST='0', SLOPE_LAMBDA_LIST='0',
    SLOPE_MASK_S_LIST='0.10', SLOPE_ROBUST='charb', SLOPE_CHARB_EPS='1e-3',
    SLOPE_HUBER_DELTA='0.05', WMSE_Q_LIST='95', WMSE_ALPHA='4.0', WMSE_S='0.10',
    WMSE_USE_ABS='1', DISABLE_OOD='1', X_NORM='zscore', X_CLIP='0', X_AUG='0',
    X_AUG_PROB='0', X_AUG_SCALE='0', X_AUG_BIAS='0', X_P_LO='1.0', X_P_HI='99.0',
    X_NODES_PER_GRAPH='256', BATCH_SIZE='256', GRAD_ACCUM_STEPS='4', LR_LIST='5e-3',
    EPOCHS='300', SCHEDULER='cosine', WARMUP_EPOCHS='5', WARMUP_START_FACTOR='0.1',
    MIN_LR='1e-6', MAX_GRAD_NORM='0', DETERMINISTIC='0', ROP_METRIC='val_rmse_phys',
    ROP_FACTOR='0.5', ROP_PATIENCE='20', ROP_THRESHOLD='1e-4', ROP_COOLDOWN='0',
    ROP_MIN_LR='1e-6', num_gpus='1', CUDA_VISIBLE_DEVICES='0', USE_AMP='1',
    AMP_DTYPE='bf16', USE_TF32='1', TORCH_THREADS='1', NUM_WORKERS='0', PIN_MEMORY='0',
    PERSISTENT_WORKERS='0', PREFETCH_FACTOR='0', MP_CONTEXT='fork',
    PYTHON_BIN='/media/volume/PACT-Data/conda_envs/torchpyg-cu124/bin/python', DO_CONDA='0',
    CONDA_ENV='/media/volume/PACT-Data/conda_envs/torchpyg-cu124', PYTHONDONTWRITEBYTECODE='1',
    USE_TMUX='1', ALL_RESULTS_ROOT='./All_results_0916/A0_HeadCapacity',
    RUN_DIR_NAME_STYLE='runname_timestamp', EXCESS_AMP_LOSS_WEIGHT='0',
    EXCESS_AMP_POOL='max', EXCESS_AMP_BETA='20', SHAPE_LOSS_WEIGHT='0',
    SEVERITY_SHAPE_EPS='1e-6', PEAK_LOSS_WEIGHT='0', PEAK_POOL='max', PEAK_POOL_BETA='20',
    CHECKPOINT_SELECTION='overall', SAVE_AUX_CHECKPOINTS='0', CHECKPOINT_OVERALL_TOL='0.01',
)
VARYING = {'STATION', 'HEAD_TYPE', 'DUAL_LOSS', 'BODY_LOSS_WEIGHT', 'EXCESS_LOSS_WEIGHT',
           'GATE_LOSS_WEIGHT', 'EXCESS_FORMULATION', 'SESSION_NAME', 'PACT_RUN_NAME',
           'PYTHON_RUN_TAG_BASE'}


def run(command, env=None):
    process = subprocess.run(command, cwd=REPO, env=env, text=True,
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=60)
    if process.returncode:
        raise RuntimeError(f'{shlex.join(map(str, command))}\n{process.stdout}')
    return process.stdout


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_values(path):
    names = []
    for line in path.read_text().splitlines():
        if not line.strip() or line.lstrip().startswith('#'):
            continue
        match = re.fullmatch(r'(?:export )?([A-Za-z_]\w*)=(.*)', line)
        assert match, (path.name, 'Only explicit assignments are allowed', line)
        name, value = match.groups()
        assert name not in names, (path.name, 'Duplicate setting', name)
        code = ' '.join(shlex.split(value, comments=True))
        assert not any(token in code for token in ('$(', '`', ';')), (path.name, line)
        if '$' in value:
            assert (name == 'SESSION_NAME' and value.startswith('"${SESSION_NAME:-')) or (
                name == 'PYTHON_RUN_TAG_BASE' and value == '"${PACT_RUN_NAME}"'), (name, value)
        names.append(name)
    script = r'''set -euo pipefail
source "$1"
shift
for a0_key in "$@"; do
    declare -n a0_value="$a0_key"
    a0_items=("${a0_value[@]}")
    printf '%s\0' "$a0_key" "${#a0_items[@]}" "${a0_items[@]}"
    unset -n a0_value
done
'''
    output = run(['bash', '--noprofile', '--norc', '-c', script, 'bash', str(path), *names],
                 env={'PATH': '/usr/bin:/bin'})
    words = iter(output.removesuffix('\0').split('\0'))
    values = {}
    for name in words:
        size = int(next(words))
        assert size == 1, (path.name, 'Each config must resolve exactly one run', name, size)
        values[name] = next(words)
    return values


def tree_snapshot(root):
    if not root.exists():
        return {}
    return {str(path.relative_to(root)): [path.stat().st_size, path.stat().st_mtime_ns]
            for path in root.rglob('*')}


def capacity(args):
    # Read only tensor shapes from one existing TRAIN-year graph file per station.
    paths = sorted(Path(args.root_dir).glob(f'*_{args.station}_*graphs.pt'))
    assert paths, args.station
    store = ForcingGraphStore.__new__(ForcingGraphStore)
    store.year_to_indices = {'_'.join(path.name.split('_')[:2]): [i] for i, path in enumerate(paths)}
    split = store.split(args.train_ratio, args.val_ratio, bool(args.shuffle_years), args.seed,
                        bool(args.future_only), args.future_year_threshold)
    sample_path = paths[split['train'][0]]
    graphs = torch.load(sample_path, map_location='meta', mmap=True, weights_only=False)
    sample = graphs[0]
    inputs, horizons = sample.x.size(-1), sample.y.numel()
    assert sample.x_hist.size(0) >= args.history_hours // 6 + 1
    metadata = station_features_from_json(load_station_json(args.station_json_dir, args.station),
        use_site_elevation=bool(args.use_site_elevation), use_bathymetry=bool(args.use_bathymetry))
    assert args.use_station_meta == 1 and metadata.numel() == 6
    values = {field.name: getattr(args, field.name) for field in fields(ModelConfig) if hasattr(args, field.name)}
    values.update(in_channels=inputs, out_channels=horizons, model='pact',
                  temporal_layers=args.transformer_layers, temporal_ff_mult=args.transformer_ff_mult,
                  temporal_dropout=args.transformer_dropout, history_steps=args.history_hours // 6,
                  station_feat_dim=metadata.numel(), peak_prior=0.05)
    context = dict(input_channels=inputs, output_horizons=horizons,
                   available_history_steps=sample.x_hist.size(0), station_feat_dim=metadata.numel(),
                   sample_file=str(sample_path),
                   split_year_groups={key: ['_'.join(paths[i].name.split('_')[:2]) for i in indices]
                                      for key, indices in split.items()})
    del graphs, sample
    result = {}
    for variant in VARIANTS:
        single = variant == 'single_base'
        severity = variant == 'dual_severity_base'
        config = ModelConfig(**dict(values, head_type='single' if single else 'dual',
            excess_formulation='severity_shape' if severity else 'direct',
            peak_threshold_norm=None if single else [0.] * horizons,
            target_y_std=[1.] * horizons if severity else None))
        torch.manual_seed(args.seed)
        model = build_model(config).cpu()
        result[variant] = dict(model_config=asdict(config), parameters=count_model_parameters(model))
        del model
    return dict(**context, variants=result)


def main():
    torch.set_num_threads(1)
    assert Path(sys.executable).resolve() == Path(COMMON['PYTHON_BIN']).resolve(), 'Use the production Python interpreter.'
    configs = sorted(HERE.glob('train_config_*.sh'))
    assert len(configs) == 12
    with (HERE / 'manifest.csv').open() as handle:
        manifest = list(csv.DictReader(handle))
    assert len(manifest) == 12
    assert Counter(row['station'] for row in manifest) == Counter({station: 3 for station in STATIONS})
    assert Counter(row['variant'] for row in manifest) == Counter({variant: 4 for variant in VARIANTS})
    assert {(row['station'], row['variant']) for row in manifest} == {
        (station, variant) for station in STATIONS for variant in VARIANTS}
    assert {REPO / row['config'] for row in manifest} == set(configs)
    commands = (HERE / 'commands_all.txt').read_text().splitlines()
    assert commands == [row['qsub_local_command'] for row in manifest]
    run(['bash', '-n', str(HERE / 'commands_all.txt')])
    run(['bash', '-n', str(REPO / 'train.sh')])
    before_results = tree_snapshot(RESULTS)
    source_paths = [REPO / 'train.sh', REPO / 'train.py', *sorted((REPO / 'emulator').rglob('*.py'))]
    source_before = {str(path.relative_to(REPO)): digest(path) for path in source_paths}
    provenance = json.loads((HERE / 'provenance.json').read_text())
    refs = {}
    for station, reference in provenance['production_references'].items():
        path = REPO / reference['path']
        assert digest(path) == reference['sha256']
        refs[station] = source_values(path)
    evidence = HERE / 'validation'
    evidence.mkdir(exist_ok=True)
    records = []
    parsed = {}
    for row in manifest:
        station, variant = row['station'], row['variant']
        path = REPO / row['config']
        single = variant == 'single_base'
        name = f'NCEP_{station}_24h_{variant}'
        label = f'A0_{station}_{variant}'
        expected = dict(COMMON, STATION=station, HEAD_TYPE='single' if single else 'dual',
            DUAL_LOSS='0' if single else '1', BODY_LOSS_WEIGHT='0' if single else '1',
            EXCESS_LOSS_WEIGHT='0' if single else '2', GATE_LOSS_WEIGHT='0' if single else '0.5',
            EXCESS_FORMULATION='severity_shape' if variant == 'dual_severity_base' else 'direct',
            SESSION_NAME=label, PACT_RUN_NAME=name, PYTHON_RUN_TAG_BASE=name)
        run(['bash', '-n', str(path)])
        shell = source_values(path)
        assert shell == expected, (path.name, {key: (shell.get(key), expected.get(key))
                                             for key in shell.keys() | expected.keys() if shell.get(key) != expected.get(key)})
        # Compare every production control except the requested head/loss/identity changes.
        permitted = VARYING | {'ALL_RESULTS_ROOT', 'TAIL_LAMBDA_LIST', 'SLOPE_LAMBDA_LIST'}
        differences = {key for key in shell.keys() | refs[station].keys() if shell.get(key) != refs[station].get(key)}
        assert differences <= permitted, (path.name, differences - permitted)
        assert row['run_name'] == name and row['queue_label'] == label
        assert row['qsub_local_command'] == f'qsub_local train.sh {label} {row["config"]}'
        assert row['results_root'] == str(RESULTS.relative_to(REPO))
        assert row['result_pattern'] == f'{row["results_root"]}/{name}__<TIMESTAMP>/'
        for field in ('head_type', 'excess_formulation', 'dual_loss', 'body_loss_weight',
                      'excess_loss_weight', 'gate_loss_weight', 'excess_amp_loss_weight',
                      'shape_loss_weight', 'peak_loss_weight', 'checkpoint_selection', 'save_aux_checkpoints'):
            assert row[field] == shell[field.upper()], (path.name, field)
        assert row['loss_mode'] == 'mse' and row['tail_lambda'] == row['slope_lambda'] == '0'
        env = dict(os.environ, DRY_RUN='1', PACT_RUNSTAMP=STAMP, PYTHONDONTWRITEBYTECODE='1')
        for key in ('SESSION_NAME', 'QSUB_LOCAL_STATUS_FILE', 'SLURM_NTASKS_PER_NODE', 'SLURM_JOB_ID'):
            env.pop(key, None)
        output = run(['bash', 'train.sh', row['config']], env=env)
        (evidence / f'{name}.dry_run.txt').write_text(output)
        emitted = [line.removeprefix('CMD: ') for line in output.splitlines() if line.startswith('CMD: ')]
        assert len(emitted) == 1, (path.name, 'Expected one command', len(emitted))
        argv = shlex.split(emitted[0])
        assert argv[0] == 'train.py'
        args = parse_args(argv[1:])  # Parser only: never invoke train.main or the emitted command.
        assert args.station == station and args.head_type == expected['HEAD_TYPE']
        assert args.excess_formulation == expected['EXCESS_FORMULATION'] and args.dual_loss == int(not single)
        assert args.loss_mode == 'mse'
        assert args.excess_amp_loss_weight == args.shape_loss_weight == args.peak_loss_weight == 0
        assert args.checkpoint_selection == 'overall' and args.save_aux_checkpoints == 0
        assert Path(args.output_dir).resolve() == RESULTS / f'{name}__{STAMP}'
        assert args.run_tag == f'{name}_{STAMP}'
        assert args.amp and args.amp_dtype == 'bf16' and args.tf32
        assert args.grad_accum_steps == 4 and 'num_gpus:      1' in output
        assert f'LAUNCHER:      {COMMON["PYTHON_BIN"]} -u' in output
        assert f'USE_TMUX=1 SESSION_NAME={label} DRY_RUN=1' in output
        if single:
            assert not any(option in argv for option in ('--dual_loss', '--body_loss_weight',
                                                        '--excess_loss_weight', '--gate_loss_weight'))
        else:
            assert (args.body_loss_weight, args.excess_loss_weight, args.gate_loss_weight) == (1, 2, .5)
        active = dict(tail=0, excess_amplitude=args.excess_amp_loss_weight, shape=args.shape_loss_weight,
                      final_peak=args.peak_loss_weight, slope=0, dual_supervision=args.dual_loss)
        records.append(dict(config=row['config'], sha256=digest(path), station=station, variant=variant,
                            syntax='PASS', dry_run='PASS', command_count=1, production_controls='PASS',
                            production_differences=sorted(differences), shell=shell,
                            parsed_cli=vars(args), active_losses=active))
        parsed[station, variant] = args
        print(f'PASS {name}: syntax, standalone settings, one dry-run command, production parser', flush=True)
    counts = {}
    for station in STATIONS:
        direct_args = vars(parsed[station, 'dual_direct_base'])
        severity_args = vars(parsed[station, 'dual_severity_base'])
        differences = {key for key in direct_args if direct_args[key] != severity_args[key]}
        assert differences == {'excess_formulation', 'run_tag', 'output_dir'}, differences
        counts[station] = capacity(parsed[station, 'dual_direct_base'])
        variants = counts[station]['variants']
        single, direct, severity = [variants[tag]['parameters'] for tag in VARIANTS]
        assert single['backbone'] == direct['backbone'] == severity['backbone']
        assert set(single['head_branches']) == {'regression'}
        assert set(direct['head_branches']) == {'body', 'excess', 'gate'}
        assert set(severity['head_branches']) == {'body', 'severity', 'shape', 'gate'}
        assert severity['total'] - direct['total'] == severity['head']['total'] - direct['head']['total'] == 33281
        assert direct['head']['total'] == 99843 and severity['head']['total'] == 133124
        assert all(value['parameters']['total'] == value['parameters']['trainable'] for value in variants.values())
        counts[station]['severity_minus_direct'] = dict(total=33281, head=33281, backbone=0,
            total_percent=100 * 33281 / direct['total'], head_percent=100 * 33281 / direct['head']['total'])
        print(f'PASS {station} capacity: direct={direct["total"]:,}; severity={severity["total"]:,}; delta=33,281', flush=True)
    assert tree_snapshot(RESULTS) == before_results, 'Dry runs must not create/alter training results.'
    assert source_before == {str(path.relative_to(REPO)): digest(path) for path in source_paths}
    report = dict(status='PASS', validated_utc=datetime.now(timezone.utc).isoformat(),
        repository=str(REPO), repository_commit=run(['git', 'rev-parse', 'HEAD']).strip(),
        python=sys.executable, configs_total=12, runs_total=12,
        per_station=dict(Counter(row['station'] for row in records)),
        per_variant=dict(Counter(row['variant'] for row in records)),
        syntax_passed=12, dry_runs_passed=12, production_parser_passed=12,
        all_auxiliary_losses_off=True, dual_weights_identical=dict(body=1, excess=2, gate=.5),
        all_checkpoint_selection='overall', all_save_aux_checkpoints=0,
        model_loss_launcher_source_unchanged=True, result_tree_unchanged=True,
        training_started=False, jobs_submitted=0, model_forward_passes=0, optimizer_steps=0,
        source_sha256=source_before, records=records)
    notes = ('Counts use emulator.models.count_model_parameters on CPU models built from parsed A0 settings, '
             'actual graph tensor dimensions, and the six existing station-coordinate features. '
             'Only one TRAIN-year file per station is memory-mapped onto the meta device for shapes. '
             'Placeholder thresholds=0, target scales=1 and prior=0.05 are used only for counting; '
             'they are non-parameter buffers/initialization and do not change registered capacity. '
             'Training still fits actual TRAIN statistics. No model forward, optimizer, training or queue submission occurs.')
    (evidence / 'parameter_counts.json').write_text(json.dumps(dict(method=notes, stations=counts), indent=2) + '\n')
    (HERE / 'validation_report.json').write_text(json.dumps(report, indent=2, default=str) + '\n')
    lines = [
        '# A0 HeadCapacity validation — PASS', '',
        f'Validated UTC: {report["validated_utc"]}  ', f'Repository commit: `{report["repository_commit"]}`', '',
        'Exactly 12 standalone configs, three per station and four per formulation. All 12 passed `bash -n`,',
        '`DRY_RUN=1 bash train.sh <config>`, and the production argument parser; each emitted exactly one run.',
        'All non-experimental controls match the validated production configs. No runtime config inheritance.', '',
        '| Station | Single base | Direct dual | Severity-shape dual | Total |',
        '| --- | ---: | ---: | ---: | ---: |',
        *[f'| {station} | 1 | 1 | 1 | 3 |' for station in STATIONS], '',
        'Tail, excess amplitude, explicit shape, final peak, and slope supervision are OFF in all 12 configs.',
        'Both duals use BODY=1, trajectory EXCESS=2, GATE=0.5, exceedance/window gating, dual loss 1 and ablation none.',
        'Single uses dual loss 0 and only a regression head. Every run selects `overall` with auxiliary checkpoints 0.', '',
        'The unchanged launcher omits tail/slope arguments under `mse`; the parser retains inactive defaults',
        '`tail_lambda=0.1` and `slope_lambda=0.01`. The config coefficients are explicitly zero and neither term',
        'enters the MSE objective. Similarly, single-head branch weights are explicitly zero in the config;',
        'the launcher omits these arguments and the parser disables dual supervision (`dual_loss=0`).', '',
        '## Parameter counts', '', notes, '',
        '| Station | Model | Total / trainable | Head | Backbone |',
        '| --- | --- | ---: | ---: | ---: |',
    ]
    for station in STATIONS:
        for variant in VARIANTS:
            p = counts[station]['variants'][variant]['parameters']
            lines.append(f'| {station} | {variant} | {p["total"]:,} | {p["head"]["total"]:,} | {p["backbone"]["total"]:,} |')
    delta = counts['CBBT']['severity_minus_direct']
    lines += ['', f'Severity-shape adds **33,281** trainable parameters over direct: one 128→256→1 branch MLP,',
              f'**{delta["total_percent"]:.3f}%** more total capacity and **{delta["head_percent"]:.3f}%** more head capacity.',
              'Direct excess capacity is 33,281; severity plus shape capacity is 66,562. Body/gate/backbone are unchanged.',
              'This is the requested three-formulation study; no fourth capacity-matched model was added.', '',
              'See [manifest.csv](manifest.csv), [commands_all.txt](commands_all.txt),',
              '[full verification](validation_report.json), and [branch counts](validation/parameter_counts.json).',
              'The `validation/` directory contains all 12 dry-run transcripts.', '',
              'Model/loss/launcher hashes and result-tree contents were unchanged by validation.',
              '**No training started, no jobs submitted, and no model forward passes or optimizer steps executed.**', '']
    (HERE / 'validation_report.md').write_text('\n'.join(lines))
    print('PASS: 12 configs / 12 dry runs / all requested controls / CPU parameter counts / no training.')


if __name__ == '__main__':
    main()
