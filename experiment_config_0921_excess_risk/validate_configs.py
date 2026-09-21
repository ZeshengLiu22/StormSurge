#!/usr/bin/env python3
"""Validate the four configs on CPU, including actual TRAIN fits; never train."""

import argparse
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
import tempfile

sys.dont_write_bytecode = True
from generate_configs import HERE, MODES, PYTHON, REFERENCE, REPO, RESULTS, START, expected_files, reference_text

sys.path.insert(0, str(REPO))
import torch
from emulator.data import ForcingGraphStore, fit_loss_thresholds, fit_statistics, load_station_json, station_features_from_json
from emulator.models import ModelConfig, build_model, count_model_parameters
from emulator.models.heads import ExceedanceHead
from emulator.training import ForecastLoss, LossConfig
from emulator.training.arguments import parse_args

ENV = dict(PATH=os.environ.get('PATH', '/usr/bin:/bin'), LANG='C.UTF-8', PYTHONDONTWRITEBYTECODE='1',
           CUDA_VISIBLE_DEVICES='')
IDENTITY = {'SESSION_NAME', 'PACT_RUN_NAME', 'PYTHON_RUN_TAG_BASE', 'ALL_RESULTS_ROOT'}
FACTORS = {'EXCESS_EVENT_NORMALIZATION', 'EXCESS_HORIZON_WEIGHTING'}
ARG_IDENTITY = {'run_tag', 'output_dir'}
ARG_FACTORS = {name.lower() for name in FACTORS}


def run(command, env=None):
    return subprocess.check_output(command, cwd=REPO, env=env or ENV, text=True, stderr=subprocess.STDOUT, timeout=60)


def source_values(path):
    names = []
    for line in path.read_text().splitlines():
        if not line.strip() or line.startswith('#'):
            continue
        match = re.fullmatch(r'(?:export )?([A-Za-z_]\w*)=(.*)', line)
        assert match and match[1] not in names, (path, line)
        assert '$(' not in line and '`' not in line and ';' not in line, (path, line)
        names.append(match[1])
    script = '''set -euo pipefail
source "$1"
shift
for setting in "$@"; do
    declare -n setting_value="$setting"
    values=("${setting_value[@]}")
    printf '%s\\0' "$setting" "${#values[@]}" "${values[@]}"
    unset -n setting_value
done
'''
    words = iter(run(['bash', '--noprofile', '--norc', '-c', script, 'bash', str(path), *names]).split('\0')[:-1])
    result = {}
    for name in words:
        assert int(next(words)) == 1, f'{path}: each setting must select exactly one run'
        result[name] = next(words)
    return result


def dry_run(path):
    output = run(['bash', 'train.sh', str(path)], env=dict(ENV, DRY_RUN='1', PACT_RUNSTAMP='20000101_000000'))
    commands = [shlex.split(line[5:]) for line in output.splitlines() if line.startswith('CMD: ')]
    assert len(commands) == 1 and commands[0][0] == 'train.py', path
    launcher = next(shlex.split(line.split(':', 1)[1]) for line in output.splitlines() if line.startswith('LAUNCHER:'))
    assert launcher == [PYTHON, '-u'], launcher
    assert re.search(r'^num_gpus:\s+1$', output, re.MULTILINE)
    return parse_args(commands[0][1:]), shlex.join(launcher + commands[0])


def differences(left, right):
    return {key: [left.get(key), right.get(key)] for key in left.keys() | right.keys() if left.get(key) != right.get(key)}


def check_protocol(args):
    required = dict(station='CBBT', root_dir='/media/share/PACT/Data/Grid4_New/NCEP/graphs',
        model='perceiver3', encoder_type='GraphSAGE', temporal_block='Transformer', history_hours=24,
        hidden_channels=128, num_layers=2, head_type='dual', head_dropout=.05, dropout=.05,
        transformer_layers=2, transformer_ff_mult=4., transformer_dropout=0.,
        node_read_heads=8, time_read_heads=8, max_time_steps=32,
        excess_formulation='direct', exceedance_head_experiment=None, exceedance_gate_pooling='mean',
        dual_ablation='none', gate_mode='window', dual_loss=1, exceedance_percentile=95.,
        body_loss_weight=1., excess_loss_weight=2., gate_loss_weight=.5, loss_mode='mse',
        lr=5e-3, epochs=300, batch_size=256, grad_accum_steps=4, scheduler='cosine',
        warmup_epochs=5, warmup_start_factor=.1, min_lr=1e-6, seed=42, deterministic=0,
        amp=True, amp_dtype='bf16', tf32=True, num_workers=0, torch_threads=1,
        checkpoint_selection='overall', save_aux_checkpoints=0, max_grad_norm=0.,
        x_norm='zscore', x_clip=0., x_aug=0, x_aug_prob=0., x_aug_scale=0., x_aug_bias=0.,
        train_ratio=.6, val_ratio=.2, shuffle_years=0, future_only=0,
        use_site_elevation=0, use_bathymetry=0, use_station_meta=1,
        excess_amp_loss_weight=0., shape_loss_weight=0., peak_loss_weight=0., excess_magnitude_alpha=1.)
    for key, value in required.items():
        assert getattr(args, key) == value, (key, getattr(args, key), value)
    assert args.loss_mode == 'mse'  # Tail/slope CLI defaults are inactive under MSE.


def data_validation(all_args):
    first = all_args[0]
    print('Loading the actual CBBT NCEP graph store on CPU...', flush=True)
    store = ForcingGraphStore(first.root_dir, first.station)
    split_names = ('train_ratio', 'val_ratio', 'shuffle_years', 'seed', 'future_only', 'future_year_threshold')
    splits = store.split(**{name: getattr(first, name) for name in split_names})
    stats = fit_statistics(store, splits['train'], first.x_norm, first.x_p_lo, first.x_p_hi,
                           first.x_nodes_per_graph, first.seed, device=torch.device('cpu'))
    station = station_features_from_json(load_station_json(first.station_json_dir, first.station),
        use_site_elevation=first.use_site_elevation, use_bathymetry=first.use_bathymetry)
    graph = store.graphs[splits['train'][0]]
    reports = []
    with torch.random.fork_rng(devices=[]):
        for (mode, _, _), args in zip(MODES, all_args):
            selected = store.split(**{name: getattr(args, name) for name in split_names})
            assert selected == splits
            fitted = fit_loss_thresholds(store, selected['train'], args.tail_frac,
                args.wmse_q, bool(args.wmse_use_abs), args.exceedance_percentile)
            values = {field.name: getattr(args, field.name) for field in fields(ModelConfig) if hasattr(args, field.name)}
            values.update(model='pact', in_channels=graph.x.size(-1), out_channels=graph.y.numel(),
                temporal_layers=args.transformer_layers, temporal_ff_mult=args.transformer_ff_mult,
                temporal_dropout=args.transformer_dropout, history_steps=args.history_hours // 6,
                station_feat_dim=station.numel(),
                peak_threshold_norm=((fitted['tau_phys'] - stats['y_mean']) / stats['y_std']).tolist(),
                peak_prior=fitted['event_prior'])
            config = ModelConfig(**values)
            torch.manual_seed(args.seed)
            model = build_model(config)  # CPU construction only: no forward or optimizer.
            assert type(model.head) is ExceedanceHead
            digest = hashlib.sha256()
            for name, value in model.state_dict().items():
                digest.update(name.encode())
                digest.update(value.detach().cpu().numpy().tobytes())
            before = torch.get_rng_state().clone()
            loss_config = LossConfig(**{field.name: getattr(args, field.name) for field in fields(LossConfig)})
            criterion = ForecastLoss(loss_config, stats, fitted['tail_threshold'], fitted['wmse_threshold'],
                event_prior=fitted['event_prior'], event_threshold=fitted['tau_phys'])
            assert criterion.event_prior == fitted['event_prior']
            assert torch.equal(before, torch.get_rng_state())
            record = dict(mode=mode, fitted=fitted, model_config=asdict(config),
                          parameters=count_model_parameters(model), initial_state_sha256=digest.hexdigest())
            if reports:
                for key in ('fitted', 'model_config', 'parameters', 'initial_state_sha256'):
                    assert record[key] == reports[0][key], f'{mode}: changed {key}'
            reports.append(record)
            print(f'{mode}: tau_phys={fitted["tau_phys"]:.12g}, prior={fitted["event_prior"]:.12g}, '
                  f'parameters={record["parameters"]["total"]}', flush=True)
    tags = '\n'.join(store.graph_tags[i] for i in splits['train'])
    return dict(runs=reports, split_windows={key: len(value) for key, value in splits.items()},
                train_tags_sha256=hashlib.sha256(tags.encode()).hexdigest(),
                y_mean=stats['y_mean'].tolist(), y_std=stats['y_std'].tolist(),
                data_source=str(first.root_dir), station_feature_dim=station.numel(),
                prior_strength_multiplier=1 / reports[0]['fitted']['event_prior'],
                weighted_excess_coefficient=2 / reports[0]['fitted']['event_prior'],
                fitting_note='Independent validation fits on the same TRAIN population; training fits once as before.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config-only', action='store_true', help='Only dry-run/compare configs; skip actual data and models.')
    parser.add_argument('--output', type=Path, help='Write validation evidence here; defaults to validation/ inside this study.')
    options = parser.parse_args()
    torch.set_num_threads(1)
    files, reference_hash = expected_files()
    for relative, content in files.items():
        assert (HERE / relative).read_text() == content, f'Generated file differs: {relative}'
    assert len(list((HERE / 'configs').glob('*.sh'))) == 4
    changed = run(['git', 'diff', '--name-only', START, '--', 'emulator']).splitlines()
    assert set(changed) <= {'emulator/training/losses.py', 'emulator/training/arguments.py'}, changed
    assert run(['git', 'rev-parse', 'main']).strip() == START, 'Local main moved since study creation'
    results_existed = Path(RESULTS).exists()
    with tempfile.TemporaryDirectory() as temporary:
        reference_path = Path(temporary) / 'reference.sh'
        reference_path.write_text(reference_text())
        old_shell = source_values(reference_path)
        old_args, _ = dry_run(reference_path)
    assert old_shell['DIRECT_DUAL_RECONSTRUCTION'] == 'soft_gate'
    assert old_shell['EXCESS_SUPERVISION_SCOPE'] == 'event'
    all_args, records = [], []
    base_shell = base_args = None
    for mode, normalization, weighting in MODES:
        path = HERE / 'configs' / f'train_config_0921_CBBT_{mode}.sh'
        run(['bash', '-n', str(path)])
        shell = source_values(path)
        args, command = dry_run(path)
        check_protocol(args)
        assert (args.excess_event_normalization, args.excess_horizon_weighting) == (normalization, weighting)
        assert shell['TAIL_LAMBDA_LIST'] == shell['SLOPE_LAMBDA_LIST'] == '0'
        assert shell['DISABLE_OOD'] == '1'
        assert shell['ALL_RESULTS_ROOT'] == RESULTS
        removed = {'DIRECT_DUAL_RECONSTRUCTION', 'EXCESS_SUPERVISION_SCOPE'}
        additions = FACTORS | {'EXCESS_MAGNITUDE_ALPHA', 'EXCEEDANCE_HEAD_EXPERIMENT'}
        historical = differences({k: v for k, v in old_shell.items() if k not in IDENTITY | removed},
                                 {k: v for k, v in shell.items() if k not in IDENTITY | additions})
        assert not historical, (mode, historical)
        resolved = vars(args)
        assert set(differences(vars(old_args), resolved)) <= ARG_IDENTITY | ARG_FACTORS
        if base_shell is None:
            base_shell, base_args = shell, resolved
        assert set(differences(base_shell, shell)) <= IDENTITY | FACTORS
        differences_from_e0 = {k: v for k, v in differences(base_args, resolved).items() if k not in ARG_IDENTITY}
        expected = {}
        if normalization != 'none':
            expected['excess_event_normalization'] = ['none', normalization]
        if weighting != 'uniform':
            expected['excess_horizon_weighting'] = ['uniform', weighting]
        assert differences_from_e0 == expected, (mode, differences_from_e0)
        records.append(dict(mode=mode, command=command, shell=shell, resolved=resolved,
            differences_from_e0=differences_from_e0,
            effective_optional_weights=dict(tail=0., slope=0., excess_amp=0., shape=0., final_peak=0.)))
        all_args.append(args)
        print(f'{mode}: dry-run and matched protocol passed; differences={differences_from_e0}', flush=True)
    data = None if options.config_only else data_validation(all_args)
    assert Path(RESULTS).exists() == results_existed, 'Validation created a results directory'
    report = dict(validated_utc=datetime.now(timezone.utc).isoformat(), starting_main=START,
        implementation_head=run(['git', 'rev-parse', 'HEAD']).strip(), reference=REFERENCE,
        reference_sha256=reference_hash, validation_device='cpu', training_launched=False,
        config_only=options.config_only, runs=records, data_validation=data,
        loss_only_source_changes=changed, production_default_path='none + uniform',
        checkpoint_selection='overall validation rmse_all; TEST unused for selection',
        caveat='DETERMINISTIC=0: matching initial CPU state does not imply identical training trajectories. '
               'Tail/slope parse as historical inactive defaults 0.1/0.01 because MSE omits those CLI flags.')
    destination = options.output or HERE / 'validation' / ('config_only.json' if options.config_only else 'validation.json')
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, default=str, allow_nan=False) + '\n')
    print(f'All checks passed. Evidence: {destination}. No training, queue submission or results creation.', flush=True)


if __name__ == '__main__':
    main()
