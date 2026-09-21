#!/usr/bin/env python3
"""Dry-run all 24 commands and validate real TRAIN splits, thresholds and initialization.

No optimizer steps, GPU jobs, queue submissions or result directories are created.
"""

import argparse
from collections import Counter
from dataclasses import fields
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys

from generate_configs import (BASE_SHA, CONDITIONS, HERE, PRIMARY, PYTHON, REPO, RESULTS,
                              STATIONS, expected_files, manifest_rows, reference_path)

sys.dont_write_bytecode = True
sys.path.insert(0, str(REPO))
import numpy as np
import torch
from torch_geometric.data import Batch, Data
from emulator.common.runtime import configure_runtime
from emulator.data import (ForcingGraphStore, ForcingGraphView, build_loader, fit_loss_thresholds,
                           fit_statistics, load_station_json, station_features_from_json)
from emulator.models import ModelConfig, build_model, count_model_parameters
from emulator.models.heads import ExceedanceHead
from emulator.training import ForecastLoss, LossConfig
from emulator.training.arguments import parse_args
from emulator.training.metrics import summarize_windows, unique_rows_and_top5_indices

STAMP = '20000101_000000'
ENV = dict(PATH=os.environ.get('PATH', '/usr/bin:/bin'), PYTHONDONTWRITEBYTECODE='1', LANG='C.UTF-8')
IDENTITY = {'SESSION_NAME', 'PACT_RUN_NAME', 'PYTHON_RUN_TAG_BASE', 'ALL_RESULTS_ROOT'}
FACTORS = {'EXCEEDANCE_PERCENTILE', 'EXCESS_LOSS_WEIGHT', 'DUAL_BODY_CAP'}
ARG_DIFFERENCES = {'exceedance_percentile', 'excess_loss_weight', 'dual_body_cap', 'run_tag', 'output_dir'}
PRIMARY_METRICS = ('rmse_all', 'mae_all', 'rmse_peak5', 'mae_peak5', 'peak_magnitude_rmse_top5',
                   'peak_magnitude_mae_top5', 'peak_bias_top5', 'peak_underprediction_fraction_top5',
                   'true_peak_point_rmse_top5', 'peak_timing_mae_steps_top5')


def run(command, *, cwd=REPO, env=ENV):
    result = subprocess.run(command, cwd=cwd, env=env, text=True, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, timeout=60)
    if result.returncode:
        raise RuntimeError(f'{shlex.join(command)}\n{result.stdout}\n{result.stderr}')
    return result.stdout


def sha(content):
    return hashlib.sha256(content).hexdigest()


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
    declare -n value="$setting"
    values=("${value[@]}")
    [[ "${#values[@]}" -eq 1 ]]
    printf '%s\\0%s\\0' "$setting" "${values[@]}"
    unset -n value
done
'''
    words = iter(run(['bash', '--noprofile', '--norc', '-c', script, 'bash', str(path), *names]).rstrip('\0').split('\0'))
    return {key: next(words) for key in words}


def dry_run(path):
    output = run(['bash', 'train.sh', str(path.relative_to(REPO))],
                 env=dict(ENV, DRY_RUN='1', PACT_RUNSTAMP=STAMP))
    commands = [shlex.split(line[5:]) for line in output.splitlines() if line.startswith('CMD: ')]
    assert len(commands) == 1 and commands[0][0] == 'train.py', path
    launcher = next(shlex.split(line.split(':', 1)[1]) for line in output.splitlines() if line.startswith('LAUNCHER:'))
    assert launcher == [PYTHON, '-u']
    assert re.search(r'^num_gpus:\s+1$', output, re.MULTILINE)
    assert re.search(r'^CUDA_VISIBLE_DEVICES:\s+0$', output, re.MULTILINE)
    return output, parse_args(commands[0][1:]), shlex.join(launcher + commands[0])


def tree_snapshot(root):
    return {str(p.relative_to(root)): (p.stat().st_size, p.stat().st_mtime_ns)
            for p in root.rglob('*') if p.is_file()} if root.exists() else {}


def check_sources():
    assert run(['git', 'rev-parse', 'HEAD'], cwd=PRIMARY).strip() == BASE_SHA
    assert run(['git', 'branch', '--show-current'], cwd=PRIMARY).strip() == 'main'
    assert run(['git', 'status', '--porcelain'], cwd=PRIMARY) == '', 'Primary main is not clean'
    assert run(['git', 'merge-base', BASE_SHA, 'HEAD']).strip() == BASE_SHA
    mutable = {'emulator/common/dual.py', 'emulator/models/heads.py', 'emulator/models/architectures.py',
               'emulator/training/arguments.py', 'train.py', 'train.sh', 'README.md',
               'configs/configs_train/train_config_common.sh', 'tests/test_dual_body_cap.py'}
    changed = run(['git', 'diff', BASE_SHA, '--name-only']).splitlines()
    assert all(name in mutable or name.startswith(HERE.name + '/') for name in changed), changed
    paths = run(['git', 'ls-tree', '-r', '--name-only', BASE_SHA, 'emulator', 'station_json',
                 'infer.py', 'infer.sh', 'infer_multi.sh', 'experiment_config_0916', 'experiment_config_0919']).splitlines()
    hashes = {}
    for name in paths:
        if name not in mutable:
            content = (REPO / name).read_bytes()
            baseline = subprocess.check_output(['git', 'show', f'{BASE_SHA}:{name}'], cwd=REPO)
            assert content == baseline, f'Frozen source/reference changed: {name}'
            hashes[name] = sha(content)
    # The only training-entrypoint change is cap metadata; the epoch, optimizer,
    # checkpoint selection and final metric/reporting code remain byte-identical.
    baseline = run(['git', 'show', f'{BASE_SHA}:train.py'])
    assert (REPO / 'train.py').read_text().replace('                          dual_body_cap=args.dual_body_cap,\n', '') == baseline
    return hashes


def model_for(args, store, splits, stats, fitted, station_feat):
    values = {f.name: getattr(args, f.name) for f in fields(ModelConfig) if hasattr(args, f.name)}
    graph = store.graphs[splits['train'][0]]
    values.update(model='pact', in_channels=graph.x.size(-1), out_channels=graph.y.numel(),
                  temporal_layers=args.transformer_layers, temporal_ff_mult=args.transformer_ff_mult,
                  temporal_dropout=args.transformer_dropout, history_steps=args.history_hours // 6,
                  station_feat_dim=station_feat.numel(), peak_prior=fitted['event_prior'],
                  peak_threshold_norm=((fitted['tau_phys'] - stats['y_mean']) / stats['y_std']).tolist())
    return build_model(ModelConfig(**values))


def data_check(station, arguments):
    first = arguments[0]
    print(f'Loading {station} real graphs and fitting TRAIN Q95/Q90...', flush=True)
    store = ForcingGraphStore(first.root_dir, station)
    split_keys = ('train_ratio', 'val_ratio', 'shuffle_years', 'seed', 'future_only', 'future_year_threshold')
    splits = store.split(**{key: getattr(first, key) for key in split_keys})
    for indices in splits.values():
        assert indices
        ForcingGraphView(store, indices, first.history_hours // 6)
    stats = fit_statistics(store, splits['train'], first.x_norm, first.x_p_lo, first.x_p_hi,
                           first.x_nodes_per_graph, first.seed, device=torch.device('cpu'))
    station_feat = station_features_from_json(load_station_json(first.station_json_dir, station),
        use_site_elevation=bool(first.use_site_elevation), use_bathymetry=bool(first.use_bathymetry))
    assert station_feat.numel() == 6
    baseline95 = fit_loss_thresholds(store, splits['train'])
    thresholds, reference_states, records = {}, {}, []
    common_parameters = common_rng = common_order = parameter_counts = None
    for condition, args in zip(CONDITIONS, arguments):
        assert splits == store.split(**{key: getattr(args, key) for key in split_keys})
        fitted = fit_loss_thresholds(store, splits['train'], args.tail_frac, args.wmse_q,
                                     bool(args.wmse_use_abs), args.exceedance_percentile)
        q = str(int(args.exceedance_percentile))
        if q == '95':
            assert fitted == baseline95, 'Q95 differs from the unchanged production default'
        assert fitted['tail_threshold'] == baseline95['tail_threshold']
        assert fitted['wmse_threshold'] == baseline95['wmse_threshold']
        if q in thresholds:
            assert thresholds[q] == fitted
        thresholds[q] = fitted
        configure_runtime(args.seed, args.torch_threads, bool(args.deterministic), bool(args.tf32))
        assert not torch.are_deterministic_algorithms_enabled()
        model = model_for(args, store, splits, stats, fitted, station_feat)
        assert type(model.head) is ExceedanceHead
        assert model.head.dual_body_cap == args.dual_body_cap
        counts = count_model_parameters(model)
        rng = torch.get_rng_state()
        # Gate bias follows the unchanged empirical-prior initialization. Changing
        # Q95 -> Q90 intentionally changes that bias and the threshold buffer only.
        parameters = {name: p.detach().clone() for name, p in model.named_parameters()
                      if name != 'head.gate.3.bias'}
        state = {name: value.clone() for name, value in model.state_dict().items()}
        if common_parameters is None:
            common_parameters, common_rng, parameter_counts = parameters, rng, counts
        assert counts == parameter_counts
        assert parameters.keys() == common_parameters.keys()
        for name, value in parameters.items():
            torch.testing.assert_close(value, common_parameters[name], rtol=0, atol=0)
        assert torch.equal(common_rng, rng)
        if q in reference_states:
            assert state.keys() == reference_states[q].keys()
            for name, value in state.items():
                torch.testing.assert_close(value, reference_states[q][name], rtol=0, atol=0)
        reference_states[q] = state
        assert abs(model.head.gate[-1].bias.sigmoid().item() - fitted['event_prior']) < 1e-7
        loader = build_loader(list(range(len(splits['train']))), None, batch_size=args.batch_size,
            num_workers=args.num_workers, pin_memory=args.pin_memory, persistent_workers=args.persistent_workers,
            prefetch_factor=args.prefetch_factor, mp_context=args.mp_context, shuffle=True,
            generator=torch.Generator().manual_seed(args.seed))
        orders = []
        for epoch in range(2):
            torch.rand(17 * (1 + int(condition[0][1:])) + epoch)  # Perturb only global RNG.
            order = torch.cat(list(loader)).numpy()
            assert len(np.unique(order)) == len(splits['train'])
            orders.append(sha(order.tobytes()))
        if common_order is None:
            common_order = orders
        assert orders == common_order and orders[0] != orders[1]
        records.append(dict(condition=condition[0], body_cap=args.dual_body_cap,
                            tau_norm=model.head.threshold.squeeze(0).tolist(),
                            gate_initial_probability=model.head.gate[-1].bias.sigmoid().item(),
                            random_parameters_sha256=sha(b''.join(p.numpy().tobytes() for p in parameters.values()))))
    # Exercise the existing TEST metric reducer with fixed synthetic predictions
    # on real TEST labels. The ten requested metrics must ignore TRAIN Q95/Q90.
    target = torch.stack([store.graphs[i].y.reshape(-1).float() for i in splits['test']])
    peaks = target.max(dim=1).values.numpy()
    n = len(peaks)
    rows = np.column_stack((np.arange(n), peaks, np.linspace(.01, .5, n), np.linspace(.1, .7, n),
                            np.linspace(-.3, .2, n), np.linspace(-.4, .1, n), np.arange(n) % target.size(1)))
    _, selected = unique_rows_and_top5_indices(rows, validation=True)
    assert len(selected) == max(1, math.ceil(.05 * n))
    np.testing.assert_array_equal(selected, np.argsort(peaks, kind='quicksort')[-len(selected):])
    metrics = [summarize_windows(rows, validation=True, event_threshold=thresholds[q]['tau_phys']) for q in ('95', '90')]
    assert all(metrics[0][key] == metrics[1][key] for key in PRIMARY_METRICS)
    result = dict(thresholds=thresholds, parameters=parameter_counts, conditions=records,
        in_channels=store.graphs[splits['train'][0]].x.size(-1), out_channels=target.size(1),
        split_counts={key: len(indices) for key, indices in splits.items()},
        split_year_groups={key: sorted({'_'.join(store.graph_tags[i].split('_')[:2]) for i in indices})
                           for key, indices in splits.items()},
        split_tag_sha256={key: sha('\n'.join(store.graph_tags[i] for i in indices).encode()) for key, indices in splits.items()},
        train_targets_sha256=sha(torch.stack([store.graphs[i].y.reshape(-1).float() for i in splits['train']]).numpy().tobytes()),
        target_mean=stats['y_mean'].tolist(), target_std=stats['y_std'].tolist(),
        matched_initialization_and_rng=True, matched_real_sample_order_first_two_epochs=common_order,
        test_windows=n, test_top5_windows=len(selected), test_top5_ids_sha256=sha(selected.tobytes()),
        primary_metrics_independent_of_train_percentile=True)
    print(f"PASS {station}: {parameter_counts['total']:,} parameters; "
          f"Q95={thresholds['95']['tau_phys']:.9f} m ({thresholds['95']['event_count']} events); "
          f"Q90={thresholds['90']['tau_phys']:.9f} m ({thresholds['90']['event_count']} events)", flush=True)
    return result


def gpu_check(groups, stations):
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('GPU BF16 smoke requested but CUDA/BF16 is unavailable; run with GPU access.')
    checks = []
    for station, arguments in groups.items():
        data = stations[station]
        for condition, args in zip(data['conditions'], arguments):
            configure_runtime(args.seed, args.torch_threads, bool(args.deterministic), bool(args.tf32))
            fitted = data['thresholds'][str(int(args.exceedance_percentile))]
            values = {f.name: getattr(args, f.name) for f in fields(ModelConfig) if hasattr(args, f.name)}
            values.update(model='pact', in_channels=data['in_channels'], out_channels=data['out_channels'],
                temporal_layers=args.transformer_layers, temporal_ff_mult=args.transformer_ff_mult,
                temporal_dropout=args.transformer_dropout, history_steps=args.history_hours // 6,
                station_feat_dim=6, peak_prior=fitted['event_prior'], peak_threshold_norm=condition['tau_norm'])
            model = build_model(ModelConfig(**values)).cuda().train()
            assert count_model_parameters(model) == data['parameters']
            batch = Batch.from_data_list([Data(x=torch.randn(6, data['in_channels']),
                x_hist=torch.randn(6, args.history_hours // 6 + 1, data['in_channels']),
                edge_index=torch.tensor([[0, 1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 0]])) for _ in range(3)]).cuda()
            stats = {key: torch.tensor(data[source], device='cuda')
                     for key, source in (('y_mean', 'target_mean'), ('y_std', 'target_std'))}
            target_norm = model.head.threshold + torch.tensor([[-1.], [0.], [1.]], device='cuda')
            target_phys = target_norm * stats['y_std'] + stats['y_mean']
            criterion = ForecastLoss(LossConfig(**{f.name: getattr(args, f.name) for f in fields(LossConfig)}),
                stats, fitted['tail_threshold'], fitted['wmse_threshold'], event_prior=fitted['event_prior'],
                event_threshold=fitted['tau_phys']).cuda()
            with torch.autocast('cuda', dtype=torch.bfloat16):
                output = model(batch, torch.zeros(6, device='cuda'))
                loss = criterion(output, output.prediction * stats['y_std'] + stats['y_mean'], target_phys)
            assert torch.isfinite(loss)
            assert all(torch.isfinite(value).all() for value in output if value is not None)
            assert (output.body <= output.threshold).all()
            torch.testing.assert_close(output.prediction, output.body + output.gate_probability * output.excess,
                                       rtol=0, atol=0)
            loss.backward()
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
            checks.append(dict(station=station, condition=condition['condition'], body_cap=args.dual_body_cap,
                               finite_predictions_loss_and_gradients=True))
    torch.cuda.synchronize()
    print('PASS: all 24 configurations on GPU with BF16/TF32; finite forward/loss/backward; no optimizer steps.')
    return dict(device=torch.cuda.get_device_name(0), amp_dtype='bf16', tf32=True, optimizer_steps=0, checks=checks)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-only', action='store_true', help='Run all checks without writing evidence.')
    parser.add_argument('--gpu-smoke', action='store_true', help='Also require GPU BF16 forward/backward on synthetic inputs.')
    args = parser.parse_args()
    torch.set_num_threads(1)
    before = tree_snapshot(RESULTS)
    frozen = check_sources()
    generated = expected_files()
    assert len(list((HERE / 'configs').glob('*.sh'))) == 24
    for name, content in generated.items():
        assert (HERE / name).read_text() == content, name
    rows = list(manifest_rows())
    assert Counter(row['station'] for row in rows) == {station: 6 for station in STATIONS}
    resolved, transcripts, groups = [], {}, {}
    for station in STATIONS:
        reference = source_values(reference_path(station))
        old = vars(dry_run(reference_path(station))[1])
        group = []
        for row in (r for r in rows if r['station'] == station):
            path = REPO / row['config_path']
            run(['bash', '-n', str(path)])
            values = source_values(path)
            assert set(values) == set(reference) | {'DUAL_BODY_CAP'}
            assert {k: v for k, v in values.items() if k not in IDENTITY | FACTORS} == {
                k: v for k, v in reference.items() if k not in IDENTITY | FACTORS}
            output, current, command = dry_run(path)
            assert {key for key in old if old[key] != vars(current)[key]} <= ARG_DIFFERENCES
            assert (current.exceedance_percentile, current.excess_loss_weight, current.dual_body_cap) == (
                row['train_percentile'], row['excess_loss_weight'], row['dual_body_cap'])
            assert (current.head_type, current.excess_formulation, current.exceedance_head_experiment,
                    current.exceedance_gate_pooling) == ('dual', 'direct', None, 'mean')
            assert (current.loss_mode, current.body_loss_weight, current.gate_loss_weight) == ('mse', 1., .5)
            assert (current.excess_amp_loss_weight, current.shape_loss_weight, current.peak_loss_weight) == (0., 0., 0.)
            assert values['TAIL_LAMBDA_LIST'] == values['SLOPE_LAMBDA_LIST'] == '0'
            assert current.tail_frac == .05 and current.checkpoint_selection == 'overall'
            assert current.dual_ablation == 'none' and current.dual_loss == 1 and current.save_aux_checkpoints == 0
            assert (current.lr, current.epochs, current.batch_size, current.grad_accum_steps) == (.005, 300, 256, 4)
            assert (current.seed, current.deterministic, current.x_aug, current.x_clip, current.max_grad_norm) == (42, 0, 0, 0., 0.)
            assert current.amp and current.amp_dtype == 'bf16' and current.tf32
            assert values['ALL_RESULTS_ROOT'] == str(RESULTS)
            assert Path(current.output_dir) == Path(row['output_dir'].replace('<TIMESTAMP>', STAMP))
            transcripts[row['run_name']] = output
            resolved.append(dict(**row, resolved_args=vars(current), resolved_command=command))
            group.append(current)
        groups[station] = group
    matched = [{k: v for k, v in vars(a).items() if k not in ARG_DIFFERENCES | {'station'}}
               for group in groups.values() for a in group]
    assert all(value == matched[0] for value in matched)
    for path in HERE.glob('run_*.sh'):
        run(['bash', '-n', str(path)])
    stations = {station: data_check(station, groups[station]) for station in STATIONS}
    assert len({v['parameters']['total'] for v in stations.values()}) == 1
    gpu = gpu_check(groups, stations) if args.gpu_smoke else None
    assert check_sources() == frozen
    assert tree_snapshot(RESULTS) == before
    report = dict(status='PASS', validated_utc=datetime.now(timezone.utc).isoformat(), base_sha=BASE_SHA,
        branch=run(['git', 'branch', '--show-current']).strip(), worktree=str(REPO),
        configs=24, dry_runs_passed=24, matched_protocol=True, real_training_jobs=0, queue_submissions=0,
        primary_main_clean=True, results_unchanged=True, evaluation='Unchanged TEST top 5% by true window maximum',
        gpu_available=torch.cuda.is_available(), gpu_execution_verified=gpu is not None, gpu_smoke=gpu,
        initialization_note='All random weights and RNG consumption match; Q90 changes only the fitted threshold '
                            'buffer and the empirical-prior gate bias under the existing initialization rule.',
        metric_note='TRAIN event metrics remain additional diagnostics; the ten primary metrics use All/TEST top5.',
        inactive_loss_note='Under mse, train.sh omits tail/slope flags; inactive parser defaults .1/.01 are not used.',
        environment=dict(python=sys.version, executable=sys.executable, torch=torch.__version__, numpy=np.__version__),
        frozen_source_sha256=frozen, stations=stations, records=resolved)
    if not args.check_only:
        evidence = HERE / 'validation'
        evidence.mkdir(exist_ok=True)
        for name, transcript in transcripts.items():
            (evidence / f'{name}.dry_run.txt').write_text('\n'.join(line.rstrip() for line in transcript.splitlines()) + '\n')
        (HERE / 'validation_report.json').write_text(json.dumps(report, indent=2, default=str) + '\n')
        (HERE / 'dry_run_commands.txt').write_text('\n\n'.join(r['resolved_command'] for r in resolved) + '\n')
    print('PASS: 24 dry runs; real splits, TRAIN thresholds, initialization, parameter counts and TEST top5 checked.')
    print('No study training launched. Primary main and external results unchanged.')
    if gpu is None:
        print('GPU execution not checked; use --gpu-smoke with GPU access.')


if __name__ == '__main__':
    main()
