#!/usr/bin/env python3
"""Measure 44 target types and calibrate 264 timing estimates with full pairs.

All inference artifacts stay in RESULTS/audit, outside the production matrices.
Year-loop timing includes loading batches, CUDA inference, CPU copies and metrics.
"""
import argparse
import contextlib
import csv
import gc
import io
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time

from generate_configs import DESTINATION, REPO, RESULTS, STATIONS, TARGETS, csv_text

sys.path.insert(0, str(REPO))


def measure(row, years, directory, batch_size):
    import torch
    import infer

    directory.mkdir(parents=True, exist_ok=True)
    timings = {'checkpoint_load_s': 0.0, 'graph_deserialize_s': 0.0}
    original_load = torch.load
    original_init = infer.ForcingGraphStore.__init__

    def timed_load(path, *args, **kwargs):
        start = time.perf_counter()
        value = original_load(path, *args, **kwargs)
        key = 'graph_deserialize_s' if str(path).endswith('graphs.pt') else 'checkpoint_load_s'
        timings[key] += time.perf_counter() - start
        return value

    def timed_init(store, *args, **kwargs):
        start = time.perf_counter()
        original_init(store, *args, **kwargs)
        timings['data_load_s'] = time.perf_counter() - start
        timings['loaded_samples'] = len(store.graphs)
        timings['nodes'] = int(store.graphs[0].x.shape[0])

    argv = ['--ckpt', row['checkpoint'], '--root_dir', row['source_root'],
            '--test_root_dir', row['target_root'], '--station', row['station'],
            '--source_name', row['source'], '--target_name', row['target'],
            '--model', 'perceiver3', '--encoder_type', 'GraphSAGE',
            '--temporal_block', 'Transformer', '--head_type', 'single', '--history_hours', '24',
            '--model_label', 'PACT_TailPeak_overall', '--station_json_dir', str(REPO / 'station_json'),
            '--years', ','.join(years), '--strict_years', '--device', 'cuda',
            '--batch_size', str(batch_size), '--torch_threads', '1', '--num_workers', '0',
            '--prefetch_factor', '0', '--mp_context', 'fork', '--amp', '--amp_dtype', 'fp16',
            '--gpu_sync_timing', '--save_npz', '--out_dir', str(directory / 'outputs')]
    torch.cuda.reset_peak_memory_stats()
    torch.load = timed_load
    infer.ForcingGraphStore.__init__ = timed_init
    start = time.perf_counter()
    try:
        with (directory / 'infer.log').open('w') as log, contextlib.redirect_stdout(log):
            infer.main(argv)
    finally:
        torch.load = original_load
        infer.ForcingGraphStore.__init__ = original_init
    timings['main_s'] = time.perf_counter() - start
    data = json.loads((directory / 'outputs/metrics.json').read_text())
    timings['reported_wall_s'] = data['wall_seconds']
    timings['year_loop_s'] = data['runtime_seconds']
    timings['other_main_s'] = timings['main_s'] - timings['data_load_s'] - timings['year_loop_s']
    timings['peak_gpu_mib'] = torch.cuda.max_memory_allocated() / 2**20
    files = [p for p in Path(row['target_root']).glob(f"*_{row['station']}_*graphs.pt")
             if '_'.join(p.name.split('_')[:2]) in years]
    result = dict(station=row['station'], group=row['group'], source=row['source'], target=row['target'],
                  years=','.join(years), year_count=len(years), samples=data['samples'],
                  input_gib=sum(p.stat().st_size for p in files) / 2**30,
                  AllRMSE=data['metrics']['all_rmse'], **timings)
    (directory / 'profile.json').write_text(json.dumps(result, indent=2) + '\n')
    (directory / 'invocation.json').write_text(json.dumps(argv, indent=2) + '\n')
    gc.collect()
    torch.cuda.empty_cache()
    return result


def base_estimate(profile, row):
    years = row['years'].split(',')
    files = [p for p in Path(row['target_root']).glob(f"*_{row['station']}_*graphs.pt")
             if '_'.join(p.name.split('_')[:2]) in years]
    gib = sum(p.stat().st_size for p in files) / 2**30
    ratio = gib / profile['input_gib']
    # Target graph storage is fixed-width within each dataset; bytes proxy sample counts.
    return profile['data_load_s'] * ratio + profile['year_loop_s'] * ratio + profile['other_main_s'] * len(years)


def summarize_estimates(rows, profiles, calibrations, output, info):
    by_target = {(r['station'], r['group'], r['target']): r for r in profiles}
    controls = {(c['group'], c['target']): c for c in calibrations}
    startup = statistics.median(c['process_overhead_s'] for c in calibrations)
    loop_factors = {}
    for c in calibrations:
        profile = by_target[c['station'], c['group'], c['target']]
        loop_factors[c['group'], c['target']] = c['year_loop_s'] / (profile['year_loop_s'] * c['input_gib'] / profile['input_gib'])
    group_loop_factors = {g: statistics.median(v for (group, target), v in loop_factors.items() if group == g) for g in TARGETS}
    other_per_year = {g: statistics.median(c['other_main_s'] / c['year_count'] for c in calibrations if c['group'] == g) for g in TARGETS}
    small = controls['future_year', 'CNRM']
    large = controls['future_year', 'EC_EARTH']
    io_slope = (large['data_load_s'] - small['data_load_s']) / (large['input_gib'] - small['input_gib'])
    past = controls['past_only', 'AWI']
    estimates = []
    for row in rows:
        group, target = row['group'], row['target']
        profile = by_target[row['station'], group, target]
        years = row['years'].split(',')
        files = [p for p in Path(row['target_root']).glob(f"*_{row['station']}_*graphs.pt") if '_'.join(p.name.split('_')[:2]) in years]
        gib = sum(p.stat().st_size for p in files) / 2**30
        ratio = gib / profile['input_gib']
        control = controls.get((group, target))
        if control:
            data_load_s = control['data_load_s'] * gib / control['input_gib']
        elif group == 'past_only':
            data_load_s = past['data_load_s'] * gib / past['input_gib']
        else:
            data_load_s = max(0., small['data_load_s'] + io_slope * (gib - small['input_gib']))
        loop_s = profile['year_loop_s'] * ratio * loop_factors.get((group, target), group_loop_factors[group])
        other_s = other_per_year[group] * len(years)
        main_s = data_load_s + loop_s + other_s
        pair_s = main_s + startup
        warm_pair_s = profile['data_load_s'] * ratio + loop_s + other_s + startup
        matched = next((c for c in calibrations if all(row[k] == c[k] for k in ('station', 'group', 'source', 'target'))), None)
        estimates.append(dict(station=row['station'], group=group, source=row['source'], target=target,
            years=len(years), input_gib=gib, estimated_data_load_s=data_load_s,
            estimated_year_loop_s=loop_s, estimated_main_s=main_s, estimated_process_s=pair_s,
            measured_process_s=matched['process_s'] if matched else '',
            warm_input_estimate_s=min(warm_pair_s, pair_s), planning_high_s=1.5 * pair_s,
            method='target-specific full-year I/O calibration plus one-year evaluation profiles'))
    (output / 'estimated_pair_times.csv').write_text(csv_text(estimates))
    groups = []
    for station in STATIONS:
        for group in TARGETS:
            for source in ('NCEP', 'AWI', 'CNRM', 'EC_EARTH', 'MPI', 'MRI'):
                selected = [r for r in estimates if (r['station'], r['group'], r['source']) == (station, group, source)]
                groups.append(dict(station=station, group=group, source=source, pairs=len(selected),
                    estimated_minutes=sum(r['estimated_process_s'] for r in selected) / 60,
                    warm_input_estimate_minutes=sum(r['warm_input_estimate_s'] for r in selected) / 60,
                    planning_high_minutes=sum(r['planning_high_s'] for r in selected) / 60))
    (output / 'estimated_source_group_times.csv').write_text(csv_text(groups))
    summary = dict(**info, startup_overhead_s=startup,
        io_model='Past load scales by bytes from the full AWI calibration. Future load interpolates CNRM and EC_EARTH full-pair observations; calibrated target anchors are kept exact.',
        source_model='All six source roles use the same architecture; estimates share target costs. Only three complete pairs were actually measured.',
        measured_one_year_targets=len(profiles), measured_full_pairs=len(calibrations),
        estimated_pairs=len(estimates), total_estimated_minutes=sum(r['estimated_process_s'] for r in estimates) / 60,
        total_warm_input_estimate_minutes=sum(r['warm_input_estimate_s'] for r in estimates) / 60,
        total_planning_high_minutes=sum(r['planning_high_s'] for r in estimates) / 60,
        range_note='Warm-input scenario and planning upper allowance, not confidence bounds. GLADE contention/caching and source-dependent metric work can change runtime.',
        stations={s: sum(r['estimated_process_s'] for r in estimates if r['station'] == s) / 60 for s in STATIONS})
    (output / 'timing_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--case-index', type=int)
    parser.add_argument('--summarize-only', action='store_true', help='Rebuild estimates from existing timing profiles without GPU inference')
    args = parser.parse_args()
    output = args.output_dir or RESULTS / 'audit' / time.strftime('runtime_benchmark_%Y%m%d_%H%M%S')
    if not output.resolve().is_relative_to((RESULTS / 'audit').resolve()):
        parser.error('Benchmark output must stay under RESULTS/audit')
    output.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader((DESTINATION / 'manifest.csv').open()))
    if args.summarize_only:
        profiles = [json.loads(p.read_text()) for p in sorted((output / 'one_year').glob('*/profile.json'))]
        calibrations = []
        for raw in csv.DictReader((output / 'full_pair_calibrations.csv').open()):
            converted = {}
            for k, v in raw.items():
                try:
                    converted[k] = float(v)
                except ValueError:
                    converted[k] = v
            calibrations.append(converted)
        info = json.loads((output / 'method.json').read_text())
        summarize_estimates(rows, profiles, calibrations, output, info)
        return
    if args.case_index is not None:
        row = rows[args.case_index]
        result = measure(row, row['years'].split(','), output, args.batch_size)
        print(json.dumps(result), flush=True)
        return

    import torch
    import torch_geometric
    if not torch.cuda.is_available():
        raise SystemExit('Run inside a Casper V100 GPU allocation.')
    info = dict(python=sys.executable, torch=torch.__version__, pyg=torch_geometric.__version__,
                gpu=torch.cuda.get_device_name(0), batch_size=args.batch_size, amp_dtype='fp16', tf32=False,
                note='Warm-process one-year target profiles; fresh-process full-pair calibration. Estimates are not 264 measured production runtimes.')
    (output / 'method.json').write_text(json.dumps(info, indent=2) + '\n')
    print('OUTPUT', output, flush=True)
    profiles = []
    for station in STATIONS:
        for group, targets in TARGETS.items():
            for target in targets:
                row = next(r for r in rows if (r['station'], r['group'], r['source'], r['target']) == (station, group, 'NCEP', target))
                year = row['years'].split(',')[0]
                directory = output / 'one_year' / f'{station}_{group}_NCEP_to_{target}'
                profile = measure(row, [year], directory, args.batch_size)
                profiles.append(profile)
                (output / 'one_year_profiles.csv').write_text(csv_text(profiles))
                print(f"[{len(profiles)}/44] {station}/{group}/{target}: load={profile['data_load_s']:.3f}s loop={profile['year_loop_s']:.3f}s main={profile['main_s']:.3f}s GPU={profile['peak_gpu_mib']:.0f}MiB", flush=True)
    by_target = {(r['station'], r['group'], r['target']): r for r in profiles}
    calibration_specs = [('Boston', 'past_only', 'NCEP', 'AWI'),
                         ('Lewes', 'future_year', 'NCEP', 'EC_EARTH'),
                         ('Boston', 'future_year', 'MRI', 'CNRM')]
    calibrations = []
    for spec in calibration_specs:
        index, row = next((i, r) for i, r in enumerate(rows) if tuple(r[k] for k in ('station', 'group', 'source', 'target')) == spec)
        directory = output / 'full_pairs' / '_'.join(spec)
        start = time.perf_counter()
        completed = subprocess.run([sys.executable, str(Path(__file__).resolve()), '--case-index', str(index),
                                    '--batch-size', str(args.batch_size), '--output-dir', str(directory)],
                                   text=True, capture_output=True, check=True)
        process_s = time.perf_counter() - start
        calibration = json.loads(completed.stdout.strip().splitlines()[-1])
        calibration['process_s'] = process_s
        calibration['process_overhead_s'] = process_s - calibration['main_s']
        baseline = base_estimate(by_target[spec[0], spec[1], spec[3]], row)
        calibration['uncalibrated_main_estimate_s'] = baseline
        calibration['calibration_factor'] = calibration['main_s'] / baseline
        calibrations.append(calibration)
        (output / 'full_pair_calibrations.csv').write_text(csv_text(calibrations))
        print(f"[FULL] {'/'.join(spec)}: {calibration['year_count']} years load={calibration['data_load_s']:.2f}s loop={calibration['year_loop_s']:.2f}s main={calibration['main_s']:.2f}s process={process_s:.2f}s", flush=True)
    summarize_estimates(rows, profiles, calibrations, output, info)


if __name__ == '__main__':
    main()
