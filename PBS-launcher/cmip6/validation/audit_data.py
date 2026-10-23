#!/usr/bin/env python3
"""Read-only, CPU-only CMIP6 graph audit; holds one source file at a time."""
import argparse
import gc
import json
import os
from pathlib import Path
import re
import sys
import time

REPO = Path('/glade/u/home/zel/scratch/StormSurge')
DATA = Path('/glade/u/home/zel/scratch/PACT-Data-Final')
GCMS = ['CMIP6_' + s for s in ['AWI', 'BCC', 'Cane5', 'CNRM', 'EC_EARTH', 'IPSL', 'MIROC', 'MPI', 'MRI']]
STATIONS = ['Battery', 'Boston', 'CBBT', 'Lewes']
NAME = re.compile(r'^(\d{4})_(\d{4})_(Battery|Boston|CBBT|Lewes)_peryear_hist(\d+)_graphs\.pt$')


def discover():
    records = []
    for gcm in GCMS:
        roots = {}
        files = {}
        for kind, name in [('past', gcm + '_PastOnly'), ('full', gcm)]:
            candidates = [p for p in [DATA/name, DATA/name/'graphs'] if list(p.glob('*graphs.pt'))]
            if len(candidates) != 1:
                raise ValueError(f'{name}: expected exactly one actual graph root, found {candidates}')
            roots[kind] = candidates[0]
            files[kind] = {s: {} for s in STATIONS}
            for p in sorted(candidates[0].glob('*graphs.pt')):
                m = NAME.fullmatch(p.name)
                if not m or int(m[2]) != int(m[1]) + 1:
                    raise ValueError(f'Malformed graph filename: {p}')
                tag = m[1] + '_' + m[2]
                if tag in files[kind][m[3]]:
                    raise ValueError(f'Duplicate station/year: {p}')
                if p.stat().st_size == 0:
                    raise ValueError(f'Empty graph file: {p}')
                files[kind][m[3]][tag] = str(p)
        for station in STATIONS:
            past = files['past'][station]
            full = files['full'][station]
            if not past or not full:
                raise ValueError(f'Missing requested station: {gcm}/{station}')
            hist = sorted(past)
            if any(int(t.split('_')[1]) >= 2030 for t in hist):
                raise ValueError(f'Future group in PastOnly: {gcm}/{station}')
            historical_full = [t for t in sorted(full) if int(t.split('_')[1]) < 2030]
            if hist != historical_full:
                raise ValueError(f'PastOnly and full historical year tags differ: {gcm}/{station}')
            future = [t for t in sorted(full) if int(t.split('_')[0]) > 2030]
            if not future or len(hist) + len(future) != len(full):
                raise ValueError(f'Missing/ambiguous future years: {gcm}/{station}')
            for label, tags in [('historical', hist), ('future', future)]:
                starts = [int(t[:4]) for t in tags]
                if starts != list(range(starts[0], starts[-1] + 1)):
                    raise ValueError(f'Missing {label} year groups: {gcm}/{station}')
            records.append(dict(gcm=gcm, station=station, past_root=str(roots['past']),
                                full_root=str(roots['full']), historical_years=hist,
                                future_years=future, historical_files=past,
                                future_files={t: full[t] for t in future},
                                full_file_count=len(full)))
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--discover-only', action='store_true')
    parser.add_argument('--future-detail', choices=['all', 'boundary'], default='all')
    parser.add_argument('--gcms', nargs='+', choices=GCMS)
    parser.add_argument('--summary-name', default='dataset_audit.json')
    parser.add_argument('--numeric-scope', choices=['all', 'boundary'], default='boundary',
                        help='All graphs get shape/time/metadata checks; numerical payload checks may sample first/last per file.')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    records = discover()
    inventory_path = args.out/'dataset_inventory.json'
    if not inventory_path.exists():
        inventory_path.write_text(json.dumps(records, indent=2) + '\n')
    else:
        assert json.loads(inventory_path.read_text()) == records
    if args.discover_only:
        print(f'Found all {len(records)} GCM/station pairs; {sum(len(r["historical_years"]) for r in records)} historical files.', flush=True)
        return
    if args.gcms:
        records = [r for r in records if r['gcm'] in args.gcms]
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(REPO))
    import numpy as np
    import torch
    import torch_geometric
    import torch.utils.serialization.config as serialization_config
    from emulator.data.graph_store import ForcingGraphStore, transform_pressure_features
    from emulator.data.targets import target_timestamps_from_graph, validate_target_timestamps
    torch.set_num_threads(1)
    # Avoid thousands of remote ZIP-header seeks for each memory-mapped file.
    # This changes only audit I/O, not graph values or training behavior.
    serialization_config.load.calculate_storage_offsets = True
    details = args.out/'data_details'
    details.mkdir(exist_ok=True)
    all_reports = []
    began = time.monotonic()

    def inspect_file(path, year, future):
        graphs = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
        assert isinstance(graphs, list) and graphs, 'Expected nonempty graph list'
        signatures = set()
        timestamps = []
        target_min, target_max = float('inf'), float('-inf')
        numeric_count = 0
        for i, g in enumerate(graphs):
            context = f'{path} graph {i}'
            x, history, y = g.x, g.x_hist, g.y
            assert x.ndim == 2 and x.shape[1] == 5 and x.is_floating_point(), context + ': x'
            assert history.ndim == 3 and history.shape[1:] == x.shape, context + ': history shape'
            assert history.shape[0] >= 7 and history.is_floating_point(), context + ': 36H unavailable'
            assert int(g.max_history_hours) >= 36, context + ': max_history_hours'
            assert history.shape[0] == int(g.max_history_hours) // 6 + 1, context + ': history metadata'
            h, w = int(g.grid_H), int(g.grid_W)
            assert h > 0 and w > 0 and h*w == x.shape[0], context + ': CNN grid'
            assert tuple(y.shape) == (6,) and y.is_floating_point(), context + ': six hourly targets'
            assert tuple(g.nc.shape) == (6,) and tuple(g.nc_tide.shape) == (6,), context + ': target metadata'
            means = g.p_mean_hist
            assert means.ndim == 1 and len(means) == history.shape[0], context + ': pressure history'
            assert means.is_floating_point() and g.p_mean_curr.numel() == 1, context + ': pressure metadata'
            if args.numeric_scope == 'all' or i in (0, len(graphs)-1):
                assert torch.isfinite(y).all(), context + ': finite target'
                assert torch.isfinite(g.nc).all() and torch.isfinite(g.nc_tide).all(), context + ': nc/tide'
                assert torch.allclose(y, g.nc-g.nc_tide, atol=2e-6, rtol=2e-5), context + ': y != nc - nc_tide'
                assert torch.isfinite(means).all() and torch.isfinite(g.p_mean_curr).all(), context + ': pressure finite'
                assert torch.isclose(g.p_mean_curr.reshape(()), means[-1], rtol=1e-5, atol=1e-5), context + ': current pressure'
                target_min = min(target_min, float(y.min()))
                target_max = max(target_max, float(y.max()))
                numeric_count += 1
            ts = target_timestamps_from_graph(g, context)
            years = ts.astype('datetime64[s]').astype('datetime64[Y]').astype(int) + 1970
            assert set(years.tolist()).issubset({int(year[:4]), int(year[5:])}), context + ': timestamp outside year group'
            assert bool(np.all(years > 2030)) if future else bool(np.all(years < 2030)), context + ': historical/future contamination'
            assert int(g.time_index) >= 0 and int(g.hour_start) >= 0, context + ': timing metadata'
            timestamps.append(ts)
            signatures.add((h, w, x.shape[0], x.shape[1], history.shape[0], int(g.max_history_hours), y.numel()))
        # Execute the existing raw-pressure validator on first/last graphs per file.
        # All graph dimensions and timestamps are checked above. Numeric checking
        # covers all graphs with --numeric-scope all, otherwise file boundaries.
        for idx in sorted({0, len(graphs)-1}):
            raw = transform_pressure_features(graphs[idx], 'raw', graph_tag=f'{path} [{idx}]')
            assert torch.isfinite(raw.x_hist).all(), f'{path}: sampled raw forcing is nonfinite'
            assert torch.equal(graphs[idx].x, graphs[idx].x_hist[-1]), f'{path}: sampled current/history mismatch'
            del raw
        ts = validate_target_timestamps(np.stack(timestamps))
        assert np.all(np.diff(ts[:, 0]) > 0), f'{path}: windows not chronological'
        result = dict(path=str(path), bytes=Path(path).stat().st_size, mtime_ns=Path(path).stat().st_mtime_ns,
                      graph_count=len(graphs), signatures=[list(s) for s in sorted(signatures)],
                      first_target_utc=str(np.datetime64(int(ts.min()), 's')),
                      last_target_utc=str(np.datetime64(int(ts.max()), 's')),
                      target_min=target_min, target_max=target_max,
                      numeric_values_scope=args.numeric_scope, numeric_graph_count=numeric_count,
                      target_hour_count=int(ts.size), status='PASS')
        del graphs, g, x, history, y, means
        return result, ts

    for record in records:
        dest = details/f'{record["gcm"]}_{record["station"]}.json'
        if dest.exists():
            cached = json.loads(dest.read_text())
            cached_files = cached.get('historical_files', []) + cached.get('future_files', [])
            unchanged = all(Path(f['path']).is_file()
                            and Path(f['path']).stat().st_size == f['bytes']
                            and Path(f['path']).stat().st_mtime_ns == f['mtime_ns'] for f in cached_files)
            numeric_satisfied = args.numeric_scope != 'all' or cached.get('numeric_values_scope', 'all') == 'all'
            if (cached.get('status') == 'PASS' and cached.get('future_detail') == args.future_detail
                    and unchanged and numeric_satisfied):
                all_reports.append(cached)
                print(f'Cached {record["gcm"]}/{record["station"]}', flush=True)
                continue
        report = dict(gcm=record['gcm'], station=record['station'], status='PASS', errors=[],
                      historical_files=[], future_files=[], future_detail=args.future_detail,
                      numeric_values_scope=args.numeric_scope)
        store = ForcingGraphStore.__new__(ForcingGraphStore)
        store.year_to_indices = {year: [idx] for idx, year in enumerate(record['historical_years'])}
        split_by_seed = {}
        for seed in [42, 123, 2026]:
            split = store.split(train_ratio=.6, val_ratio=.2, shuffle_years=False, seed=seed,
                                future_only=False, future_year_threshold=2030)
            split_by_seed[str(seed)] = {part: [record['historical_years'][i] for i in ids] for part, ids in split.items()}
        assert split_by_seed['42'] == split_by_seed['123'] == split_by_seed['2026']
        report['split_by_seed'] = split_by_seed
        split = split_by_seed['42']
        assert [len(split[p]) for p in ['train', 'val', 'test']] == [22, 7, 7]
        assert max(split['train']) < min(split['val']) and max(split['val']) < min(split['test'])
        report['split_groups'] = split
        time_parts = {p: [] for p in split}
        future_times = []
        for kind in ['historical', 'future']:
            files = record[kind+'_files']
            tags = sorted(files)
            if kind == 'future' and args.future_detail == 'boundary':
                tags = [tags[0], tags[-1]]
            for year in tags:
                try:
                    info, ts = inspect_file(files[year], year, kind == 'future')
                    report[kind+'_files'].append(info)
                    if kind == 'historical':
                        part = next(p for p, groups in split.items() if year in groups)
                        time_parts[part].append(ts)
                    else:
                        future_times.append(ts)
                except Exception as exc:
                    report['errors'].append(f'{files[year]}: {type(exc).__name__}: {exc}')
                    report['status'] = 'FAIL'
                if len(report[kind+'_files']) % 12 == 0:
                    print(f'{record["gcm"]}/{record["station"]} {kind}: {len(report[kind+"_files"])}/{len(tags)} files validated', flush=True)
            gc.collect()
        try:
            times = {p: validate_target_timestamps(np.concatenate(blocks)) for p, blocks in time_parts.items()}
            validate_target_timestamps(np.concatenate(list(times.values())))
            validate_target_timestamps(np.concatenate(future_times))
            assert times['train'].max() < times['val'].min() and times['val'].max() < times['test'].min()
            assert max(t.max() for t in times.values()) < min(t.min() for t in future_times)
            report['split_time_ranges'] = {p: dict(first=str(np.datetime64(int(ts.min()), 's')),
                                                  last=str(np.datetime64(int(ts.max()), 's')),
                                                  graph_count=int(ts.shape[0]), target_hours=int(ts.size)) for p, ts in times.items()}
            signatures = {tuple(s) for kind in ['historical', 'future'] for f in report[kind+'_files'] for s in f['signatures']}
            assert len(signatures) == 1, f'Nonuniform historical/future model dimensions: {signatures}'
            report['dimensions'] = list(signatures.pop())
        except Exception as exc:
            report['errors'].append(f'Temporal/dimension validation: {type(exc).__name__}: {exc}')
            report['status'] = 'FAIL'
        temporary = dest.with_name(dest.name + f'.{os.getpid()}.tmp')
        temporary.write_text(json.dumps(report, indent=2) + '\n')
        temporary.replace(dest)
        all_reports.append(report)
        print(f'{report["status"]} {record["gcm"]}/{record["station"]}: historical={len(report["historical_files"])} future={len(report["future_files"])}; elapsed={time.monotonic()-began:.1f}s', flush=True)
        del time_parts, future_times
    summary = dict(status='PASS' if all(r['status'] == 'PASS' for r in all_reports) else 'FAIL',
                   runtime=dict(python=sys.executable, torch=torch.__version__, pyg=torch_geometric.__version__, cuda_visible_devices=os.environ['CUDA_VISIBLE_DEVICES']),
                   audit_scope='Every historical and future graph: dimensions, 36H support, CNN grid, pressure/target metadata and target timestamps. Numeric target/pressure checks and raw forcing finiteness cover first/last graphs per file (some completed pairs also have all target/pressure values checked). One memory-mapped file per worker; no full tensor checksum.',
                   historical_file_count=sum(len(r['historical_files']) for r in all_reports),
                   future_file_count=sum(len(r['future_files']) for r in all_reports),
                   graph_count=sum(f['graph_count'] for r in all_reports for kind in ['historical', 'future'] for f in r[kind+'_files']),
                   seconds=time.monotonic()-began, stations=all_reports)
    (args.out/args.summary_name).write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps({k: v for k, v in summary.items() if k != 'stations'}), flush=True)
    sys.exit(0 if summary['status'] == 'PASS' else 1)


if __name__ == '__main__':
    main()
