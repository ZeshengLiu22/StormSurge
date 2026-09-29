from pathlib import Path
from collections import defaultdict
from datetime import datetime, timezone
import argparse, json, sys
import numpy as np
import torch
sys.path.insert(0, '/home/exouser/StormSurge')
from emulator.data.graph_store import ForcingGraphStore
from emulator.data.targets import target_timestamps_from_graph, validate_target_timestamps, threshold_population
from emulator.training.episode_peaks import build_episode_peak_targets

torch.set_num_threads(1)
repo = Path('/home/exouser/StormSurge')
parser = argparse.ArgumentParser(description='Validate Boston data and the requested chronological split.')
parser.add_argument('group', choices=['past_only', 'future_year'])
parser.add_argument('--reuse-validation', type=Path)
args = parser.parse_args()
data_folder = 'Grid4_New_PastOnly' if args.group == 'past_only' else 'Grid4_New'
ratios = (.6, .2) if args.group == 'past_only' else (.4545454545, .0909090909)
root = repo / 'Data' / data_folder
output = repo / 'configs/CMIP6_Boston_QuickCheck_0929' / args.group / 'dataset_audit.json'
prior = json.loads(args.reuse_validation.read_text()) if args.reuse_validation else None
datasets = sorted(p for p in root.glob('CMIP6_*') if p.is_dir())
expected_years = sorted({'_'.join(p.name.split('_')[:2]) for d in datasets for p in (d/'dicts').glob('*_Boston_*_dict.npz')})
if args.group == 'past_only':
    expected_years = sorted('_'.join(p.name.split('_')[:2]) for p in (repo/'Data/Grid4_New/NCEP/graphs').glob('*_Boston_*graphs.pt'))
report = dict(audited_at_utc=datetime.now(timezone.utc).isoformat(), data_root=str(root.resolve()),
              scope='All file inventories; full Boston tensor/target validation against source dictionaries.',
              expected_year_groups=expected_years, group=args.group, train_ratio=ratios[0], val_ratio=ratios[1], datasets={})
for dataset in datasets:
    result = dict(included=False, reasons=[], files=[])
    report['datasets'][dataset.name] = result
    if dataset.name == 'CMIP6_Cane5':
        result['reasons'].append('Excluded by user request; no configs or runs.')
        output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
        continue
    print('Checking', args.group, dataset.name, flush=True)
    try:
        graphs = sorted((dataset/'graphs').glob('*_graphs.pt'))
        dictionaries = sorted((dataset/'dicts').glob('*_dict.npz'))
        if args.group == 'past_only':
            unused = [p for p in dictionaries if '_'.join(p.name.split('_')[:2]) not in expected_years]
            assert all(int(p.name[:4]) >= 2070 for p in unused), 'Unexpected nonhistorical dictionary inventory'
            result['unused_future_dictionaries'] = len(unused)
            dictionaries = [p for p in dictionaries if '_'.join(p.name.split('_')[:2]) in expected_years]
        assert graphs and dictionaries, 'Missing graph or dictionary inventory'
        assert {p.name.removesuffix('_graphs.pt') for p in graphs} == {p.name.removesuffix('_dict.npz') for p in dictionaries}, 'Graph/dictionary filenames do not match'
        assert all(p.stat().st_size > 0 for p in graphs + dictionaries), 'Empty graph or dictionary file'
        boston = [p for p in graphs if p.name.split('_')[2] == 'Boston']
        years = ['_'.join(p.name.split('_')[:2]) for p in boston]
        assert years == expected_years, 'Missing or duplicate Boston year groups'
        result.update(all_graph_files=len(graphs), boston_graph_files=len(boston))
        if args.group == 'past_only':
            ncep_years = sorted('_'.join(p.name.split('_')[:2]) for p in (repo/'Data/Grid4_New/NCEP/graphs').glob('*_Boston_*graphs.pt'))
            assert years == ncep_years, 'Past-only years differ from NCEP'
        reuse_files = {}
        if prior is not None:
            assert args.group == 'future_year' and prior['datasets'][dataset.name]['included']
            reuse_files = {f['name']: f for f in prior['datasets'][dataset.name]['files']}
            assert sorted(reuse_files) == sorted(p.name for p in boston)
        result['tensor_validation'] = 'Reused full same-session tensor audit after checking all file sizes and mtimes' if reuse_files else 'Full tensor validation in this audit'
        labels, timestamps = [], []
        store = ForcingGraphStore.__new__(ForcingGraphStore)
        store.year_to_indices = defaultdict(list)
        signature = None
        for file_index, path in enumerate(boston):
            before = path.stat()
            if reuse_files:
                old = reuse_files[path.name]
                assert (before.st_size, before.st_mtime_ns) == (old['bytes'], old['mtime_ns']), f'Previously validated file changed: {path}'
            data = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
            assert len(data) > 0, f'Empty graph list: {path.name}'
            source = dataset/'dicts'/path.name.replace('_graphs.pt', '_dict.npz')
            with np.load(source, allow_pickle=False) as reference:
                nc, tide = reference['nc'], reference['nc_tide']
                assert len(nc) == len(tide) == len(data)*6, f'Graph/target population mismatch: {path.name}'
                source_target = (nc-tide).reshape(len(data), 6)
            year = '_'.join(path.name.split('_')[:2])
            first, last = None, None
            for index, graph in enumerate(data):
                x, hist, y, edges = graph.x, graph.x_hist, graph.y, graph.edge_index
                current = (tuple(x.shape), tuple(hist.shape), tuple(y.shape), tuple(edges.shape))
                signature = signature or current
                assert current == signature, f'Inconsistent tensor dimensions: {path.name}:{index}'
                assert x.ndim == 2 and x.shape[1] == 5 and hist.ndim == 3 and hist.shape[0] >= 5 and tuple(hist.shape[1:]) == tuple(x.shape), f'Invalid 24h history: {path.name}:{index}'
                assert y.numel() == 6 and edges.ndim == 2 and edges.shape[0] == 2 and edges.numel(), f'Invalid targets/edges: {path.name}:{index}'
                assert edges.dtype == torch.int64 and int(edges.min()) >= 0 and int(edges.max()) < len(x), f'Invalid graph edges: {path.name}:{index}'
                if not reuse_files:
                    assert all(np.isfinite(t.numpy()).all() for t in (x, hist, y)), f'Nonfinite tensors: {path.name}:{index}'
                row = y.numpy().reshape(-1).astype(np.float64)
                np.testing.assert_allclose(row, source_target[index], rtol=1e-5, atol=1e-6, err_msg=f'{path.name}:{index} source targets')
                times = target_timestamps_from_graph(graph, f'{path.name}:{index}')
                first = int(times[0]) if first is None else first
                last = int(times[-1])
                store.year_to_indices[year].append(len(labels))
                labels.append(row)
                timestamps.append(times)
            after = path.stat()
            assert (before.st_size,before.st_mtime_ns) == (after.st_size,after.st_mtime_ns), f'File changed during validation: {path.name}'
            result['files'].append(dict(name=path.name, bytes=before.st_size, mtime_ns=before.st_mtime_ns, graphs=len(data), first_target_utc=str(np.datetime64(first,'s')), last_target_utc=str(np.datetime64(last,'s'))))
            del data, graph, x, hist, y, edges
            if (file_index+1) % 12 == 0:
                print(f'  {file_index+1}/{len(boston)} Boston archives passed', flush=True)
        truth = np.stack(labels)
        times = validate_target_timestamps(np.stack(timestamps))
        splits = store.split(train_ratio=ratios[0], val_ratio=ratios[1], shuffle_years=False, seed=42, future_only=False, future_year_threshold=2030)
        assert all(splits.values()), 'Empty TRAIN/VAL/TEST split'
        tau = float(np.quantile(truth[splits['train']].ravel(), .95, method='linear'))
        split_report = {}
        for part, indices in splits.items():
            selected = set(indices)
            pop = threshold_population(truth[indices], times[indices], tau)
            assert pop['extreme_hour_count'] > 0, f'Empty {part} exceedance population'
            split_report[part] = dict(year_groups=[year for year, ix in store.year_to_indices.items() if selected.intersection(ix)], windows=len(indices), **pop)
        expected_counts = (22,7,7) if args.group == 'past_only' else (30,6,30)
        assert tuple(len(split_report[p]['year_groups']) for p in ('train','val','test')) == expected_counts
        if args.group == 'future_year':
            assert all(int(y.split('_')[0]) < 2070 for p in ('train','val') for y in split_report[p]['year_groups'])
            assert split_report['test']['year_groups'] == [f'{y}_{y+1}' for y in range(2070,2100)]
        peaks = build_episode_peak_targets(truth[splits['train']], times[splits['train']], tau)
        assert peaks.metadata['canonical_peak_target_count'] > 0
        result.update(included=True, tensor_shapes=signature, total_graphs=len(truth), tau_physical=tau, split=split_report, episode_peak_metadata=peaks.metadata)
        print(f'PASS {dataset.name}: {len(boston)} files, {len(truth)} graphs, train/val/test years='+ '/'.join(str(len(split_report[p]['year_groups'])) for p in ('train','val','test')), flush=True)
    except Exception as exc:
        result['reasons'].append(f'{type(exc).__name__}: {exc}')
        print('EXCLUDE',dataset.name,result['reasons'],flush=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
print('Included:', ', '.join(name for name,r in report['datasets'].items() if r['included']), flush=True)
print('Excluded:', ', '.join(name for name,r in report['datasets'].items() if not r['included']) or 'none', flush=True)
