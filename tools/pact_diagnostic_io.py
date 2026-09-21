"""Read-only checkpoint replay shared by the isolated 0921 analysis tools."""
from pathlib import Path
import hashlib
import json
import re
import sys

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if not (REPO/'emulator/models/heads.py').exists():
    REPO = Path('/media/volume/PACT-Data/StormSurge')
RESULTS = Path('/media/share/PACT/Results/All_results_0921_true_peak_amp_factorial')
METHODS = ('S0', 'D0', 'D1', 'D2', 'D3')
STATIONS = ('CBBT', 'Lewes')


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for data in iter(lambda: handle.read(2**20), b''):
            digest.update(data)
    return digest.hexdigest()


def write_json(path, value):
    def encode(x):
        if isinstance(x, Path): return str(x)
        if isinstance(x, np.ndarray): return x.tolist()
        if isinstance(x, np.generic): return x.item()
        raise TypeError(type(x))
    Path(path).write_text(json.dumps(value, indent=2, default=encode, allow_nan=False) + '\n')


def one(directory, pattern):
    matches = list(Path(directory).glob(pattern))
    if len(matches) != 1:
        raise ValueError(f'Expected exactly one {pattern} in {directory}: {matches}')
    return matches[0]


def discover(root, *, require_all=True):
    runs, excluded = {}, []
    for path in sorted(Path(root).glob('0921*/config_*.json')):
        config = json.loads(path.read_text())
        match = re.search(r'0921_(CBBT|Lewes)_(S0|D[0-3])_', config.get('run_tag', ''))
        if not match: continue
        directory = path.parent
        if not (directory/'exit_status').exists() or (directory/'exit_status').read_text().strip() != '0':
            excluded.append(dict(path=str(directory), reason='No successful completion marker'))
            continue
        key = match.groups()
        if key in runs: raise ValueError(f'Ambiguous completed run for {key}')
        summary_path = one(directory, 'summary_*.json')
        summary = json.loads(summary_path.read_text())
        if config['station'] != key[0] or config['checkpoint_selection'] != 'overall':
            raise ValueError(f'Unexpected station/selection: {path}')
        epochs = [json.loads(s) for s in one(directory, 'metrics_*.jsonl').read_text().splitlines()]
        if [r['epoch'] for r in epochs] != list(range(1, config['epochs']+1)):
            raise ValueError(f'Incomplete epoch history: {directory}')
        if min(epochs, key=lambda r: r['val']['rmse_all'])['epoch'] != summary['best_epoch']:
            raise ValueError(f'Not the minimum VAL overall checkpoint: {directory}')
        runs[key] = dict(directory=directory, config_path=path, config=config,
                         checkpoint=one(directory, 'best_*.pth'), summary_path=summary_path,
                         summary=summary, prediction=one(directory, 'test_preds_*.npz'), epochs=epochs)
    if require_all and set(runs) != {(s, m) for s in STATIONS for m in METHODS}:
        raise ValueError(f'Missing completed runs: {set(runs)}')
    return runs, excluded


def align(arrays, tags, truth=None):
    actual = arrays['tags'].tolist()
    if len(set(actual)) != len(actual) or set(actual) != set(tags):
        raise ValueError('Missing, extra, or duplicate sample IDs; events must never be dropped')
    lookup = {t: i for i, t in enumerate(actual)}
    order = [lookup[t] for t in tags]
    result = {k: v[order] for k, v in arrays.items()}
    if truth is not None and not np.array_equal(result['y_true'], truth):
        raise ValueError('GT differs after exact ID alignment')
    if not all(np.isfinite(result[k]).all() for k in result if k != 'tags'):
        raise ValueError('Nonfinite replay values')
    return result


def source_hashes(repo):
    # These are the actual inference/target-definition dependencies. No trainer import.
    paths = [*Path(repo).glob('emulator/models/*.py'), *Path(repo).glob('emulator/data/*.py'),
             *Path(repo).glob('emulator/common/*.py'), Path(repo)/'emulator/training/engine.py',
             Path(repo)/'emulator/training/metrics.py']
    return {str(p.relative_to(repo)): sha(p) for p in sorted(paths)}


def replay_station(repo, station, runs, output, device='cuda'):
    import torch
    sys.path.insert(0, str(repo))
    from emulator.common.runtime import configure_runtime
    from emulator.data import ForcingGraphStore, ForcingGraphView, build_loader
    from emulator.models import ModelConfig, build_model
    from emulator.training.engine import run_epoch

    cache = output/'arrays'
    cache.mkdir(exist_ok=True, parents=True)
    hashes = source_hashes(repo)
    station_runs = {m: r for (s, m), r in runs.items() if s == station}
    ref = torch.load(station_runs['S0']['checkpoint'], map_location='cpu', weights_only=False)
    config = station_runs['S0']['config']
    configure_runtime(config['seed'], config['torch_threads'], bool(config['deterministic']), bool(config['tf32']))
    root = (Path(repo)/config['root_dir']).resolve()
    print(f'{station}: loading saved split targets and graph store', flush=True)
    store = ForcingGraphStore(root, station)
    splits = store.split(**ref['split_config'])
    for split, indices in splits.items():
        tags = np.asarray([store.graph_tags[i] for i in indices])
        assert tags.tolist() == ref['split_tags'][split]
        truth = torch.stack([store.graphs[i].y.reshape(-1).float() for i in indices]).numpy()
        np.savez_compressed(cache/f'{station}_{split}_targets.npz', tags=tags, y_true=truth,
                            sample_id=np.asarray(indices))
    audit = []
    for method, info in station_runs.items():
        checkpoint = torch.load(info['checkpoint'], map_location='cpu', weights_only=False)
        cfg = info['config']
        assert checkpoint['epoch'] == info['summary']['best_epoch']
        assert checkpoint['split_tags'] == ref['split_tags']
        assert checkpoint['loss_thresholds'] == ref['loss_thresholds']
        assert all(torch.equal(v, ref['normalization'][k]) for k, v in checkpoint['normalization'].items())
        expected = dict(checkpoint_sha256=sha(info['checkpoint']), source_sha256=hashes,
                        config_sha256=sha(info['config_path']), device=device,
                        torch_version=torch.__version__)
        model = None
        for split in ('val', 'test'):
            path = cache/f'{station}_{method}_{split}.npz'
            meta_path = path.with_suffix('.json')
            if path.exists() and meta_path.exists():
                meta = json.loads(meta_path.read_text())
                if meta['inputs'] == expected and meta['arrays_sha256'] == sha(path):
                    audit.append(meta)
                    print(f'{station} {method} {split}: verified cache', flush=True)
                    continue
            if model is None:
                configure_runtime(cfg['seed'], cfg['torch_threads'], bool(cfg['deterministic']), bool(cfg['tf32']))
                target_device = torch.device(device)
                if target_device.type == 'cuda' and not torch.cuda.is_available():
                    raise RuntimeError('CUDA unavailable. Use the existing GPU environment; do not change baseline precision.')
                model = build_model(ModelConfig(**checkpoint['model_config'])).to(target_device)
                model.load_state_dict(checkpoint['model_state'], strict=True)
                model.eval()
            view = ForcingGraphView(store, splits[split], checkpoint['model_config']['history_steps'])
            loader = build_loader(view, None, **{k: cfg[k] for k in
                ('batch_size','num_workers','pin_memory','persistent_workers','prefetch_factor','mp_context')})
            stats = {k: v.to(target_device) for k, v in checkpoint['normalization'].items()}
            logits = []
            # Observe the already returned gate tensor. Return None: no replacement or extra forward.
            def capture(_module, _inputs, value):
                if value.gate_logits is not None:
                    logits.append(value.gate_logits.detach().float().reshape(-1).cpu())
            hook = model.register_forward_hook(capture) if method != 'S0' else None
            try:
                result = run_epoch(model, loader, target_device, stats,
                    station_feat=checkpoint['station_feat'].to(target_device) if checkpoint['station_feat'] is not None else None,
                    use_amp=bool(cfg['amp']) and target_device.type == 'cuda',
                    amp_dtype={'bf16': torch.bfloat16, 'fp16': torch.float16}[cfg['amp_dtype']],
                    x_clip=cfg['x_clip'], save_predictions=True, save_dual_diagnostics=method != 'S0',
                    event_threshold=checkpoint['loss_thresholds']['tau_phys'])
            finally:
                if hook is not None: hook.remove()
            arrays = result.predictions
            if method != 'S0': arrays['gate_logit'] = torch.cat(logits).numpy()
            targets = dict(np.load(cache/f'{station}_{split}_targets.npz'))
            arrays = align(arrays, targets['tags'], targets['y_true'])
            reconstruction_error = None
            if method != 'S0':
                reconstruction = arrays['body_phys'] + arrays['gate_probability'][:,None]*arrays['excess_phys']
                reconstruction_error = float(np.max(np.abs(reconstruction-arrays['y_pred'])))
                if reconstruction_error > 3e-7: raise ValueError('Branch reconstruction failed')
                np.testing.assert_allclose(1/(1+np.exp(-arrays['gate_logit'])), arrays['gate_probability'], atol=1e-7)
            replay_difference = None
            if split == 'test':
                original = align(dict(np.load(info['prediction'])), arrays['tags'], arrays['y_true'])
                replay_difference = float(np.max(np.abs(original['y_pred']-arrays['y_pred'])))
            np.savez_compressed(path, **arrays)
            meta = dict(station=station, method=method, split=split, inputs=expected,
                        arrays_sha256=sha(path), metrics=result.metrics,
                        reconstruction_max_abs_m=reconstruction_error,
                        archived_test_max_abs_difference_m=replay_difference)
            write_json(meta_path, meta)
            audit.append(meta)
            print(f'{station} {method} {split}: {len(arrays["tags"])} windows; replay delta={replay_difference}', flush=True)
        del model
        if device == 'cuda': torch.cuda.empty_cache()
    return audit
