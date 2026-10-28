"""Shared, CPU-safe contracts for the fixed CMIP6 future inference matrix."""
from __future__ import annotations

import contextlib
import fcntl
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import re
import sys
import uuid
import zipfile

REPO = Path('/glade/u/home/zel/scratch/StormSurge')
CKPT_ROOT = Path('/glade/u/home/zel/scratch/PACT_StormSurge_FormalRuns_1021_LR5e-3_Clip0.25/cmip6')
DATA_ROOT = REPO / 'dataset'
RESULT_ROOT = CKPT_ROOT.parent / 'cmip6_inference'
LAUNCHER = REPO / 'PBS-launcher/cmip6_inference'
PYTHON = '/glade/work/zel/conda-envs/derecho-ml/bin/python'
SOURCES = ('CMIP6_AWI', 'CMIP6_BCC', 'CMIP6_Cane5', 'CMIP6_CNRM', 'CMIP6_EC_EARTH',
           'CMIP6_IPSL', 'CMIP6_MIROC', 'CMIP6_MPI', 'CMIP6_MRI')
STATIONS = ('Battery', 'Boston', 'CBBT', 'Lewes')
SEEDS = (42, 123, 2026)
MODELS = {'CNN_GRU_MSE': 'baseline_cnn_gru', 'PACT_MSE': 'perceiver3',
          'PACT_Peak': 'perceiver3', 'PACT_Both': 'perceiver3'}
LOSSES = {'CNN_GRU_MSE': (0.0, 0.0), 'PACT_MSE': (0.0, 0.0),
          'PACT_Peak': (0.0, 0.01), 'PACT_Both': (0.025, 0.01)}
YEARS = [f'{year}_{year + 1}' for year in range(2070, 2100)]
RESOURCES = {'derecho': 'select=1:ncpus=64:mpiprocs=4:ompthreads=1:ngpus=4:mem=487GB',
             'casper': 'select=1:ncpus=32:mpiprocs=1:ngpus=1:gpu_type=a100_80gb:mem=240GB'}
DEFAULT_RUNTIME = dict(batch_size=32, num_workers=0, torch_threads=1, amp=True,
                       amp_dtype='bf16', tf32=True)
sys.path.insert(0, str(REPO))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def combinations():
    return itertools.product(SOURCES, STATIONS, SEEDS, MODELS)


def identity(source, station, seed, model):
    return f'{source}/{station}/seed{seed}/{model}'


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def digest_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    temp = path.with_name(f'.{path.name}.{uuid.uuid4().hex}.tmp')
    try:
        with temp.open('w') as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def completed_training(path):
    """Training's own exit status and final summary are completion evidence."""
    status = path.parent / 'exit_status'
    summaries = sorted(path.parent.glob('summary_*.json'))
    if not status.is_file() or status.read_text().strip() != '0' or len(summaries) != 1:
        return False
    summary = json.loads(summaries[0].read_text())
    return summary.get('checkpoint_selection', {}).get('checkpoint_selection_mode') == 'overall'


def resolve_checkpoint(root, source, station, seed, model, selections=None):
    """Never rank by mtime or substitute another checkpoint role."""
    key = identity(source, station, seed, model)
    directory = Path(root) / key
    candidates = sorted(p.resolve() for p in directory.glob('*/best_overall.pt') if completed_training(p))
    selected = (selections or {}).get(key)
    if selected is not None:
        chosen = Path(selected).expanduser().resolve()
        require(chosen in candidates, f'{key}: selected checkpoint is absent, incomplete, or outside its hierarchy: {chosen}')
        return chosen
    require(len(candidates) == 1,
            f'{key}: expected one completed best_overall.pt, found {len(candidates)}: '
            f'{[str(p) for p in candidates]}; use --selection JSON or --reuse-manifest for explicit resolution')
    return candidates[0]


def inspect_checkpoint(path, source, station, seed, model, *, check_weights=True):
    """Read trusted local checkpoints; reuse the repository's architecture loader."""
    import torch
    from dataclasses import asdict
    from emulator.models import build_model, model_config_from_checkpoint
    torch.set_num_threads(1)
    path = Path(path)
    require(path.name == 'best_overall.pt', f'Wrong checkpoint filename: {path}')
    before = path.stat()
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    config = model_config_from_checkpoint(checkpoint)
    training = checkpoint['training_config']
    threshold = checkpoint['threshold_metadata']
    require(checkpoint.get('checkpoint_role') == 'overall', f'{path}: checkpoint_role must be overall')
    require(checkpoint.get('selection_split') == 'val' and checkpoint.get('selection_metric_key') == 'all_rmse',
            f'{path}: invalid overall checkpoint selection metadata')
    require(training.get('checkpoint_selection') == 'overall', f'{path}: training selection is not overall')
    require(checkpoint.get('station') == station and training.get('station') == station, f'{path}: station mismatch')
    require(training.get('seed') == seed, f'{path}: seed mismatch')
    actual = 'perceiver3' if config.model == 'pact' else config.model
    require(actual == MODELS[model] and training.get('model') == actual, f'{path}: model identity mismatch')
    weights = tuple(float(training.get(k, -1)) for k in ('exceedance_loss_weight', 'episode_gt_aligned_peak_weight'))
    require(weights == LOSSES[model], f'{path}: loss variant {weights} does not identify {model}')
    source_root = training.get('root_dir')
    require(source_root and Path(source_root).is_absolute(), f'{path}: missing absolute original training root')
    require(Path(source_root).parent.name in (source, source + '_PastOnly'), f'{path}: cross-GCM source root {source_root}')
    require(Path(source_root).name == 'graphs', f'{path}: unexpected training graph root')
    require(threshold.get('fitted_on') == 'train' and threshold.get('exceedance_percentile') == 95,
            f'{path}: threshold was not fitted on TRAIN at Q95')
    require(threshold.get('metric_schema') == 'hourly_q95_v1' and
            threshold.get('threshold_schema') == 'train_hourly_q95_v1', f'{path}: incompatible threshold schema')
    require(math.isfinite(float(threshold['tau_physical'])), f'{path}: invalid saved Q95')
    pressure = training.get('pressure_mode', 'anomaly')
    require(pressure in ('raw', 'anomaly', 'anomaly_plus_mean'), f'{path}: invalid pressure mode')
    norm = checkpoint['normalization']
    for name, width in (('x_center', config.in_channels), ('x_scale', config.in_channels),
                        ('y_mean', config.out_channels), ('y_std', config.out_channels)):
        value = norm.get(name)
        require(isinstance(value, torch.Tensor) and tuple(value.shape) == (width,) and
                bool(torch.isfinite(value).all()), f'{path}: invalid saved normalization {name}')
        if name in ('x_scale', 'y_std'):
            require(bool((value > 0).all()), f'{path}: nonpositive saved {name}')
    feature = checkpoint['station_feat']
    if config.station_feat_dim:
        require(isinstance(feature, torch.Tensor) and tuple(feature.shape) == (config.station_feat_dim,) and
                bool(torch.isfinite(feature).all()), f'{path}: missing/incompatible saved station features')
    for key in ('deterministic', 'x_clip', 'use_site_elevation', 'use_bathymetry'):
        require(key in training, f'{path}: missing training setting {key}')
    if check_weights:
        network = build_model(config)
        network.load_state_dict(checkpoint['model_state'], strict=True)
    summary = json.loads(next(path.parent.glob('summary_*.json')).read_text())
    require(summary['checkpoint_selection']['selected_epoch'] == checkpoint['epoch'], f'{path}: summary/checkpoint epoch mismatch')
    checksum = sha256(path)
    after = path.stat()
    require((before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns), f'{path}: changed during inspection')
    return dict(sha256=checksum, size=after.st_size, source_root=str(source_root),
                model_config=asdict(config), pressure_mode=pressure, epoch=checkpoint['epoch'],
                threshold_metadata=threshold, normalization={k: v.tolist() for k, v in norm.items()},
                normalization_origin='source_checkpoint_train', evaluation_threshold_origin='source_checkpoint_train')


def graph_inventory(root, station):
    """Mirror ForcingGraphStore's filename filters without deserializing data."""
    root = Path(root)
    require(root.is_dir(), f'Missing graph directory: {root}')
    selected = {}
    for path in sorted(root.glob('*graphs.pt')):
        parts = path.stem.split('_')
        require(len(parts) >= 3, f'Invalid graph filename: {path}')
        if parts[2] != station:
            continue
        year = '_'.join(parts[:2])
        if year not in YEARS:
            continue
        require(re.fullmatch(r'\d{4}_\d{4}_' + re.escape(station) + r'_.+_graphs\.pt', path.name),
                f'Unexpected station graph filename: {path}')
        require(year not in selected, f'Duplicate station/year file: {root}, {station}, {year}')
        stat = path.stat()
        require(stat.st_size > 0, f'Empty graph file: {path}')
        selected[year] = dict(path=str(path.absolute()), size=stat.st_size, mtime_ns=stat.st_mtime_ns)
    require(sorted(selected) == YEARS, f'{root}/{station}: missing years {sorted(set(YEARS) - set(selected))}')
    return [selected[year] for year in YEARS]


def memory_estimate(inventory, workers=0):
    """Budget 3x serialized data plus 8 GiB per process; account for worker copies."""
    gib = sum(item['size'] for item in inventory) / 2**30
    return (3 * gib + 8) * (1 + workers)


def load_manifest(path):
    manifest = json.loads(Path(path).read_text())
    require(manifest.get('schema_version') == 1, 'Unsupported manifest schema')
    require(manifest.get('years') == YEARS, 'Manifest must select exactly the 30 future winters')
    expected = {identity(*combo) for combo in combinations()}
    rows = manifest['runs']
    require(len(rows) == 432 and {row['id'] for row in rows} == expected, 'Manifest must contain exactly the 432 unique combinations')
    require(len({str(Path(row['output_dir']).resolve()) for row in rows}) == 432, 'Duplicate output destinations')
    for row in rows:
        require(row['id'] == identity(row['source'], row['station'], row['seed'], row['model']), 'Manifest identity mismatch')
        require(row['model_key'] == MODELS[row['model']], 'Wrong model mapping')
        require(Path(row['checkpoint']).is_absolute() and Path(row['checkpoint']).name == 'best_overall.pt', 'Wrong checkpoint role/path')
    return manifest


def runtime_from_env():
    runtime = dict(DEFAULT_RUNTIME)
    for key in runtime:
        value = os.environ.get('INFER_' + key.upper())
        if value is None:
            continue
        if isinstance(runtime[key], bool):
            require(value in ('0', '1'), f'INFER_{key.upper()} must be 0 or 1')
            runtime[key] = value == '1'
        elif isinstance(runtime[key], int):
            runtime[key] = int(value)
        else:
            runtime[key] = value
    require(runtime['batch_size'] > 0 and runtime['torch_threads'] > 0 and runtime['num_workers'] >= 0, 'Invalid inference size/thread settings')
    require(runtime['amp_dtype'] in ('bf16', 'fp16'), 'Invalid INFER_AMP_DTYPE')
    return runtime


def inference_command(row, output, runtime, python=PYTHON, repo=REPO):
    command = [str(python), '-u', str(Path(repo) / 'infer.py'), '--ckpt', row['checkpoint'],
               '--root_dir', row['source_root'], '--test_root_dir', row['test_root_dir'],
               '--source_name', row['source'], '--target_name', row['source'], '--station', row['station'],
               '--model', row['model_key'], '--model_label', row['model'], '--years', ','.join(YEARS),
               '--strict_years', '--save_npz', '--device', 'cuda', '--out_dir', str(output)]
    # Architecture, station features, history, and pressure all come from the checkpoint.
    for key in ('batch_size', 'num_workers', 'torch_threads', 'amp_dtype'):
        command += ['--' + key, str(runtime[key])]
    for key in ('amp', 'tf32'):
        if runtime[key]:
            command.append('--' + key)
    return command


@contextlib.contextmanager
def run_lock(directory):
    """Never unlink lock inodes. Inference inherits the fd to survive parent SIGKILL."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / '.run.lock').open('a+') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f'Another process holds {directory}/.run.lock') from error
        try:
            yield lock.fileno()
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def validate_outputs(directory, row):
    """Fully read NPZ arrays and provenance; an existing filename is insufficient."""
    import numpy as np
    directory = Path(directory)
    reports = sorted(directory.glob('metrics_per_year_*.json'))
    require(len(reports) == 1, f'{directory}: expected exactly one per-year report')
    required = [directory / 'metrics.json', directory / 'metrics.md', directory / 'predictions.npz', *reports]
    require(all(p.is_file() and p.stat().st_size for p in required), f'{directory}: incomplete required outputs')
    metrics = json.loads((directory / 'metrics.json').read_text())
    report = json.loads(reports[0].read_text())
    expected = dict(requested_years=YEARS, evaluated_years=YEARS, requested_year_count=30,
                    evaluated_year_count=30, strict_years=True, station=row['station'], model=row['model_key'],
                    source_name=row['source'], target_name=row['source'], checkpoint=str(Path(row['checkpoint']).resolve()),
                    source_root=str(Path(row['source_root']).resolve()), target_root=str(Path(row['test_root_dir']).resolve()),
                    normalization_origin='source_checkpoint_train', evaluation_threshold_origin='source_checkpoint_train',
                    pressure_mode=row['checkpoint_metadata']['pressure_mode'],
                    source_tau_physical=row['checkpoint_metadata']['threshold_metadata']['tau_physical'],
                    tau_physical=row['checkpoint_metadata']['threshold_metadata']['tau_physical'],
                    exceedance_percentile=95.0, source_exceedance_percentile=95.0,
                    metric_schema='hourly_q95_v1', threshold_schema='train_hourly_q95_v1')
    for document in (metrics, report):
        for key, value in expected.items():
            require(document.get(key) == value, f'{directory}: incorrect {key}: {document.get(key)!r}')
        require(document.get('threshold_metadata') == row['checkpoint_metadata']['threshold_metadata'], 'Saved threshold metadata changed')
        require(set(YEARS).issubset(document.get('results', {})), 'Missing per-year metrics')
        require(all(document['results'][year]['samples'] > 0 for year in YEARS), 'Empty evaluated winter')
        require(math.isfinite(float(document['metrics']['all_rmse'])), 'Invalid aggregate RMSE')
    require(report.get('years_evaluated') == YEARS and metrics.get('years') == YEARS, 'Wrong reported year set')
    with np.load(directory / 'predictions.npz', allow_pickle=False) as archive:
        # Reading all members checks ZIP CRCs, including a partially written archive.
        arrays = {name: archive[name] for name in archive.files}
    for key, value in expected.items():
        require(key in arrays and arrays[key].tolist() == value, f'NPZ metadata mismatch: {key}')
    pred, truth, tags, times = (arrays[k] for k in ('y_pred', 'y_true', 'tags', 'target_timestamps'))
    require(pred.ndim == 2 and pred.shape == truth.shape == times.shape and len(tags) == len(pred) > 0,
            'Incompatible/empty prediction arrays')
    require(pred.shape[1] == row['checkpoint_metadata']['model_config']['out_channels'], 'Incorrect prediction horizon count')
    require(bool(np.isfinite(pred).all()), 'Nonfinite physical predictions')
    require(not bool(np.isinf(truth).any()), 'Infinite physical ground truth')
    require(len(pred) == metrics['samples'] == sum(metrics['results'][y]['samples'] for y in YEARS), 'Prediction/sample count mismatch')
    require({'_'.join(str(tag).split('_')[:2]) for tag in tags} == set(YEARS), 'NPZ contains incorrect years')
    require(all(str(tag).split('_')[2] == row['station'] for tag in tags), 'NPZ contains another station')
    require(set(arrays['split_ids'].tolist()) == {'external'}, 'NPZ is not external future evaluation')
    return {p.name: sha256(p) for p in required}


def verified_completion(directory, row, fingerprint):
    directory = Path(directory)
    marker = directory / 'COMPLETE.json'
    if not marker.is_file():
        return False
    try:
        saved = json.loads(marker.read_text())
        provenance = json.loads((directory / 'provenance.json').read_text())
        status = json.loads((directory / 'status.json').read_text())
        return (saved['fingerprint'] == fingerprint == provenance['fingerprint'] and
                provenance['run'] == row and status['state'] == 'success' and
                Path(status['log']).is_file() and
                saved['files'] == validate_outputs(directory / 'outputs', row))
    except (OSError, ValueError, KeyError, EOFError, zipfile.BadZipFile):
        return False
