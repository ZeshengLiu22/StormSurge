#!/usr/bin/env python3
"""Audit 48 source roles, then generate exactly 264 pinned pair configurations.

All paths are fixed to the requested production roots. --check compares rendered
files in memory: it does not write configs or run inference.
"""
import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import shlex
import sys

REPO = Path('/home/exouser/StormSurge')
DESTINATION = REPO / 'configs/cross_dataset_infer_4stations'
FORMAL = Path('/home/exouser/media/share/PACT/FormalRuns_0925')
RESULTS = FORMAL / 'CrossDatasetInference_4Stations'
STATIONS = ('Boston', 'CBBT', 'Lewes', 'Battery')
SOURCES = ('NCEP', 'AWI', 'CNRM', 'EC_EARTH', 'MPI', 'MRI')
TARGETS = {'past_only': SOURCES, 'future_year': SOURCES[1:]}
ORIGIN = 'source_checkpoint_train'
CANDIDATE = 'G0_T0500_EP0100'
sys.path.insert(0, str(REPO))


def years(first, last):
    return [f'{year}_{year + 1}' for year in range(first, last + 1)]


YEARS = {'past_only': years(2008, 2014), 'future_year': years(2070, 2099)}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def dataset_root(group, name):
    if name == 'NCEP':
        return './Data/Grid4_New/NCEP/graphs'
    tree = 'Grid4_New_PastOnly' if group == 'past_only' else 'Grid4_New'
    return f'./Data/{tree}/CMIP6_{name}/graphs'


def shell_scalars(path):
    """Read the resolved training snapshot without executing shell code."""
    values = {}
    for line in path.read_text().splitlines():
        if line.startswith('declare -a '):
            continue
        for word in shlex.split(line, comments=True):
            key, sep, value = word.partition('=')
            if sep and key.isidentifier():
                values[key] = value
    return values


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def discover(station, group, source):
    if source == 'NCEP':
        area = FORMAL / ('single_tail_episodepeak_4x4' if group == 'past_only' else 'NCEP_future_transfer')
        stem = f'NCEP_{station}' + ('' if group == 'past_only' else '_future_transfer')
    else:
        area = FORMAL / f'CMIP6_{station}_QuickCheck_0929' / group
        stem = f'CMIP6_{source}_{station}_{group}'
    candidates = sorted(p for p in area.glob(f'{stem}_{CANDIDATE}__*') if p.is_dir())
    require(len(candidates) == 1,
            f'{station}/{group}/{source}: expected one active final run, found {list(map(str, candidates))}')
    return candidates[0] / 'best_overall.pt'


def training_provenance(run, checkpoint):
    """Prove training finished and the checkpoint is the full-run VAL optimum.

Two inspected Lewes runs completed all epochs but their old final TEST exporter
rejected missing labels. Preserve exit=1 and accept only this exact failure after
the final re-evaluation marker; this is not a blanket acceptance of failed runs.
"""
    records = list(run.glob('metrics_*.jsonl'))
    require(len(records) == 1, f'{run}: ambiguous training metrics')
    epochs = [json.loads(line) for line in records[0].read_text().splitlines()]
    count = checkpoint['training_config']['epochs']
    require([row['epoch'] for row in epochs] == list(range(1, count + 1)), f'{run}: incomplete training')
    require(all(math.isfinite(row['val']['all_rmse']) for row in epochs), f'{run}: nonfinite VAL score')
    best = min(epochs, key=lambda row: row['val']['all_rmse'])
    require(checkpoint['epoch'] == best['epoch'], f'{run}: checkpoint is not full-run best overall epoch')
    require(checkpoint['val']['all_rmse'] == best['val']['all_rmse'], f'{run}: saved VAL score mismatch')
    require(checkpoint['selection_metric_key'] == 'all_rmse' and checkpoint['selection_split'] == 'val',
            f'{run}: wrong selection policy')
    status = int((run / 'exit_status').read_text().strip())
    note = 'training_and_final_reporting_complete'
    evidence = {'training_metrics': str(records[0]), 'training_metrics_sha256': sha256(records[0])}
    if status != 0:
        allowed = {
            'CMIP6_MRI_Lewes_past_only_G0_T0500_EP0100__20261003_051753',
            'CMIP6_EC_EARTH_Lewes_future_year_G0_T0500_EP0100__20261003_051753',
        }
        require(status == 1 and run.name in allowed, f'{run}: unvalidated nonzero exit status {status}')
        logs = list(run.glob('train_*.log'))
        require(len(logs) == 1, f'{run}: ambiguous training log')
        log = logs[0].read_text()
        marker = 'FINAL MULTI-CHECKPOINT RE-EVALUATION'
        require(marker in log, f'{run}: no completed-training final evaluation marker')
        before, after = log.split(marker, 1)
        require(f'Epoch {count:03d}/{count}' in before and 'Traceback' not in before,
                f'{run}: training failed before completion')
        require(after.rstrip().endswith('ValueError: Cannot evaluate nonfinite y_true.'),
                f'{run}: unexpected final reporting failure')
        note = 'all_epochs_complete_best_overall_verified_final_test_reporting_failed_on_missing_labels'
        evidence.update(training_log=str(logs[0]), training_log_sha256=sha256(logs[0]))
    return dict(exit_status=status, training_epochs_completed=count, completion_provenance=note, **evidence)


def audit_checkpoint(station, group, source, path):
    import torch
    from emulator.models import ModelConfig, build_model

    prefix = f'{station}/{group}/{source}: '
    require(path.is_file() and path.name == 'best_overall.pt', prefix + f'missing {path}')
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    model, training, threshold = (checkpoint[k] for k in ('model_config', 'training_config', 'threshold_metadata'))
    saved = shell_scalars(path.parent / 'config_used.sh')
    require(checkpoint['station'] == station, prefix + 'checkpoint station mismatch')
    require(checkpoint['checkpoint_role'] == 'overall', prefix + 'checkpoint role mismatch')
    for key, value in dict(model='pact', encoder_type='GraphSAGE', temporal_block='Transformer',
                           head_type='single', history_steps=4, in_channels=5, out_channels=6).items():
        require(model[key] == value, prefix + f'model {key} mismatch: {model[key]!r}')
    for key, value in dict(station=station, model='perceiver3', encoder_type='GraphSAGE',
                           temporal_block='Transformer', head_type='single', history_hours=24,
                           use_site_elevation=0, use_bathymetry=0, loss_mode='mse',
                           exceedance_loss_mode='mse', exceedance_loss_weight=.05,
                           episode_gt_aligned_peak_weight=.01, x_norm='zscore').items():
        require(training[key] == value, prefix + f'training {key} mismatch: {training[key]!r}')
        if key.upper() in saved:
            observed = saved[key.upper()]
            require(float(observed) == value if isinstance(value, (int, float)) else observed == value,
                    prefix + f'config_used.sh {key} mismatch')
        else:
            require(key in ('history_hours', 'loss_mode'), prefix + f'config_used.sh missing {key}')
    source_root = dataset_root(group, source)
    for value in (training['root_dir'], saved['ROOT_DIR']):
        require((REPO / value).resolve() == (REPO / source_root).resolve(), prefix + 'source root mismatch')
    require(threshold['fitted_on'] == 'train', prefix + 'threshold not fitted on TRAIN')
    require(threshold['exceedance_percentile'] == 95., prefix + 'threshold is not Q95')
    require(threshold['threshold_schema'] == 'train_hourly_q95_v1', prefix + 'threshold schema mismatch')
    require(threshold['metric_schema'] == 'hourly_q95_v1', prefix + 'metric schema mismatch')
    require(math.isfinite(threshold['tau_physical']) and checkpoint['tau_physical'] == threshold['tau_physical'],
            prefix + 'invalid or inconsistent saved tau')
    groups = {split: sorted({'_'.join(tag.split('_')[:2]) for tag in tags})
              for split, tags in checkpoint['split_tags'].items()}
    expected = (dict(train=years(1979, 2000), val=years(2001, 2007), test=YEARS['past_only'])
                if group == 'past_only' else
                dict(train=years(1979, 2008), val=years(2009, 2014),
                     test=[] if source == 'NCEP' else YEARS['future_year']))
    require(groups == expected, prefix + f'split provenance mismatch: {groups}')
    tags = [tag for values in checkpoint['split_tags'].values() for tag in values]
    require(len(tags) == len(set(tags)), prefix + 'overlapping split samples')
    require(all(tag.split('_')[2] == station for tag in tags), prefix + 'split station mismatch')
    require(threshold['train_windows'] == len(checkpoint['split_tags']['train']), prefix + 'TRAIN population mismatch')
    # Current checkpoints may omit missing target hours; require the saved valid population.
    require(0 < threshold['train_target_hour_count'] <= 6 * threshold['train_windows'],
            prefix + 'invalid TRAIN hourly count')
    normalization = checkpoint['normalization']
    for key, width in dict(x_center=5, x_scale=5, y_mean=6, y_std=6).items():
        value = normalization[key]
        require(tuple(value.shape) == (width,) and bool(torch.isfinite(value).all()), prefix + f'invalid {key}')
        if key in ('x_scale', 'y_std'):
            require(bool((value > 0).all()), prefix + f'nonpositive {key}')
    network = build_model(ModelConfig(**model))
    network.load_state_dict(checkpoint['model_state'], strict=True)
    require(all(bool(torch.isfinite(v).all()) for v in checkpoint['model_state'].values()), prefix + 'nonfinite weights')
    station_feat = checkpoint['station_feat']
    require(station_feat is not None and station_feat.numel() == model['station_feat_dim']
            and bool(torch.isfinite(station_feat).all()), prefix + 'invalid source station features')
    provenance = training_provenance(path.parent, checkpoint)
    return dict(station=station, group=group, source=source, checkpoint=str(path), sha256=sha256(path),
                epoch=checkpoint['epoch'], model='perceiver3', encoder_type=model['encoder_type'],
                temporal_block=model['temporal_block'], head_type=model['head_type'],
                history_steps=model['history_steps'], history_hours=24, candidate=CANDIDATE,
                normalization_type=training['x_norm'], normalization_origin=ORIGIN,
                normalization={k: v.tolist() for k, v in normalization.items()},
                tau_physical=threshold['tau_physical'], source_tau_physical=threshold['tau_physical'],
                source_exceedance_percentile=threshold['exceedance_percentile'],
                threshold_schema=threshold['threshold_schema'], threshold_metadata=threshold,
                evaluation_threshold_origin=ORIGIN, source_root=source_root, split_years=groups,
                train_group_count=len(groups['train']), val_group_count=len(groups['val']),
                test_group_count=len(groups['test']), model_state_strict_load=True,
                loss_mode=training['loss_mode'], exceedance_loss_weight=training['exceedance_loss_weight'],
                episode_gt_aligned_peak_weight=training['episode_gt_aligned_peak_weight'],
                config_used=str(path.parent / 'config_used.sh'), config_used_sha256=sha256(path.parent / 'config_used.sh'),
                **provenance)


def audit_targets():
    rows = []
    for station in STATIONS:
        metadata = json.loads((REPO / 'station_json' / f'{station}.json').read_text())
        require(metadata['station_key'] == station, f'{station}: station JSON token mismatch')
        for group, targets in TARGETS.items():
            for target in targets:
                root = REPO / dataset_root(group, target)
                files = sorted(root.glob(f'*_{station}_*graphs.pt'))
                available = sorted({'_'.join(p.name.split('_')[:2]) for p in files})
                require(set(YEARS[group]) <= set(available),
                        f'{station}/{group}/{target}: missing years {sorted(set(YEARS[group]) - set(available))}')
                rows.append(dict(station=station, group=group, target=target, target_root=str(root),
                                 requested_years=YEARS[group], available_years=available))
    return rows


def csv_text(rows):
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=list(dict.fromkeys(k for row in rows for k in row)), lineterminator='\n')
    writer.writeheader()
    writer.writerows({k: json.dumps(v, sort_keys=True) if isinstance(v, (dict, list)) else v
                     for k, v in row.items()} for row in rows)
    return output.getvalue()


def render(checkpoints):
    files, rows = {}, []
    source_common = '#!/usr/bin/env bash\nsource "$(dirname "${BASH_SOURCE[0]}")/../common.sh"\n\n'
    for station in STATIONS:
        files[f'{station}/common.sh'] = source_common + f'STATION="{station}"\n'
        for group, targets in TARGETS.items():
            year_list = ','.join(YEARS[group])
            files[f'{station}/{group}/common.sh'] = (
                source_common + f'EXPERIMENT_GROUP="{group}"\nYEARS="{year_list}"\n'
                f'EXPECTED_YEAR_COUNT={len(YEARS[group])}\n'
                f'INFERENCE_RESULTS_ROOT="{RESULTS}/{station}/{group}"\n')
            for source in SOURCES:
                for target in targets:
                    checkpoint = str(checkpoints[station, group, source])
                    require(Path(checkpoint).is_absolute() and Path(checkpoint).name == 'best_overall.pt'
                            and not any(c in checkpoint for c in '*?['), f'unpinned checkpoint: {checkpoint}')
                    relative = f'{station}/{group}/{source}/{source}_to_{target}.sh'
                    source_root, target_root = dataset_root(group, source), dataset_root(group, target)
                    files[relative] = (source_common + f'NAME="{group}_{station}_{source}_to_{target}"\n'
                        f'SOURCE_NAME="{source}"\nTARGET_NAME="{target}"\n\n'
                        f'CKPT_PATH={shlex.quote(checkpoint)}\n\n'
                        f'SOURCE_ROOT="{source_root}"\nTARGET_ROOT="{target_root}"\n\n'
                        'ROOT_DIR="${SOURCE_ROOT}"\nTEST_ROOT_DIR="${TARGET_ROOT}"\n')
                    rows.append(dict(station=station, group=group, source=source, target=target,
                        checkpoint=checkpoint, source_root=source_root, target_root=target_root,
                        years=year_list, year_count=len(YEARS[group]),
                        config_path=str(DESTINATION / relative), threshold_origin=ORIGIN))
    require(len(rows) == 264, 'Expected 264 configs')
    files['manifest.csv'] = csv_text(rows)
    return files, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    import torch
    torch.set_num_threads(1)
    checkpoints, audits = {}, []
    for station in STATIONS:
        for group in TARGETS:
            for source in SOURCES:
                path = discover(station, group, source)
                row = audit_checkpoint(station, group, source, path)
                checkpoints[station, group, source] = path
                audits.append(row)
                print(f"PASS {station}/{group}/{source}: epoch={row['epoch']} exit={row['exit_status']} "
                      f"split={row['train_group_count']}/{row['val_group_count']}/{row['test_group_count']}", flush=True)
    targets = audit_targets()
    files, rows = render(checkpoints)
    files['checkpoint_audit.csv'] = csv_text(audits)
    files['checkpoint_audit.json'] = json.dumps(dict(checkpoints=audits, target_years=targets), indent=2) + '\n'
    for station in STATIONS:
        actual = {str(p.relative_to(DESTINATION)) for p in (DESTINATION / station).rglob('*.sh')}
        require(actual <= set(files), f'{station}: unexpected config files: {sorted(actual - set(files))}')
    if args.check:
        for name, content in files.items():
            require((DESTINATION / name).is_file() and (DESTINATION / name).read_text() == content,
                    f'Generated file drift: {DESTINATION / name}')
    else:
        for name, content in files.items():
            path = DESTINATION / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)
    print(f'PASS: 48 source roles, {len(targets)} target year audits, {len(rows)} configs; '
          f'{"checked without writes" if args.check else "written under " + str(DESTINATION)}')


if __name__ == '__main__':
    main()
