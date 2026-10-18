#!/usr/bin/env python3
"""Validate raw pair artifacts and summarize AllRMSE without recomputing metrics."""
import csv
import json
import math
from pathlib import Path
import sys

import numpy as np

from generate_configs import DESTINATION, REPO, RESULTS, STATIONS, SOURCES, TARGETS, ORIGIN, csv_text, require


def identity(row):
    return tuple(row[key] for key in ('station', 'group', 'source', 'target'))


def validate_metadata(data, row, audit):
    expected = dict(station=row['station'], source_name=row['source'], target_name=row['target'],
        model='perceiver3', encoder_type='GraphSAGE', temporal_block='Transformer', head_type='single',
        history_hours=24, history_steps=4, strict_years=True, normalization_origin=ORIGIN,
        normalization_type=audit['normalization_type'], evaluation_threshold_origin=ORIGIN,
        tau_physical=audit['tau_physical'], source_tau_physical=audit['tau_physical'],
        source_exceedance_percentile=95., threshold_schema='train_hourly_q95_v1', metric_schema='hourly_q95_v1')
    for key, value in expected.items():
        require(data.get(key) == value, f'{key}: {data.get(key)!r} != {value!r}')
    for key in ('checkpoint', 'source_root', 'target_root'):
        require(Path(data[key]).resolve() == (REPO / row[key]).resolve(), f'{key} mismatch')
    years = row['years'].split(',')
    for key in ('requested_years', 'evaluated_years'):
        require(data[key] == years, f'{key} mismatch: {data[key]}')
    for key in ('requested_year_count', 'evaluated_year_count'):
        require(data[key] == len(years) == int(row['year_count']), f'{key} mismatch')
    require(data['threshold_metadata'] == audit['threshold_metadata'], 'source threshold metadata changed')
    require(data['scope'] == 'external_all_years', 'pair did not use explicit external evaluation')
    value = data['metrics']['all_rmse']
    require(isinstance(value, (int, float)) and math.isfinite(value) and value >= 0, 'invalid final AllRMSE')
    require(value == data['results']['_overall']['all_rmse'], 'final AllRMSE artifacts disagree')
    require(not any(key.startswith('target_tau') or key.startswith('target_threshold') for key in data),
            'unexpected target threshold fields')


def validate_predictions(path, data):
    require(path.is_file(), f'missing {path.name}')
    with np.load(path, allow_pickle=False) as arrays:
        for key in ('station', 'source_name', 'target_name', 'checkpoint', 'source_root', 'target_root',
                    'requested_years', 'evaluated_years', 'requested_year_count', 'evaluated_year_count',
                    'tau_physical', 'source_tau_physical', 'source_exceedance_percentile',
                    'evaluation_threshold_origin', 'normalization_origin', 'threshold_schema',
                    'model', 'encoder_type', 'temporal_block', 'head_type', 'history_hours'):
            require(arrays[key].tolist() == data[key], f'predictions metadata mismatch: {key}')
        prediction, truth = arrays['y_pred'], arrays['y_true']
        require(prediction.shape == truth.shape == (data['samples'], 6), 'prediction/target shape mismatch')
        valid = np.isfinite(truth)
        require(bool(np.isfinite(prediction[valid]).all()), 'nonfinite prediction for a finite target')
        tags = arrays['tags'].tolist()
        require(len(tags) == data['samples'], 'sample/tag count mismatch')
        require(sorted({'_'.join(tag.split('_')[:2]) for tag in tags}) == data['requested_years'],
                'prediction tags have the wrong year population')
        require(all(tag.split('_')[2] == data['station'] for tag in tags), 'prediction tag station mismatch')
        require(set(arrays['split_ids'].tolist()) == {'external'}, 'prediction splits are not external')
        require(arrays['target_timestamps'].shape == truth.shape, 'target timestamp shape mismatch')
        return dict(finite_target_hours=int(valid.sum()), missing_target_hours=int((~valid).sum()),
                    episode_eligible_windows=int(valid.all(axis=1).sum()))


def matrix_rows(station, group, values):
    return [dict(source=source, **{target: values.get((station, group, source, target), '')
                                  for target in TARGETS[group]}) for source in SOURCES]


def summarize():
    manifest = list(csv.DictReader((DESTINATION / 'manifest.csv').open()))
    require(len(manifest) == 264 and len({identity(r) for r in manifest}) == 264, 'manifest is not exactly 264 unique pairs')
    expected = {identity(row): row for row in manifest}
    audit_rows = json.loads((DESTINATION / 'checkpoint_audit.json').read_text())['checkpoints']
    audits = {(r['station'], r['group'], r['source']): r for r in audit_rows}
    found, issues, attempted = {}, [], set()
    for station in STATIONS:
        for group in TARGETS:
            for run in sorted((RESULTS / station / group).glob('*')):
                if not run.is_dir():
                    continue
                matches = [key for key in expected if key[:2] == (station, group)
                           and run.name.startswith(f'{group}_{station}_{key[2]}_to_{key[3]}_')]
                if len(matches) != 1:
                    issues.append(dict(output_dir=str(run), error='unrecognized pair output directory'))
                    continue
                key = matches[0]
                attempted.add(key)
                found.setdefault(key, []).append(run / 'outputs')
    rows, values, failed = [], {}, set()
    for key, row in expected.items():
        outputs = found.get(key, [])
        if not outputs:
            continue
        if len(outputs) != 1:
            issues.append(dict(pair=list(key), error='duplicate pair outputs', output_dirs=list(map(str, outputs))))
            failed.add(key)
            continue
        output = outputs[0]
        try:
            data = json.loads((output / 'metrics.json').read_text())
            validate_metadata(data, row, audits[key[:3]])
            counts = validate_predictions(output / 'predictions.npz', data)
            for name in ('infer_config_used.sh', 'command_used.sh'):
                require((output.parent / name).is_file(), f'missing reproducibility artifact {name}')
            reports = list(output.glob('metrics_per_year_*.json'))
            require(len(reports) == 1, 'missing/ambiguous physical-unit report')
            report = json.loads(reports[0].read_text())
            require(report['metric_space'] == 'physical' and report['metrics'] == data['metrics'],
                    'physical-unit report disagrees with final metrics')
            value = data['metrics']['all_rmse']
            values[key] = value
            rows.append(dict(station=key[0], group=key[1], source=key[2], target=key[3], AllRMSE=value,
                checkpoint=row['checkpoint'], tau_physical=data['tau_physical'], threshold_origin=ORIGIN,
                requested_year_count=data['requested_year_count'], evaluated_year_count=data['evaluated_year_count'],
                output_dir=str(output), requested_years=row['years'], evaluated_years=','.join(data['evaluated_years']),
                samples=data['samples'], **counts))
        except (OSError, ValueError, KeyError, TypeError) as error:
            failed.add(key)
            issues.append(dict(pair=list(key), output_dir=str(output), error=str(error)))
    missing = [list(key) for key in expected if key not in values]
    summary = RESULTS / 'summary'
    summary.mkdir(parents=True, exist_ok=True)
    if rows:
        (summary / 'all_pairs.csv').write_text(csv_text(rows))
    else:
        (summary / 'all_pairs.csv').write_text('station,group,source,target,AllRMSE,checkpoint,tau_physical,threshold_origin,requested_year_count,evaluated_year_count,output_dir\n')
    report = ['# Four-station AllRMSE matrices', '',
        'Rows = source; columns = target. AllRMSE is in meters and is read directly from each final metrics.json.', '',
        'Every pair uses source checkpoint normalization and source TRAIN Q95. Diagonals use the same explicit target path and fixed years as transfers.', '',
        'Current repository missing-label semantics are unchanged: hourly metrics omit nonfinite target hours; episode metrics omit their entire windows. Counts are recorded in all_pairs.csv.', '']
    matrices = []
    for station in STATIONS:
        for group in TARGETS:
            matrix = matrix_rows(station, group, values)
            filename = f'{station}_{group}_AllRMSE.csv'
            (summary / filename).write_text(csv_text(matrix))
            population = sum((station, group, source, target) in values for source in SOURCES for target in TARGETS[group])
            count = len(SOURCES) * len(TARGETS[group])
            matrices.append(dict(station=station, group=group, expected=count, populated=population, file=filename))
            report += [f'## {station} — {group}', '', f'Expected cells: {count}; successfully populated: {population}.', '',
                       '| Source | ' + ' | '.join(TARGETS[group]) + ' |',
                       '| --- | ' + ' | '.join('---:' for _ in TARGETS[group]) + ' |']
            for row in matrix:
                report.append('| ' + row['source'] + ' | ' + ' | '.join(
                    'MISSING' if row[target] == '' else f'{row[target]:.6f}' for target in TARGETS[group]) + ' |')
            report.append('')
    if issues or missing:
        report += ['## Incomplete or invalid pairs', '']
        report += [f'- {" / ".join(pair)}' for pair in missing]
        report += ['', 'See completeness.json for validation errors and duplicate output paths.', '']
    (summary / 'AllRMSE_matrices.md').write_text('\n'.join(report))
    status = dict(expected=264, completed=len(rows), failed=len(failed), missing=len(missing),
                  unattempted=len(set(expected) - attempted), missing_pairs=missing, issues=issues,
                  matrices=matrices, complete_matrices=sum(m['populated'] == m['expected'] for m in matrices))
    (summary / 'completeness.json').write_text(json.dumps(status, indent=2) + '\n')
    print(json.dumps({k: status[k] for k in ('expected', 'completed', 'failed', 'missing', 'unattempted', 'complete_matrices')}, indent=2))
    return 0 if len(rows) == 264 and not issues and not missing else 1


if __name__ == '__main__':
    sys.exit(summarize())
