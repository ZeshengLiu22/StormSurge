#!/usr/bin/env python3
"""Check CMIP6 configs, year-file inventory, actual splits, and launcher dry-runs."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
STATION = ROOT.name.split('_')[1]
BOSTON = REPO / 'configs/CMIP6_Boston_QuickCheck_0929'
RESULTS = Path('/home/exouser/media/share/PACT/FormalRuns_0925') / ROOT.name
DATASETS = ['CMIP6_AWI', 'CMIP6_CNRM', 'CMIP6_EC_EARTH', 'CMIP6_MPI', 'CMIP6_MRI']
VARIANTS = {'T0000_EP0000': (0., 0.), 'T0500_EP0000': (.05, 0.),
            'T0000_EP0100': (0., .01), 'T0500_EP0100': (.05, .01)}
sys.path.insert(0, str(REPO))
import train
from emulator.data.graph_store import ForcingGraphStore
from emulator.training.checkpoint_selection import ROLES


def rows_at(path):
    with path.open() as handle:
        return list(csv.DictReader(handle))


def assignments(path):
    return dict(re.findall(r'^([A-Za-z_][A-Za-z_0-9]*)=(.*)$', path.read_text(), re.M))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dry_command(path, stamp):
    env = dict(os.environ, DRY_RUN='1', USE_TMUX='0', PACT_RUNSTAMP=stamp)
    run = subprocess.run(['bash', 'train.sh', str(path)], cwd=REPO, env=env,
                         text=True, capture_output=True, check=True)
    commands = [shlex.split(line[5:])[1:] for line in run.stdout.splitlines() if line.startswith('CMD: ')]
    assert len(commands) == 1, f'Expected one training command: {path}'
    return vars(train.parse_args(commands[0])), run.stdout


def audit(group):
    folder = ROOT / group
    rows = rows_at(folder / 'manifest.csv')
    combined = rows_at(ROOT / 'manifest.csv')
    assert len(combined) == 40 and len(rows) == 20
    assert rows == [r for r in combined if r['group'] == group]
    assert {(r['dataset'], r['variant']) for r in rows} == {(d, v) for d in DATASETS for v in VARIANTS}
    for key in ['config_path', 'result_path', 'queue_label']:
        assert len({r[key] for r in combined}) == 40, f'Duplicate {key}'
    assert {str(p) for p in folder.glob('train_config_*.sh')} == {r['config_path'] for r in rows}
    assert set(ROLES) == {'overall', 'exceedance'}
    metadata_path = REPO / 'station_json' / f'{STATION}.json'
    assert metadata_path.is_file()
    metadata = json.loads(metadata_path.read_text())
    assert metadata['station_key'].lower() == STATION.lower()
    for command_file, selected in [(ROOT / 'qsub_all.txt', combined), (folder / 'qsub_all.txt', rows)]:
        lines = command_file.read_text().splitlines()
        commands = [shlex.split(line) for line in lines if line.startswith('qsub_local ')]
        expected = [['qsub_local', 'train.sh', r['queue_label'], str(Path(r['config_path']).relative_to(REPO))] for r in selected]
        assert commands == expected, f'Submission commands disagree: {command_file}'
        assert len({r['runstamp'] for r in selected}) == 1
        assert f'export PACT_RUNSTAMP={selected[0]["runstamp"]} USE_TMUX=1 QSUB_LOCAL_SLOTS=1 DRY_RUN=0' in lines
    data_folder = 'Grid4_New_PastOnly' if group == 'past_only' else 'Grid4_New'
    ratios = (.6, .2) if group == 'past_only' else (.4545454545, .0909090909)
    ranges = {'train': range(1979, 2001), 'val': range(2001, 2008), 'test': range(2008, 2015)} if group == 'past_only' else {
        'train': range(1979, 2009), 'val': range(2009, 2015), 'test': range(2070, 2100)}
    expected_split = {part: [f'{year}_{year+1}' for year in years] for part, years in ranges.items()}
    expected_years = [y for years in expected_split.values() for y in years]
    inventory = {}
    for dataset in DATASETS:
        data_root = REPO / 'Data' / data_folder / dataset / 'graphs'
        files = sorted(data_root.glob(f'*_{STATION}_*graphs.pt'))
        years = ['_'.join(p.name.split('_')[:2]) for p in files]
        assert years == expected_years, f'Unexpected station/year inventory: {data_root}'
        assert all(p.stat().st_size > 0 for p in files), f'Empty graph archive: {dataset}'
        store = ForcingGraphStore.__new__(ForcingGraphStore)
        store.year_to_indices = {year: [i] for i, year in enumerate(years)}
        split = store.split(train_ratio=ratios[0], val_ratio=ratios[1], shuffle_years=False,
                            seed=42, future_only=False, future_year_threshold=2030)
        actual_split = {part: [years[i] for i in indices] for part, indices in split.items()}
        assert actual_split == expected_split, f'Unexpected split: {dataset}'
        if group == 'past_only':
            ncep = sorted((REPO / 'Data/Grid4_New/NCEP/graphs').glob(f'*_{STATION}_*graphs.pt'))
            assert ['_'.join(p.name.split('_')[:2]) for p in ncep] == years
        inventory[dataset] = {'root': str(data_root), 'file_count': len(files), 'split': actual_split,
                              'files': [{'name': p.name, 'bytes': p.stat().st_size, 'mtime_ns': p.stat().st_mtime_ns} for p in files]}
    cached_sources, audit_rows, dry_logs = {}, [], []
    for row in rows:
        config, source = Path(row['config_path']), Path(row['template_path'])
        dataset, variant = row['dataset'], row['variant']
        assert row['group'] == group and config.parent == folder
        assert source == REPO / 'configs/single_tail_episodepeak_4x4' / f'train_config_NCEP_{STATION}_G0_{variant}.sh'
        run_name = f'{dataset}_{STATION}_{group}_G0_{variant}'
        result = RESULTS / group / f'{run_name}__{row["runstamp"]}'
        assert row['result_path'] == str(result) and row['primary_checkpoint'] == 'overall'
        assert (float(row['tail_mse_weight']), float(row['episode_gt_aligned_peak_mse_weight'])) == VARIANTS[variant]
        boston_config = BOSTON / group / f'train_config_{dataset}_Boston_{group}_G0_{variant}.sh'
        assert config.read_text() == boston_config.read_text().replace('Boston', STATION), f'Unexpected Boston template changes: {config}'
        before, after = assignments(source), assignments(config)
        assert before.keys() == after.keys()
        expected_changes = {'ROOT_DIR', 'ALL_RESULTS_ROOT', 'PACT_RUN_NAME', 'PYTHON_RUN_TAG_BASE', 'CHECKPOINT_SELECTION'}
        if group == 'future_year':
            expected_changes |= {'TRAIN_RATIO', 'VAL_RATIO'}
        assert {k for k in before if before[k] != after[k]} == expected_changes, config
        assert after['STATION'] == f'"{STATION}"'
        assert after['ALL_RESULTS_ROOT'] == f'"{RESULTS / group}"'
        assert after['PACT_RUN_NAME'] == after['PYTHON_RUN_TAG_BASE'] == f'"{run_name}"'
        subprocess.run(['bash', '-n', str(config)], check=True)
        if source not in cached_sources:
            cached_sources[source] = dry_command(source, row['runstamp'])[0]
        original = cached_sources[source]
        resolved, log = dry_command(config, row['runstamp'])
        allowed = {'root_dir', 'checkpoint_selection', 'run_tag', 'output_dir', 'train_ratio', 'val_ratio'}
        assert original.keys() == resolved.keys()
        assert {k for k in original if original[k] != resolved[k]} <= allowed, config
        expected = dict(root_dir=f'./Data/{data_folder}/{dataset}/graphs', station=STATION, head_type='single',
                        history_hours=24, encoder_type='GraphSAGE', temporal_block='Transformer', hidden_channels=128,
                        lr=.005, epochs=300, batch_size=256, grad_accum_steps=4, seed=42, loss_mode='mse',
                        scheduler='cosine', amp=True, amp_dtype='bf16', tf32=True, checkpoint_selection='overall',
                        train_ratio=ratios[0], val_ratio=ratios[1], shuffle_years=False, future_only=False,
                        exceedance_loss_weight=VARIANTS[variant][0], episode_gt_aligned_peak_weight=VARIANTS[variant][1],
                        output_dir=str(result), run_tag=f'{run_name}_{row["runstamp"]}')
        assert all(resolved[k] == value for k, value in expected.items()), f'Resolved arguments mismatch: {config}'
        for other in audit_rows:
            if other['dataset'] == dataset:
                assert {k for k in resolved if other['resolved_arguments'][k] != resolved[k]} <= {
                    'exceedance_loss_weight', 'episode_gt_aligned_peak_weight', 'run_tag', 'output_dir'}
        audit_rows.append(dict(row, config_sha256=digest(config), template_sha256=digest(source),
                               boston_template_sha256=digest(boston_config), resolved_arguments=resolved))
        dry_logs.append(log)
    now = datetime.now(timezone.utc).isoformat()
    (folder / 'dataset_inventory.json').write_text(json.dumps(dict(station=STATION, group=group, audited_at_utc=now,
        scope='Station/year filenames, nonempty archives, metadata, and actual chronological split; tensor contents were not loaded.',
        datasets=inventory), indent=2) + '\n')
    (folder / 'config_audit.json').write_text(json.dumps(dict(passed=True, station=STATION, group=group,
        audited_at_utc=now, config_count=len(rows), checkpoint_roles=list(ROLES), primary_checkpoint='overall',
        split_years=expected_split, code_sha256={name: digest(REPO / name) for name in [
        'train.py', 'train.sh', 'emulator/data/graph_store.py', 'emulator/training/checkpoint_selection.py']}, rows=audit_rows), indent=2, default=str) + '\n')
    (folder / 'dry_run.log').write_text('\n'.join(dry_logs))
    print(f'PASS {STATION} {group}: {len(rows)} configs; syntax, templates, resolved arguments, paths, and all five year inventories.')
    print('PASS split: ' + '; '.join(f'{part}={len(years)} ({years[0]} to {years[-1]})' for part, years in expected_split.items()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('group', nargs='?', choices=['past_only', 'future_year', 'all'], default='all')
    selected = parser.parse_args().group
    for group in (['past_only', 'future_year'] if selected == 'all' else [selected]):
        audit(group)
