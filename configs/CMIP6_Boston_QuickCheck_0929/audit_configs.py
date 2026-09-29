#!/usr/bin/env python3
"""Audit this fixed Boston quick check against the current NCEP formal templates."""
from collections import Counter
from datetime import datetime, timezone
import argparse
import csv
import difflib
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys

REPO = Path(__file__).resolve().parents[2]
ROOT_FOLDER = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('group', choices=['past_only','future_year'])
GROUP = parser.parse_args().group
FOLDER = ROOT_FOLDER / GROUP
DATA_FOLDER = 'Grid4_New_PastOnly' if GROUP == 'past_only' else 'Grid4_New'
RATIOS = (.6, .2) if GROUP == 'past_only' else (.4545454545, .0909090909)
RESULTS = Path('/home/exouser/media/share/PACT/FormalRuns_0925/CMIP6_Boston_QuickCheck_0929') / GROUP
sys.path.insert(0, str(REPO))
import train
from emulator.training.checkpoint_selection import ROLES
from emulator.data.graph_store import ForcingGraphStore

VARIANTS = {'T0000_EP0000': (0., 0.), 'T0500_EP0100': (.05, .01)}
ASSIGNMENT_CHANGES = {'ROOT_DIR', 'ALL_RESULTS_ROOT', 'PACT_RUN_NAME', 'PYTHON_RUN_TAG_BASE', 'CHECKPOINT_SELECTION'}
ARGUMENT_CHANGES = {'root_dir', 'checkpoint_selection', 'run_tag', 'output_dir'}
if GROUP == 'future_year':
    ASSIGNMENT_CHANGES |= {'TRAIN_RATIO','VAL_RATIO'}
    ARGUMENT_CHANGES |= {'train_ratio','val_ratio'}

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def assignments(path):
    return dict(re.findall(r'^([A-Za-z_][A-Za-z_0-9]*)=(.*)$', path.read_text(), re.M))

def dry_command(path, stamp):
    environment = dict(os.environ, DRY_RUN='1', USE_TMUX='0', PACT_RUNSTAMP=stamp)
    result = subprocess.run(['bash', 'train.sh', str(path)], cwd=REPO, env=environment,
                            text=True, capture_output=True, check=True)
    commands = [shlex.split(line[5:])[1:] for line in result.stdout.splitlines() if line.startswith('CMD: ')]
    assert len(commands) == 1, f'{path}: expected exactly one training command'
    return vars(train.parse_args(commands[0])), result.stdout

def main():
    data = json.loads((FOLDER/'dataset_audit.json').read_text())
    discovered = sorted(p.name for p in (REPO/'Data'/DATA_FOLDER).glob('CMIP6_*') if p.is_dir())
    assert sorted(data['datasets']) == discovered, 'Dataset audit is incomplete or inventory changed'
    included = sorted(k for k,v in data['datasets'].items() if v['included'])
    assert len(included) == 5 and 'CMIP6_Cane5' not in included, 'Expected five eligible datasets; Cane5 is excluded'
    assert data['group'] == GROUP and (data['train_ratio'], data['val_ratio']) == RATIOS
    rows = list(csv.DictReader((FOLDER/'manifest.csv').open()))
    assert Counter(r['dataset'] for r in rows) == Counter({k:2 for k in included})
    assert {(r['dataset'],r['variant']) for r in rows} == {(k,v) for k in included for v in VARIANTS}
    assert len({r['config_path'] for r in rows}) == len({r['result_path'] for r in rows}) == len(rows)
    assert sorted(str(p) for p in FOLDER.glob('train_config_*.sh')) == sorted(r['config_path'] for r in rows)
    assert set(ROLES) == {'overall','exceedance'}, 'Training must retain both checkpoint roles'
    for name in included:
        graph_root = REPO/'Data'/DATA_FOLDER/name/'graphs'
        files = data['datasets'][name]['files']
        assert sorted(p.name for p in graph_root.glob('*_Boston_*graphs.pt')) == sorted(f['name'] for f in files)
        for f in files:
            stat = (graph_root/f['name']).stat()
            assert (stat.st_size, stat.st_mtime_ns) == (f['bytes'], f['mtime_ns']), f'Data changed: {name}/{f["name"]}'
    source_cache, audit_rows, diffs, dry_logs = {}, [], [], []
    for row in rows:
        assert row['group'] == GROUP
        path = Path(row['config_path'])
        result = Path(row['result_path'])
        source = Path(row['template_path'])
        assert path.parent == FOLDER and path.resolve().parent == FOLDER.resolve()
        assert result.parent == RESULTS and result.resolve().parent == RESULTS.resolve()
        assert 'NCEP' not in path.name and 'NCEP' not in result.name
        subprocess.run(['bash','-n',str(path)], check=True)
        before, after = assignments(source), assignments(path)
        assert before.keys() == after.keys(), f'Added/removed config assignments: {path}'
        changes = {k:dict(template=before[k], quick_check=after[k]) for k in before if before[k] != after[k]}
        assert set(changes) == ASSIGNMENT_CHANGES, f'Unexpected assignment changes: {path}: {changes}'
        run_name = f'{row["dataset"]}_Boston_{GROUP}_G0_{row["variant"]}'
        assert after['ROOT_DIR'] == f'"./Data/{DATA_FOLDER}/{row["dataset"]}/graphs"'
        assert after['ALL_RESULTS_ROOT'] == f'"{RESULTS}"'
        assert after['PACT_RUN_NAME'] == after['PYTHON_RUN_TAG_BASE'] == f'"{run_name}"'
        assert after['CHECKPOINT_SELECTION'] == '"overall"'
        assert row['primary_checkpoint'] == 'overall'
        if str(source) not in source_cache:
            source_cache[str(source)] = dry_command(source, row['runstamp'])[0]
        original = source_cache[str(source)]
        resolved, dry_log = dry_command(path, row['runstamp'])
        changed_args = {k:dict(template=original[k], quick_check=resolved[k]) for k in original if original[k] != resolved[k]}
        assert set(changed_args) <= ARGUMENT_CHANGES, f'Unexpected CLI changes: {path}: {changed_args}'
        assert Path(resolved['output_dir']) == result, (resolved['output_dir'],result)
        assert f'TRAIN_DATA_TAG:{row["dataset"]}' in dry_log
        assert f'TEST_DATA_TAG: {row["dataset"]}' in dry_log
        assert resolved['root_dir'] == f'./Data/{DATA_FOLDER}/{row["dataset"]}/graphs'
        assert (resolved['exceedance_loss_weight'],resolved['episode_gt_aligned_peak_weight']) == VARIANTS[row['variant']]
        expected = dict(station='Boston',head_type='single',encoder_type='GraphSAGE',temporal_block='Transformer',
                        history_hours=24,hidden_channels=128,lr=.005,epochs=300,batch_size=256,grad_accum_steps=4,
                        seed=42,loss_mode='mse',scheduler='cosine',amp=True,amp_dtype='bf16',tf32=True,
                        checkpoint_selection='overall',train_ratio=RATIOS[0],val_ratio=RATIOS[1],shuffle_years=False,future_only=False)
        assert all(resolved[k] == value for k,value in expected.items()), f'Required settings mismatch: {path}'
        # Exercise the real splitter on the resolved config and compare exact year memberships.
        years = ['_'.join(f['name'].split('_')[:2]) for f in data['datasets'][row['dataset']]['files']]
        store = ForcingGraphStore.__new__(ForcingGraphStore)
        store.year_to_indices = {year:[i] for i,year in enumerate(years)}
        selected = store.split(**{key:resolved[key] for key in ('train_ratio','val_ratio','shuffle_years','seed','future_only','future_year_threshold')})
        split_years = {part:[years[i] for i in indices] for part,indices in selected.items()}
        assert split_years == {part:values['year_groups'] for part,values in data['datasets'][row['dataset']]['split'].items()}
        if GROUP == 'future_year':
            assert split_years['test'] == [f'{year}_{year+1}' for year in range(2070,2100)]
            assert all(int(year[:4]) < 2070 for part in ('train','val') for year in split_years[part])
        else:
            ncep_years = sorted('_'.join(p.name.split('_')[:2]) for p in (REPO/'Data/Grid4_New/NCEP/graphs').glob('*_Boston_*graphs.pt'))
            assert years == ncep_years
            ncep_store = ForcingGraphStore.__new__(ForcingGraphStore)
            ncep_store.year_to_indices = {year:[i] for i,year in enumerate(ncep_years)}
            assert selected == ncep_store.split(**{key:original[key] for key in ('train_ratio','val_ratio','shuffle_years','seed','future_only','future_year_threshold')})
        for other in audit_rows:
            if other['dataset'] == row['dataset']:
                pair_changes = {k for k in other['resolved_arguments'] if other['resolved_arguments'][k] != resolved[k]}
                assert pair_changes <= {'exceedance_loss_weight','episode_gt_aligned_peak_weight','run_tag','output_dir'}
        diffs.extend(difflib.unified_diff(source.read_text().splitlines(True),path.read_text().splitlines(True),fromfile=str(source),tofile=str(path)))
        dry_logs.append(dry_log)
        audit_rows.append(dict(**row, passed=True, config_sha256=digest(path), template_sha256=digest(source),
                               assignment_changes=changes, argument_changes=changed_args, resolved_arguments=resolved))
    report = dict(passed=True, group=GROUP, audited_at_utc=datetime.now(timezone.utc).isoformat(), config_count=len(rows),
                  configs_per_dataset=2, split_years=split_years, included=included, excluded={k:v['reasons'] for k,v in data['datasets'].items() if not v['included']},
                  checkpoint_roles=list(ROLES), primary_checkpoint='overall', results_root=str(RESULTS),
                  git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
                  code_sha256={name:digest(REPO/name) for name in ['train.py','train.sh','emulator/training/checkpoint_selection.py','emulator/training/losses.py','emulator/data/graph_store.py']}, rows=audit_rows)
    (FOLDER/'config_audit.json').write_text(json.dumps(report,indent=2,default=str)+'\n')
    (FOLDER/'template_diffs.patch').write_text(''.join(diffs))
    (FOLDER/'dry_run.log').write_text('\n'.join(dry_logs))
    print(f'PASS {GROUP}: {len(rows)} configs; exactly 2 per dataset; template and resolved-argument audit passed.')
    print('PASS split: '+ '; '.join(f'{part}={len(years)} groups ({years[0]} to {years[-1]})' for part,years in split_years.items()))
    print('PASS: both checkpoint roles retained; primary=overall; all config/result paths isolated from NCEP runs.')
    print('Config root: '+str(FOLDER))
    print('Result root: '+str(RESULTS))
    print('Dataset | Loss variant | Config path (relative to config root) | Result path (relative to result root)')
    print('--- | --- | --- | ---')
    for row in rows:
        print(' | '.join([row['dataset'],row['variant'],Path(row['config_path']).name,Path(row['result_path']).name]))

if __name__ == '__main__':
    main()
