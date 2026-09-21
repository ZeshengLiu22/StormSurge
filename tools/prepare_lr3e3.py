#!/usr/bin/env python3
"""Copy exact 0921 parents, or validate every resolved LR-only training argument."""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

from pact_diagnostic_io import REPO, RESULTS, METHODS, STATIONS, discover, sha, write_json

ALLOWED = {'lr', 'run_tag', 'output_dir'}
SHELL_ALLOWED = {'LR_LIST', 'ALL_RESULTS_ROOT', 'PACT_RUN_NAME', 'PYTHON_RUN_TAG_BASE'}


def assignments(text):
    return {line.split('=',1)[0]:line.split('=',1)[1] for line in text.splitlines()
            if line.strip() and not line.startswith('#') and '=' in line}


def validate(repo, destination, results):
    sys.path.insert(0,str(repo))
    from emulator.training.arguments import parse_args
    manifest=json.loads((destination/'parents.json').read_text())
    rows=[]
    expected={item['config'] for item in manifest}
    actual={p.name for p in (destination/'configs').glob('*.sh')}
    if actual != expected or len(expected)!=10: raise ValueError('The LR sweep must contain exactly the ten paired configs')
    for item in manifest:
        parent=Path(item['parent_shell']); new=destination/'configs'/item['config']
        if sha(parent)!=item['parent_shell_sha256']: raise ValueError(f'Parent changed: {parent}')
        if sha(item['parent_json'])!=item['parent_json_sha256']: raise ValueError('Frozen parent JSON changed')
        p,n=assignments(parent.read_text()),assignments(new.read_text())
        changes={k for k in p.keys()|n.keys() if p.get(k)!=n.get(k)}
        if changes != SHELL_ALLOWED: raise ValueError(f'Unexpected shell config changes: {changes}')
        # Check complete non-comment lines too: assignments alone must not conceal a command or source.
        def science_lines(text):
            return [s for s in text.splitlines() if s.strip() and not s.lstrip().startswith('#')
                    and s.split('=',1)[0] not in SHELL_ALLOWED]
        if science_lines(parent.read_text()) != science_lines(new.read_text()):
            raise ValueError('Unexpected non-assignment shell change')
        result=subprocess.run(['bash','train.sh',str(new)],cwd=repo,env=dict(os.environ,DRY_RUN='1',USE_TMUX='0'),
                              text=True,capture_output=True,check=True)
        commands=[shlex.split(s[5:])[1:] for s in result.stdout.splitlines() if s.startswith('CMD: ')]
        if len(commands)!=1: raise ValueError('Config must resolve to one command')
        resolved=json.loads(json.dumps(vars(parse_args(commands[0])),default=str))
        old=json.loads(Path(item['parent_json']).read_text())
        differences={k:dict(parent=old.get(k),new=resolved.get(k)) for k in old.keys()|resolved.keys()
                     if old.get(k)!=resolved.get(k)}
        if set(differences)-ALLOWED: raise ValueError(f'Non-LR scientific difference in {new}: {differences}')
        assert old['lr']==.005 and resolved['lr']==.003
        assert resolved['epochs']==300 and resolved['min_lr']==1e-6
        assert resolved['checkpoint_selection']=='overall' and resolved['save_aux_checkpoints']==0
        assert not resolved.get('track_peakaware',0)
        assert Path(resolved['output_dir']).parent.resolve()==results.resolve()
        (destination/'validation'/f'{item["run_name"]}.dry_run.txt').write_text(
            '\n'.join(line.rstrip() for line in result.stdout.splitlines()) + '\n')
        write_json(destination/'validation'/f'{item["run_name"]}.resolved.json',resolved)
        rows.append(dict(run_name=item['run_name'],status='PASS',shell_changed=sorted(changes),
                         resolved_differences=differences,parent_sha256=sha(parent),config_sha256=sha(new)))
    write_json(destination/'validation'/'config_diff.json',dict(status='PASS',count=len(rows),runs=rows,
        only_scientific_change='lr: 0.005 -> 0.003',training_launched=False))
    print('PASS: 10/10 exact pairs; LR is the only scientific change; overall-only selection; no training')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo',type=Path,default=REPO)
    p.add_argument('--baseline-results',type=Path,default=RESULTS)
    p.add_argument('--config-dir',type=Path)
    p.add_argument('--result-dir',type=Path,default=Path('/media/share/PACT/Results/All_results_0921_lr3e3'))
    p.add_argument('--prepare',action='store_true',help='Create copies; otherwise only validate')
    args=p.parse_args(); dest=args.config_dir or args.repo/'experiment_config_0921_lr3e3'
    if args.prepare:
        runs,_=discover(args.baseline_results); manifest=[]
        for folder in ('configs','validation'): (dest/folder).mkdir(parents=True,exist_ok=True)
        for station in STATIONS:
            for method in METHODS:
                run=runs[(station,method)]
                old_name=run['config']['run_tag'].rsplit('_',2)[0]
                parent=args.repo/'experiment_config_0921_true_peak_amp_factorial'/'configs'/f'train_config_{old_name}.sh'
                new_name=old_name+'_LR3e3'
                original=parent.read_text()
                assert 'LR_LIST=("5e-3")' in original
                updated=original.replace('LR_LIST=("5e-3")','LR_LIST=("3e-3")')
                updated=updated.replace('ALL_RESULTS_ROOT="/media/share/PACT/Results/All_results_0921_true_peak_amp_factorial"',
                                        f'ALL_RESULTS_ROOT="{args.result_dir}"')
                for key in ('PACT_RUN_NAME','PYTHON_RUN_TAG_BASE'):
                    updated=updated.replace(f'{key}="{old_name}"',f'{key}="{new_name}"')
                updated=updated.replace('# Formal matched factorial:', '# LR-only copy of the frozen factorial:')
                path=dest/'configs'/f'train_config_{new_name}.sh'
                if path.exists() and path.read_text()!=updated: raise ValueError(f'Refusing to overwrite edited config {path}')
                path.write_text(updated)
                manifest.append(dict(run_name=new_name,config=path.name,parent_shell=str(parent),
                    parent_shell_sha256=sha(parent),parent_json=str(run['config_path']),
                    parent_json_sha256=sha(run['config_path']),parent_run=str(run['directory'])))
        write_json(dest/'parents.json',manifest)
    validate(args.repo,dest,args.result_dir)


if __name__=='__main__': main()
