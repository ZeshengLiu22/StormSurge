#!/usr/bin/env python3
"""Prepare a future checkpoint-policy experiment from one chosen config. Never launches."""
import argparse
from pathlib import Path
import re
import shlex

from pact_diagnostic_io import REPO


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--selection',choices=['overall','peakaware'],default='peakaware')
    parser.add_argument('--refs',type=Path,default=REPO/'checkpoint_score_refs.json')
    parser.add_argument('--result-dir',type=Path,default=Path('/media/share/PACT/Results/All_results_0921_peakaware'))
    args=parser.parse_args()
    text=args.parent.read_text()
    name=re.search(r'^PACT_RUN_NAME=(.+)$',text,re.M)
    if not name: raise ValueError('Parent must explicitly define PACT_RUN_NAME')
    new_name=shlex.split(name.group(1))[0]+'_PeakAware'
    changes=dict(CHECKPOINT_SELECTION=args.selection,TRACK_PEAKAWARE='1',SAVE_AUX_CHECKPOINTS='0',
        CHECKPOINT_SCORE_REFS=str(args.refs.resolve()),CKPT_SCORE_W_ALL='0.65',CKPT_SCORE_W_TOP5='0.20',
        CKPT_SCORE_W_TRUEPEAK='0.15',ALL_RESULTS_ROOT=str(args.result_dir),
        PACT_RUN_NAME=new_name,PYTHON_RUN_TAG_BASE=new_name)
    for key,value in changes.items():
        replacement=f'{key}={shlex.quote(value)}'
        if re.search(r'^'+key+'=',text,re.M):
            text=re.sub(r'^'+key+r'=.*$',lambda _:replacement,text,flags=re.M)
        else: text+='\n'+replacement+'\n'
    if args.output.exists(): raise ValueError('Use a new output config path; existing files are never overwritten')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(text)
    print(f'Prepared future run {new_name}; LR and losses copied unchanged; no training launched.')
    print('Dry run: DRY_RUN=1 USE_TMUX=0 bash train.sh '+shlex.quote(str(args.output)))
    print('Future explicit launch: USE_TMUX=0 bash train.sh '+shlex.quote(str(args.output)))


if __name__=='__main__': main()
