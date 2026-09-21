#!/usr/bin/env python3
"""Paired 5e-3/3e-3 tables, including explicitly pending runs; no ranking or training."""
import argparse
import json
from pathlib import Path

import numpy as np

from pact_diagnostic_io import REPO, RESULTS, STATIONS, METHODS, discover, replay_station, align
from peak_branch_diagnosis import csv_write

KEYS=dict(AllRMSE='rmse_all',AllMAE='mae_all',Top5RMSE='rmse_peak5',Top5MAE='mae_peak5',
          PeakBias='true_peak_bias_top5',UnderPercent='true_peak_underprediction_fraction_top5',
          TruePeakRMSE='true_peak_rmse_top5',TimingSteps='peak_timing_mae_steps_top5')


def metrics(run, station, method, split, cache):
    result={name:run['summary'][split].get(key) for name,key in KEYS.items()}
    if result['UnderPercent'] is not None: result['UnderPercent']*=100
    result.update(PeakRMSE=None,PeakMAE=None,peak_maximum_source='unavailable; use --replay for VAL')
    path=run['prediction'] if split=='test' else cache/'arrays'/f'{station}_{method}_val.npz'
    if path.exists():
        if split=='val':
            from pact_diagnostic_io import sha
            meta=json.loads(path.with_suffix('.json').read_text())
            assert meta['inputs']['checkpoint_sha256']==sha(run['checkpoint'])
            assert meta['arrays_sha256']==sha(path)
        arrays=dict(np.load(path)); y,p=arrays['y_true'].astype(float),arrays['y_pred'].astype(float)
        top=np.argsort(y.max(axis=1).astype(np.float32),kind='quicksort')[-max(1,int(np.ceil(.05*len(y)))):]
        err=p.max(axis=1)[top]-y.max(axis=1)[top]
        result.update(PeakRMSE=float(np.sqrt(np.mean(err**2))),PeakMAE=float(np.mean(abs(err))),
                      peak_maximum_source='archived TEST predictions' if split=='test' else 'VAL checkpoint replay')
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo',type=Path,default=REPO)
    p.add_argument('--baseline-results',type=Path,default=RESULTS)
    p.add_argument('--new-results',type=Path,default=Path('/media/share/PACT/Results/All_results_0921_lr3e3'))
    p.add_argument('--baseline-cache',type=Path,default=Path('/home/exouser/peak_branch_diagnosis_0921'))
    p.add_argument('--output',type=Path,default=Path('/home/exouser/lr_comparison_0921'))
    p.add_argument('--replay',action='store_true',help='Evaluate completed new sweep checkpoints, never train')
    args=p.parse_args(); args.output.mkdir(exist_ok=True,parents=True)
    old,_=discover(args.baseline_results); new,excluded=discover(args.new_results,require_all=False)
    for key,run in old.items():
        if run['config']['lr'] != .005: raise ValueError(f'Expected frozen 5e-3 baseline for {key}')
    for key,run in new.items():
        if run['config']['lr'] != .003: raise ValueError(f'Expected 3e-3 new run for {key}')
        original=old[key]['config']; current=run['config']
        changes={k for k in original.keys()|current.keys() if original.get(k)!=current.get(k)}
        if changes-{'lr','run_tag','output_dir'}:
            raise ValueError(f'Non-LR differences in completed pair {key}: {changes}')
    if args.replay:
        for station in STATIONS:
            if all((station,m) in new for m in METHODS): replay_station(args.repo,station,new,args.output)
            elif any(s==station for s,m in new): raise ValueError(f'Complete all five {station} runs before paired replay')
    pairs,curves=[] ,[]
    for station in STATIONS:
        for method in METHODS:
            key=station,method
            for split in ('val','test'):
                a=metrics(old[key],station,method,split,args.baseline_cache)
                b=metrics(new[key],station,method,split,args.output) if key in new else {}
                row=dict(station=station,method=method,split=split,status='complete' if key in new else 'pending_3e3',
                         baseline_run=str(old[key]['directory']),new_run=str(new[key]['directory']) if key in new else '')
                for metric in (*KEYS,'PeakRMSE','PeakMAE'):
                    row[metric+'_5e3'],row[metric+'_3e3']=a[metric],b.get(metric)
                    row[metric+'_delta_3e3_minus_5e3']=b[metric]-a[metric] if b.get(metric) is not None and a[metric] is not None else None
                row['PeakRMSE_source_5e3'],row['PeakRMSE_source_3e3']=a['peak_maximum_source'],b.get('peak_maximum_source')
                pairs.append(row)
            for rate,runs in [(.005,old),(.003,new)]:
                row=dict(station=station,method=method,lr=rate,status='complete' if key in runs else 'pending')
                if key in runs:
                    r=runs[key]; late=[e for e in r['epochs'] if 250<=e['epoch']<=300]
                    row.update(best_epoch=r['summary']['best_epoch'],
                        epoch_min_Val_AllRMSE=min(r['epochs'],key=lambda e:e['val']['rmse_all'])['epoch'],
                        late_epoch_start=250,late_epoch_end=300,late_epoch_count=len(late))
                    for label,k in [('AllRMSE','rmse_all'),('TruePeakRMSE','true_peak_rmse_top5')]:
                        v=[e['val'][k] for e in late if e['val'].get(k) is not None]
                        row['late_Val_'+label+'_mean']=float(np.mean(v)) if v else None
                        row['late_Val_'+label+'_std']=float(np.std(v)) if v else None
                curves.append(row)
    csv_write(args.output/'paired_metrics.csv',pairs); csv_write(args.output/'learning_curves.csv',curves)
    (args.output/'README.md').write_text('All amplitude metrics are meters; UnderPercent is percentage points and TimingSteps is steps. '
        'Delta = 3e-3 minus 5e-3. Empty new-run columns mean not completed, not zero. '
        'All/Top5 and true-peak metrics use original final checkpoint summaries. PeakRMSE/MAE use free window maxima on the same GT top5 subset; '
        'TEST uses original exports, VAL requires evaluation-only replay. PeakBias/Under/TruePeakRMSE/TimingSteps use first GT argmax on top5 windows, '
        'matching the current trainer display. Late epochs are inclusive 250–300, std ddof=0. No LR conclusions are generated.\n')
    print(f'{len(new)}/10 LR3e3 runs completed; paired tables written to {args.output}')


if __name__=='__main__': main()
