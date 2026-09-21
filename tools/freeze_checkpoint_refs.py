#!/usr/bin/env python3
"""Freeze saved S0 checkpoint-epoch VAL metrics. No TEST input or inference."""
import argparse
from pathlib import Path
import json
import torch

from pact_diagnostic_io import REPO, RESULTS, STATIONS, discover, sha, write_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results',type=Path,default=RESULTS)
    p.add_argument('--output',type=Path,default=REPO/'checkpoint_score_refs.json')
    args=p.parse_args(); runs,_=discover(args.results)
    doc={'_meta':dict(schema_version=1,split='val',metric_population='top5',
        source='saved checkpoint epoch validation metrics; S0 best overall at LR=5e-3',
        metric_keys=dict(all_rmse='rmse_all',top5_rmse='rmse_peak5',true_peak_rmse='true_peak_rmse_top5'),
        sources={})}
    for station in STATIONS:
        r=runs[(station,'S0')]; c=torch.load(r['checkpoint'],map_location='cpu',weights_only=False)
        assert r['config']['lr']==.005 and c['epoch']==r['summary']['best_epoch']
        assert c['selection_metric_key']=='rmse_all'
        doc[station]={k:c['val'][v] for k,v in doc['_meta']['metric_keys'].items()}
        assert all(v>0 for v in doc[station].values())
        doc['_meta']['sources'][station]=dict(checkpoint=str(r['checkpoint']),sha256=sha(r['checkpoint']),
                                              epoch=c['epoch'],run_tag=r['config']['run_tag'])
    if args.output.exists() and json.loads(args.output.read_text())!=doc:
        raise ValueError('Frozen references already exist and differ; use a separate experiment file')
    write_json(args.output,doc)
    print(json.dumps({s:doc[s] for s in STATIONS},indent=2))


if __name__=='__main__': main()
