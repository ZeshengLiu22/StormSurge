"""Optional second checkpoint tracker. Does not change the legacy selector or losses."""
import csv
import hashlib
import json
import math
from pathlib import Path

import torch

from .checkpoints import atomic_save

TERMS = ('rmse_all', 'rmse_peak5', 'true_peak_rmse_top5')
REF_KEYS = ('all_rmse', 'top5_rmse', 'true_peak_rmse')
DEFAULT_WEIGHTS = (.65, .20, .15)
DEFAULT_REFS = Path(__file__).resolve().parents[2] / 'checkpoint_score_refs.json'


def enabled(args):
    return args.checkpoint_selection == 'peakaware' or bool(getattr(args, 'track_peakaware', 0))


def read_settings(args):
    """Freeze station VAL references in memory once; never accept a TEST metric."""
    if args.checkpoint_selection not in ('overall', 'peakaware'):
        raise ValueError('Peak-aware tracking is isolated to overall or peakaware selection.')
    if args.save_aux_checkpoints:
        raise ValueError('The optional peak-aware tracker uses its own two artifacts; leave save_aux_checkpoints=0.')
    weights = tuple(getattr(args, 'ckpt_score_w_'+name, default)
                    for name, default in zip(('all','top5','truepeak'), DEFAULT_WEIGHTS))
    if any(not math.isfinite(v) or v < 0 for v in weights) or not math.isclose(sum(weights), 1., abs_tol=1e-12):
        raise ValueError('Checkpoint score weights must be finite, nonnegative, and sum to one.')
    path = Path(getattr(args, 'checkpoint_score_refs', DEFAULT_REFS))
    try:
        raw = path.read_bytes()
        document = json.loads(raw)
        refs = tuple(float(document[args.station][key]) for key in REF_KEYS)
    except (OSError, KeyError, ValueError, TypeError) as error:
        raise ValueError(f'Cannot read frozen checkpoint references for {args.station}: {path}') from error
    if document.get('_meta', {}).get('split') != 'val':
        raise ValueError('Reference file must explicitly declare _meta.split=val.')
    if document.get('_meta', {}).get('metric_keys') != dict(zip(REF_KEYS, TERMS)):
        raise ValueError('Reference metric keys must match the existing top5 true-peak definition.')
    if any(not math.isfinite(v) or v <= 0 for v in refs):
        raise ValueError('Checkpoint reference values must be finite and strictly positive.')
    return dict(station=args.station, weights=dict(zip(TERMS, weights)),
                references=dict(zip(TERMS, refs)), reference_path=str(path.resolve()),
                reference_sha256=hashlib.sha256(raw).hexdigest(),
                reference_provenance=document['_meta'])


def normalized_score(val, settings):
    values = [val.get(key) for key in TERMS]
    if any(v is None or not math.isfinite(v) or v < 0 for v in values):
        raise ValueError('Peak-aware score requires finite nonnegative VAL All/Top5/TruePeak RMSE.')
    score = sum(settings['weights'][key] * (val[key] / settings['references'][key]) for key in TERMS)
    if not math.isfinite(score):
        raise ValueError('Nonfinite normalized checkpoint score; check the frozen reference scales.')
    return score


class PeakAwareTracker:
    """Rank-zero scalar bookkeeping and two durable aliases; strict < retains earliest ties."""
    def __init__(self, output, settings):
        self.output, self.settings = Path(output), settings
        self.best = {'overall': None, 'peakaware': None}
        self.paths = {role: self.output/f'best_{role}.pt' for role in self.best}
        if any(p.exists() for p in self.paths.values()):
            raise ValueError('Peak-aware tracking requires a fresh run output directory.')
        self.last_epoch = 0

    def observe(self, epoch, val, checkpoint_factory):
        if epoch <= self.last_epoch:
            raise ValueError('Peak-aware validation epochs must increase strictly.')
        score = normalized_score(val, self.settings)
        candidate = dict(epoch=epoch, val=dict(val), score=score)
        improved = []
        for role in self.best:
            old = self.best[role]
            value = val['rmse_all'] if role == 'overall' else score
            previous = (old['val']['rmse_all'] if role == 'overall' else old['score']) if old else math.inf
            if value < previous:
                checkpoint = checkpoint_factory()
                metric = 'rmse_all' if role == 'overall' else 'peakaware_score'
                checkpoint.update(checkpoint_role=role, checkpoint_roles=[role], checkpoint_candidate=False,
                    checkpoint_selection_mode=role, selection_metric_key=metric, selection_metric_value=value,
                    selection_by_role={role:dict(selection_metric_key=metric,selection_metric_value=value)},
                    peakaware_score=score, peakaware_settings=self.settings)
                atomic_save(checkpoint, self.paths[role])
                self.best[role] = candidate
                improved.append(role)
        self.last_epoch = epoch
        return score, improved

    def selection_summary(self, original, mode):
        selected = self.best[mode]
        return dict(original, checkpoint_selection_mode=mode,
                    checkpoint_selection_metric_key='rmse_all' if mode=='overall' else 'peakaware_score',
                    selected_epoch=selected['epoch'], selected_val_rmse_all=selected['val']['rmse_all'],
                    selected_val_rmse_peak5=selected['val']['rmse_peak5'],
                    selected_val_true_peak_rmse_top5=selected['val']['true_peak_rmse_top5'],
                    selected_peakaware_score=selected['score'], best_peakaware_epoch=self.best['peakaware']['epoch'],
                    peakaware_settings=self.settings,
                    peakaware_val_all_degradation_fraction=self.degradation() if math.isfinite(self.degradation()) else None,
                    peakaware_val_all_degradation_gt_1pct=self.exceeds_guardrail())

    def exceeds_guardrail(self):
        # Direct comparison avoids flagging exactly 1% because ratio-1 rounded up.
        return self.best['peakaware']['val']['rmse_all'] > 1.01*self.best['overall']['val']['rmse_all']

    def degradation(self):
        overall = self.best['overall']['val']['rmse_all']
        peak = self.best['peakaware']['val']['rmse_all']
        return peak/overall-1 if overall else (0. if peak==0 else math.inf)

    def compare(self, network, selected_role, primary_results, evaluate, log):
        """Post-training full VAL/TEST re-evaluation only; never feeds selection."""
        other = 'peakaware' if selected_role == 'overall' else 'overall'
        reports = {selected_role: primary_results}
        selected_path = self.paths[selected_role]
        if self.best[other]['epoch'] == self.best[selected_role]['epoch']:
            reports[other] = primary_results
        else:
            state = torch.load(self.paths[other], map_location='cpu', weights_only=False)
            network.load_state_dict(state['model_state'], strict=True)
            try:
                reports[other] = {split:evaluate(split) for split in ('val','test')}
            finally:
                state = torch.load(selected_path, map_location='cpu', weights_only=False)
                network.load_state_dict(state['model_state'], strict=True)
        columns = dict(AllRMSE='rmse_all', Top5RMSE='rmse_peak5', TruePeakRMSE='true_peak_rmse_top5',
                       PeakBias='true_peak_bias_top5', UnderPercent='true_peak_underprediction_fraction_top5')
        rows=[]
        for split in ('val','test'):
            row = dict(split=split, overall_epoch=self.best['overall']['epoch'], peakaware_epoch=self.best['peakaware']['epoch'])
            for label,key in columns.items():
                factor=100 if label=='UnderPercent' else 1
                a,b=reports['overall'][split].get(key),reports['peakaware'][split].get(key)
                a,b=a*factor if a is not None else None,b*factor if b is not None else None
                row[label+'_overall'],row[label+'_peakaware']=a,b
                row['Delta_'+label]=b-a if a is not None and b is not None else None
            rows.append(row)
        path=self.output/'checkpoint_policy_comparison.csv'
        with path.open('w',newline='') as handle:
            writer=csv.DictWriter(handle,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
        document=dict(selection_uses='VAL ONLY', selected_role=selected_role, settings=self.settings,
            checkpoints={role:dict(self.best[role], path=str(self.paths[role])) for role in self.best},
            reevaluated=reports, deltas=rows,
            selection_val_all_degradation_fraction=self.degradation() if math.isfinite(self.degradation()) else None,
            selection_val_all_degradation_gt_1pct=self.exceeds_guardrail(),
            reevaluated_val_all_degradation_gt_1pct=reports['peakaware']['val']['rmse_all']>1.01*reports['overall']['val']['rmse_all'])
        (self.output/'checkpoint_policy_comparison.json').write_text(json.dumps(document,indent=2,allow_nan=False)+'\n')
        log('CHECKPOINT POLICY COMPARISON: values are overall / peakaware / delta (peakaware-overall)')
        log('Split  '+ '  '.join(columns))
        for row in rows:
            def display(key):
                return 'NA' if row[key] is None else f'{row[key]:+.6f}'
            log(row['split'].upper()+'  '+'  '.join(' / '.join(display(key) for key in
                (k+'_overall',k+'_peakaware','Delta_'+k)) for k in columns))
        log(f'VAL AllRMSE degradation >1%: {self.exceeds_guardrail()} (selection-epoch metrics); no automatic fallback')
        return document
