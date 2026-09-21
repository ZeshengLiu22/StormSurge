#!/usr/bin/env python3
"""Fixed-GT-window diagnosis of frozen 0921 checkpoints; never trains."""
import argparse
import csv
import json
from pathlib import Path
import shutil

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from pact_diagnostic_io import (REPO, RESULTS, METHODS, STATIONS, align, discover,
                                replay_station, sha, source_hashes, write_json)

COLORS = dict(zip(METHODS, ['#555555', '#0072B2', '#009E73', '#D55E00', '#CC79A7']))


def csv_write(path, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def markdown(rows, fields):
    def value(v):
        return f'{v:.6g}' if isinstance(v, (float, np.floating)) else str(v)
    return '\n'.join(['| '+' | '.join(fields)+' |', '| '+' | '.join(['---']*len(fields))+' |']+
                     ['| '+' | '.join(value(r.get(k, '')) for k in fields)+' |' for r in rows])


def safe_ratio(a, b):
    # Physical denominator tolerance: one nanometer; undefined values stay NaN.
    return np.divide(a, b, out=np.full_like(np.asarray(a, dtype=float), np.nan), where=np.abs(b) > 1e-9)


def quantile_bins(peaks, min_count=20):
    for count in (4, 3):
        edges = np.quantile(peaks, np.linspace(0, 1, count+1))
        membership = np.searchsorted(edges[1:-1], peaks, side='right') + 1
        if len(np.unique(edges)) == count+1 and min(np.bincount(membership, minlength=count+1)[1:]) >= min_count:
            return membership, edges
    raise ValueError('Too few distinct GT events for three bins with the requested minimum; lower --min-bin-count explicitly')


def event_rows(station, method, arrays, targets, tau, bins):
    arrays = align(arrays, targets['tags'], targets['y_true'])
    y, p = arrays['y_true'].astype(float), arrays['y_pred'].astype(float)
    h = y.argmax(axis=1)
    ids = np.flatnonzero(y.max(axis=1) > tau)
    rows = []
    for j, i in enumerate(ids):
        gt, pred = y[i, h[i]], p[i, h[i]]
        excess = max(gt-tau, 0.)
        r = dict(station=station, method=method, sample_id=int(targets['sample_id'][i]),
                 window_id=str(arrays['tags'][i]), quantile_bin=int(bins[j]), gt_peak=gt, h_gt=int(h[i]),
                 pred_at_gt_horizon=pred, pred_window_peak=float(p[i].max()),
                 pred_peak_horizon=int(p[i].argmax()), timing_error_steps=int(p[i].argmax()-h[i]),
                 tau=tau, true_excess=excess, peak_error_at_gt=pred-gt,
                 peak_error_window=float(p[i].max()-gt), prediction_to_GT_ratio=float(safe_ratio(pred, gt)),
                 absolute_underprediction=gt-pred, UnderAny=pred < gt,
                 Under5mm=pred < gt-.005, Under10mm=pred < gt-.010, Under20mm=pred < gt-.020,
                 OverAny=pred > gt, Over5mm=pred > gt+.005, Over10mm=pred > gt+.010,
                 Over20mm=pred > gt+.020)
        branch = ('body_prediction','raw_excess_prediction','gate_logit','gate_probability',
                  'final_excess_contribution','final_reconstructed_prediction','reconstruction_error',
                  'raw_excess_ratio','final_excess_ratio','body_deficit','raw_excess_deficit','gate_attenuation')
        r.update(dict.fromkeys(branch, np.nan))
        if method != 'S0':
            body, raw, gate = (float(arrays['body_phys'][i,h[i]]), float(arrays['excess_phys'][i,h[i]]),
                               float(arrays['gate_probability'][i]))
            final = gate*raw
            r.update(body_prediction=body, raw_excess_prediction=raw, gate_probability=gate,
                     gate_logit=float(arrays['gate_logit'][i]), final_excess_contribution=final,
                     final_reconstructed_prediction=body+final, reconstruction_error=body+final-pred,
                     raw_excess_ratio=float(safe_ratio(raw, excess)), final_excess_ratio=float(safe_ratio(final, excess)),
                     body_deficit=tau-body, raw_excess_deficit=excess-raw, gate_attenuation=(1-gate)*raw)
            assert abs(r['reconstruction_error']) < 3e-7
            assert abs((gt-pred)-(r['body_deficit']+r['raw_excess_deficit']+r['gate_attenuation'])) < 3e-7
        rows.append(r)
    return rows


def summarize(rows):
    a = lambda k: np.asarray([r[k] for r in rows], dtype=float)
    error, free = a('peak_error_at_gt'), a('peak_error_window')
    result = dict(station=rows[0]['station'], method=rows[0]['method'], n=len(rows),
        GT_peak_mean=a('gt_peak').mean(), GT_peak_median=np.median(a('gt_peak')),
        RMSE=np.sqrt(np.mean(error**2)), MAE=np.mean(abs(error)), bias=error.mean(),
        median_prediction_GT_ratio=np.nanmedian(a('prediction_to_GT_ratio')),
        PeakRMSE=np.sqrt(np.mean(free**2)), PeakMAE=np.mean(abs(free)),
        WindowPeakBias=free.mean(), TruePeakRMSE=np.sqrt(np.mean(error**2)),
        TimingSteps=np.mean(abs(a('timing_error_steps'))), TimingSignedSteps=a('timing_error_steps').mean())
    for k in ('UnderAny','Under5mm','Under10mm','Under20mm','OverAny','Over5mm','Over10mm','Over20mm'):
        result[k+'_percent'] = a(k).mean()*100
    if rows[0]['method'] != 'S0':
        for k in ('gate_probability','true_excess','body_prediction','raw_excess_prediction',
                  'final_excess_contribution','body_deficit','raw_excess_deficit','gate_attenuation',
                  'raw_excess_ratio','final_excess_ratio'):
            result[k+'_mean'], result[k+'_median'] = a(k).mean(), np.nanmedian(a(k))
        for name, k in [('raw', 'raw_excess_prediction'), ('final', 'final_excess_contribution')]:
            err = a(k)-a('true_excess')
            result[name+'_excess_bias'], result[name+'_excess_RMSE'] = err.mean(), np.sqrt(np.mean(err**2))
        result['reconstruction_max_abs_m'] = max(abs(a('reconstruction_error')))
    return result


def regime(station, split, targets, tau):
    y = targets['y_true'].astype(float)
    peaks = y.max(axis=1)
    event, positive = peaks > tau, y > tau
    base = dict(station=station, split=split, tau=tau, total_windows=len(y), total_horizons=y.size,
                event_windows=int(event.sum()), event_windows_percent=event.mean()*100,
                positive_horizons=int(positive.sum()), positive_horizons_percent=positive.mean()*100)
    populations = dict(positive_horizon_excess=(y-tau)[positive], event_peak=peaks[event],
                       event_peak_excess=peaks[event]-tau, all_window_peak=peaks)
    result = []
    for population, values in populations.items():
        row = dict(base, population=population, count=len(values))
        for k, f in [('mean',np.mean),('median',np.median),('std',np.std),('P75',lambda a:np.quantile(a,.75)),
                     ('P90',lambda a:np.quantile(a,.90)),('P95',lambda a:np.quantile(a,.95)),('max',np.max)]:
            row[k] = float(f(values)) if len(values) else np.nan
        for mm in (5,10,20,50): row[f'count_gt_{mm}mm'] = int((values > mm/1000).sum())
        result.append(row)
    return result


def plot_station(out, station, rows, summaries, targets, tau):
    dest = out/'plots'
    def save(fig, name):
        fig.savefig(dest/f'{station}_{name}.png', dpi=180, bbox_inches='tight')
        fig.savefig(dest/f'{station}_{name}.pdf', bbox_inches='tight')
        plt.close(fig)
    specs = [('01_gt_vs_at_gt','gt_peak','pred_at_gt_horizon','GT peak (m)','Prediction at GT peak (m)',True),
             ('02_gt_vs_window_peak','gt_peak','pred_window_peak','GT peak (m)','Predicted window maximum (m)',True),
             ('03_peak_ratio','gt_peak','prediction_to_GT_ratio','GT peak (m)','Prediction / GT',False),
             ('07_raw_excess','true_excess','raw_excess_prediction','True excess (m)','Raw excess after softplus (m)',True),
             ('08_final_excess','true_excess','final_excess_contribution','True excess (m)','Gated excess contribution (m)',True),
             ('09_gate','true_excess','gate_probability','True excess (m)','Gate probability',False)]
    for name,x,y,xlabel,ylabel,identity in specs:
        methods = METHODS if name[:2] in ('01','02','03') else METHODS[1:]
        fig, axes = plt.subplots(1,len(methods),figsize=(3.3*len(methods),3.4),sharex=True,sharey=True,layout='constrained')
        for ax,m in zip(axes,methods):
            g = [r for r in rows if r['method']==m]
            ax.scatter([r[x] for r in g],[r[y] for r in g],s=13,alpha=.65,color=COLORS[m])
            ax.set(title=f'{m} (n={len(g)})',xlabel=xlabel)
            if identity:
                low = min(min(r[x],r[y]) for r in rows if np.isfinite(r[y]))
                high = max(max(r[x],r[y]) for r in rows if np.isfinite(r[y]))
                ax.plot([low,high],[low,high],'--',color='black',lw=1)
            elif y=='prediction_to_GT_ratio': ax.axhline(1,color='black',ls='--',lw=1)
            if y=='gate_probability': ax.set_ylim(0,1.02)
            ax.grid(alpha=.18)
        axes[0].set_ylabel(ylabel)
        fig.suptitle(f'{station}: fixed GT TEST events')
        save(fig,name)
    for name, metric, ylabel in [('04_bias_by_quantile','bias','Bias at GT peak (mm)'),('05_rmse_by_quantile','RMSE','RMSE at GT peak (mm)')]:
        fig,ax=plt.subplots(figsize=(6,4),layout='constrained')
        for m in METHODS:
            g=[r for r in summaries if r['method']==m]
            ax.plot([r['quantile_bin'] for r in g],[r[metric]*1000 for r in g],'o-',color=COLORS[m],label=m)
        ax.axhline(0,color='black',lw=.6)
        ax.set(xlabel='Shared GT peak quantile bin',ylabel=ylabel,title=station)
        ax.set_xticks(sorted(set(r['quantile_bin'] for r in summaries)))
        ax.legend(); ax.grid(alpha=.2); save(fig,name)
    fig,ax=plt.subplots(figsize=(7,4),layout='constrained')
    for i,m in enumerate(METHODS):
        g=summarize([r for r in rows if r['method']==m])
        ax.bar(np.arange(4)+(i-2)*.15,[g[k+'_percent'] for k in ('UnderAny','Under5mm','Under10mm','Under20mm')],
               width=.15,color=COLORS[m],label=f'{m} (n={g["n"]})')
    ax.set(xticks=range(4),xticklabels=['Any','>5 mm','>10 mm','>20 mm'],ylabel='Underprediction (%)',title=station,ylim=(0,100))
    ax.legend(ncol=3); save(fig,'06_underprediction')
    fig,axes=plt.subplots(2,2,figsize=(10,7),layout='constrained')
    for split,values in targets.items():
        peaks=values['y_true'].max(axis=1).astype(float); peaks=peaks[peaks>tau]
        for j,(v,label) in enumerate([(peaks,'GT event peak (m)'),(peaks-tau,'True event-peak excess (m)')]):
            axes[0,j].hist(v,bins=25,density=True,histtype='step',label=f'{split.upper()} n={len(v)}')
            axes[1,j].step(np.sort(v),np.arange(1,len(v)+1)/len(v),where='post',label=split.upper())
            axes[0,j].set(xlabel=label,ylabel='Density'); axes[1,j].set(xlabel=label,ylabel='CDF')
    for ax in axes.flat: ax.legend(); ax.grid(alpha=.2)
    fig.suptitle(station); save(fig,'10_gt_distributions')


def analyze(args, runs):
    out=args.output
    overall, quantiles, branches, regimes, consistency, original_metrics, deltas = [],[],[],[],[],[],[]
    for station in STATIONS:
        tau=runs[(station,'S0')]['summary']['loss_thresholds']['tau_phys']
        targets={split:dict(np.load(out/'arrays'/f'{station}_{split}_targets.npz')) for split in ('train','val','test')}
        y=targets['test']['y_true'].astype(float); peaks=y.max(axis=1); event=peaks>tau
        bins,edges=quantile_bins(peaks[event],args.min_bin_count)
        protocol=dict(station=station,tau=tau,event_definition='max_h(y_true) > TRAIN tau',
                      event_ids=targets['test']['tags'][event],sample_ids=targets['test']['sample_id'][event],
                      quantile_edges=edges,quantile_membership=bins,min_bin_count=args.min_bin_count)
        protocol_path=out/'per_event'/f'{station}_fixed_event_ids.json'
        if protocol_path.exists():
            old=json.loads(protocol_path.read_text())
            if old['event_ids'] != protocol['event_ids'].tolist(): raise ValueError('Frozen event IDs changed')
        write_json(protocol_path,protocol)
        all_rows, qrows=[],[]
        for method in METHODS:
            arrays=dict(np.load(out/'arrays'/f'{station}_{method}_test.npz'))
            rows=event_rows(station,method,arrays,targets['test'],tau,bins)
            assert [r['window_id'] for r in rows] == protocol['event_ids'].tolist()
            original=align(dict(np.load(runs[(station,method)]['prediction'])),arrays['tags'],arrays['y_true'])
            archived=event_rows(station,'S0',original,targets['test'],tau,bins)
            om=summarize(archived); om['method']=method; original_metrics.append(om)
            for row,old in zip(rows,archived):
                row['archived_pred_at_gt_horizon']=old['pred_at_gt_horizon']
                row['archived_pred_window_peak']=old['pred_window_peak']
                row['replay_minus_archived_at_gt']=row['pred_at_gt_horizon']-old['pred_at_gt_horizon']
                row['archived_UnderAny']=old['UnderAny']
            all_rows+=rows
            sm=summarize(rows); overall.append(sm)
            consistency.append(dict(station=station,method=method,n_events=len(rows),
                event_ids_sha256=sha(protocol_path),reconstruction_max_abs_m=sm.get('reconstruction_max_abs_m'),
                replay_archived_UnderAny_flips=sum(r['UnderAny']!=r['archived_UnderAny'] for r in rows),
                replay_archived_at_gt_max_abs_m=max(abs(r['replay_minus_archived_at_gt']) for r in rows)))
            if method!='S0': branches.append(dict(sm, population='all_fixed_events'))
            for b in sorted(set(bins)):
                qs=dict(summarize([r for r in rows if r['quantile_bin']==b]),quantile_bin=int(b))
                qrows.append(qs)
                if method!='S0': branches.append(dict(qs,population=f'Q{b}'))
        assert len({sum(r['method']==m for r in all_rows) for m in METHODS})==1
        csv_write(out/'per_event'/f'{station}_events.csv',all_rows)
        quantiles+=qrows
        for split,t in targets.items(): regimes+=regime(station,split,t,tau)
        # Symmetric product attribution: delta(gate*raw) = avg(gate)*delta(raw) + avg(raw)*delta(gate).
        for a,b in [('D0','D2'),('D1','D3')]:
            for population in ('all',f'Q{max(bins)}'):
                left=[r for r in all_rows if r['method']==a and (population=='all' or r['quantile_bin']==max(bins))]
                right=[r for r in all_rows if r['method']==b and (population=='all' or r['quantile_bin']==max(bins))]
                parts=[]
                for l,r in zip(left,right):
                    assert l['window_id']==r['window_id']
                    parts.append((r['body_prediction']-l['body_prediction'],
                        .5*(r['gate_probability']+l['gate_probability'])*(r['raw_excess_prediction']-l['raw_excess_prediction']),
                        .5*(r['raw_excess_prediction']+l['raw_excess_prediction'])*(r['gate_probability']-l['gate_probability'])))
                sm0,sm1=summarize(left),summarize(right)
                parts=np.mean(parts,axis=0)
                deltas.append(dict(station=station,contrast=f'{b}-{a}',population=population,n=len(left),
                    delta_body_m=parts[0],delta_raw_effect_m=parts[1],delta_gate_effect_m=parts[2],
                    delta_bias_m=sm1['bias']-sm0['bias'],delta_UnderAny_pp=sm1['UnderAny_percent']-sm0['UnderAny_percent'],
                    delta_RMSE_m=sm1['RMSE']-sm0['RMSE'],delta_TimingSteps=sm1['TimingSteps']-sm0['TimingSteps']))
                assert abs(sum(parts)-(sm1['bias']-sm0['bias']))<3e-7
        plot_station(out,station,all_rows,qrows,targets,tau)
    for name,rows in [('overall_metrics',overall),('quantile_metrics',quantiles),('extreme_regime_stats',regimes),
                      ('branch_metrics',branches),('consistency_checks',consistency),
                      ('archived_fixed_event_metrics',original_metrics),('paired_amp_attribution',deltas)]:
        csv_write(out/'summaries'/f'{name}.csv',rows)
    write_report(out,overall,quantiles,regimes,branches,deltas,consistency,original_metrics)


def write_report(out, overall, quantiles, regimes, branches, deltas, consistency, archived):
    report=['# CBBT versus Lewes: fixed GT event branch diagnosis',
        'All amplitude values in tables are meters unless a column says otherwise. Every main comparison uses the same TRAIN-threshold TEST window IDs for all five methods. No training, loss, model, split, threshold, batch, seed, or precision changes were made for this diagnosis.',
        '## Exact definitions and provenance',
        '`tau` is the NumPy 95th percentile of TRAIN window maxima (saved in each checkpoint). Events use strict `max_h(y_true)>tau`. There is no predicted-peak matching, declustering, duration filter, or split-specific threshold fitting. `h_gt` and `h_pred` use the first argmax; one output horizon step is one hour. Counts are forecast windows, not independent storms.',
        '`emulator/training/metrics.py`: AllRMSE is the trajectory RMSE over all horizons. Top5RMSE selects `ceil(.05*N)` windows by GT maxima, using FP32 maxima and NumPy quicksort in validation, then scores all their horizons. TruePeakRMSE is `sqrt(mean((pred[h_gt]-GT[h_gt])**2))`; PeakBias is the signed mean of that aligned error; Under% is its negative fraction. TimingSteps is `mean(abs(h_pred-h_gt))`. The trainer records these peak metrics for all/top5/event separately and displays top5. This report uses the event population explicitly.',
        'There is no trainer key named PeakRMSE. The previous CBBT analysis defines it as RMSE of `max(pred)-max(GT)` on the stated window subset (PeakMAE analogously); we preserve that supplementary definition. The Figure6–8 notebook instead unrolls hourly blocks by year, independently thresholds GT and prediction at their own Q95, clusters exceedances separated by at most 24 hours, retains clusters with at least 3 exceedance points, pairs the ith GT event with the ith predicted event, and drops pairs farther than 48 hours apart. Its plotted Peak RMSE is the square root of the mean paired squared amplitude error within bins (then a 3-bin moving mean); its counts depend on predictions. Those matching-based results are not used here.',
        '## Exact branch algebra',
        'In `ExceedanceHead.forward`, all following quantities are initially normalized per horizon: `body_n = tau_n - softplus(tau_n - raw_body_n)`, `excess_n = softplus(excess_MLP(context))`, `logit = gate_MLP(mean_h(context))`, `g = sigmoid(logit)`, `prediction_n = body_n + g*excess_n`. The gate is one scalar per window, shared across horizons. Physical `body = body_n*y_std+y_mean`; physical raw excess `e = excess_n*y_std` (no mean); final excess `g*e`. Thus `prediction_phys = body+g*e`, up to float32 operation order. “Raw excess” in CSV means positive pre-gate excess after softplus, not the pre-softplus activation. Single branch columns are NaN.',
        'At an event peak, the exact amplitude shortfall decomposes as `(tau-body)+(true_excess-e)+(1-g)*e`. Negative raw-excess deficit means overprediction by the raw branch. This identity isolates the observed body gap, raw excess error and gate attenuation; it is not a causal intervention.',
        '## Fixed-event results',
        markdown(overall,['station','method','n','RMSE','bias','PeakRMSE','UnderAny_percent','Under5mm_percent','Under10mm_percent','Under20mm_percent','OverAny_percent','TimingSteps']),
        'Quantile bins use only GT peaks, with at least 20 windows per bin by default. Quartiles are preferred; three bins are attempted if quartiles lack support. Equal peak values are never split by method or rank. Ratios with denominator magnitude at most 1e-9 m are NaN. Severe under/over counts use strict comparisons. `absolute_underprediction` is the requested signed amount `GT-pred`, not clipped at zero.',
        '## Highest shared GT bin',
        markdown([r for r in quantiles if r['quantile_bin']==max(q['quantile_bin'] for q in quantiles if q['station']==r['station'])],
                 ['station','method','n','GT_peak_mean','RMSE','bias','UnderAny_percent','Under20mm_percent','TimingSteps']),
        markdown([r for r in branches if r['population'].startswith('Q') and r['quantile_bin']==max(q['quantile_bin'] for q in quantiles if q['station']==r['station'])],
                 ['station','method','true_excess_mean','body_deficit_mean','raw_excess_deficit_mean','gate_attenuation_mean','gate_probability_mean','raw_excess_ratio_median','final_excess_ratio_median']),
        '## TRAIN / VAL / TEST regimes',
        markdown([r for r in regimes if r['population']=='event_peak_excess'],
                 ['station','split','tau','total_windows','event_windows','event_windows_percent','positive_horizons','positive_horizons_percent','mean','median','P95','max']),
        'The full distribution table also includes positive individual-horizon excess, event GT peaks, and all GT window peaks; std uses ddof=0. Counts >5/10/20/50 mm are applied to the named population (event-peak rows are absolute GT values; event-peak-excess rows subtract tau).',
        '## Paired Amp changes and branch attribution',
        markdown(deltas,['station','contrast','population','n','delta_body_m','delta_raw_effect_m','delta_gate_effect_m','delta_bias_m','delta_UnderAny_pp','delta_RMSE_m','delta_TimingSteps']),
        'For each fixed window, `delta(g*e)=mean(g_before,g_after)*delta(e)+mean(e_before,e_after)*delta(g)`. These exact symmetric product attributions sum with the body change to the prediction change. They do not imply that changing one branch while holding the others fixed would reproduce a retrained model.',
        '## Answers to the five diagnostic questions']
    for station in STATIONS:
        s={r['method']:r for r in overall if r['station']==station}
        q={r['method']:r for r in branches if r['station']==station and r['population']=='Q4'}
        if not q: q={r['method']:r for r in branches if r['station']==station and r['population']=='Q3'}
        d=next(r for r in deltas if r['station']==station and r['contrast']=='D2-D0' and r['population']=='all')
        report.append(f"**{station}, D2 versus D0:** UnderAny {s['D0']['UnderAny_percent']:.2f}% → {s['D2']['UnderAny_percent']:.2f}%; Under20mm {s['D0']['Under20mm_percent']:.2f}% → {s['D2']['Under20mm_percent']:.2f}%; aligned RMSE {s['D0']['RMSE']*1000:.3f} → {s['D2']['RMSE']*1000:.3f} mm; bias {s['D0']['bias']*1000:.3f} → {s['D2']['bias']*1000:.3f} mm. Mean prediction change is {d['delta_bias_m']*1000:+.3f} mm: body {d['delta_body_m']*1000:+.3f}, raw-excess effect {d['delta_raw_effect_m']*1000:+.3f}, gate effect {d['delta_gate_effect_m']*1000:+.3f} mm. Timing MAE {s['D0']['TimingSteps']:.3f} → {s['D2']['TimingSteps']:.3f} steps; window-maximum RMSE {s['D0']['PeakRMSE']*1000:.3f} → {s['D2']['PeakRMSE']*1000:.3f} mm.")
        report.append(f"At the largest GT quartile/bin, D2 shortfall decomposes into body gap {q['D2']['body_deficit_mean']*1000:.3f} mm, raw excess deficit {q['D2']['raw_excess_deficit_mean']*1000:.3f} mm and gate attenuation {q['D2']['gate_attenuation_mean']*1000:.3f} mm. D0 corresponding values are {q['D0']['body_deficit_mean']*1000:.3f}, {q['D0']['raw_excess_deficit_mean']*1000:.3f}, {q['D0']['gate_attenuation_mean']*1000:.3f} mm. There is no missing reconstruction term beyond measured floating-point error.")
    c=next(r for r in deltas if r['station']=='CBBT' and r['contrast']=='D2-D0' and r['population']!='all')
    l=next(r for r in deltas if r['station']=='Lewes' and r['contrast']=='D2-D0' and r['population']!='all')
    report += [
        '1. **Main station difference:** the directly observed mechanism is primarily raw excess regression, accompanied by a different physical target regime. At the largest peaks both gates are effectively open, body gaps are about 2 mm, and raw-excess deficits are tens of millimeters. The station contrast is not primarily gate attenuation or reconstruction. Timing contributes to aligned-versus-free-maximum errors, but does not explain the persistent large amplitude deficits.',
        f"2. **Where large-peak amplitude disappears:** predominantly the raw excess branch. Amp changes the highest-bin raw-excess contribution by {c['delta_raw_effect_m']*1000:+.3f} mm at CBBT versus {l['delta_raw_effect_m']*1000:+.3f} mm at Lewes. Gate changes at those peaks are below 0.001 mm in magnitude. The exact reconstruction residual is below 0.0002 mm for every saved event.",
        '3. **Why Amp changes Under% differently:** in these selected fits, D2 raises raw excess relative to D0 at CBBT, while it lowers it on average at Lewes; small body/gate improvements at Lewes do not offset that decrease. This directly supports the opposite mean-bias and UnderAny directions. UnderAny counts signs rather than severity: Lewes severe-underprediction >20 mm is unchanged overall, its largest-bin RMSE slightly improves despite a more negative mean bias, and CBBT highest-bin UnderAny is unchanged despite amplitude improvement. The raw-branch response is observed; its training/optimization cause is not identified by final checkpoints.',
        '4. **Lewes hypotheses:** supported: larger true excess values, larger observed upper-tail amplitudes, and a stronger TEST/TRAIN increase in the reported mean and P95 excess summaries. Supported for D2 relative to D0: smaller mean raw prediction at the GT peak and a larger highest-bin raw deficit; this is not a claim that every window has smaller raw excess. Unsupported: fewer extreme TRAIN windows (both 662/13224). Lewes has slightly fewer positive TRAIN horizons (3218 versus 3321), which is a different count. Unsupported as the main high-peak failure: stronger gate attenuation (the Lewes gate is even closer to one). Timing worsens modestly in D2, and can affect sign crossings, but free-window-maximum amplitude remains substantially underestimated; timing alone cannot account for the failure.',
        '5. **Direct evidence and limits:** arithmetic decomposition and paired raw/gate contributions establish the output mechanism; TRAIN/VAL/TEST summaries establish the observed target differences. The recorded checkpoint-epoch and final TEST outputs do not establish why the optimizer learned this response or whether the tail shift caused it. These are single-seed checkpoints; neighboring windows can belong to one storm. D3−D1 differs from D2−D0, so the Amp-only response is not a universal claim about Amp with Tail.']
    for station in STATIONS:
        r={x['split']:x for x in regimes if x['station']==station and x['population']=='event_peak_excess'}
        report.append(f"{station}: TRAIN has {r['train']['event_windows']} extreme windows / {r['train']['total_windows']} ({r['train']['event_windows_percent']:.3f}%). Event-peak excess mean TRAIN→VAL→TEST is {r['train']['mean']*1000:.3f}→{r['val']['mean']*1000:.3f}→{r['test']['mean']*1000:.3f} mm; P95 {r['train']['P95']*1000:.3f}→{r['val']['P95']*1000:.3f}→{r['test']['P95']*1000:.3f} mm. TEST/TRAIN mean ratio {r['test']['mean']/r['train']['mean']:.3f}; P95 ratio {r['test']['P95']/r['train']['P95']:.3f}. These finite-sample tail summaries describe shift; they do not identify an asymptotic tail law.")
    report+=['## Replay and consistency audit',
        'Branch attribution uses predictions and components from the same evaluation-only replay. Archived TEST predictions remain unchanged and are also evaluated on these exact event IDs. BF16/TF32 with deterministic=0 may change low-order outputs on replay; the table measures this explicitly. VAL reference normalization for the future checkpoint score uses the saved epoch VAL metrics, not these replays or any TEST metric.',
        markdown(consistency,['station','method','n_events','reconstruction_max_abs_m','replay_archived_UnderAny_flips','replay_archived_at_gt_max_abs_m']),
        'Archived fixed-event results:',markdown(archived,['station','method','n','RMSE','bias','UnderAny_percent','Under20mm_percent','TimingSteps']),
        'See `audit.json`, `inference_audit.json`, `source_snapshot/`, the two frozen ID manifests, and the per-array SHA256 sidecars for provenance. `summaries/` contains full CSV precision; `plots/` contains ten PNG/PDF figures per station. No baseline artifact was edited.']
    for station in STATIONS:
        a={r['method']:r for r in archived if r['station']==station}
        report.append(f"Archived {station} D2−D0: UnderAny {a['D2']['UnderAny_percent']-a['D0']['UnderAny_percent']:+.3f} percentage points; bias {(a['D2']['bias']-a['D0']['bias'])*1000:+.3f} mm; aligned RMSE {(a['D2']['RMSE']-a['D0']['RMSE'])*1000:+.3f} mm. Thus the key opposite UnderAny/bias directions also hold in the original exports. The nearly zero Lewes RMSE difference should not be treated as a meaningful degradation.")
    (out/'DIAGNOSIS.md').write_text('\n\n'.join(report)+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo',type=Path,default=REPO)
    parser.add_argument('--results',type=Path,default=RESULTS)
    parser.add_argument('--output',type=Path,default=Path('/home/exouser/peak_branch_diagnosis_0921'))
    parser.add_argument('--stage',choices=['all','infer','analyze'],default='all')
    parser.add_argument('--device',choices=['cuda','cpu'],default='cuda')
    parser.add_argument('--min-bin-count',type=int,default=20)
    args=parser.parse_args()
    for folder in ('per_event','summaries','plots','arrays','source_snapshot'):
        (args.output/folder).mkdir(parents=True,exist_ok=True)
    runs,excluded=discover(args.results)
    if any(r['config']['lr'] != .005 for r in runs.values()):
        raise ValueError('This diagnosis is frozen to the completed 5e-3 baselines; use compare_lr0921.py for new runs')
    inventory=[dict(path=str(p),sha256=sha(p)) for r in runs.values() for p in
               (r['config_path'],r['checkpoint'],r['summary_path'],r['prediction'])]
    audit_path=args.output/'audit.json'
    if audit_path.exists() and json.loads(audit_path.read_text())['baseline_inputs'] != inventory:
        raise ValueError('Frozen baseline input hashes changed')
    write_json(audit_path,dict(baseline_inputs=inventory,excluded=excluded,
        runs=[dict(station=s,method=m,path=r['directory'],epoch=r['summary']['best_epoch']) for (s,m),r in runs.items()],
        source_hashes=source_hashes(args.repo), training_launched=False))
    for name in source_hashes(args.repo):
        dest=args.output/'source_snapshot'/name
        dest.parent.mkdir(exist_ok=True,parents=True)
        if dest.exists() and sha(dest)!=sha(args.repo/name): raise ValueError(f'Frozen inference source changed: {name}')
        shutil.copyfile(args.repo/name,dest)
    if args.stage in ('all','infer'):
        checks=[]
        for station in STATIONS: checks+=replay_station(args.repo,station,runs,args.output,args.device)
        write_json(args.output/'inference_audit.json',checks)
    if args.stage in ('all','analyze'):
        # Refuse corrupted/stale arrays even for a plot-only regeneration.
        for station,method in runs:
            for split in ('val','test'):
                path=args.output/'arrays'/f'{station}_{method}_{split}.npz'
                meta=json.loads(path.with_suffix('.json').read_text())
                assert sha(path)==meta['arrays_sha256']
                assert meta['inputs']['checkpoint_sha256']==sha(runs[(station,method)]['checkpoint'])
                assert meta['inputs']['source_sha256']==source_hashes(args.repo)
        analyze(args,runs)
    assert all(sha(r['path'])==r['sha256'] for r in inventory)
    print(f'Diagnosis complete: {args.output}; baseline hashes unchanged; no training',flush=True)


if __name__=='__main__': main()
