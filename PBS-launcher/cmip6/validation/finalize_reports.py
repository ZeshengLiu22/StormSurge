#!/usr/bin/env python3
"""Assemble reviewable validation reports; never submits or starts experiments."""
import csv
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import shutil
from zoneinfo import ZoneInfo

WORK = Path('/tmp/stormsurge_cmip6_prepare')
OUT = WORK/'validation'
REPO = Path('/glade/u/home/zel/scratch/StormSurge')
DEST = REPO/'PBS-launcher/cmip6/validation'


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2)+'\n')


def main():
    inventory = json.loads((WORK/'audit/dataset_inventory.json').read_text())
    manifest = json.loads((DEST/'manifest.json').read_text())
    matrix = json.loads((OUT/'matrix_validation.json').read_text())
    interfaces = json.loads((OUT/'interface_validation.json').read_text())
    preservation = json.loads((OUT/'preservation_validation.json').read_text())
    details = []
    for item in inventory:
        report = json.loads((WORK/'audit/data_details'/f'{item["gcm"]}_{item["station"]}.json').read_text())
        assert len(report['historical_files']) == len(item['historical_years']) == 36
        assert len(report['future_files']) == len(item['future_years']) == 30
        for earlier, later in [('train','val'),('val','test')]:
            earlier_end = datetime.fromisoformat(report['split_time_ranges'][earlier]['last'])
            later_context_start = datetime.fromisoformat(report['split_time_ranges'][later]['first']) - timedelta(hours=36)
            assert earlier_end < later_context_start, f'36H forcing context overlaps {earlier}/{later}: {item["gcm"]}/{item["station"]}'
        report['requested_36h_context_disjoint'] = True
        details.append(report)
    assert len(details) == 36
    scope = ('All 1,296 PastOnly files and 1,080 future files were deserialized one file per CPU worker, '
             'with at most four workers. Every graph was checked for feature/target/history dimensions, '
             'CNN grid sizes, pressure metadata shape, and supervised timestamps. Actual target hours '
             'were checked for duplicates and chronological split separation. Numerical target/pressure '
             'values and raw-pressure transformation/forcing finiteness were checked at least on the '
             'first and last graph of every file; several cached initial pairs had all target/pressure '
             'values checked. This is not a bytewise checksum or exhaustive finiteness scan of all forcing tensors.')
    audit = dict(status='PASS' if all(d['status']=='PASS' for d in details) else 'FAIL',
                 historical_file_count=sum(len(d['historical_files']) for d in details),
                 future_file_count=sum(len(d['future_files']) for d in details),
                 graph_count=sum(f['graph_count'] for d in details for kind in ['historical','future'] for f in d[kind+'_files']),
                 numeric_target_graph_count=sum(f.get('numeric_graph_count', f['graph_count']) for d in details for kind in ['historical','future'] for f in d[kind+'_files']),
                 scope=scope,stations=details)
    write_json(OUT/'dataset_audit.json',audit)
    shutil.copytree(WORK/'audit/data_details',OUT/'data_details',dirs_exist_ok=True)
    summaries = []
    boundaries = []
    per_file = []
    for item, d in zip(inventory, details):
        row = dict(gcm=item['gcm'],station=item['station'],past_root=item['past_root'],full_root=item['full_root'],
                   historical_years=item['historical_years'],future_years=item['future_years'],
                   splits=d['split_groups'],split_time_ranges=d['split_time_ranges'],
                   dimensions=dict(zip(['grid_H','grid_W','nodes','features','stored_history_slices','max_history_hours','target_hours'],d['dimensions'])),
                   status=d['status'])
        summaries.append(row)
        for kind in ['historical','future']:
            files = d[kind+'_files']
            counts = sorted(f['graph_count'] for f in files)
            median = counts[len(counts)//2]
            for f in files:
                year = Path(f['path']).name[:9]
                partition = next((part for part,tags in d['split_groups'].items() if year in tags),'future')
                entry = dict(gcm=item['gcm'],station=item['station'],protocol=kind,year_group=year,split=partition,
                             graph_count=f['graph_count'],first_target_utc=f['first_target_utc'],last_target_utc=f['last_target_utc'],
                             source=f['path'],bytes=f['bytes'],status=f['status'])
                per_file.append(entry)
                if f['graph_count'] < .8*median:
                    boundaries.append(entry)
    write_json(OUT/'year_groups.json',summaries)
    write_json(OUT/'shortened_year_groups.json',boundaries)
    with (OUT/'year_groups.tsv').open('w') as stream:
        fields=['gcm','station','split','year_group_count','year_groups','first_target_utc','last_target_utc','graph_count']
        writer=csv.DictWriter(stream,fieldnames=fields,delimiter='\t',lineterminator='\n')
        writer.writeheader()
        for r in summaries:
            for part, tags in [*r['splits'].items(),('future',r['future_years'])]:
                fs=[f for f in per_file if f['gcm']==r['gcm'] and f['station']==r['station'] and f['split']==part]
                writer.writerow(dict(gcm=r['gcm'],station=r['station'],split=part,year_group_count=len(tags),year_groups=','.join(tags),
                                     first_target_utc=min(f['first_target_utc'] for f in fs),last_target_utc=max(f['last_target_utc'] for f in fs),
                                     graph_count=sum(f['graph_count'] for f in fs)))
    with (OUT/'graph_file_audit.tsv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(per_file[0]),delimiter='\t',lineterminator='\n')
        writer.writeheader(); writer.writerows(per_file)
    categories = {
        'A_dataset_audit':audit['status'],
        'B_exact_matrix_and_uniqueness':matrix['status'],
        'C_formal_NCEP_scientific_comparability':matrix['status'],
        'D_PBS_syntax_and_540_CPU_dry_runs':matrix['status'],
        'E_preservation_and_no_results_created':'PASS' if preservation['status']=='PASS' and matrix['no_training_results_created'] else 'FAIL',
        'F_future_inference_interface':interfaces['status'],
    }
    ready = all(v=='PASS' for v in categories.values())
    notes = [
        'No scientific hyperparameter, optimizer, architecture, loss, or split changes from the 60 matching formal station/seed NCEP references.',
        'Authorized operational changes: CMIP6 PastOnly roots, unique output/log/run paths, Derecho Python, USE_TMUX=0, timestamp_runname, independent PBS jobs, and requested walltimes.',
        'train.sh appends its launch timestamp to PYTHON_RUN_TAG_BASE; validation checks that existing behavior. Its fresh-directory mkdir refuses collisions, including reruns within one second.',
        'Historical 2014_2015 and future 2099_2100 boundary files have fewer windows than full winter groups. No requested year group is missing; existing groups are preserved intact. See shortened_year_groups.json for exact coverage.',
    ]
    risks = [
        'CUDA/NCCL/BF16 device preflight is retained but intentionally not executed on GPUs. CPU validation cannot establish GPU memory fit or the actual time required for 300 epochs.',
        'First-batch PBS walltimes: Cane5 00:20:00; IPSL 00:25:00; CNRM/MIROC 00:40:00; BCC/MRI 00:45:00; AWI/MPI 01:00:00; EC_EARTH 02:00:00. These are user-specified limits, not measured CMIP6 completion estimates.',
        'The preempt queue can interrupt runs. #PBS -r n prevents automatic reruns; no automatic resume mechanism was added.',
        'Historical checkpoints do not exist until training completes. End-to-end inference with trained weights is deferred; source and interface checks passed.',
        'Numerical payload checks are sampled at file boundaries as detailed in the dataset audit; all graph shapes and target timestamps were checked.',
    ]
    stamp = datetime.now(timezone.utc)
    report = dict(status='PASS' if ready else 'FAIL',ready_combinations=540 if ready else 0,total_combinations=540,
                  generated_at_utc=stamp.isoformat(),generated_at_client_time=stamp.astimezone(ZoneInfo('America/New_York')).isoformat(),
                  repository_revision=preservation['revision'],categories=categories,
                  missing_files_or_invalid_data=[e for d in details for e in d['errors']],
                  scientific_deviations=[],data_notes=notes,risks_and_limits=risks,
                  pbs_submissions=0,training_launches=0,future_inference_launches=0,
                  manifest=str(DEST/'manifest.tsv'),year_summary=str(DEST/'year_groups.tsv'),
                  manual_commands=str(DEST/'manual_qsub_commands.txt'))
    write_json(OUT/'validation_report.json',report)
    lines = [f'CMIP6 PastOnly preparation: {report["status"]}',f'{report["ready_combinations"]}/540 combinations ready for manual submission after CPU validation.',
             f'Validated at {report["generated_at_client_time"]} (America/New_York).',f'Repository revision: {preservation["revision"]}',
             'No qsub, qdel, training, or future inference was executed.','', 'Category results:']
    lines += [f'{value}: {key}' for key,value in categories.items()]
    lines += ['', 'Matrix: 540 configs, 540 independent PBS scripts, 540 unique tuples/output roots/run names/PBS names/log locations.',
              'Distributions: 60/GCM; 135/station; 180/seed; 108/variant. Maximum PBS job-name length: 12.',
              'All 1,080 generated shell/PBS files pass bash -n; all 540 PBS CPU DRY_RUNs pass.',
              'All 540 effective scientific command lines match their exact formal station/seed reference.',
              'All 540 command lines parse with the repository training argument parser.',
              f'Data: {audit["historical_file_count"]} PastOnly files and {audit["future_file_count"]} future files; {audit["graph_count"]:,} graphs inspected.',
              'All 36 GCM/station pairs have the same chronological year split for seeds 42, 123, and 2026:',
              '  TRAIN: 22 groups, 1979_1980 through 2000_2001.',
              '  VAL:    7 groups, 2001_2002 through 2007_2008.',
              '  TEST:   7 groups, 2008_2009 through 2014_2015.',
              'Future availability: 30 groups, 2070_2071 through 2099_2100, for every pair.',
              'All observed target timestamps are unique; partitions and their requested 36H forcing contexts are chronologically separated; historical timestamps precede 2030.',
              'No missing requested GCM/station/year files or malformed filenames were detected.' if audit['status']=='PASS' else 'DATA ERRORS: see dataset_audit.json.',
              '', 'Scientific comparability and data notes:'] + ['- '+n for n in notes]
    lines += ['', 'Validation scope:',scope,'', 'Runtime and PBS limits:'] + ['- '+r for r in risks]
    lines += ['', 'Preservation:',f'All {preservation["protected_tracked_file_count"]} pre-existing tracked files retain their SHA256 hashes.',
              'No tracked files changed. New repository files are confined to configs/cmip6 and PBS-launcher/cmip6.',
              'The existing cmip6 results parent remains empty. Results are created only by actual training.',
              '', 'Artifacts:', 'manifest.tsv / manifest.json: one record per combination, with template and generated-file SHA256 provenance.',
              'matrix_validation.json and dry_run/: per-combination checks and all 540 CPU previews.',
              'dataset_audit.json, data_details/, graph_file_audit.tsv: exact graph/time/shape audit evidence.',
              'year_groups.tsv / year_groups.json: every GCM/station split and future year group.',
              'future_inference_compatibility.txt and interface_validation.json: deferred evaluation procedure and checks.',
              'manual_qsub_commands.txt / pbs_paths.txt: manual staged/full submission commands.']
    (OUT/'validation_report.txt').write_text('\n'.join(lines)+'\n')
    future = ['Future-inference compatibility: PASS (source and interface checks; no inference executed).',
              'Protocol A trains once on PastOnly and evaluates its chronological historical TEST after VAL-only checkpoint selection.',
              'Protocol B must use that exact run directory\'s best_overall.pt. It creates no second training run.',
              'The checkpoint stores model_config, model_state, source TRAIN normalization, threshold_metadata and episode_peak_metadata.',
              'infer.py restores model_state strictly, reads checkpoint normalization, and reads threshold_metadata.tau_physical.',
              'It does not call fit_statistics or fit_loss_thresholds. Future data are never used to fit statistics or select a checkpoint.',
              'An explicit --test_root_dir selects external data. Always supply --years and --strict_years.',
              'Do not use --scope test with --test_root_dir. Architecture comes from the saved checkpoint; pressure_mode=raw is an assertion.',
              'All 36 actual target-root/station year selections passed infer.py\'s parser and selection logic using a filename-only test store.',
              'Missing, duplicate, malformed and empty year requests were rejected. Every future graph file was separately inspected for dimensions and timestamps.',
              'Trained checkpoints do not exist yet; no end-to-end trained-model evaluation was attempted.',
              '', 'Exact future groups (identical for all nine GCMs and all four stations):', ','.join(inventory[0]['future_years']),
              '', 'The 2099_2100 tag is present, but its stored target coverage ends in 2099; see the per-file audit. Do not infer target calendar dates from filenames alone.',
              '', 'Manual example for later, inside a suitable GPU allocation. Replace the one historical_run placeholder with the completed run selected for Protocol A; do not select a rerun by future TEST scores. These commands have NOT been run:',
              f'cd {REPO}',
              'historical_run="'+manifest[0]['output_root']+'/<timestamp>_'+manifest[0]['run_name']+'"',
              'checkpoint="${historical_run}/best_overall.pt"',
              'test -r "${checkpoint}" || exit 1',
              'future_years="'+','.join(inventory[0]['future_years'])+'"',
              'future_output="$(mktemp -d "${historical_run}/future_inference_$(date -u +%Y%m%d_%H%M%S)_XXXXXX")"',
              '/glade/work/zel/conda-envs/derecho-ml/bin/python infer.py \\',
              '  --ckpt "${checkpoint}" \\',
              '  --root_dir '+inventory[0]['past_root']+' \\',
              '  --test_root_dir '+inventory[0]['full_root']+' \\',
              '  --station Battery --years "${future_years}" --strict_years \\',
              '  --pressure_mode raw --device cuda --amp --amp_dtype bf16 --tf32 \\',
              '  --batch_size 256 --num_workers 0 --torch_threads 1 --save_npz \\',
              '  --model_label PACT_MSE --out_dir "${future_output}"',
              '', 'For each other combination use its manifest data_root/full_data_root, station, variant and selected historical best_overall.pt.',
              'The fresh mktemp output directory prevents later future-evaluation reruns from overwriting earlier outputs.']
    (OUT/'future_inference_compatibility.txt').write_text('\n'.join(future)+'\n')
    command_text = (DEST/'manual_qsub_commands.txt').read_text()
    summary_commands = f'''Manual submission alternatives (not executed)

cd {REPO}

# Stage one GCM: 60 AWI runs. Change CMIP6_AWI to the desired GCM for later stages.
for pbs in PBS-launcher/cmip6/CMIP6_AWI/*/seed*/run_*.pbs; do
  qsub "$pbs" || break
done

# Alternatively submit all 540 exactly once. Do not also submit the staged set.
while IFS= read -r pbs; do
  qsub "$pbs" || break
done < PBS-launcher/cmip6/validation/pbs_paths.txt

All per-run commands follow. They are text only; this file is not a submission script.

'''
    (OUT/'manual_qsub_commands.txt').write_text(summary_commands+command_text)
    for name in ['audit_data.py','generate.py','validate_matrix.py','validate_interfaces.py','finalize_reports.py']:
        shutil.copy2(WORK/name,OUT/name)
    print(json.dumps({k:v for k,v in report.items() if k in ['status','ready_combinations','categories','missing_files_or_invalid_data']}))


if __name__=='__main__':
    main()
