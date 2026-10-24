#!/usr/bin/env python3
"""Generate only the authorized CMIP6 configs, launchers, and preparation artifacts."""
import csv
import hashlib
import itertools
import json
from pathlib import Path
import re
import shlex
import shutil
import sys

from audit_data import discover, GCMS, STATIONS, REPO
from pbs_failure_mail import add_failure_mail

STAGE = Path('/tmp/stormsurge_cmip6_prepare/package')
RESULTS = Path('/glade/u/home/zel/scratch/PACT_StormSurge_FormalRuns_1021_LR5e-3_Clip0.25/cmip6')
PYTHON = '/glade/work/zel/conda-envs/derecho-ml/bin/python'
SEEDS = [42, 123, 2026]
VARIANTS = ['PACT_MSE', 'PACT_Peak', 'PACT_Both', 'CNN_GRU_MSE', 'CNN_GRU_Tail']
WALLTIMES = {
    'CMIP6_AWI': '01:00:00',
    'CMIP6_BCC': '00:45:00',
    'CMIP6_Cane5': '00:20:00',
    'CMIP6_CNRM': '00:40:00',
    'CMIP6_EC_EARTH': '02:00:00',
    'CMIP6_IPSL': '00:25:00',
    'CMIP6_MIROC': '00:40:00',
    'CMIP6_MPI': '01:00:00',
    'CMIP6_MRI': '00:45:00',
}
PBS_SOURCE = REPO/'PBS-launcher/baselines_PeakAware/baseline_cnn_gru/Battery/seed42/run_T0250_EP0000.pbs'
GCODES = dict(zip(GCMS, ['AW', 'BC', 'CA', 'CN', 'EC', 'IP', 'MI', 'MP', 'MR']))
SCODES = dict(zip(STATIONS, ['Bt', 'Bo', 'Cb', 'Le']))
VCODES = dict(zip(VARIANTS, ['PM', 'PP', 'PB', 'GM', 'GT']))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_for(station, seed, variant):
    if variant == 'PACT_MSE':
        root = REPO/f'configs/architecture_avgrank_multiseed_LR5e-3_Clip0.25/CNN_MLP_36H/{station}/seed{seed}'
        return root/f'train_config_NCEP_{station}_CNN_MLP_36H_G0_T0000_EP0000_seed{seed}.sh'
    if variant in ['PACT_Peak', 'PACT_Both']:
        root = REPO/f'configs/loss_weight_sweep_5e-3_Clip0.25/{station}' if seed == 42 else REPO/f'configs/loss_weight_sweep_5e-3_Clip0.25_multiseed/{station}/seed{seed}'
        tail = '0000' if variant == 'PACT_Peak' else '0250'
        return root/f'train_config_NCEP_{station}_CNN_MLP_36H_G0_T{tail}_EP0100_seed{seed}.sh'
    if variant == 'CNN_GRU_MSE':
        return REPO/f'configs/baselines_MSEOnly/Derecho/baseline_cnn_gru/{station}/seed{seed}/train_config.sh'
    return REPO/f'configs/baselines_PeakAware/baseline_cnn_gru/{station}/seed{seed}/train_config_T0250_EP0000.sh'


def replace_assignments(text, values):
    for key, value in values.items():
        pattern = rf'^{re.escape(key)}=.*$'
        replacement = key + '=' + shlex.quote(str(value))
        text, n = re.subn(pattern, lambda _: replacement, text, flags=re.M)
        if n > 1:
            raise ValueError(f'Duplicate assignment: {key}')
        if n == 0:
            text += '\n' + replacement + '\n'
    return text


def main():
    inventory = discover()
    data = {(r['gcm'], r['station']): r for r in inventory}
    artifacts = STAGE/'PBS-launcher/cmip6/validation'
    artifacts.mkdir(parents=True, exist_ok=True)
    (artifacts/'dataset_inventory.json').write_text(json.dumps(inventory, indent=2)+'\n')
    source_pbs = PBS_SOURCE.read_text()
    rows = []
    for gcm, station, seed, variant in itertools.product(GCMS, STATIONS, SEEDS, VARIANTS):
        source = source_for(station, seed, variant)
        assert source.is_file(), f'Missing matching formal reference: {source}'
        config_rel = Path(f'configs/cmip6/{gcm}/{station}/seed{seed}/train_config_{variant}.sh')
        pbs_rel = Path(f'PBS-launcher/cmip6/{gcm}/{station}/seed{seed}/run_{variant}.pbs')
        log_rel = pbs_rel.parent/'pbs_logs'/variant
        output = RESULTS/gcm/station/f'seed{seed}'/variant
        run_name = f'{gcm}_{station}_seed{seed}_{variant}_36H_lr5e-3_clip0.25'
        pbs_name = f'c6{GCODES[gcm]}{SCODES[station]}{seed}{VCODES[variant]}'
        walltime = WALLTIMES[gcm]
        overrides = dict(ROOT_DIR=data[gcm, station]['past_root'], TEST_ROOT_DIR='',
                         PYTHON_BIN=PYTHON, USE_TMUX=0, DO_CONDA=0,
                         ALL_RESULTS_ROOT=str(output), RUN_DIR_NAME_STYLE='timestamp_runname',
                         PACT_RUN_NAME=run_name, PYTHON_RUN_TAG_BASE=run_name)
        source_text = source.read_text()
        body = source_text[source_text.index('TRAIN_PY='):]
        config = (f'#!/usr/bin/env bash\n# Formal CMIP6 historical PastOnly: {gcm}, {station}, seed {seed}, {variant}.\n'
                  f'# Formal NCEP source: {source.relative_to(REPO)}\n'
                  f'# Source SHA256: {digest(source)}\n'
                  '# Assignment-only; one 36H/LR combination; historical best_overall.pt is reused for future inference.\n\n'
                  + replace_assignments(body, overrides))
        pbs = source_pbs
        pbs = pbs.replace('# Formal BF16 Extreme-Aware ablation: baseline_cnn_gru, Battery, seed 42, T0250_EP0000.',
                          f'# Formal CMIP6 historical PastOnly: {gcm}, {station}, seed {seed}, {variant}.')
        pbs = pbs.replace('# One independent station run; four DDP ranks; 20-minute preempt queue request.',
                          '# One independent configuration; four DDP ranks; no automatic rerun after preemption.')
        pbs = re.sub(r'^#PBS -N .*$', '#PBS -N ' + pbs_name, pbs, flags=re.M)
        pbs = re.sub(r'^#PBS -l walltime=.*$', '#PBS -l walltime=' + walltime, pbs, flags=re.M)
        pbs = re.sub(r'^#PBS -o .*$', '#PBS -o ' + str(REPO/log_rel) + '/', pbs, flags=re.M)
        pbs = re.sub(r'^config=.*$', 'config=' + shlex.quote(str(REPO/config_rel)), pbs, flags=re.M)
        pbs = pbs.replace('stormsurge-peakaware-${PBS_JOBID}', 'stormsurge-cmip6-${PBS_JOBID}')
        pbs = add_failure_mail(pbs, REPO/log_rel)
        for relative, contents in [(config_rel, config), (pbs_rel, pbs)]:
            path = STAGE/relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(contents)
        (STAGE/log_rel).mkdir(parents=True, exist_ok=True)
        (STAGE/log_rel/'.gitkeep').write_text('')
        rows.append(dict(gcm=gcm, station=station, seed=seed, variant=variant,
                         config=str(REPO/config_rel), pbs=str(REPO/pbs_rel),
                         data_root=data[gcm, station]['past_root'],
                         full_data_root=data[gcm, station]['full_root'],
                         output_root=str(output), log_dir=str(REPO/log_rel),
                         pbs_name=pbs_name, run_name=run_name, python_run_tag_base=run_name,
                         walltime=walltime, source_config=str(source), source_config_sha256=digest(source),
                         source_pbs=str(PBS_SOURCE), source_pbs_sha256=digest(PBS_SOURCE),
                         config_sha256=digest(STAGE/config_rel), pbs_sha256=digest(STAGE/pbs_rel),
                         checkpoint_pattern=str(output/'*'/ 'best_overall.pt'),
                         future_years=','.join(data[gcm, station]['future_years'])))
    (artifacts/'manifest.json').write_text(json.dumps(rows, indent=2)+'\n')
    with (artifacts/'manifest.tsv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter='\t', lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    (artifacts/'pbs_paths.txt').write_text(''.join(r['pbs']+'\n' for r in rows))
    (artifacts/'manual_qsub_commands.txt').write_text(
        '# Manual commands only. These commands have NOT been executed.\n'
        '# One selected trial run:\nqsub ' + rows[0]['pbs'] + '\n\n'
        '# Staged submission: all 60 CMIP6_AWI runs (execute once).\n'
        + ''.join('qsub '+r['pbs']+'\n' for r in rows if r['gcm']=='CMIP6_AWI')
        + '\n# Full submission: all 540 runs (do not also submit the staged set).\n'
        + ''.join('qsub '+r['pbs']+'\n' for r in rows))
    (artifacts/'pbs_name_codes.json').write_text(json.dumps(dict(prefix='c6', format='c6<GCM2><station2><seed><variant2>',
        gcm=GCODES, station=SCODES, variant=VCODES), indent=2)+'\n')
    for name in ['generate.py', 'audit_data.py', 'pbs_failure_mail.py', 'preservation_before.json']:
        shutil.copy2(Path(__file__).parent/name, artifacts/name)
    print(f'Generated {len(rows)} configs, {len(rows)} independent PBS scripts, and {len(rows)} log parents under {STAGE}.')


if __name__ == '__main__':
    main()
