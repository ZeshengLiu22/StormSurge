#!/usr/bin/env python3
"""Validate the actual 648 installed launchers without training or submission."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import itertools
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import time

from pbs_failure_mail import MAIL_TO, add_failure_mail

REPO = Path('/glade/u/home/zel/scratch/StormSurge')
VALIDATION = REPO/'PBS-launcher/cmip6/validation'
RESULTS = Path('/glade/u/home/zel/scratch/PACT_StormSurge_FormalRuns_1021_LR5e-3_Clip0.25/cmip6')
PYTHON = '/glade/work/zel/conda-envs/derecho-ml/bin/python'
GCMS = ['CMIP6_'+x for x in ['AWI','BCC','Cane5','CNRM','EC_EARTH','IPSL','MIROC','MPI','MRI']]
STATIONS = ['Battery','Boston','CBBT','Lewes']
SEEDS = [42,123,2026]
VARIANTS = ['PACT_MSE','PACT_Peak','PACT_Both','CNN_GRU_MSE','CNN_GRU_Tail','CNN_GRU_Peak']
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
CHANGED = {'ROOT_DIR','TEST_ROOT_DIR','PYTHON_BIN','USE_TMUX','DO_CONDA','ALL_RESULTS_ROOT',
           'RUN_DIR_NAME_STYLE','PACT_RUN_NAME','PYTHON_RUN_TAG_BASE'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def assignments(text):
    found = re.findall(r'^([A-Za-z_][A-Za-z_0-9]*)=(.*)$', text, re.M)
    assert len(found) == len(dict(found)), 'Duplicate assignments'
    for line in text.splitlines():
        if line.strip() and not line.lstrip().startswith('#'):
            assert re.match(r'^[A-Za-z_][A-Za-z_0-9]*=', line), f'Nonassignment config line: {line}'
    return dict(found)


def source_values(path, names):
    code = 'source "$1"; shift; for key in "$@"; do declare -n value="$key"; printf "%s\\0%s\\0" "$key" "${value[*]}"; unset -n value; done'
    env = dict(os.environ, S0_PYTHON_BIN=PYTHON, PYTHON_BIN=PYTHON, USE_TMUX='0')
    data = subprocess.check_output(['bash','--noprofile','--norc','-c',code,'bash',str(path),*names],env=env,cwd=REPO)
    values = data.decode().split('\0')[:-1]
    return dict(zip(values[::2], values[1::2]))


def cli_from_log(text):
    commands = [line[5:] for line in text.splitlines() if line.startswith('CMD: ')]
    assert len(commands) == 1, f'Expected exactly one training invocation, found {len(commands)}'
    tokens = shlex.split(commands[0])
    assert tokens[0] == 'train.py'
    options = {}
    i = 1
    while i < len(tokens):
        key = tokens[i]
        assert key.startswith('--') and key not in options, f'Invalid/repeated CLI option: {key}'
        if i+1 < len(tokens) and not tokens[i+1].startswith('--'):
            options[key] = tokens[i+1]
            i += 2
        else:
            options[key] = True
            i += 1
    return tokens, options


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rows = json.loads((VALIDATION/'manifest.json').read_text())
    inventory = json.loads((VALIDATION/'dataset_inventory.json').read_text())
    roots = {(r['gcm'],r['station']):r for r in inventory}
    errors = []
    expected = set(itertools.product(GCMS, STATIONS, SEEDS, VARIANTS))
    actual = [(r['gcm'],r['station'],r['seed'],r['variant']) for r in rows]
    assert len(rows) == 648 and len(set(actual)) == 648 and set(actual) == expected
    assert len(list((REPO/'configs/cmip6').rglob('*.sh'))) == 648
    assert len(list((REPO/'PBS-launcher/cmip6').rglob('*.pbs'))) == 648
    for key in ['config','pbs','data_root','output_root','log_dir','pbs_name','run_name','python_run_tag_base']:
        if key != 'data_root':
            assert len({r[key] for r in rows}) == 648, key
    for key, number in [('gcm',72),('station',162),('seed',216),('variant',108)]:
        assert set(Counter(r[key] for r in rows).values()) == {number}
    for gcm, station, seed in itertools.product(GCMS,STATIONS,SEEDS):
        directory = REPO/f'configs/cmip6/{gcm}/{station}/seed{seed}'
        assert sorted(p.name for p in directory.iterdir()) == sorted('train_config_'+v+'.sh' for v in VARIANTS)
    dry_env = dict(os.environ, CUDA_VISIBLE_DEVICES='', DRY_RUN='1', PYTHON_BIN=PYTHON,
                   S0_PYTHON_BIN=PYTHON, USE_TMUX='0', PYTHONDONTWRITEBYTECODE='1',
                   OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    for key in ['SLURM_NTASKS_PER_NODE','SLURM_JOB_ID','PACT_RUNSTAMP','PACT_RUN_NAME','SESSION_NAME','PYTHON_RUN_TAG_BASE']:
        dry_env.pop(key, None)
    references = {}
    for path in sorted({r['source_config'] for r in rows}):
        run = subprocess.run(['bash','train.sh',path],cwd=REPO,env=dry_env,text=True,capture_output=True,timeout=30)
        assert run.returncode == 0, f'Reference dry run failed: {path}: {run.stderr}'
        tokens, options = cli_from_log(run.stdout)
        references[path] = dict(options=options, assignments=assignments(Path(path).read_text()))
    print(f'Validated {len(references)} formal reference command lines without loading NCEP data.', flush=True)

    def validate(row):
        checks = {}
        try:
            config = Path(row['config'])
            pbs = Path(row['pbs'])
            assert sha(config) == row['config_sha256']
            assert sha(pbs) == row['pbs_sha256']
            assert sha(row['source_config']) == row['source_config_sha256']
            assert sha(row['source_pbs']) == row['source_pbs_sha256']
            cfg = assignments(config.read_text())
            original = references[row['source_config']]['assignments']
            assert {k:v for k,v in cfg.items() if k not in CHANGED} == {k:v for k,v in original.items() if k not in CHANGED}, 'Scientific source assignments changed'
            vals = source_values(config, sorted(cfg))
            common = dict(HISTORY_HOURS_LIST='36',PRESSURE_MODE='raw',LR_LIST='5e-3',MAX_GRAD_NORM='0.25',
                          EPOCHS='300',BATCH_SIZE='256',GRAD_ACCUM_STEPS='1',SCHEDULER='cosine',
                          WARMUP_EPOCHS='5',WARMUP_START_FACTOR='0.1',MIN_LR='1e-6',WEIGHT_DECAY_POLICY='module',
                          NODE_ATTENTION_BACKEND='math',X_NORM='zscore',DISABLE_OOD='1',USE_AMP='1',AMP_DTYPE='bf16',
                          USE_TF32='1',num_gpus='4',USE_TMUX='0',CHECKPOINT_SELECTION='overall',EXCEEDANCE_PERCENTILE='95',
                          TRAIN_RATIO='0.6',VAL_RATIO='0.2',SHUFFLE_YEARS='0',FUTURE_ONLY='0',FUTURE_YEAR_THRESHOLD='2030',
                          TEST_ROOT_DIR='',RUN_DIR_NAME_STYLE='timestamp_runname',PYTHON_BIN=PYTHON,
                          ROOT_DIR=roots[row['gcm'],row['station']]['past_root'],STATION=row['station'],SEED=str(row['seed']),
                          ALL_RESULTS_ROOT=row['output_root'],PACT_RUN_NAME=row['run_name'],PYTHON_RUN_TAG_BASE=row['run_name'])
            for key,value in common.items():
                assert vals.get(key) == value, f'{key}: {vals.get(key)!r} != {value!r}'
            tail, peak = {'PACT_MSE':(0,0),'PACT_Peak':(0,.01),'PACT_Both':(.025,.01),
                          'CNN_GRU_MSE':(0,0),'CNN_GRU_Tail':(.025,0),'CNN_GRU_Peak':(0,.01)}[row['variant']]
            assert float(vals['EXCEEDANCE_LOSS_WEIGHT']) == tail
            assert float(vals['EPISODE_GT_ALIGNED_PEAK_WEIGHT']) == peak
            if row['variant'].startswith('PACT'):
                architecture = dict(MODEL='perceiver3',ENCODER_TYPE='CNN',TEMPORAL_BLOCK='MLP',CNN_INTERMEDIATE_CHANNEL='29',
                                    HIDDEN_CHANNELS='128',NUM_LAYERS='2',DROPOUT='0.05',HEAD_DROPOUT='0.05',
                                    NODE_READ_HEADS='8',TIME_READ_HEADS='8',TRANSFORMER_LAYERS='2',TRANSFORMER_FF_MULT='4.0',
                                    TRANSFORMER_DROPOUT='0.0',MAX_TIME_STEPS='32')
            else:
                architecture = dict(MODEL='baseline_cnn_gru',BASELINE_POOLING='adaptive',BASELINE_POOL_GRID='4',
                                    BASELINE_STEM_CHANNELS='32',HIDDEN_CHANNELS='128',DROPOUT='0.05')
            for key, value in architecture.items():
                assert vals[key] == value, f'Architecture {key}'
            checks['scientific_config'] = 'PASS'
            assert row['walltime'] == WALLTIMES[row['gcm']], 'Manifest walltime'
            text = pbs.read_text()
            directives = [l for l in text.splitlines() if l.startswith('#PBS ')]
            required = ['#PBS -A ULHI0005','#PBS -q preempt@desched1',
                        '#PBS -l select=1:ncpus=64:mem=487GB:mpiprocs=4:ompthreads=1:ngpus=4',
                        '#PBS -r n','#PBS -j oe','#PBS -m a','#PBS -M '+MAIL_TO,'#PBS -N '+row['pbs_name'],
                        '#PBS -o '+row['log_dir']+'/',
                        '#PBS -l walltime='+WALLTIMES[row['gcm']]]
            assert sorted(directives) == sorted(required), 'PBS directives'
            assert re.fullmatch(r'[A-Za-z][A-Za-z0-9]{0,14}', row['pbs_name'])
            assert Path(row['log_dir']).is_dir()
            assert len(re.findall(r'^bash train\.sh "\$\{config\}"$',text,re.M)) == 1
            assert not re.search(r'\b(qsub|qdel|sbatch|srun)\b|^#PBS -[Jt]\b',text,re.M), 'Hidden job operations/arrays'
            for required_text in ['module load cuda/12.9.0', 'torch.cuda.device_count() != 4',
                                  'dist.is_nccl_available()', 'get_device_capability(i)[0] < 8',
                                  'import torch_geometric','OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1',
                                  'PYTHONUNBUFFERED=1','export USE_TMUX=0','export TMPDIR=', 'export MPLCONFIGDIR=',
                                  'if [[ "${DRY_RUN}" != 1 ]]; then', '[[ -n "${PBS_JOBID:-}" ]]']:
                assert required_text in text, f'Missing PBS runtime feature: {required_text}'
            # Only documented paths and failure notification differ from the formal PBS.
            template = add_failure_mail(Path(row['source_pbs']).read_text(), row['log_dir'])
            def normalize_pbs(t):
                t = '\n'.join(l for l in t.splitlines() if not l.startswith('#') or l.startswith('#!'))
                t = re.sub(r'^config=.*$', 'config=CONFIG', t, flags=re.M)
                return t.replace('stormsurge-peakaware-', 'stormsurge-cmip6-')
            assert normalize_pbs(text) == normalize_pbs(template), 'Formal PBS implementation drift'
            checks['pbs_static'] = 'PASS'
            for script in [config,pbs]:
                proc = subprocess.run(['bash','-n',str(script)],capture_output=True,text=True,timeout=30)
                assert proc.returncode == 0, f'bash -n {script}: {proc.stderr}'
            checks['bash_syntax'] = 'PASS'
            proc = subprocess.run(['bash',str(pbs)],cwd=REPO,env=dry_env,capture_output=True,text=True,timeout=30)
            log_relative = Path('dry_run')/row['gcm']/row['station']/f'seed{row["seed"]}'/(row['variant']+'.log')
            log = args.out/log_relative
            log.parent.mkdir(parents=True,exist_ok=True)
            log.write_text(proc.stdout+proc.stderr)
            assert proc.returncode == 0, f'CPU DRY_RUN exit={proc.returncode}: {proc.stderr}'
            assert not proc.stderr.strip(), f'Unexpected DRY_RUN stderr: {proc.stderr}'
            tokens, options = cli_from_log(proc.stdout)
            expected_options = references[row['source_config']]['options']
            operational = {'--root_dir','--output_dir','--run_tag'}
            assert {k:v for k,v in options.items() if k not in operational} == {k:v for k,v in expected_options.items() if k not in operational}, 'Effective scientific CLI differs from NCEP source'
            assert options['--root_dir'] == row['data_root'] and 'NCEP' not in options['--root_dir']
            assert not options.get('--test_root_dir')
            assert re.fullmatch(re.escape(row['run_name']) + r'_\d{8}_\d{6}', options['--run_tag']), 'train.sh must append its timestamp to the configured run-tag base'
            assert options['--output_dir'].startswith(row['output_root']+'/')
            assert not Path(options['--output_dir']).exists(), 'CPU preview created its output directory'
            checks['no_training_output'] = 'PASS'
            assert '--nproc_per_node=4' in proc.stdout and 'num_gpus:      4' in proc.stdout
            assert f'Config file:   {row["config"]}' in proc.stdout
            assert 'AMP=1 AMP_DTYPE=bf16 TF32=1' in proc.stdout
            assert 'USE_TMUX=0' in proc.stdout and 'DRY_RUN=1' in proc.stdout
            assert proc.stdout.count('[START] ') == 1
            checks['cpu_dry_run'] = 'PASS'
            checks['effective_reference_comparison'] = 'PASS'
            return dict(gcm=row['gcm'],station=row['station'],seed=row['seed'],variant=row['variant'],status='PASS',
                        checks=checks,log=str(VALIDATION/log_relative),argv=tokens[1:],effective_options=options)
        except Exception as exc:
            import traceback
            return dict(gcm=row['gcm'],station=row['station'],seed=row['seed'],variant=row['variant'],status='FAIL',
                        checks=checks,error=f'{type(exc).__name__}: {exc}', traceback=traceback.format_exc())
    began = time.monotonic()
    results = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        for i,result in enumerate(pool.map(validate,rows),1):
            results.append(result)
            if result['status']=='FAIL':
                print(f'FAIL {result}', flush=True)
            if i%60 == 0:
                print(f'{i}/648 installed scripts checked; {sum(r["status"]=="PASS" for r in results)} PASS.',flush=True)
    # Existing jobs may write their own results while these CPU previews run.
    no_results_created = all(r['checks'].get('no_training_output') == 'PASS' for r in results)
    report = dict(status='PASS' if all(r['status']=='PASS' for r in results) and no_results_created else 'FAIL',
                  total=648,passed=sum(r['status']=='PASS' for r in results),failed=sum(r['status']=='FAIL' for r in results),
                  no_training_results_created=no_results_created,
                  output_check='Each CPU preview reported a run directory that does not exist; existing training outputs are not traversed.',
                  reference_count=len(references),
                  max_pbs_name_length=max(len(r['pbs_name']) for r in rows),
                  distributions={key:dict(Counter(r[key] for r in rows)) for key in ['gcm','station','seed','variant']},
                  seconds=time.monotonic()-began,combinations=results)
    (args.out/'matrix_validation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ['combinations','distributions']}),flush=True)
    raise SystemExit(0 if report['status']=='PASS' else 1)


if __name__=='__main__':
    main()
