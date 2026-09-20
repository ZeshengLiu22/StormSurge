#!/usr/bin/env python3
"""Render 12 training and four F0-derived evaluation configs; never launch jobs."""

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import re
import shlex


HERE = Path(__file__).resolve().parent
REPO = HERE.parent
RESULTS = Path('/home/exouser/media/volume/PACT-Data/StormSurge/All_results_0920')
PYTHON = '/media/volume/PACT-Data/conda_envs/torchpyg-cu124/bin/python'
STATIONS = ('CBBT', 'Lewes', 'Battery', 'Boston')
FORMULATIONS = ('F0_Soft', 'F1_Hard', 'F2_AdditiveEvent', 'F3_AdditiveAll')
FACTORS = {'F0_Soft': ('soft_gate', 'event'), 'F2_AdditiveEvent': ('additive', 'event'),
           'F3_AdditiveAll': ('additive', 'all')}
IMPLEMENTATION_COMMIT = 'e62b42289615dc47986a442f8ee0484f525eba57'


def run_name(station, formulation):
    return f'0920_{station}_{formulation}'


def config_path(station, formulation):
    prefix = 'postprocess' if formulation == 'F1_Hard' else 'train'
    return f'experiment_config_0920/configs/{prefix}_config_{run_name(station, formulation)}.sh'


def reference_path(station):
    return REPO / f'experiment_config_0919/configs/train_config_0919_{station}_Legacy_Mean.sh'


def render_training(station, formulation):
    source = reference_path(station).read_text()
    name = run_name(station, formulation)
    changes = dict(SESSION_NAME=f'"${{SESSION_NAME:-{name}}}"', PACT_RUN_NAME=f'"{name}"',
                   ALL_RESULTS_ROOT=f'"{RESULTS}"')
    for key, value in changes.items():
        source, count = re.subn(rf'^{key}=.*$', lambda _: f'{key}={value}', source, flags=re.M)
        assert count == 1, (key, count)
    source, count = re.subn(r'^EXCEEDANCE_HEAD_EXPERIMENT=.*\n', '', source, flags=re.M)
    assert count == 1  # Omission selects the production head, not the capacity experiment.
    reconstruction, scope = FACTORS[formulation]
    source = source.replace('EXCESS_FORMULATION="direct"\n', 'EXCESS_FORMULATION="direct"\n'
                            f'DIRECT_DUAL_RECONSTRUCTION="{reconstruction}"\nEXCESS_SUPERVISION_SCOPE="{scope}"\n')
    source = source.replace('# 0919 direct-dual excess-capacity / gate-pooling study.',
                            '# 0920 direct-dual reconstruction / supervision-scope study.')
    return source


def render_postprocess(station):
    name = run_name(station, 'F1_Hard')
    source = run_name(station, 'F0_Soft')
    return f'''#!/usr/bin/env bash
# Evaluation only. Use run_f1.py; this is not a train.sh config.
# Supply the already-selected F0 checkpoint via --checkpoint or F0_CHECKPOINT.
MODE="postprocess"
FORMULATION="F1_Hard"
STATION="{station}"
PACT_RUN_NAME="{name}"
SOURCE_RUN_NAME="{source}"
SOURCE_CONFIG="{config_path(station, 'F0_Soft')}"
SOURCE_CHECKPOINT="{RESULTS}/{source}__<TIMESTAMP>/best_*.pth"
F0_CHECKPOINT="${{F0_CHECKPOINT:-}}"
HARD_GATE_THRESHOLD=0.5

INFER_PY="infer.py"
PYTHON_BIN="{PYTHON}"
ROOT_DIR="/media/share/PACT/Data/Grid4_New/NCEP/graphs"
STATION_JSON_DIR="/media/volume/PACT-Data/StormSurge/station_json"
ALL_RESULTS_ROOT="{RESULTS}"
EVALUATION_SCOPE="test"
MODEL="perceiver3"
HEAD_TYPE="dual"
ENCODER_TYPE="GraphSAGE"
TEMPORAL_BLOCK="Transformer"
HISTORY_HOURS=24
USE_SITE_ELEVATION=0
USE_BATHYMETRY=0
BATCH_SIZE=256
TORCH_THREADS=1
NUM_WORKERS=0
PIN_MEMORY=0
PERSISTENT_WORKERS=0
PREFETCH_FACTOR=0
MP_CONTEXT="fork"
DEVICE="cuda"
export CUDA_VISIBLE_DEVICES=0
USE_AMP=1
AMP_DTYPE="bf16"
USE_TF32=1
DUAL_DIAGNOSTICS=1
SAVE_NPZ=1
export PYTHONDONTWRITEBYTECODE=1
'''


def manifest_rows():
    for station in STATIONS:
        for formulation in FORMULATIONS:
            postprocess = formulation == 'F1_Hard'
            name = run_name(station, formulation)
            path = config_path(station, formulation)
            reconstruction, scope = FACTORS['F0_Soft' if postprocess else formulation]
            command = (shlex.join([PYTHON, '-B', 'experiment_config_0920/run_f1.py', path]) if postprocess else
                       shlex.join(['qsub_local', 'train.sh', name, path]))
            yield dict(station=station, formulation=formulation, mode='postprocess' if postprocess else 'train',
                       run_name=name, config_path=path,
                       source_checkpoint=run_name(station, 'F0_Soft') if postprocess else '',
                       source_config=config_path(station, 'F0_Soft') if postprocess else '',
                       source_checkpoint_pattern=str(RESULTS / f'{run_name(station, "F0_Soft")}__<TIMESTAMP>/best_*.pth')
                       if postprocess else '', hard_gate_threshold='0.5' if postprocess else '',
                       direct_dual_reconstruction=reconstruction, excess_supervision_scope=scope,
                       head_type='dual', excess_formulation='direct', gate_pooling='mean',
                       loss_mode='mse', body_loss_weight=1, excess_loss_weight=2, gate_loss_weight=.5,
                       seed=42, deterministic=0, lr='5e-3', epochs='' if postprocess else 300,
                       history_hours=24, hidden_channels=128, batch_size=256,
                       grad_accum_steps='' if postprocess else 4, amp_dtype='bf16', tf32=1,
                       checkpoint_selection='source_F0_overall' if postprocess else 'overall',
                       output_dir=str(RESULTS / f'{name}__<TIMESTAMP>'), command=command)


def expected_files():
    files = {}
    for station in STATIONS:
        for formulation in FORMULATIONS:
            relative = str(Path(config_path(station, formulation)).relative_to('experiment_config_0920'))
            files[relative] = (render_postprocess(station) if formulation == 'F1_Hard' else
                               render_training(station, formulation))
    rows = list(manifest_rows())
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
    writer.writeheader()
    writer.writerows(rows)
    files['manifest.csv'] = stream.getvalue()
    files['commands_all.txt'] = ('# 12 training submissions and 4 evaluation commands; never run this as a batch script.\n'
        '# F1 requires F0_CHECKPOINT set to that station\'s already-selected F0 checkpoint.\n' +
        ''.join(f'# {row["mode"]}: {row["run_name"]}\n{row["command"]}\n' for row in rows))
    return files


def source_hashes():
    paths = [*sorted((REPO / 'emulator').rglob('*.py')),
             *[REPO / name for name in ('train.py', 'train.sh', 'infer.py', 'infer.sh')]]
    return {str(path.relative_to(REPO)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


def reference_hashes():
    return {str(path.relative_to(REPO)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted((REPO / 'experiment_config_0919').rglob('*')) if path.is_file()
            and '__pycache__' not in path.parts}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    expected = expected_files()
    actual = {str(path.relative_to(HERE)) for path in (HERE / 'configs').glob('*.sh')}
    assert not actual - set(expected), 'Unexpected config files'
    for relative, content in expected.items():
        path = HERE / relative
        if args.check:
            assert path.read_text() == content, f'Generated file differs: {relative}'
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)
    provenance = dict(implementation_commit=IMPLEMENTATION_COMMIT,
                      canonical_reference='experiment_config_0919', source_sha256=source_hashes(),
                      reference_sha256=reference_hashes())
    path = HERE / 'provenance.json'
    if path.exists():
        assert json.loads(path.read_text()) == provenance, 'Implementation or frozen 0919 reference changed'
    elif args.check:
        raise AssertionError('Missing provenance.json')
    else:
        path.write_text(json.dumps(provenance, indent=2) + '\n')
    print(f'{"Checked" if args.check else "Generated"}: 16 configs = 12 training + 4 F1 post-processing; no jobs launched.')


if __name__ == '__main__':
    main()
