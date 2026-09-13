#!/usr/bin/env python3
"""Generate the frozen standalone P2 suite after launcher regression passes."""
import argparse
import csv
import hashlib
import itertools
import json
from pathlib import Path
import re

STATIONS = ("CBBT", "Lewes", "Battery", "Boston")
RELATIVE = "experiment_config/P2_PeakAwareFactorial"
# Literal frozen template: generation and runtime never source P0/P1 configs.
TEMPLATE = r'''#!/usr/bin/env bash
# P2 peak-aware factorial screen. Standalone; no config inheritance.
# Run from the StormSurge repository: bash train.sh <this-config.sh>
TRAIN_PY="train.py"
ROOT_DIR="/media/share/PACT/Data/Grid4_New/NCEP/graphs"
STATION_JSON_DIR="/media/volume/PACT-Data/StormSurge/station_json"
TEST_ROOT_DIR=""
STATION="CBBT"

# Fixed GraphSAGE + Transformer, 24-hour history (no architecture sweep).
MODEL="perceiver3"
ENCODER_TYPE="GraphSAGE"
CNN_INTERMEDIATE_CHANNEL=29
TEMPORAL_BLOCK="Transformer"
HISTORY_HOURS_LIST=(24)
HIDDEN_CHANNELS=128
NUM_LAYERS=2
DROPOUT=0.05
HEAD_DROPOUT=0.05
NODE_READ_HEADS=8
TIME_READ_HEADS=8
TRANSFORMER_LAYERS=2
TRANSFORMER_FF_MULT="4.0"
TRANSFORMER_DROPOUT="0.0"
MAX_TIME_STEPS=32

# Chronological year-group split; station feature toggles are fixed.
TRAIN_RATIO="0.6"
VAL_RATIO="0.2"
SHUFFLE_YEARS=0
FUTURE_ONLY=0
FUTURE_YEAR_THRESHOLD=2030
SEED=42
USE_SITE_ELEVATION=0
USE_BATHYMETRY=0

# Fixed supervised dual exceedance head.
HEAD_TYPE="dual"
GATE_MODE="window"
DUAL_MODE="exceedance"
DUAL_LOSS=1
DUAL_ABLATION="none"
EXCEEDANCE_PERCENTILE=95
LOSS_MODE_LIST=("mse")

# Frozen branch weights. Tail is active ONLY under mse_tail.
BODY_LOSS_WEIGHT=1
EXCESS_LOSS_WEIGHT=2
GATE_LOSS_WEIGHT=0.5
TAIL_FRAC="0.05"
TAIL_LAMBDA_LIST=("0.025")  # Inactive under mse; active coefficient under mse_tail.
# P0 slope controls retained but inactive: P2 never uses a slope loss mode.
SLOPE_LAMBDA_LIST=("0.01")
SLOPE_MASK_S_LIST=("0.10")
SLOPE_ROBUST="charb"
SLOPE_CHARB_EPS="1e-3"
SLOPE_HUBER_DELTA="0.05"
# Threshold infrastructure only; P0 never selects a WMSE loss mode.
WMSE_Q_LIST=("95")
WMSE_ALPHA="4.0"
WMSE_S="0.10"
WMSE_USE_ABS=1

# OOD is completely disabled, including clipping and feature augmentation.
DISABLE_OOD=1
X_NORM="zscore"
X_CLIP="0"
X_AUG=0
X_AUG_PROB="0"
X_AUG_SCALE="0"
X_AUG_BIAS="0"
# Passed by the current launcher; unused by zscore statistics.
X_P_LO="1.0"
X_P_HI="99.0"
X_NODES_PER_GRAPH=256

# Fixed optimizer protocol: train.py uses Adam with weight_decay=1e-5.
BATCH_SIZE=256
GRAD_ACCUM_STEPS=4
LR_LIST=("5e-3")
EPOCHS=300
SCHEDULER="cosine"
WARMUP_EPOCHS=5
WARMUP_START_FACTOR="0.1"
MIN_LR="1e-6"
MAX_GRAD_NORM=0
DETERMINISTIC=0
# Inactive with the fixed cosine scheduler; explicit for resolved snapshots.
ROP_METRIC="val_rmse_phys"
ROP_FACTOR="0.5"
ROP_PATIENCE=20
ROP_THRESHOLD="1e-4"
ROP_COOLDOWN=0
ROP_MIN_LR="1e-6"

# One local H100. Accumulation is supported only with num_gpus=1.
num_gpus=1
export CUDA_VISIBLE_DEVICES=0
USE_AMP=1
AMP_DTYPE="bf16"
USE_TF32=1
TORCH_THREADS=1
NUM_WORKERS=0
PIN_MEMORY=0
PERSISTENT_WORKERS=0
PREFETCH_FACTOR=0
MP_CONTEXT="fork"

# Use the verified environment directly; no activation or HPC modules.
PYTHON_BIN="/media/volume/PACT-Data/conda_envs/torchpyg-cu124/bin/python"
DO_CONDA=0
CONDA_ENV="/media/volume/PACT-Data/conda_envs/torchpyg-cu124"
export PYTHONDONTWRITEBYTECODE=1

# Queue/tmux identity is separate from the semantic artifact name.
USE_TMUX=1
SESSION_NAME="${SESSION_NAME:-P2_CBBT_direct_T0_A0_S0_P0}"
PACT_RUN_NAME="NCEP_CBBT_24h_dual_direct_T0_A0_S0_P0"
ALL_RESULTS_ROOT="./All_Results/P2_PeakAwareFactorial"
RUN_DIR_NAME_STYLE="runname_timestamp"
PYTHON_RUN_TAG_BASE="${PACT_RUN_NAME}"

# Peak-aware mechanism controls: the only P2 variations besides loss mode.
EXCESS_FORMULATION="direct"
EXCESS_AMP_LOSS_WEIGHT=0
EXCESS_AMP_POOL="max"
EXCESS_AMP_BETA=20
SHAPE_LOSS_WEIGHT=0
SEVERITY_SHAPE_EPS=1e-6
PEAK_LOSS_WEIGHT=0
PEAK_POOL="max"
PEAK_POOL_BETA=20

# VAL overall RMSE remains the primary checkpoint criterion for every run.
CHECKPOINT_SELECTION="overall"
SAVE_AUX_CHECKPOINTS=0
CHECKPOINT_OVERALL_TOL=0.01
'''


def save(path, content):
    if path.exists():
        assert path.read_text() == content, f'Refusing to replace a changed generated file: {path}'
    else:
        path.write_text(content)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument('--output-dir', type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    repo, output = args.repo.resolve(), args.output_dir.resolve()
    regression = json.loads((output / 'validation/launcher_regression.json').read_text())
    assert regression['status'] == 'PASS' and regression['old_configs_checked'] == 64
    assert regression['launcher_sha256'] == hashlib.sha256((repo / 'train.sh').read_bytes()).hexdigest()
    output.mkdir(parents=True, exist_ok=True)
    rows, commands = [], {station: [] for station in STATIONS}
    for station in STATIONS:
        for formulation in ('direct', 'severity_shape'):
            form = 'direct' if formulation == 'direct' else 'severity'
            for tail, amp, shape, peak in itertools.product((0, 1), (0, 1), (0,) if form == 'direct' else (0, 1), (0, 1)):
                name = f'NCEP_{station}_24h_dual_{form}_T{tail}_A{amp}_S{shape}_P{peak}'
                label = f'P2_{station}_{form}_T{tail}_A{amp}_S{shape}_P{peak}'
                filename = f'train_config_{name}.sh'
                relative = f'{RELATIVE}/{filename}'
                mode = 'mse_tail' if tail else 'mse'
                amp_weight = ('0.0007' if form == 'direct' else '0.0025') if amp else '0'
                shape_weight, peak_weight = ('0.00003' if shape else '0'), ('0.0035' if peak else '0')
                settings = dict(STATION=f'"{station}"', LOSS_MODE_LIST=f'("{mode}")',
                                PACT_RUN_NAME=f'"{name}"', SESSION_NAME='"${SESSION_NAME:-' + label + '}"',
                                EXCESS_FORMULATION=f'"{formulation}"', EXCESS_AMP_LOSS_WEIGHT=amp_weight,
                                SHAPE_LOSS_WEIGHT=shape_weight, PEAK_LOSS_WEIGHT=peak_weight)
                text = TEMPLATE
                for key, value in settings.items():
                    text, count = re.subn(rf'^{key}=.*$', lambda match: f'{key}={value}', text, flags=re.M)
                    assert count == 1, key
                save(output / filename, text)
                (output / filename).chmod(0o755)
                command = f'qsub_local train.sh {label} {relative}'
                commands[station].append(command)
                rows.append(dict(experiment_id=len(rows) + 1, config_path=relative, station=station,
                                 formulation=formulation, tail_enabled=tail, tail_lambda='0.025' if tail else '0',
                                 configured_tail_lambda='0.025', amp_enabled=amp, excess_amp_loss_weight=amp_weight,
                                 shape_enabled=shape, shape_loss_weight=shape_weight, peak_enabled=peak,
                                 peak_loss_weight=peak_weight, loss_mode=mode, seed=42,
                                 checkpoint_selection='overall', save_aux_checkpoints=0,
                                 checkpoint_overall_tol='0.01', run_tag=name, run_name=name,
                                 python_run_tag_base=name, history_hours=24, excess_amp_pool='max',
                                 excess_amp_beta=20, peak_pool='max', peak_pool_beta=20,
                                 severity_shape_eps='1e-6', qsub_label=label, qsub_command=command,
                                 results_root='All_Results/P2_PeakAwareFactorial'))
    assert len(rows) == 96
    import io
    csv_text = io.StringIO(newline='')
    writer = csv.DictWriter(csv_text, fieldnames=list(rows[0]), lineterminator='\n')
    writer.writeheader()
    writer.writerows(rows)
    save(output / 'manifest.csv', csv_text.getvalue())
    for station in STATIONS:
        assert len(commands[station]) == 24
        save(output / f'commands_{station}.txt', '\n'.join(commands[station]) + '\n')
    save(output / 'commands_all.txt', '\n'.join(command for station in STATIONS for command in commands[station]) + '\n')
    save(output / 'RUN_ORDER.md', '''# P2 suggested run order

Run from the StormSurge repository root in the existing shell that defines
`qsub_local`. These are deferred execution instructions; generation/validation
submits nothing. Use one station list at a time with the known single-H100 queue.

| Batch | Station | Experiment IDs | Direct | Severity-shape | Command list |
| --- | --- | --- | ---: | ---: | --- |
| 1 | CBBT | 1–24 | 8 | 16 | `commands_CBBT.txt` |
| 2 | Lewes | 25–48 | 8 | 16 | `commands_Lewes.txt` |
| 3 | Battery | 49–72 | 8 | 16 | `commands_Battery.txt` |
| 4 | Boston | 73–96 | 8 | 16 | `commands_Boston.txt` |

Within each station, direct precedes severity-shape. Binary order is Tail,
Amplitude, Shape (always zero for direct), Peak, with Peak varying fastest.
The first config is direct T0 A0 S0 P0. All interaction combinations follow.

`commands_all.txt` concatenates the four station lists in the order above.
Choose the combined list or the station lists to avoid duplicate submissions.
The command form is `qsub_local train.sh <label> <config_path>`.

Scientific settings do not change between batches. Each config is a single
seed-42 run, with one history, learning rate and loss-mode entry. The queue's
existing single-slot behavior serializes jobs; no queue settings were changed.
Artifact directories use the semantic name followed by the execution timestamp.
''')
    print('Generated 96 standalone configs, manifest, 96 combined commands, four 24-command station lists and RUN_ORDER.md.')


if __name__ == '__main__':
    main()
