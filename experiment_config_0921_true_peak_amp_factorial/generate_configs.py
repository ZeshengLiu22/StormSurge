#!/usr/bin/env python3
"""Write the fixed 20-run factorial and command lists. Never submit jobs."""

import csv
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
EXPERIMENT_ID = "0921_true_peak_amp_factorial"
MAIN_SHA = "af53de756860765a9809b0c5c44bc828c2a08359"
RESULTS_ROOT = "/media/share/PACT/Results/All_results_0921_true_peak_amp_factorial"
PYTHON_BIN = "/media/volume/PACT-Data/conda_envs/torchpyg-cu124/bin/python"
STATIONS = ("CBBT", "Lewes", "Battery", "Boston")
CONDITIONS = (
    ("S0", "Single", "single", False, False),
    ("D0", "DualBase", "dual", False, False),
    ("D1", "Tail", "dual", True, False),
    ("D2", "Amp", "dual", False, True),
    ("D3", "TailAmp", "dual", True, True),
)

# Current production interface only; no sourced historical experiment profiles.
TEMPLATE = '''#!/usr/bin/env bash
# Formal matched factorial: {station} / {condition}. Exactly one run.
# Production main: {main_sha}
# Tail and true-peak Amp coefficients are fixed by the completed diagnosis.

TRAIN_PY="train.py"
PYTHON_BIN="{python_bin}"
DO_CONDA=0
num_gpus=1
USE_TMUX="${{USE_TMUX:-1}}"

ROOT_DIR="./Data/Grid4_New/NCEP/graphs"
TEST_ROOT_DIR=""
STATION="{station}"
MODEL="perceiver3"
ENCODER_TYPE="GraphSAGE"
CNN_INTERMEDIATE_CHANNEL=29
TEMPORAL_BLOCK="Transformer"
HEAD_TYPE="{head}"
HISTORY_HOURS_LIST=(24)
HIDDEN_CHANNELS=128
NUM_LAYERS=2
DROPOUT=0.05
HEAD_DROPOUT=0.05
NODE_READ_HEADS=8
TIME_READ_HEADS=8
TRANSFORMER_LAYERS=2
TRANSFORMER_FF_MULT=4.0
TRANSFORMER_DROPOUT=0.0
MAX_TIME_STEPS=32

TRAIN_RATIO="0.6"
VAL_RATIO="0.2"
SHUFFLE_YEARS=0
FUTURE_ONLY=0
FUTURE_YEAR_THRESHOLD=2030
SEED=42
STATION_JSON_DIR="./station_json"
USE_SITE_ELEVATION=0
USE_BATHYMETRY=0

X_NORM="zscore"
DISABLE_OOD=1
X_P_LO="1.0"
X_P_HI="99.0"
X_NODES_PER_GRAPH=0
X_CLIP=0
X_AUG=0
X_AUG_PROB=0
X_AUG_SCALE=0
X_AUG_BIAS=0

BATCH_SIZE=256
GRAD_ACCUM_STEPS=4
LR_LIST=("5e-3")
EPOCHS=300
SCHEDULER="cosine"
WARMUP_EPOCHS=5
WARMUP_START_FACTOR=0.1
MIN_LR=1e-6
MAX_GRAD_NORM=0
DETERMINISTIC=0
# Inactive ReduceLROnPlateau defaults.
ROP_METRIC="val_rmse_phys"
ROP_FACTOR=0.5
ROP_PATIENCE=20
ROP_THRESHOLD=1e-4
ROP_COOLDOWN=0
ROP_MIN_LR=1e-6

USE_AMP=1
AMP_DTYPE="bf16"
USE_TF32=1
NUM_WORKERS=0
TORCH_THREADS=1
PIN_MEMORY=0
PERSISTENT_WORKERS=0
PREFETCH_FACTOR=0
MP_CONTEXT="fork"

LOSS_MODE_LIST=("{loss_mode}")
TAIL_FRAC="0.05"
# In MSE runs the launcher omits tail_lambda; its production default is inactive.
TAIL_LAMBDA_LIST=("{tail_lambda}")
EXCESS_AMP_LOSS_WEIGHT={amp_weight}
# Inactive WMSE and slope defaults; neither objective is selected.
WMSE_Q_LIST=("95")
WMSE_ALPHA="4.0"
WMSE_S="0.10"
WMSE_USE_ABS=1
SLOPE_LAMBDA_LIST=("0.01")
SLOPE_MASK_S_LIST=("0.10")
SLOPE_ROBUST="charb"
SLOPE_CHARB_EPS="1e-3"
SLOPE_HUBER_DELTA="0.05"

# These head/branch settings are inactive for Single.
DUAL_MODE="exceedance"
EXCESS_FORMULATION="direct"
EXCEEDANCE_HEAD_EXPERIMENT=""
EXCEEDANCE_GATE_POOLING="mean"
GATE_MODE="window"
DUAL_ABLATION="none"
DUAL_LOSS={dual_loss}
EXCEEDANCE_PERCENTILE=95
BODY_LOSS_WEIGHT=1
EXCESS_LOSS_WEIGHT={excess_weight}
GATE_LOSS_WEIGHT={gate_weight}
SHAPE_LOSS_WEIGHT=0
SEVERITY_SHAPE_EPS=1e-6

CHECKPOINT_SELECTION="overall"
CHECKPOINT_OVERALL_TOL="0.01"
SAVE_AUX_CHECKPOINTS=0

ALL_RESULTS_ROOT="{results_root}"
RUN_DIR_NAME_STYLE="runname_timestamp"
PACT_RUN_NAME="{run_name}"
PYTHON_RUN_TAG_BASE="{run_name}"
'''


def main():
    configs = HERE / "configs"
    configs.mkdir(exist_ok=True)
    rows = []
    for station in STATIONS:
        commands = []
        for condition, label, head, tail, amp in CONDITIONS:
            dual = head == "dual"
            run_name = f"0921_{station}_{condition}_{label}"
            config = configs / f"train_config_{run_name}.sh"
            loss_mode = "mse_tail" if tail else "mse"
            tail_lambda = "0.025" if tail else "0.10"
            amp_weight = "0.003" if amp else "0"
            config.write_text(TEMPLATE.format(
                station=station, condition=condition, main_sha=MAIN_SHA,
                python_bin=PYTHON_BIN, head=head, loss_mode=loss_mode,
                tail_lambda=tail_lambda, amp_weight=amp_weight,
                dual_loss=int(dual), excess_weight=2 if dual else 1,
                gate_weight="0.5" if dual else "1", results_root=RESULTS_ROOT,
                run_name=run_name,
            ))
            relative = config.relative_to(REPO).as_posix()
            command = f"qsub_local train.sh {run_name} {relative}"
            commands.append(command)
            rows.append(dict(
                experiment_id=EXPERIMENT_ID, station=station, condition=condition,
                config_path=relative, head_type=head, loss_mode=loss_mode,
                tail_enabled=int(tail), tail_lambda="0.025" if tail else "0",
                amp_enabled=int(amp), amp_weight=amp_weight,
                exceedance_percentile="95" if dual else "N/A",
                body_loss_weight="1" if dual else "N/A",
                excess_loss_weight="2" if dual else "N/A",
                gate_loss_weight="0.5" if dual else "N/A", lr="5e-3", seed=42,
                checkpoint_selection="overall", run_name=run_name,
                results_root=RESULTS_ROOT, tail_lambda_config=tail_lambda,
                qsub_local_command=command,
            ))
        (HERE / f"commands_{station}.txt").write_text("\n".join(commands) + "\n")
    with (HERE / "manifest.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    (HERE / "commands_all.txt").write_text("\n".join(row["qsub_local_command"] for row in rows) + "\n")
    print(f"Generated {len(rows)} configs: five each for {', '.join(STATIONS)}. No jobs submitted.")


if __name__ == "__main__":
    main()
