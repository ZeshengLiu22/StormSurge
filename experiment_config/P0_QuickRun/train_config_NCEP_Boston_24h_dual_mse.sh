#!/usr/bin/env bash
# P0 frozen protocol. Fully independent config; no runtime inheritance.
# Run from the StormSurge repository: bash train.sh <this-config.sh>
TRAIN_PY="train.py"
ROOT_DIR="/media/share/PACT/Data/Grid4_New/NCEP/graphs"
STATION_JSON_DIR="/media/volume/PACT-Data/StormSurge/station_json"
TEST_ROOT_DIR=""
STATION="Boston"

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

# Current supervised exceedance head; single-head parser sets dual_loss=0.
HEAD_TYPE="dual"
GATE_MODE="window"
DUAL_MODE="exceedance"
DUAL_LOSS=1
DUAL_ABLATION="none"
EXCEEDANCE_PERCENTILE=95
LOSS_MODE_LIST=("mse")

# FINAL weights, identical in all 32 configs. Terms activate by head/mode.
BODY_LOSS_WEIGHT=1
EXCESS_LOSS_WEIGHT=2
GATE_LOSS_WEIGHT=0.5
TAIL_FRAC="0.05"
TAIL_LAMBDA_LIST=("0.025")
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
SESSION_NAME="${SESSION_NAME:-P0_Boston_dual_mse}"
PACT_RUN_NAME="NCEP_Boston_24h_dual_mse"
ALL_RESULTS_ROOT="./All_Results/P0_QuickRun"
RUN_DIR_NAME_STYLE="runname_timestamp"
