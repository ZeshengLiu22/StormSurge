#!/usr/bin/env bash
# CMIP6 PastOnly pressure-mode ablation: AWI, CBBT, CNN_GRU_36H, pressure=anomaly_plus_mean, seed=42.
# Reference objective unchanged: Global-MSE, T=0.0, EP=0.0.
# Primary selector: overall; secondary: exceedance (both retained by active code).
# Assignment-only configuration; contains no launch or submission command.

TRAIN_PY="train.py"
DO_CONDA=0
# Explicit runtime survives queue workers and tmux shell initialization.
PYTHON_BIN="${S0_PYTHON_BIN:-/glade/work/zel/conda-envs/derecho-ml/bin/python}"
num_gpus=4
USE_TMUX="${USE_TMUX:-1}"

ROOT_DIR="/glade/u/home/zel/scratch/PACT-Data-Final/CMIP6_AWI_PastOnly/graphs"
TEST_ROOT_DIR=""
PRESSURE_MODE="anomaly_plus_mean"
STATION="CBBT"
MODEL="perceiver3"
ENCODER_TYPE="CNN"
CNN_INTERMEDIATE_CHANNEL=29
TEMPORAL_BLOCK="GRU"
HISTORY_HOURS_LIST=(36)
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
GRAD_ACCUM_STEPS=1
LR_LIST=("5e-3")
EPOCHS=300
SCHEDULER="cosine"
WARMUP_EPOCHS=5
WARMUP_START_FACTOR=0.1
MIN_LR=1e-6
MAX_GRAD_NORM=0
DETERMINISTIC=0
# Inactive ReduceLROnPlateau defaults.
ROP_METRIC="val_all_rmse"
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

EXCEEDANCE_LOSS_WEIGHT=0.0
EPISODE_GT_ALIGNED_PEAK_WEIGHT=0.0

EXCEEDANCE_PERCENTILE=95

CHECKPOINT_SELECTION="overall"

ALL_RESULTS_ROOT="/glade/u/home/zel/scratch/PACT_StormSurge_FormalRuns/pressure_mode_ablation_cmip6/AWI/CBBT/anomaly_plus_mean"
RUN_DIR_NAME_STYLE="runname_timestamp"
PACT_RUN_NAME="CMIP6_AWI_CBBT_CNN_GRU_36H_panomaly_plus_mean_seed42_G0_T0000_EP0000"
PYTHON_RUN_TAG_BASE="CMIP6_AWI_CBBT_CNN_GRU_36H_panomaly_plus_mean_seed42_G0_T0000_EP0000"
