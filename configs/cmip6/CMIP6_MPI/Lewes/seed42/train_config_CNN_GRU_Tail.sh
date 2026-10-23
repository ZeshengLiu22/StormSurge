#!/usr/bin/env bash
# Formal CMIP6 historical PastOnly: CMIP6_MPI, Lewes, seed 42, CNN_GRU_Tail.
# Formal NCEP source: configs/baselines_PeakAware/baseline_cnn_gru/Lewes/seed42/train_config_T0250_EP0000.sh
# Source SHA256: 665e402e9dce054cdbafa4d21438e09b03a12f1cc4f951828a65f85c601961ba
# Assignment-only; one 36H/LR combination; historical best_overall.pt is reused for future inference.

TRAIN_PY="train.py"
num_gpus=4
USE_TMUX=0

ROOT_DIR=/glade/u/home/zel/scratch/PACT-Data-Final/CMIP6_MPI_PastOnly/graphs
TEST_ROOT_DIR=''
STATION="Lewes"
MODEL="baseline_cnn_gru"
NODE_ATTENTION_BACKEND="math"
WEIGHT_DECAY_POLICY="module"
# Static models require 0; recurrent histories may be any supported multiple of 6.
HISTORY_HOURS_LIST=(36)
BASELINE_POOLING="adaptive"
BASELINE_POOL_GRID=4
BASELINE_STEM_CHANNELS=32
HIDDEN_CHANNELS=128
DROPOUT=0.05
PRESSURE_MODE="raw"

TRAIN_RATIO=0.6
VAL_RATIO=0.2
SHUFFLE_YEARS=0
FUTURE_ONLY=0
FUTURE_YEAR_THRESHOLD=2030
SEED=42
STATION_JSON_DIR="./station_json"
USE_SITE_ELEVATION=0
USE_BATHYMETRY=0

X_NORM="zscore"
DISABLE_OOD=1
X_P_LO=1.0
X_P_HI=99.0
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
MAX_GRAD_NORM=0.25
DETERMINISTIC=0
ROP_METRIC="val_all_rmse"
ROP_FACTOR=0.5
ROP_PATIENCE=20
ROP_THRESHOLD=1e-4
ROP_COOLDOWN=0
ROP_MIN_LR=1e-6

# Tail-only: Global MSE + 0.025 Tail + 0.0 GT-aligned Episode Peak; existing ForecastLoss.
EXCEEDANCE_PERCENTILE=95
EXCEEDANCE_LOSS_WEIGHT=0.025
EPISODE_GT_ALIGNED_PEAK_WEIGHT=0.0
CHECKPOINT_SELECTION="overall"

USE_AMP=1
AMP_DTYPE="bf16"
USE_TF32=1
NUM_WORKERS=0
TORCH_THREADS=1
PIN_MEMORY=0
PERSISTENT_WORKERS=0
PREFETCH_FACTOR=0
MP_CONTEXT="fork"

ALL_RESULTS_ROOT=/glade/u/home/zel/scratch/PACT_StormSurge_FormalRuns_1021_LR5e-3_Clip0.25/cmip6/CMIP6_MPI/Lewes/seed42/CNN_GRU_Tail
RUN_DIR_NAME_STYLE=timestamp_runname

PACT_RUN_NAME=CMIP6_MPI_Lewes_seed42_CNN_GRU_Tail_36H_lr5e-3_clip0.25
PYTHON_RUN_TAG_BASE=CMIP6_MPI_Lewes_seed42_CNN_GRU_Tail_36H_lr5e-3_clip0.25

PYTHON_BIN=/glade/work/zel/conda-envs/derecho-ml/bin/python

DO_CONDA=0
