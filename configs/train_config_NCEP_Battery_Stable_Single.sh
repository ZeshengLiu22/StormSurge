#!/usr/bin/env bash
# Shared training protocol; single-head accuracy control. Stable checkpoint version is automatic.
source "$(dirname "${BASH_SOURCE[0]}")/configs_train/train_config_common.sh"
ROOT_DIR="/home/exouser/media/volume/PACT-Data/Emulator/Data/Grid4_New/NCEP/graphs"
STATION_JSON_DIR="/home/exouser/media/volume/PACT-Data/Emulator/station_json"
STATION="Battery"
MODEL="perceiver3"
ENCODER_TYPE="GraphSAGE"
TEMPORAL_BLOCK="Transformer"
HISTORY_HOURS_LIST=(12)
HEAD_TYPE="single"
DUAL_MODE="exceedance"
USE_SITE_ELEVATION=0
USE_BATHYMETRY=0
TRAIN_RATIO="0.6"
VAL_RATIO="0.2"
num_gpus=1
GRAD_ACCUM_STEPS=4
LR_LIST=("5e-3")
SCHEDULER="cosine"
WARMUP_EPOCHS=5
WARMUP_START_FACTOR=0.1
MIN_LR=1e-6
MAX_GRAD_NORM=0
# H100 GNN timing: strict determinism costs about 5x per training step.
# Override to 1 for exact same-environment reproduction/diagnostics.
DETERMINISTIC=0
USE_TMUX=0
DO_CONDA=0
# Activate the torchpyg environment before calling train.sh.
ALL_RESULTS_ROOT="./All_Results/Stable_V3"
