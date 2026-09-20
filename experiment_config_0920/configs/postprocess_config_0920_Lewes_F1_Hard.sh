#!/usr/bin/env bash
# Evaluation only. Use run_f1.py; this is not a train.sh config.
# Supply the already-selected F0 checkpoint via --checkpoint or F0_CHECKPOINT.
MODE="postprocess"
FORMULATION="F1_Hard"
STATION="Lewes"
PACT_RUN_NAME="0920_Lewes_F1_Hard"
SOURCE_RUN_NAME="0920_Lewes_F0_Soft"
SOURCE_CONFIG="experiment_config_0920/configs/train_config_0920_Lewes_F0_Soft.sh"
SOURCE_CHECKPOINT="/home/exouser/media/volume/PACT-Data/StormSurge/All_results_0920/0920_Lewes_F0_Soft__<TIMESTAMP>/best_*.pth"
F0_CHECKPOINT="${F0_CHECKPOINT:-}"
HARD_GATE_THRESHOLD=0.5

INFER_PY="infer.py"
PYTHON_BIN="/media/volume/PACT-Data/conda_envs/torchpyg-cu124/bin/python"
ROOT_DIR="/media/share/PACT/Data/Grid4_New/NCEP/graphs"
STATION_JSON_DIR="/media/volume/PACT-Data/StormSurge/station_json"
ALL_RESULTS_ROOT="/home/exouser/media/volume/PACT-Data/StormSurge/All_results_0920"
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
