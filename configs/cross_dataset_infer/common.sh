#!/usr/bin/env bash
# Final candidate runtime. Pair files contain only identities and resolved paths.
INFER_PY="infer.py"
STATION="Boston"
MODEL="perceiver3"
ENCODER_TYPE="GraphSAGE"
TEMPORAL_BLOCK="Transformer"
HEAD_TYPE="single"
HISTORY_HOURS=24
MODEL_LABEL="PACT_TailPeak_overall"
STATION_JSON_DIR="./station_json"
USE_SITE_ELEVATION=0
USE_BATHYMETRY=0

PYTHON_BIN="${S0_PYTHON_BIN:-/home/exouser/.conda/envs/torchpyg-cu12x/bin/python}"
DO_CONDA=0
: "${USE_TMUX:=0}"
USE_AMP=1
AMP_DTYPE="bf16"
USE_TF32=1
# Same batch as training; evaluation has no gradients. Validated on H100 80 GB.
BATCH_SIZE=256
NUM_WORKERS=0
PIN_MEMORY=0
PERSISTENT_WORKERS=0
PREFETCH_FACTOR=0
MP_CONTEXT="fork"
TORCH_THREADS=1
DUAL_DIAGNOSTICS=0
STRICT_YEARS=1
