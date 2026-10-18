#!/usr/bin/env bash
# Final candidate runtime. Pair files contain only identities and resolved paths.
INFER_PY="infer.py"
MODEL="perceiver3"
ENCODER_TYPE="GraphSAGE"
TEMPORAL_BLOCK="Transformer"
HISTORY_HOURS=24
MODEL_LABEL="PACT_TailPeak_overall"
STATION_JSON_DIR="./station_json"
USE_SITE_ELEVATION=0
USE_BATHYMETRY=0

FOUR_STATION_RESULTS_ROOT="/glade/u/home/zel/scratch/PACT-Derecho/PACT/FormalRuns_0925/CrossDatasetInference_4Stations"
PYTHON_BIN="${S0_PYTHON_BIN:-/glade/work/zel/conda-envs/pact-casper-cu126/bin/python}"
DO_CONDA=0
: "${USE_TMUX:=0}"
# Volta V100: FP16 Tensor Cores; BF16 and TF32 require newer GPUs.
: "${USE_AMP:=1}"
: "${AMP_DTYPE:=fp16}"
: "${USE_TF32:=0}"
# Conservative starting batch for the Casper V100 32 GB; may be overridden.
: "${BATCH_SIZE:=64}"
NUM_WORKERS=0
PIN_MEMORY=0
PERSISTENT_WORKERS=0
PREFETCH_FACTOR=0
MP_CONTEXT="fork"
TORCH_THREADS=1
STRICT_YEARS=1
