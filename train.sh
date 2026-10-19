#!/usr/bin/env bash
# Run single-node training with the already activated derecho-ml environment.
# Usage: bash train.sh <train-config.sh>
# The config argument is relative to the caller's working directory.
# Paths inside the config are relative to this script's directory (repo root).
set -euo pipefail

# Stop direct child processes when the launcher receives SIGINT or SIGTERM.
# Inputs: none. Prints a trap message; returns success even if no child exists.
cleanup() {
  echo "[TRAP] Received SIGINT/SIGTERM; stopping child processes."
  pkill -P "$$" 2>/dev/null || true
}
trap cleanup INT TERM

# Locate the repository and config before changing the working directory.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
WORKDIR="${SCRIPT_DIR}"
CONFIG_PATH="${1:-}"
if [[ -z "${CONFIG_PATH}" || "${CONFIG_PATH}" == "--_tmux_inner" ]]; then
  echo "Usage: bash train.sh <train-config.sh>" >&2
  exit 2
fi
if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "[FATAL] Config not found: ${CONFIG_PATH}" >&2
  exit 2
fi
CONFIG_PATH="$(cd -- "$(dirname -- "${CONFIG_PATH}")" && pwd)/$(basename -- "${CONFIG_PATH}")"
cd -- "${WORKDIR}"

SCRIPT_PATH="${SCRIPT_DIR}/$(basename -- "${BASH_SOURCE[0]}")"

# Use the active environment; support interpreter overrides in older configs.
PYTHON_BIN="${PYTHON_BIN:-${S0_PYTHON_BIN:-${CONDA_PREFIX:+${CONDA_PREFIX}/bin/python}}}"
: "${PYTHON_BIN:=python}"
: "${S0_PYTHON_BIN:=${PYTHON_BIN}}"

# Load experiment settings; apply defaults for omitted values below.
set +u
# shellcheck disable=SC1090
source "${CONFIG_PATH}"
set -u

# Runtime and data loading
: "${TRAIN_PY:=./train.py}"
: "${num_gpus:=1}"
: "${USE_AMP:=0}"
: "${AMP_DTYPE:=bf16}"             # bf16 | fp16
: "${USE_TF32:=0}"
: "${TORCH_THREADS:=1}"
: "${NUM_WORKERS:=4}"
: "${PIN_MEMORY:=0}"
: "${PERSISTENT_WORKERS:=0}"
: "${PREFETCH_FACTOR:=2}"
: "${MP_CONTEXT:=fork}"
: "${USE_TMUX:=1}"
: "${DRY_RUN:=0}"

# Data, chronological splits, and TRAIN normalization
: "${ROOT_DIR:=./dataset/NCEP/graphs}"
: "${TEST_ROOT_DIR:=}"
: "${PRESSURE_MODE:=anomaly}"       # anomaly | raw | anomaly_plus_mean
: "${STATION:=}"
: "${SEED:=42}"
: "${TRAIN_RATIO:=0.6}"
: "${VAL_RATIO:=0.2}"
: "${SHUFFLE_YEARS:=0}"
: "${FUTURE_ONLY:=0}"
: "${FUTURE_YEAR_THRESHOLD:=2030}"
: "${STATION_JSON_DIR:=./station_json}"
: "${USE_SITE_ELEVATION:=1}"
: "${USE_BATHYMETRY:=0}"
: "${X_NORM:=robust}"              # zscore | robust | mag
: "${X_P_LO:=1.0}"
: "${X_P_HI:=99.0}"
: "${X_NODES_PER_GRAPH:=256}"
: "${X_CLIP:=5.0}"
: "${X_AUG:=1}"
: "${X_AUG_PROB:=1.0}"
: "${X_AUG_SCALE:=0.05}"
: "${X_AUG_BIAS:=0.02}"
: "${DISABLE_OOD:=0}"

# Model and history
: "${MODEL:=baseline}"
: "${ENCODER_TYPE:=GraphSAGE}"
: "${CNN_INTERMEDIATE_CHANNEL:=29}"
: "${TEMPORAL_BLOCK:=Transformer}"
: "${HIDDEN_CHANNELS:=64}"
: "${NUM_LAYERS:=2}"
: "${DROPOUT:=0.05}"
: "${HEAD_DROPOUT:=0.0}"
: "${NODE_READ_HEADS:=8}"
: "${TIME_READ_HEADS:=8}"
: "${TRANSFORMER_LAYERS:=2}"
: "${TRANSFORMER_FF_MULT:=4.0}"
: "${TRANSFORMER_DROPOUT:=0.05}"
: "${MAX_TIME_STEPS:=32}"
if [[ -z "${HISTORY_HOURS_LIST+x}" ]]; then HISTORY_HOURS_LIST=(24); fi

# Optimization and scheduling
: "${BATCH_SIZE:=256}"
: "${GRAD_ACCUM_STEPS:=1}"
: "${EPOCHS:=300}"
: "${SCHEDULER:=cosine}"           # cosine | rop
: "${WARMUP_EPOCHS:=5}"
: "${WARMUP_START_FACTOR:=0.1}"
: "${MIN_LR:=1e-6}"
: "${MAX_GRAD_NORM:=0}"
: "${DETERMINISTIC:=0}"
: "${ROP_METRIC:=val_all_rmse}"   # val_all_rmse | val_exceedance_rmse
: "${ROP_FACTOR:=.5}"
: "${ROP_PATIENCE:=20}"
: "${ROP_THRESHOLD:=1e-4}"
: "${ROP_COOLDOWN:=0}"
: "${ROP_MIN_LR:=1e-6}"
if [[ -z "${LR_LIST+x}" ]]; then LR_LIST=("3e-3"); fi

# Objective: physical Global MSE + strict Tail MSE + episode peak MSE
: "${EXCEEDANCE_PERCENTILE:=95}"
: "${EXCEEDANCE_LOSS_WEIGHT:=0}"
: "${EPISODE_GT_ALIGNED_PEAK_WEIGHT:=0.0}"

# Checkpoint selection and output
: "${CHECKPOINT_SELECTION:=exceedance}"  # exceedance (primary) | overall (sensitivity)
: "${ALL_RESULTS_ROOT:=./All_Results}"
: "${RUN_DIR_NAME_STYLE:=timestamp_runname}"
: "${PYTHON_RUN_TAG_BASE:=}"

# Validate experiment settings before creating artifacts.
PRESSURE_TAG=""
case "${PRESSURE_MODE}" in
  anomaly) ;;
  raw) PRESSURE_TAG="_praw" ;;
  anomaly_plus_mean) PRESSURE_TAG="_panomaly_mean" ;;
  *) echo "[FATAL] PRESSURE_MODE must be anomaly, raw, or anomaly_plus_mean; got '${PRESSURE_MODE}'" >&2; exit 2 ;;
esac

case "${SHUFFLE_YEARS,,}" in
  1|true|yes|y|on) SHUFFLE_YEARS=1 ;;
  0|false|no|n|off) SHUFFLE_YEARS=0 ;;
  *) echo "[FATAL] SHUFFLE_YEARS must be 0/1 or true/false; got '${SHUFFLE_YEARS}'"; exit 1 ;;
esac

case "${FUTURE_ONLY,,}" in
  1|true|yes|y|on) FUTURE_ONLY=1 ;;
  0|false|no|n|off) FUTURE_ONLY=0 ;;
  *) echo "[FATAL] FUTURE_ONLY must be 0/1 or true/false; got '${FUTURE_ONLY}'"; exit 1 ;;
esac

case "${ENCODER_TYPE}" in
  GraphSAGE|CNN) ;;
  *) echo "[FATAL] ENCODER_TYPE must be GraphSAGE or CNN; got '${ENCODER_TYPE}'"; exit 1 ;;
esac

case "${TEMPORAL_BLOCK,,}" in
  transformer|attn) TEMPORAL_BLOCK="Transformer" ;;
  mlp) TEMPORAL_BLOCK="MLP" ;;
  lstm) TEMPORAL_BLOCK="LSTM" ;;
  gru) TEMPORAL_BLOCK="GRU" ;;
  *) echo "[FATAL] TEMPORAL_BLOCK must be MLP, LSTM, GRU, or Transformer; got '${TEMPORAL_BLOCK}'"; exit 1 ;;
esac

case "${CHECKPOINT_SELECTION}" in
  overall|exceedance) ;;
  *) echo "[FATAL] CHECKPOINT_SELECTION must be overall or exceedance." >&2; exit 2 ;;
esac

if [[ ! "${GRAD_ACCUM_STEPS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "[FATAL] GRAD_ACCUM_STEPS must be a positive integer; got '${GRAD_ACCUM_STEPS}'"
  exit 1
fi

# Resolve the interpreter before creating artifacts or starting tmux.
: "${PYTHON_BIN:=${S0_PYTHON_BIN}}"
if ! PYTHON_EXECUTABLE="$(command -v -- "${PYTHON_BIN}")" || [[ ! -x "${PYTHON_EXECUTABLE}" ]]; then
  echo "[FATAL] Python interpreter not found or not executable: ${PYTHON_BIN}" >&2
  exit 2
fi
PYTHON_BIN="${PYTHON_EXECUTABLE}"
if [[ "${PYTHON_BIN}" != /* ]]; then
  PYTHON_BIN="${WORKDIR}/${PYTHON_BIN}"
fi

# Print the dataset label for a graph directory.
# Args: $1 is a path; a leaf named graphs uses its parent directory name.
# Output: one label on stdout; returns the basename command status.
infer_dataset_tag () {
  local p="$1"
  p="${p%/}"
  if [[ "$(basename "$p")" == "graphs" ]]; then
    basename "$(dirname "$p")"
  else
    basename "$p"
  fi
}

TRAIN_DATA_TAG="$(infer_dataset_tag "$ROOT_DIR")"
if [[ -n "${TEST_ROOT_DIR}" ]]; then
  TEST_DATA_TAG="$(infer_dataset_tag "$TEST_ROOT_DIR")"
else
  TEST_DATA_TAG="${TRAIN_DATA_TAG}"
fi

# Apply the in-distribution baseline settings when requested.
if [[ "${DISABLE_OOD}" == "1" ]]; then
  X_NORM="zscore"
  X_CLIP="0"
  X_AUG="0"
fi

# Configure distributed training on the current node.
_ddp_ngpu="${SLURM_NTASKS_PER_NODE:-${num_gpus}}"
num_gpus="${_ddp_ngpu}"

if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  IFS=',' read -r -a _CVD_ARR <<< "${CUDA_VISIBLE_DEVICES}"
  _NGPU_VISIBLE="${#_CVD_ARR[@]}"
  if [[ "${_NGPU_VISIBLE}" -gt 0 && "${num_gpus}" -gt "${_NGPU_VISIBLE}" ]]; then
    echo "[WARN] num_gpus=${num_gpus} > CUDA_VISIBLE_DEVICES count=${_NGPU_VISIBLE}. Clamping."
    num_gpus="${_NGPU_VISIBLE}"
  fi
fi

if (( GRAD_ACCUM_STEPS > 1 && num_gpus != 1 )); then
  echo "[FATAL] GRAD_ACCUM_STEPS=${GRAD_ACCUM_STEPS} is supported only with num_gpus=1; got num_gpus=${num_gpus}."
  exit 1
fi

MASTER_ADDR="127.0.0.1"
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  MASTER_PORT=$((29500 + (SLURM_JOB_ID % 1000)))
else
  MASTER_PORT=$((29500 + (RANDOM % 1000)))
fi
export MASTER_ADDR MASTER_PORT

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${TORCH_THREADS}"
export MKL_NUM_THREADS="${TORCH_THREADS}"

unset NCCL_ASYNC_ERROR_HANDLING
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_DISTRIBUTED_DEBUG=OFF

# Use the same interpreter for single-process and distributed training.
TORCH_LAUNCH=("${PYTHON_BIN}" -m torch.distributed.run --nproc_per_node="${num_gpus}" --master_addr="${MASTER_ADDR}" --master_port="${MASTER_PORT}")
if [[ "${num_gpus}" -le 1 ]]; then
  TRAIN_LAUNCH=("${PYTHON_BIN}" -u)
else
  TRAIN_LAUNCH=("${TORCH_LAUNCH[@]}")
fi

# The inner tmux invocation owns the run directory and launcher log.
TMUX_INNER=0
if [[ "${2:-}" == "--_tmux_inner" ]]; then
  TMUX_INNER=1
fi

# Launcher logging
RUNSTAMP="${PACT_RUNSTAMP:-$(date +"%Y%m%d_%H%M%S")}"

CONFIG_RUN_NAME="$(basename "${CONFIG_PATH}")"
CONFIG_RUN_NAME="${CONFIG_RUN_NAME%.sh}"
CONFIG_RUN_NAME="${CONFIG_RUN_NAME#train_config_}"
RUN_NAME="${PACT_RUN_NAME:-${SESSION_NAME:-${CONFIG_RUN_NAME}${PRESSURE_TAG}}}"
RUN_NAME_SAFE="${RUN_NAME//[^A-Za-z0-9._-]/_}"
if [[ -z "${RUN_NAME_SAFE}" ]]; then
  RUN_NAME_SAFE="train"
fi
PACT_RUN_NAME="${RUN_NAME}"
PACT_RUNSTAMP="${RUNSTAMP}"
if [[ "${USE_TMUX}" == "1" ]]; then
  SESSION_NAME="${SESSION_NAME:-${RUN_NAME_SAFE}_${RUNSTAMP}}"
fi

if [[ "${ALL_RESULTS_ROOT}" != /* ]]; then
  ALL_RESULTS_ROOT="${WORKDIR}/${ALL_RESULTS_ROOT#./}"
fi
case "${RUN_DIR_NAME_STYLE}" in
  timestamp_runname) RUN_DIR="${ALL_RESULTS_ROOT}/${RUNSTAMP}_${RUN_NAME_SAFE}" ;;
  runname_timestamp) RUN_DIR="${ALL_RESULTS_ROOT}/${RUN_NAME_SAFE}__${RUNSTAMP}" ;;
  *) echo "[FATAL] Unknown RUN_DIR_NAME_STYLE: ${RUN_DIR_NAME_STYLE}" >&2; exit 2 ;;
esac

# Save effective configuration values using Bash declarations.
# Args: $1 is the destination file; its parent directory must exist.
# Reads: CONFIG_PATH and the configuration variables listed below.
# Output: overwrites $1 with a sourceable snapshot, preserving array values.
# Returns: nonzero if the output redirection or writing fails.
write_resolved_config() {
  local output_path="$1"
  local name
  local config_vars=(
    TRAIN_PY PYTHON_BIN num_gpus ROOT_DIR TEST_ROOT_DIR PRESSURE_MODE STATION MODEL ENCODER_TYPE
    CNN_INTERMEDIATE_CHANNEL
    TEMPORAL_BLOCK STATION_JSON_DIR USE_SITE_ELEVATION USE_BATHYMETRY
    BATCH_SIZE GRAD_ACCUM_STEPS
    EPOCHS HIDDEN_CHANNELS NUM_LAYERS DROPOUT HEAD_DROPOUT SEED TRAIN_RATIO
    VAL_RATIO SHUFFLE_YEARS FUTURE_ONLY FUTURE_YEAR_THRESHOLD LR_LIST
    HISTORY_HOURS_LIST EXCEEDANCE_LOSS_WEIGHT
    EPISODE_GT_ALIGNED_PEAK_WEIGHT
    SCHEDULER ROP_METRIC ROP_FACTOR ROP_PATIENCE
    ROP_THRESHOLD ROP_COOLDOWN ROP_MIN_LR
    WARMUP_EPOCHS WARMUP_START_FACTOR MIN_LR MAX_GRAD_NORM
    DETERMINISTIC EXCEEDANCE_PERCENTILE
    CHECKPOINT_SELECTION
    X_NORM X_P_LO X_P_HI X_NODES_PER_GRAPH X_CLIP X_AUG X_AUG_PROB
    X_AUG_SCALE X_AUG_BIAS DISABLE_OOD USE_AMP AMP_DTYPE USE_TF32
    TORCH_THREADS NUM_WORKERS PIN_MEMORY PERSISTENT_WORKERS PREFETCH_FACTOR
    MP_CONTEXT USE_TMUX
    NODE_READ_HEADS TIME_READ_HEADS TRANSFORMER_LAYERS
    TRANSFORMER_FF_MULT TRANSFORMER_DROPOUT MAX_TIME_STEPS ALL_RESULTS_ROOT
    RUN_DIR_NAME_STYLE PACT_RUN_NAME PACT_RUNSTAMP SESSION_NAME PYTHON_RUN_TAG_BASE
  )

  {
    echo '#!/usr/bin/env bash'
    echo '# Fully resolved training configuration.'
    printf '# Original config: %s\n' "${CONFIG_PATH}"
    printf '# Generated UTC: %s\n\n' "$(date -u +"%Y-%m-%d %H:%M:%S")"
    for name in "${config_vars[@]}"; do
      if declare -p "${name}" >/dev/null 2>&1; then
        declare -p "${name}"
      fi
    done
  } > "${output_path}"
}

# Create one artifact directory per launch, including launches through tmux.
if [[ "${DRY_RUN}" != "1" ]] && { [[ "${USE_TMUX}" != "1" || "${TMUX_INNER}" == "1" ]] || ! command -v tmux >/dev/null 2>&1; }; then
  if [[ "${RUN_DIR_NAME_STYLE}" == "runname_timestamp" || "${PRESSURE_MODE}" != "anomaly" ]]; then
    mkdir -p "${ALL_RESULTS_ROOT}"
    # A repeated name within the same second must never overwrite a prior run.
    if ! mkdir "${RUN_DIR}"; then
      echo "[FATAL] Cannot create a fresh run directory: ${RUN_DIR}" >&2
      exit 2
    fi
  else
    mkdir -p "${RUN_DIR}"
  fi
  exec > >(tee -a "${RUN_DIR}/launcher.log") 2>&1
  trap 'status=$?; printf "%s\n" "$status" > "${RUN_DIR}/exit_status"; exit "$status"' EXIT
  write_resolved_config "${RUN_DIR}/config_used.sh"
fi

echo "========================================="
echo "Config file:   ${CONFIG_PATH}"
echo "Host:          $(hostname)"
echo "Workdir:       ${WORKDIR}"
echo "Run name:      ${RUN_NAME}"
echo "Run dir:       ${RUN_DIR}"
echo "Run dir style: ${RUN_DIR_NAME_STYLE}"
echo "Results root:  ${ALL_RESULTS_ROOT}"
echo "Python:        ${PYTHON_BIN}"
echo "tmux:          USE_TMUX=${USE_TMUX} SESSION_NAME=${SESSION_NAME:-<unset>} DRY_RUN=${DRY_RUN}"
echo "LAUNCHER:      ${TRAIN_LAUNCH[*]}"
echo "Train script:  ${TRAIN_PY}"
echo "Model:         ${MODEL}"
echo "Encoder:       ${ENCODER_TYPE}"
echo "CNN width:     ${CNN_INTERMEDIATE_CHANNEL} (intermediate channels)"
echo "Temporal:      ${TEMPORAL_BLOCK}"
echo "Station feats: site_elevation=${USE_SITE_ELEVATION} bathymetry=${USE_BATHYMETRY} (PACT)"
if (( GRAD_ACCUM_STEPS > 1 )); then
  echo "Grad accum:    ${GRAD_ACCUM_STEPS} (nominal effective batch=$((BATCH_SIZE * GRAD_ACCUM_STEPS)))"
fi
echo "ROOT_DIR:      ${ROOT_DIR}"
echo "TEST_ROOT_DIR: ${TEST_ROOT_DIR:-<none>}"
echo "Pressure mode: ${PRESSURE_MODE}"
echo "TRAIN_DATA_TAG:${TRAIN_DATA_TAG}"
echo "TEST_DATA_TAG: ${TEST_DATA_TAG}"
echo "LR_LIST:       ${LR_LIST[*]}"
echo "Global prediction loss: MSE (physical units)"
echo "Tail MSE weight: ${EXCEEDANCE_LOSS_WEIGHT} (strict GT > TRAIN Q95)"
echo "Episode GT-aligned peak MSE weight: ${EPISODE_GT_ALIGNED_PEAK_WEIGHT}"
echo "Checkpoint roles (VAL-only): overall, exceedance; primary=${CHECKPOINT_SELECTION}"
echo "H_LIST:        ${HISTORY_HOURS_LIST[*]}"
echo "Split:         train=${TRAIN_RATIO} val=${VAL_RATIO} shuffle_years=${SHUFFLE_YEARS} future_only=${FUTURE_ONLY} future_year_threshold=${FUTURE_YEAR_THRESHOLD} seed=${SEED}"
echo "MASTER_ADDR:   ${MASTER_ADDR}"
echo "MASTER_PORT:   ${MASTER_PORT}"
echo "SLURM_JOB_ID:  ${SLURM_JOB_ID:-<none>}"
echo "CUDA_VISIBLE_DEVICES:  ${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "num_gpus:      ${num_gpus}"
echo "Scheduler:     ${SCHEDULER} (ROP_METRIC=${ROP_METRIC})"
echo "DISABLE_OOD:   ${DISABLE_OOD} (x_norm=${X_NORM}, x_clip=${X_CLIP}, x_aug=${X_AUG})"
echo "DL:            workers=${NUM_WORKERS} pin=${PIN_MEMORY} pers=${PERSISTENT_WORKERS} prefetch=${PREFETCH_FACTOR} mp=${MP_CONTEXT}"
echo "Precision:     AMP=${USE_AMP} AMP_DTYPE=${AMP_DTYPE} TF32=${USE_TF32}"
echo "========================================="

# Optionally launch in a detached tmux session.
if [[ "${DRY_RUN}" != "1" && "${USE_TMUX}" == "1" && "${TMUX_INNER}" == "0" ]] && command -v tmux >/dev/null 2>&1; then
  # Preserve caller-provided session names and optional queue status files.
  STATUS_HANDOFF=""
  if [[ -n "${QSUB_LOCAL_STATUS_FILE:-}" ]]; then
    printf -v STATUS_HANDOFF 'printf "%%s\n" "$status" > %q; ' "${QSUB_LOCAL_STATUS_FILE}"
  fi
  printf -v TMUX_INNER_CMD 'PYTHON_BIN=%q S0_PYTHON_BIN=%q DRY_RUN=0 PRESSURE_MODE=%q SESSION_NAME=%q PACT_RUNSTAMP=%q PACT_RUN_NAME=%q bash %q %q --_tmux_inner; status=$?; %s if [[ ${status} -ne 0 ]]; then echo; echo "[tmux] train.sh exited with status ${status}; closing failed session so the local queue can continue."; fi; exit "${status}"' "${PYTHON_BIN}" "${PYTHON_BIN}" "${PRESSURE_MODE}" "${SESSION_NAME}" "${RUNSTAMP}" "${RUN_NAME}" "${SCRIPT_PATH}" "${CONFIG_PATH}" "${STATUS_HANDOFF}"
  printf -v TMUX_CMD 'bash -c %q' "${TMUX_INNER_CMD}"
  echo "[INFO] launching inside tmux session: ${SESSION_NAME}"
  # Carry the active environment and GPU allocation into the detached session.
  TMUX_ENV=(-e "PATH=${PATH}")
  for name in CONDA_PREFIX CONDA_DEFAULT_ENV LD_LIBRARY_PATH CUDA_VISIBLE_DEVICES; do
    if [[ -v "${name}" ]]; then
      TMUX_ENV+=(-e "${name}=${!name}")
    fi
  done
  tmux new-session -d -s "${SESSION_NAME}" -c "${WORKDIR}" "${TMUX_ENV[@]}" "${TMUX_CMD}"
  echo "Attach: tmux attach -t ${SESSION_NAME}"
  echo "Artifacts: ${RUN_DIR}"
  exit 0
fi

# Build descriptive run tags.
EXTRA_TAG=""
case "${MODEL}" in
  baseline) EXTRA_TAG="" ;;
  perceiver3)
    EXTRA_TAG="_nrh${NODE_READ_HEADS}_trh${TIME_READ_HEADS}_L${TRANSFORMER_LAYERS}_ff${TRANSFORMER_FF_MULT}_td${TRANSFORMER_DROPOUT}"
    # Preserve existing automatic Single run names.
    EXTRA_TAG+="_stable1v3_exceedance_hsingle"
    ;;
  *) echo "[FATAL] Unknown MODEL='${MODEL}'. Use baseline|perceiver3"; exit 1 ;;
esac

SPLIT_TAG=""
if [[ "${SHUFFLE_YEARS}" == "1" ]]; then
  SPLIT_TAG="_yshuffle"
fi
if [[ "${FUTURE_ONLY}" == "1" ]]; then
  SPLIT_TAG+="_futuregt${FUTURE_YEAR_THRESHOLD}"
fi

# Label the optional CNN encoder in automatic run names.
ENCODER_TAG=""
if [[ "${ENCODER_TYPE}" == "CNN" ]]; then
  ENCODER_TAG="_encCNN"
fi

# Label temporal blocks other than the Transformer.
TEMPORAL_TAG=""
if [[ "${MODEL}" == "perceiver3" && "${TEMPORAL_BLOCK}" != "Transformer" ]]; then
  TEMPORAL_TAG="_t${TEMPORAL_BLOCK}"
fi

# Record accumulation when more than one microbatch is used.
ACCUM_TAG=""
if (( GRAD_ACCUM_STEPS > 1 )); then
  ACCUM_TAG="_ga${GRAD_ACCUM_STEPS}"
fi

# Run the learning-rate and history-window sweep.
SWEEP_START_SECONDS=${SECONDS}
for LR_CUR in "${LR_LIST[@]}"; do
  for H in "${HISTORY_HOURS_LIST[@]}"; do
    LOSS_TAG="_lossmse_ex${EXCEEDANCE_LOSS_WEIGHT}"
    LOSS_ARGS=(
      --exceedance_loss_weight "${EXCEEDANCE_LOSS_WEIGHT}"
      --episode_gt_aligned_peak_weight "${EPISODE_GT_ALIGNED_PEAK_WEIGHT}"
    )

    SCHED_ARGS=(--scheduler "${SCHEDULER}" --warmup_epochs "${WARMUP_EPOCHS}"
                --warmup_start_factor "${WARMUP_START_FACTOR}" --min_lr "${MIN_LR}")
    if [[ "${SCHEDULER}" == "rop" ]]; then
      SCHED_ARGS+=(--rop_factor "${ROP_FACTOR}")
      SCHED_ARGS+=(--rop_patience "${ROP_PATIENCE}")
      SCHED_ARGS+=(--rop_threshold "${ROP_THRESHOLD}")
      SCHED_ARGS+=(--rop_cooldown "${ROP_COOLDOWN}")
      SCHED_ARGS+=(--rop_min_lr "${ROP_MIN_LR}")
      SCHED_ARGS+=(--rop_metric "${ROP_METRIC}")
    fi

    AUTO_RUN_TAG="${STATION:-ALL}_${TRAIN_DATA_TAG}to${TEST_DATA_TAG}_${MODEL}${ENCODER_TAG}${TEMPORAL_TAG}${ACCUM_TAG}${EXTRA_TAG}${SPLIT_TAG}${LOSS_TAG}_hist${H}h_hid${HIDDEN_CHANNELS}_L${NUM_LAYERS}_bs${BATCH_SIZE}_lr${LR_CUR}_ep${EPOCHS}_sch${SCHEDULER}_xn${X_NORM}${PRESSURE_TAG}"
    RUN_TAG="${PYTHON_RUN_TAG_BASE:-${AUTO_RUN_TAG}}"
    LOG_TAG="${RUN_TAG}"
    if (( ${#LOG_TAG} > 240 )); then
      LOG_HASH="$(printf '%s' "${RUN_TAG}" | sha256sum)"
      LOG_TAG="${RUN_TAG:0:220}_${LOG_HASH:0:12}"
    fi
    LOG_FILE="${RUN_DIR}/train_${LOG_TAG}.log"
    if [[ "${DRY_RUN}" != "1" ]]; then : > "${LOG_FILE}"; fi

    BASE_CMD=(
      "${TRAIN_PY}"
      --root_dir "${ROOT_DIR}"
      --pressure_mode "${PRESSURE_MODE}"
      --model "${MODEL}"
      --exceedance_percentile "${EXCEEDANCE_PERCENTILE}"
      --checkpoint_selection "${CHECKPOINT_SELECTION}"
      --encoder_type "${ENCODER_TYPE}"
      --cnn_intermediate_channel "${CNN_INTERMEDIATE_CHANNEL}"
      --batch_size "${BATCH_SIZE}"
      --lr "${LR_CUR}"
      --epochs "${EPOCHS}"
      --hidden_channels "${HIDDEN_CHANNELS}"
      --num_layers "${NUM_LAYERS}"
      --dropout "${DROPOUT}"
      --head_dropout "${HEAD_DROPOUT}"
      --max_grad_norm "${MAX_GRAD_NORM}"
      --deterministic "${DETERMINISTIC}"
      --history_hours "${H}"
      --seed "${SEED}"
      --train_ratio "${TRAIN_RATIO}"
      --val_ratio "${VAL_RATIO}"
      --shuffle_years "${SHUFFLE_YEARS}"
      --future_only "${FUTURE_ONLY}"
      --future_year_threshold "${FUTURE_YEAR_THRESHOLD}"
      --run_tag "${RUN_TAG}_${RUNSTAMP}"
      --output_dir "${RUN_DIR}"
      --torch_threads "${TORCH_THREADS}"
      --num_workers "${NUM_WORKERS}"
      --prefetch_factor "${PREFETCH_FACTOR}"
      --mp_context "${MP_CONTEXT}"
      --x_norm "${X_NORM}"
      --x_p_lo "${X_P_LO}"
      --x_p_hi "${X_P_HI}"
      --x_nodes_per_graph "${X_NODES_PER_GRAPH}"
      --x_clip "${X_CLIP}"
      --x_aug "${X_AUG}"
      --x_aug_prob "${X_AUG_PROB}"
      --x_aug_scale "${X_AUG_SCALE}"
      --x_aug_bias "${X_AUG_BIAS}"
      --station_json_dir "${STATION_JSON_DIR}"
      --use_site_elevation "${USE_SITE_ELEVATION}"
      --use_bathymetry "${USE_BATHYMETRY}"
    )

    [[ -n "${STATION}" ]]       && BASE_CMD+=(--station "${STATION}")
    [[ -n "${TEST_ROOT_DIR}" ]] && BASE_CMD+=(--test_root_dir "${TEST_ROOT_DIR}")
    if (( GRAD_ACCUM_STEPS > 1 )); then
      BASE_CMD+=(--grad_accum_steps "${GRAD_ACCUM_STEPS}")
    fi

    # Configure precision and data loading.
    if [[ "${USE_AMP}" -eq 1 ]]; then
      BASE_CMD+=(--amp --amp_dtype "${AMP_DTYPE}")
    fi
    if [[ "${USE_TF32}" -eq 1 ]]; then
      BASE_CMD+=(--tf32)
    fi
    if [[ "${PIN_MEMORY}" -eq 1 ]]; then
      BASE_CMD+=(--pin_memory)
    fi
    if [[ "${PERSISTENT_WORKERS}" -eq 1 ]]; then
      BASE_CMD+=(--persistent_workers)
    fi

    # Add the temporal architecture options for perceiver3.
    if [[ "${MODEL}" == "perceiver3" ]]; then
      BASE_CMD+=(--temporal_block "${TEMPORAL_BLOCK}"
                 --node_read_heads "${NODE_READ_HEADS}"
                 --time_read_heads "${TIME_READ_HEADS}"
                 --transformer_layers "${TRANSFORMER_LAYERS}"
                 --transformer_ff_mult "${TRANSFORMER_FF_MULT}"
                 --transformer_dropout "${TRANSFORMER_DROPOUT}"
                 --max_time_steps "${MAX_TIME_STEPS}")
    fi

    BASE_CMD+=("${SCHED_ARGS[@]}")
    BASE_CMD+=("${LOSS_ARGS[@]}")

    if [[ "${DRY_RUN}" == "1" ]]; then
      printf 'CMD: '
      printf '%q ' "${BASE_CMD[@]}"
      printf '\n'
      continue
    fi

    {
      echo "-----------------------------------------"
      echo "RUN_TAG:  ${RUN_TAG}"
      echo "LOG:      ${LOG_FILE}"
      echo "LAUNCHER: ${TRAIN_LAUNCH[*]}"
      echo "-----------------------------------------"
      printf "CMD: "
      printf "%q " "${BASE_CMD[@]}"
      echo
    } | tee -a "${LOG_FILE}"

    "${TRAIN_LAUNCH[@]}" "${BASE_CMD[@]}" 2>&1 | tee -a "${LOG_FILE}"
  done
done

if [[ "${DRY_RUN}" != "1" ]]; then
  printf '[%s] Wall time (all combinations): %s s | Artifacts: %s\n' "$(date +'%Y-%m-%d|%H:%M:%S')" "$((SECONDS - SWEEP_START_SECONDS))" "${RUN_DIR}"
fi
