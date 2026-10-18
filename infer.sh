#!/usr/bin/env bash
# Evaluate a checkpoint with the already activated derecho-ml environment.
# Usage: bash infer.sh <infer-config.sh>
# The config argument is relative to the caller's working directory.
# Paths inside the config are relative to this script's directory (repo root).
set -euo pipefail

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
  echo "Usage: bash infer.sh <infer-config.sh>" >&2
  exit 2
fi
if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "[FATAL] Config not found: ${CONFIG_PATH}" >&2
  exit 2
fi
CONFIG_PATH="$(cd -- "$(dirname -- "${CONFIG_PATH}")" && pwd)/$(basename -- "${CONFIG_PATH}")"
cd -- "${WORKDIR}"

# shellcheck source=emulator/common/inference_artifacts.sh
source "${SCRIPT_DIR}/emulator/common/inference_artifacts.sh"

# Use the active environment; support interpreter overrides in older configs.
PYTHON_BIN="${PYTHON_BIN:-${S0_PYTHON_BIN:-${CONDA_PREFIX:+${CONDA_PREFIX}/bin/python}}}"
: "${PYTHON_BIN:=python}"
: "${S0_PYTHON_BIN:=${PYTHON_BIN}}"

# Load experiment settings; apply defaults for omitted values below.
set +u
# shellcheck disable=SC1090
source "${CONFIG_PATH}"
set -u

# Map optional source and target aliases used by pair configs.
if [[ -v SOURCE_ROOT ]]; then ROOT_DIR="${SOURCE_ROOT}"; fi
if [[ -v TARGET_ROOT ]]; then TEST_ROOT_DIR="${TARGET_ROOT}"; fi
if [[ -n "${SOURCE_NAME:-}${TARGET_NAME:-}${EXPERIMENT_GROUP:-}" ]] && [[ -z "${TEST_ROOT_DIR:-}" ]]; then
  echo "[FATAL] Named pair configs require a nonempty TARGET_ROOT / TEST_ROOT_DIR, including diagonal pairs."
  exit 2
fi

# Data, architecture, and output defaults
: "${INFER_PY:=./infer.py}"
: "${ROOT_DIR:=./dataset/NCEP/graphs}"
: "${TEST_ROOT_DIR:=}"
: "${STATION:=Battery}"
: "${MODEL:=baseline}"
: "${MODEL_LABEL:=${MODEL}}"
: "${INFERENCE_RESULTS_ROOT:=./All_Inference_Results}"
: "${ENCODER_TYPE:=GraphSAGE}"
: "${CNN_INTERMEDIATE_CHANNEL:=}"
: "${TEMPORAL_BLOCK:=Transformer}"
: "${HISTORY_HOURS:=12}"
: "${BATCH_SIZE:=1}"
: "${STATION_JSON_DIR:=./station_json}"
: "${USE_SITE_ELEVATION:=1}"
: "${USE_BATHYMETRY:=0}"
: "${YEARS:=}"
: "${STRICT_YEARS:=0}"
: "${DRY_RUN:=0}"

# Runtime and optional diagnostics
: "${CUDA_LAUNCH_BLOCKING_FLAG:=0}"
: "${TORCH_GPU_PROBE:=0}"
: "${FAIL_IF_NO_SLURM:=0}"
: "${USE_TMUX:=1}"

# Precision and data loading
: "${USE_AMP:=1}"
: "${AMP_DTYPE:=bf16}"
: "${USE_TF32:=1}"
: "${TORCH_THREADS:=1}"

: "${NUM_WORKERS:=0}"
: "${PIN_MEMORY:=0}"
: "${PERSISTENT_WORKERS:=0}"
: "${PREFETCH_FACTOR:=0}"
: "${MP_CONTEXT:=fork}"

# Required checkpoint path or glob
: "${CKPT_PATH:=}"

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

# Derive dataset labels and safe artifact directory names.
infer_dataset_tag () {
  local root="${1%/}"
  local leaf
  leaf="$(basename -- "${root}")"
  if [[ "${leaf,,}" == "graphs" ]]; then
    basename -- "$(dirname -- "${root}")"
  else
    printf '%s\n' "${leaf}"
  fi
}

safe_run_component () {
  local value="$1"
  value="${value//[!A-Za-z0-9._-]/_}"
  printf '%s\n' "${value}"
}

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

# Resolve the checkpoint once so replay uses the same file.
CKPT_RESOLVED="$(resolve_ckpt "${CKPT_PATH}" || true)"
if [[ -z "${CKPT_RESOLVED}" ]]; then
  echo "[FATAL] CKPT_PATH does not resolve to a file: ${CKPT_PATH}"
  exit 2
fi

# Resolve output paths relative to the repository root.
case "${INFERENCE_RESULTS_ROOT}" in
  /*) RESULTS_ROOT="${INFERENCE_RESULTS_ROOT%/}" ;;
  *) RESULTS_ROOT="${WORKDIR}/${INFERENCE_RESULTS_ROOT#./}"; RESULTS_ROOT="${RESULTS_ROOT%/}" ;;
esac

SOURCE_TAG="${SOURCE_NAME:-$(infer_dataset_tag "${ROOT_DIR}")}"
SOURCE_ROOT="${SOURCE_ROOT:-${ROOT_DIR}}"
TARGET_ROOT="${TEST_ROOT_DIR}"
TARGET_TAG="${TARGET_NAME:-$(infer_dataset_tag "${TEST_ROOT_DIR:-${ROOT_DIR}}")}"
RUN_BASE="$(safe_run_component "${STATION}_${MODEL_LABEL}_${SOURCE_TAG}_To_${TARGET_TAG}")"
if [[ -n "${EXPERIMENT_GROUP:-}" ]]; then
  RUN_BASE="$(safe_run_component "${EXPERIMENT_GROUP}_${STATION}_${SOURCE_TAG}_to_${TARGET_TAG}")"
fi
RUNSTAMP=$(date +"%Y%m%d_%H%M%S")
RUN_FOLDER_NAME="${RUN_BASE}_${RUNSTAMP}"
RUN_DIR="${RESULTS_ROOT}/${RUN_FOLDER_NAME}"
SESSION_NAME="${SESSION_NAME:-${RUN_FOLDER_NAME}}"

OUT_DIR="${RUN_DIR}/outputs"

TEST_TAG="${TARGET_TAG}"

echo "========================================="
echo "Config file:   ${CONFIG_PATH}"
echo "Workdir:       ${WORKDIR}"
echo "Run dir:       ${RUN_DIR}"
echo "Out dir:       ${OUT_DIR}"
echo "Station:       ${STATION}"
echo "Model label:   ${MODEL_LABEL}"
echo "Source:        ${SOURCE_TAG}"
echo "Target:        ${TARGET_TAG}"
echo "Python:        ${PYTHON_BIN}"
echo "Infer script:  ${INFER_PY}"
echo "Checkpoint:    ${CKPT_RESOLVED}"
echo "ROOT_DIR:      ${ROOT_DIR}"
echo "TEST_ROOT_DIR: ${TEST_ROOT_DIR:-<empty => ROOT_DIR year-split test>}"
echo "Model:         ${MODEL}"
echo "Encoder:       ${ENCODER_TYPE}"
echo "CNN width:     ${CNN_INTERMEDIATE_CHANNEL:-<checkpoint>}"
echo "Temporal:      ${TEMPORAL_BLOCK}"
echo "Station feats: site_elevation=${USE_SITE_ELEVATION} bathymetry=${USE_BATHYMETRY} (PACT)"
echo "History hours: ${HISTORY_HOURS}"
echo "Batch size:    ${BATCH_SIZE}"
echo "Years:         ${YEARS:-<all>}"
echo "Strict years:  ${STRICT_YEARS}"
if [[ -n "${YEARS}" ]]; then
  IFS=',' read -r -a REQUESTED_YEAR_TAGS <<< "${YEARS}"
  echo "Year count:    ${#REQUESTED_YEAR_TAGS[@]}"
fi
echo "Threshold:     source_checkpoint_train (saved tau; no refit)"
echo "========================================="

if [[ "${DRY_RUN}" == "1" ]]; then
  echo "[DRY RUN] Configuration resolved; infer.py was not executed and no result directory was created."
  exit 0
fi
mkdir -p "${OUT_DIR}"

# Save the resolved configuration and a self-contained replay script.
RUNNER="${RUN_DIR}/run_infer.sh"
CONFIG_SNAPSHOT="${RUN_DIR}/infer_config_used.sh"
write_infer_config_snapshot "${CONFIG_SNAPSHOT}"
{
  printf '#!/usr/bin/env bash\nset -euo pipefail\n'
  printf 'cd -- %q\n' "${WORKDIR}"
  printf 'source %q\n' "${CONFIG_SNAPSHOT}"
  # Embed the command writer so replay does not depend on the helper script.
  declare -f write_infer_command
  cat <<'RUNNER_BODY'
# Use the interpreter captured from the activated environment at launch time.
if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "[FATAL] Python interpreter not found or not executable: ${PYTHON_BIN}" >&2
  exit 2
fi

export PYTHONUNBUFFERED=1

# Enable synchronous CUDA execution when requested.
if [[ "${CUDA_LAUNCH_BLOCKING_FLAG}" -eq 1 ]]; then
  export CUDA_LAUNCH_BLOCKING=1
fi

# Honor an explicit Slurm GPU assignment when present.
if [[ -n "${SLURM_STEP_GPUS:-}" ]]; then
  export CUDA_VISIBLE_DEVICES="${SLURM_STEP_GPUS}"
elif [[ -n "${SLURM_JOB_GPUS:-}" ]]; then
  export CUDA_VISIBLE_DEVICES="${SLURM_JOB_GPUS}"
fi

# Use the checkpoint selected before tmux started; never re-expand the glob.
if [[ ! -f "${CKPT_RESOLVED}" ]]; then
  echo "[FATAL] Selected checkpoint no longer exists: ${CKPT_RESOLVED}"
  exit 2
fi

LOG_FILE="${RUN_DIR}/infer_${STATION}_${TEST_TAG}_${MODEL}_hist${HISTORY_HOURS}h_${RUNSTAMP}.log"
: > "${LOG_FILE}"

# Log GPU diagnostics; optionally probe devices through PyTorch.
{
  echo "========================================="
  echo "[tmux] host=$(hostname)"
  echo "[tmux] SLURM_JOB_ID=${SLURM_JOB_ID:-<none>}"
  echo "[tmux] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
  echo "[tmux] CUDA_LAUNCH_BLOCKING=${CUDA_LAUNCH_BLOCKING:-0}"
  echo "[tmux] ckpt=${CKPT_RESOLVED}"
  echo "========================================="

  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi || true
  fi

  if [[ "${TORCH_GPU_PROBE}" -eq 1 ]]; then
    "${PYTHON_BIN}" - <<'PY'
import torch
print("[tmux][torch] torch:", torch.__version__)
print("[tmux][torch] cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("[tmux][torch] device_count:", torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        try:
            print(f"[tmux][torch] {i}: {torch.cuda.get_device_name(i)}")
        except Exception as e:
            print(f"[tmux][torch] {i}: <error> {e}")
PY
  fi
  echo "========================================="
} 2>&1 | tee -a "${LOG_FILE}"

if [[ "${FAIL_IF_NO_SLURM}" -eq 1 ]] && [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "[FATAL] SLURM_JOB_ID is empty." | tee -a "${LOG_FILE}"
  exit 2
fi

TF32_ARGS=()
if [[ "${USE_TF32}" -eq 1 ]]; then TF32_ARGS=(--tf32); fi

AMP_ARGS=()
if [[ "${USE_AMP}" -eq 1 ]]; then AMP_ARGS=(--amp --amp_dtype "${AMP_DTYPE}"); fi

STATION_ARGS=()
if [[ -n "${STATION}" ]]; then STATION_ARGS=(--station "${STATION}"); fi

TEST_ARGS=()
if [[ -n "${TEST_ROOT_DIR}" ]]; then TEST_ARGS=(--test_root_dir "${TEST_ROOT_DIR}"); fi

YEARS_ARGS=()
if [[ -n "${YEARS}" ]]; then YEARS_ARGS=(--years "${YEARS}"); fi
if [[ "${STRICT_YEARS}" == "1" ]]; then YEARS_ARGS+=(--strict_years); fi

DL_ARGS=(--num_workers "${NUM_WORKERS}" --prefetch_factor "${PREFETCH_FACTOR}" --mp_context "${MP_CONTEXT}")
if [[ "${PIN_MEMORY}" -eq 1 ]]; then DL_ARGS+=(--pin_memory); fi
if [[ "${PERSISTENT_WORKERS}" -eq 1 ]]; then DL_ARGS+=(--persistent_workers); fi

THREAD_ARGS=(--torch_threads "${TORCH_THREADS}")
ARCH_ARGS=()
if [[ -n "${CNN_INTERMEDIATE_CHANNEL}" ]]; then ARCH_ARGS+=(--cnn_intermediate_channel "${CNN_INTERMEDIATE_CHANNEL}"); fi

# Build and record the inference command.
CMD=(
  "${PYTHON_BIN}" -u "${INFER_PY}"
  --root_dir "${ROOT_DIR}"
  --source_name "${SOURCE_NAME:-}"
  --target_name "${TARGET_NAME:-}"
  "${TEST_ARGS[@]}"
  "${STATION_ARGS[@]}"
  --station_json_dir "${STATION_JSON_DIR}"
  --use_site_elevation "${USE_SITE_ELEVATION}"
  --use_bathymetry "${USE_BATHYMETRY}"
  --model "${MODEL}"
  --model_label "${MODEL_LABEL}"
  --encoder_type "${ENCODER_TYPE}"
  "${ARCH_ARGS[@]}"
  --temporal_block "${TEMPORAL_BLOCK}"
  --history_hours "${HISTORY_HOURS}"
  --batch_size "${BATCH_SIZE}"
  --ckpt "${CKPT_RESOLVED}"
  --out_dir "${OUT_DIR}"
  --save_npz
  "${YEARS_ARGS[@]}"
  "${AMP_ARGS[@]}"
  "${TF32_ARGS[@]}"
  "${THREAD_ARGS[@]}"
  "${DL_ARGS[@]}"
)
write_infer_command "${RUN_DIR}/command_used.sh" "${CMD[@]}"
"${CMD[@]}" 2>&1 | tee -a "${LOG_FILE}"

echo "[DONE] outputs in: ${OUT_DIR}" | tee -a "${LOG_FILE}"
RUNNER_BODY
} > "${RUNNER}"

# Execute directly or start a detached tmux session.
chmod +x "${RUNNER}"
printf -v RUNNER_COMMAND 'bash %q' "${RUNNER}"
if [[ "${USE_TMUX}" == "1" ]] && command -v tmux >/dev/null 2>&1; then
  # Carry the active environment and GPU allocation into the detached session.
  TMUX_ENV=(-e "PATH=${PATH}")
  for name in CONDA_PREFIX CONDA_DEFAULT_ENV LD_LIBRARY_PATH CUDA_VISIBLE_DEVICES; do
    if [[ -v "${name}" ]]; then
      TMUX_ENV+=(-e "${name}=${!name}")
    fi
  done
  tmux new-session -d -s "${SESSION_NAME}" -c "${WORKDIR}" "${TMUX_ENV[@]}" "${RUNNER_COMMAND}"
  echo "Started inference in tmux: ${SESSION_NAME}"
  echo "Attach: tmux attach -t ${SESSION_NAME}"
else
  bash "${RUNNER}"
fi
echo "Run dir: ${RUN_DIR}"
echo "Runner: ${RUNNER}"
