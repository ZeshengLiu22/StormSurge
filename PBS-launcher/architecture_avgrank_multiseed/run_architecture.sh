#!/usr/bin/env bash
# One architecture, four stations, three seeds; one four-GPU DDP run at a time.
set -euo pipefail
set +m  # setsid's child PID must also be its process-group ID.

fatal() { printf '[FATAL] %s\n' "$*" >&2; exit 2; }
[[ $# -eq 2 ]] || fatal "Usage: $0 <LR3e-3|LR5e-3_Clip0.25> <architecture>"
group="$1"
architecture="$2"
case "${group}:${architecture}" in
  LR3e-3:CNN_GRU_48H|LR3e-3:CNN_LSTM_42H|LR3e-3:CNN_GRU_30H) group_letter=A ;;
  LR5e-3_Clip0.25:CNN_Attn_48H|LR5e-3_Clip0.25:CNN_MLP_36H|LR5e-3_Clip0.25:CNN_LSTM_30H|LR5e-3_Clip0.25:CNN_Attn_24H|LR5e-3_Clip0.25:CNN_LSTM_24H) group_letter=C ;;
  *) fatal "Unknown group/architecture: ${group}/${architecture}" ;;
esac

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd -- "${PROJECT_ROOT}"
export DRY_RUN="${DRY_RUN:-0}"
case "${DRY_RUN}" in 0|1) ;; *) fatal "DRY_RUN must be 0 or 1" ;; esac

# Preserve the reference Derecho interpreter, CUDA setup, thread limits and DDP.
export PYTHON_BIN="${PYTHON_BIN:-/glade/work/zel/conda-envs/derecho-ml/bin/python}"
export S0_PYTHON_BIN="${PYTHON_BIN}"
export USE_TMUX=0 NODE_ATTENTION_BACKEND=math WEIGHT_DECAY_POLICY=module
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
# train.sh reads these two Slurm variables when selecting GPU count/master port.
unset SLURM_NTASKS_PER_NODE SLURM_JOB_ID
# Always let train.sh create a fresh timestamp for each new run.
unset PACT_RUNSTAMP
[[ -x "${PYTHON_BIN}" ]] || fatal "Python is not executable: ${PYTHON_BIN}"
[[ -r "${PROJECT_ROOT}/train.sh" && -r "${PROJECT_ROOT}/train.py" ]] || fatal "Missing training entry point"
command -v setsid >/dev/null || fatal "setsid is required for DDP process-group cleanup"

STATIONS=(Battery Boston CBBT Lewes)
SEEDS=(42 123 2026)
configs=() run_stations=() run_seeds=()
for station in "${STATIONS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    config="${PROJECT_ROOT}/configs/architecture_avgrank_multiseed_${group}/${architecture}/${station}/seed${seed}/train_config_NCEP_${station}_${architecture}_G0_T0000_EP0000_seed${seed}.sh"
    [[ -r "${config}" ]] || fatal "Missing config: ${config}"
    configs+=("${config}")
    run_stations+=("${station}")
    run_seeds+=("${seed}")
  done
done

# CPU-only, assignment-only validation of every config against its exact reference.
# This happens before module loading, output creation or any training process.
"${PYTHON_BIN}" "${SCRIPT_DIR}/experiments.py" validate --repo "${PROJECT_ROOT}" \
  --group "${group}" --architecture "${architecture}"
printf '[PBS] job=%s group=%s LR_group=%s architecture=%s runs=12 GPUs/run=4 DRY_RUN=%s\n' \
  "${PBS_JOBID:-dry-run}" "${group_letter}" "${group}" "${architecture}" "${DRY_RUN}"
for index in "${!configs[@]}"; do
  printf 'CONFIG\t%s\t%s\t%s\t%s\n' "$((index + 1))" "${run_stations[index]}" "${run_seeds[index]}" "${configs[index]}"
done
if [[ "${DRY_RUN}" == 1 ]]; then
  printf '[DRY_RUN] Validated 12 configs; no modules, CUDA checks, training or outputs.\n'
  exit 0
fi

[[ -n "${PBS_JOBID:-}" ]] || fatal "A PBS allocation is required; use DRY_RUN=1 on login nodes"
module load cuda/12.9.0
export TMPDIR="/glade/derecho/scratch/zel/tmp/stormsurge-${PBS_JOBID}"
mkdir -p -- "${TMPDIR}"
"${PYTHON_BIN}" - <<'PYTHON'
import sys
import torch
import torch.distributed as dist
import torch_geometric

if not torch.cuda.is_available() or torch.cuda.device_count() != 4:
    raise SystemExit("[FATAL] Exactly four visible CUDA GPUs are required for these DDP jobs.")
if not dist.is_available() or not dist.is_nccl_available():
    raise SystemExit("[FATAL] PyTorch distributed/NCCL support is required.")
for index in range(4):
    if torch.cuda.get_device_capability(index)[0] < 8:
        raise SystemExit("[FATAL] These BF16/TF32 configs require Ampere or newer GPUs.")
    print(f"[GPU {index}] {torch.cuda.get_device_name(index)}")
print(f"[RUNTIME] {sys.executable}; torch={torch.__version__}; CUDA={torch.version.cuda}; PyG={torch_geometric.__version__}")
PYTHON

log_root="${PROJECT_ROOT}/PBS-launcher/architecture_avgrank_multiseed_${group}/${architecture}"
mkdir -p -- "${log_root}/manifests"
job_label="${PBS_JOBID//[^[:alnum:]._-]/_}"
manifest="$(mktemp "${log_root}/manifests/$(date -u +%Y%m%dT%H%M%SZ)_${job_label}_XXXXXX.tsv")"
launch_id="$(basename -- "${manifest}" .tsv)"
printf 'status\tgroup\tarchitecture\tstation\tseed\tconfig\tstart_utc\tend_utc\texit_code\texecution_log\n' > "${manifest}"
printf '[MANIFEST] %s\n' "${manifest}"

active_pid=""
current_index=-1
run_open=0
failures=0
start_time="-"
execution_log="-"

record() {
  local index="$1" status="$2" start="$3" end="$4" code="$5" log="$6"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "${status}" "${group_letter}" "${architecture}" "${run_stations[index]}" "${run_seeds[index]}" \
    "${configs[index]}" "${start}" "${end}" "${code}" "${log}" >> "${manifest}"
}

group_alive() { kill -0 -- "-${active_pid}" 2>/dev/null; }

# Signal the whole session's process group: train.sh, torch launcher, all ranks,
# dataloader children and tee processes. Bound grace time, then force survivors.
stop_group() {
  [[ -n "${active_pid}" ]] || return 0
  local attempt
  kill -TERM -- "-${active_pid}" 2>/dev/null || true
  for ((attempt=0; attempt<100; attempt++)); do
    group_alive || break
    sleep 0.1
  done
  if group_alive; then kill -KILL -- "-${active_pid}" 2>/dev/null || true; fi
  wait "${active_pid}" 2>/dev/null || true
  for ((attempt=0; attempt<20; attempt++)); do
    group_alive || return 0
    sleep 0.1
  done
  ! group_alive
}

finish_run() {
  local status="$1" code="$2" end_time
  end_time="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  record "${current_index}" "${status}" "${start_time}" "${end_time}" "${code}" "${execution_log}"
  printf '[END] group=%s architecture=%s station=%s seed=%s config=%s start=%s end=%s exit_code=%s status=%s log=%s\n' \
    "${group_letter}" "${architecture}" "${run_stations[current_index]}" "${run_seeds[current_index]}" \
    "${configs[current_index]}" "${start_time}" "${end_time}" "${code}" "${status}" "${execution_log}"
  run_open=0
}

on_exit() {
  local code=$? index
  trap - EXIT
  trap '' TERM INT
  # Also handle a signal delivered between background launch and PID assignment.
  if [[ -z "${active_pid}" ]]; then active_pid="$(jobs -pr)"; fi
  if [[ -n "${active_pid}" ]]; then
    if ! stop_group; then
      printf '[FATAL] Process group %s did not disappear; stopping the sequence.\n' "${active_pid}" >&2
      [[ "${code}" -ne 0 ]] || code=125
    fi
  fi
  if [[ "${run_open}" == 1 ]]; then
    [[ "${code}" -ne 0 ]] || code=125
    if [[ "${code}" == 143 || "${code}" == 130 ]]; then
      finish_run INTERRUPTED "${code}"
    else
      finish_run FAILED "${code}"
    fi
  fi
  for ((index=current_index+1; index<${#configs[@]}; index++)); do
    record "${index}" NOT_RUN - "$(date -u +%Y-%m-%dT%H:%M:%SZ)" - -
  done
  exit "${code}"
}
trap on_exit EXIT
trap 'exit 143' TERM
trap 'exit 130' INT

for index in "${!configs[@]}"; do record "${index}" PENDING - - - -; done
for index in "${!configs[@]}"; do
  current_index="${index}"
  execution_dir="${log_root}/execution_logs/${run_stations[index]}/seed${run_seeds[index]}"
  mkdir -p -- "${execution_dir}"
  execution_log="${execution_dir}/${launch_id}.log"
  start_time="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  run_open=1
  record "${index}" RUNNING "${start_time}" - - "${execution_log}"
  printf '[START] group=%s architecture=%s station=%s seed=%s config=%s start=%s log=%s\n' \
    "${group_letter}" "${architecture}" "${run_stations[index]}" "${run_seeds[index]}" \
    "${configs[index]}" "${start_time}" "${execution_log}"
  # No pipeline/background overlap: wait on this isolated session before advancing.
  setsid bash "${PROJECT_ROOT}/train.sh" "${configs[index]}" > "${execution_log}" 2>&1 &
  active_pid=$!
  if wait "${active_pid}"; then code=0; else code=$?; fi

  # Allow train.sh's log tee to drain. Unexpected surviving ranks make this run fail.
  for ((attempt=0; attempt<20; attempt++)); do
    group_alive || break
    sleep 0.1
  done
  if group_alive; then
    printf '[ERROR] Residual processes after %s; cleaning up before any next run.\n' "${configs[index]}" >&2
    [[ "${code}" -ne 0 ]] || code=125
    stop_group || exit 125
  fi
  active_pid=""
  if [[ "${code}" == 0 ]]; then
    finish_run SUCCEEDED 0
  else
    failures=$((failures + 1))
    finish_run FAILED "${code}"
  fi
done
printf '[COMPLETE] group=%s architecture=%s attempted=12 succeeded=%s failed=%s manifest=%s\n' \
  "${group_letter}" "${architecture}" "$((12 - failures))" "${failures}" "${manifest}"
[[ "${failures}" -eq 0 ]]
