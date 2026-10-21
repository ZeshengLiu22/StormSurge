#!/usr/bin/env bash
# Shared single-node Derecho launcher: eight sequential four-GPU DDP runs.
# Usage: DRY_RUN=1 bash run_station.sh Boston 0H
set -euo pipefail
export NODE_ATTENTION_BACKEND="${NODE_ATTENTION_BACKEND:-math}"
export WEIGHT_DECAY_POLICY="${WEIGHT_DECAY_POLICY:-module}"

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 <Boston|CBBT|Lewes|Battery> <0H|6H|12H|18H|24H|30H|36H|42H|48H>" >&2
  exit 2
fi
station="$1"
history="$2"
case "${station}" in
  Boston|CBBT|Lewes|Battery) ;;
  *) echo "[FATAL] Unknown station: ${station}" >&2; exit 2 ;;
esac
case "${history}" in
  0H|6H|12H|18H|24H|30H|36H|42H|48H) ;;
  *) echo "[FATAL] Unknown history: ${history}" >&2; exit 2 ;;
esac

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd -- "${PROJECT_ROOT}"
export DRY_RUN="${DRY_RUN:-0}"
case "${DRY_RUN}" in
  0|1) ;;
  *) echo "[FATAL] DRY_RUN must be 0 or 1." >&2; exit 2 ;;
esac

# Configs read S0_PYTHON_BIN; set both variables to the same Derecho runtime.
export PYTHON_BIN="${PYTHON_BIN:-/glade/work/zel/conda-envs/derecho-ml/bin/python}"
export S0_PYTHON_BIN="${PYTHON_BIN}"
export USE_TMUX=0
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1
# train.sh otherwise lets inherited Slurm settings override num_gpus=4.
unset SLURM_NTASKS_PER_NODE SLURM_JOB_ID

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "[FATAL] Python is not executable: ${PYTHON_BIN}" >&2
  exit 2
fi

architectures=(GNN_MLP GNN_LSTM GNN_GRU GNN_Attn CNN_MLP CNN_LSTM CNN_GRU CNN_Attn)
configs=()
for architecture in "${architectures[@]}"; do
  config="${PROJECT_ROOT}/configs/architecture_ablation_LR1e-3/${history}/${station}/train_config_NCEP_${station}_${architecture}_${history}_G0_T0000_EP0000.sh"
  if [[ ! -r "${config}" ]]; then
    echo "[FATAL] Missing config: ${config}" >&2
    exit 2
  fi
  configs+=("${config}")
done

if [[ "${DRY_RUN}" != 1 ]]; then
  if [[ -z "${PBS_JOBID:-}" ]]; then
    echo "[FATAL] Submit a station .pbs script with qsub; use DRY_RUN=1 for a login-node preview." >&2
    exit 2
  fi
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
fi

printf '[PBS] job=%s station=%s history=%s runs=8 GPUs/run=4 DRY_RUN=%s\n' \
  "${PBS_JOBID:-dry-run}" "${station}" "${history}" "${DRY_RUN}"
start_seconds=${SECONDS}
for index in "${!configs[@]}"; do
  printf '[RUN %d/8] %s\n' "$((index + 1))" "${architectures[index]}"
  # train.sh starts one torch.distributed.run with --nproc_per_node=4.
  # Wait for all four ranks to finish before starting the next architecture.
  bash "${PROJECT_ROOT}/train.sh" "${configs[index]}"
done
printf '[COMPLETE] station=%s history=%s runs=8 elapsed=%ss DRY_RUN=%s\n' \
  "${station}" "${history}" "$((SECONDS - start_seconds))" "${DRY_RUN}"
