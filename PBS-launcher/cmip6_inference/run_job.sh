#!/usr/bin/env bash
# Shared PBS environment. DRY_RUN never loads CUDA, creates output, or runs inference.
set -euo pipefail
runner_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export PYTHON_BIN=${PYTHON_BIN:-/glade/work/zel/conda-envs/derecho-ml/bin/python}
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${INFER_TORCH_THREADS:-1}
export MKL_NUM_THREADS=$OMP_NUM_THREADS OPENBLAS_NUM_THREADS=$OMP_NUM_THREADS
unset RANK LOCAL_RANK WORLD_SIZE LOCAL_WORLD_SIZE MASTER_ADDR MASTER_PORT
case ${DRY_RUN:-0} in
  1) exec "$PYTHON_BIN" "$runner_dir/launch.py" "$@" --manifest "$runner_dir/manifest.json" --dry-run ;;
  0) ;;
  *) echo 'DRY_RUN must be 0 or 1' >&2; exit 2 ;;
esac
[[ -n ${PBS_JOBID:-} ]] || { echo 'A PBS allocation is required; use DRY_RUN=1 for a CPU preview.' >&2; exit 2; }
if [[ ${CUDA_MODULE:-cuda/12.9.0} != none ]]; then
  if ! type module >/dev/null 2>&1; then
    source /etc/profile.d/modules.sh
  fi
  module load "${CUDA_MODULE:-cuda/12.9.0}"
fi
exec "$PYTHON_BIN" "$runner_dir/launch.py" "$@" --manifest "$runner_dir/manifest.json"
