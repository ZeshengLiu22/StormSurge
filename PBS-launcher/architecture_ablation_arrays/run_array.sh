#!/usr/bin/env bash
# One array index, one configuration, one four-GPU DDP experiment.
set -euo pipefail
if [[ $# -ne 3 ]]; then
  echo "Usage: $0 <LR3e-3|LR1e-3|LR5e-3_Clip0.25> <CNN|GNN> <0..143>" >&2
  exit 2
fi
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export DRY_RUN="${DRY_RUN:-1}"
case "${DRY_RUN}" in 0|1) ;; *) echo "DRY_RUN must be 0 or 1" >&2; exit 2 ;; esac
exec /glade/work/zel/conda-envs/derecho-ml/bin/python "${SCRIPT_DIR}/sweep.py" run \
  --group "$1" --encoder "$2" --index "$3" --max-attempts "${MAX_ATTEMPTS:-3}"
