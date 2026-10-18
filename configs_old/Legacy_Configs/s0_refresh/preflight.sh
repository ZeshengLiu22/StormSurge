#!/usr/bin/env bash
set -euo pipefail
# Use the same config-selected interpreter as train.sh, before submitting jobs.
source "$1"
"$PYTHON_BIN" - <<'PY'
import sys
import torch
import torch_geometric
import train
if not torch.cuda.is_available():
    raise SystemExit("WQE/Tail factorial requires an available CUDA device; no jobs submitted.")
print(f"[Preflight OK] Python={sys.executable}; torch={torch.__version__}; GPU={torch.cuda.get_device_name(0)}")
PY
