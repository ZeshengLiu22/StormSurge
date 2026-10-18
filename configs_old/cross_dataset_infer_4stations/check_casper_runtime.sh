#!/usr/bin/env bash
# Check the exact interpreter and CUDA kernels before creating inference outputs.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
source "${SCRIPT_DIR}/common.sh"
[[ -x "${PYTHON_BIN}" ]] || { echo "[FATAL] Python not found: ${PYTHON_BIN}" >&2; exit 2; }
"${PYTHON_BIN}" - "${USE_AMP}" "${AMP_DTYPE}" "${USE_TF32}" "${BATCH_SIZE}" <<'PYTHON'
import sys
import torch
import torch_geometric
from torch_geometric.nn import SAGEConv

amp, dtype, tf32, batch = sys.argv[1:]
if not torch.cuda.is_available():
    raise SystemExit("CUDA unavailable. Run inside a Casper PBS GPU allocation.")
capability = torch.cuda.get_device_capability(0)
arch = "sm_" + "".join(map(str, capability))
if arch not in torch.cuda.get_arch_list():
    raise SystemExit(f"Torch build lacks {arch}: {torch.cuda.get_arch_list()}")
if amp == "1" and dtype == "bf16" and capability[0] < 8:
    raise SystemExit("This GPU needs AMP_DTYPE=fp16; native BF16 requires Ampere or newer.")
if tf32 == "1" and capability[0] < 8:
    raise SystemExit("This GPU needs USE_TF32=0.")
torch.backends.cuda.matmul.allow_tf32 = tf32 == "1"
torch.backends.cudnn.allow_tf32 = tf32 == "1"
x = torch.randn(8, 16, device="cuda")
edges = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], device="cuda")
layer = SAGEConv(16, 16).cuda().eval()
with torch.inference_mode(), torch.autocast("cuda", dtype={"fp16": torch.float16, "bf16": torch.bfloat16}[dtype], enabled=amp == "1"):
    y = layer(x, edges)
    product = x @ x.T
assert torch.isfinite(y).all() and torch.isfinite(product).all(), "Nonfinite CUDA probe"
torch.cuda.synchronize()
print(f"PASS: {sys.executable}")
print(f"Torch {torch.__version__}; CUDA {torch.version.cuda}; PyG {torch_geometric.__version__}")
print(f"GPU {torch.cuda.get_device_name(0)}; {arch}; AMP={amp}/{dtype}; TF32={tf32}; batch={batch}")
print("PASS: CUDA GraphSAGE and AMP matrix multiply")
PYTHON
