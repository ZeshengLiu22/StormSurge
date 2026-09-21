#!/usr/bin/env bash
# Explicit launch command only; ten sequential new LR runs, no baseline reruns.
set -euo pipefail
PACT_SWEEP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PACT_REPO_DIR="$(cd "${PACT_SWEEP_DIR}/.." && pwd)"
cd "${PACT_REPO_DIR}"
PACT_PYTHON="/media/volume/PACT-Data/conda_envs/torchpyg-cu124/bin/python"
"${PACT_PYTHON}" tools/prepare_lr3e3.py
for station in CBBT Lewes; do
  for variant in S0_Single D0_DualBase D1_Tail D2_Amp D3_TailAmp; do
    config="${PACT_SWEEP_DIR}/configs/train_config_0921_${station}_${variant}_LR3e3.sh"
    USE_TMUX=0 bash train.sh "${config}"
  done
done
