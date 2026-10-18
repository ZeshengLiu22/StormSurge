#!/usr/bin/env bash
set -euo pipefail
cd /home/exouser/StormSurge
CONFIG_ROOT=/home/exouser/StormSurge/configs/CMIP6_Boston_QuickCheck_0929
PYTHON_BIN="${S0_PYTHON_BIN:-/home/exouser/.conda/envs/torchpyg-cu12x/bin/python}"
# Audit both groups before any submission.
"$PYTHON_BIN" "$CONFIG_ROOT/audit_configs.py" past_only
"$PYTHON_BIN" "$CONFIG_ROOT/audit_configs.py" future_year
bash "$CONFIG_ROOT/past_only/launch_all.sh"
bash "$CONFIG_ROOT/future_year/launch_all.sh"
