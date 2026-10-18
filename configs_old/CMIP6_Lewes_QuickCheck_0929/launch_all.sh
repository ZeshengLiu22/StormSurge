#!/usr/bin/env bash
set -euo pipefail
CONFIG_ROOT=/home/exouser/StormSurge/configs/CMIP6_Lewes_QuickCheck_0929
bash "$CONFIG_ROOT/past_only/launch_all.sh"
bash "$CONFIG_ROOT/future_year/launch_all.sh"
