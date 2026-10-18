#!/usr/bin/env bash
if [[ "${DRY_RUN:-0}" != "1" ]] && ! command -v qsub_local >/dev/null 2>&1 && [[ $- != *i* ]]; then
  exec bash -i "$0" "$@"
fi
set -euo pipefail
cd /home/exouser/StormSurge

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  qsub_local() { DRY_RUN=1 USE_TMUX=0 bash "$1" "$3"; }
else
  bash /home/exouser/StormSurge/configs/s0_refresh/preflight.sh configs/s0_refresh/train_config_NCEP_CBBT_G1_T0.sh
fi

qsub_local train.sh S0F_CBBT_G1T0 configs/s0_refresh/train_config_NCEP_CBBT_G1_T0.sh
qsub_local train.sh S0F_Boston_G1T0 configs/s0_refresh/train_config_NCEP_Boston_G1_T0.sh
qsub_local train.sh S0F_Battery_G1T0 configs/s0_refresh/train_config_NCEP_Battery_G1_T0.sh
qsub_local train.sh S0F_Lewes_G1T0 configs/s0_refresh/train_config_NCEP_Lewes_G1_T0.sh
