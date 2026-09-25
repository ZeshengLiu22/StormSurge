#!/usr/bin/env bash
if [[ "${DRY_RUN:-0}" != "1" ]] && ! command -v qsub_local >/dev/null 2>&1 && [[ $- != *i* ]]; then
  exec bash -i "$0" "$@"
fi
set -euo pipefail
cd /home/exouser/StormSurge

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  qsub_local() { DRY_RUN=1 USE_TMUX=0 bash "$1" "$3"; }
else
  bash /home/exouser/StormSurge/configs/wqe_factorial_multickpt/preflight.sh configs/wqe_factorial_multickpt/train_config_NCEP_CBBT_G1_E0_T1.sh
fi

qsub_local train.sh WQEF_CBBT_G1E0T1 configs/wqe_factorial_multickpt/train_config_NCEP_CBBT_G1_E0_T1.sh
qsub_local train.sh WQEF_Boston_G1E0T1 configs/wqe_factorial_multickpt/train_config_NCEP_Boston_G1_E0_T1.sh
qsub_local train.sh WQEF_Battery_G1E0T1 configs/wqe_factorial_multickpt/train_config_NCEP_Battery_G1_E0_T1.sh
qsub_local train.sh WQEF_Lewes_G1E0T1 configs/wqe_factorial_multickpt/train_config_NCEP_Lewes_G1_E0_T1.sh
