#!/usr/bin/env bash
if ! command -v qsub_local >/dev/null 2>&1 && [[ $- != *i* ]]; then
  exec bash -i "$0" "$@"
fi
set -euo pipefail
cd /home/exouser/StormSurge

bash /home/exouser/StormSurge/configs/wqe_factorial_multickpt/preflight.sh configs/wqe_factorial_multickpt/train_config_NCEP_CBBT_G0_E0_T0.sh

qsub_local train.sh WQEF_CBBT_G0E0T0 configs/wqe_factorial_multickpt/train_config_NCEP_CBBT_G0_E0_T0.sh
qsub_local train.sh WQEF_Boston_G0E0T0 configs/wqe_factorial_multickpt/train_config_NCEP_Boston_G0_E0_T0.sh
qsub_local train.sh WQEF_Battery_G0E0T0 configs/wqe_factorial_multickpt/train_config_NCEP_Battery_G0_E0_T0.sh
qsub_local train.sh WQEF_Lewes_G0E0T0 configs/wqe_factorial_multickpt/train_config_NCEP_Lewes_G0_E0_T0.sh
