#!/usr/bin/env bash
if [[ "${DRY_RUN:-0}" != "1" ]] && ! command -v qsub_local >/dev/null 2>&1 && [[ $- != *i* ]]; then
  exec bash -i "$0" "$@"
fi
set -euo pipefail
cd /home/exouser/StormSurge
CONFIG_ROOT=/home/exouser/StormSurge/configs/CMIP6_Battery_QuickCheck_0929
CONFIG_FOLDER="$CONFIG_ROOT/future_year"
RESULTS_ROOT=/home/exouser/media/share/PACT/FormalRuns_0925/CMIP6_Battery_QuickCheck_0929/future_year
PYTHON_BIN="${S0_PYTHON_BIN:-/home/exouser/.conda/envs/torchpyg-cu12x/bin/python}"
export PACT_RUNSTAMP=20261003_051753 USE_TMUX=1 QSUB_LOCAL_SLOTS=1
unset QSUB_LOCAL_SESSION_NAME
"$PYTHON_BIN" "$CONFIG_ROOT/audit_configs.py" future_year
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo '[DRY_RUN] Validated all 20 commands; no jobs submitted. See the group dry_run.log.'
  exit 0
fi
command -v qsub_local >/dev/null 2>&1 || { echo '[FATAL] qsub_local is unavailable.' >&2; exit 2; }
"$PYTHON_BIN" - <<'CHECK_RUNTIME'
import torch
assert torch.cuda.is_available(), 'CUDA is required'
assert torch.cuda.is_bf16_supported(), 'BF16 is required'
CHECK_RUNTIME
RECEIPT_DIR="$RESULTS_ROOT/_orchestration/$PACT_RUNSTAMP"
mkdir -p "$RESULTS_ROOT/_orchestration"
mkdir "$RECEIPT_DIR" || { echo "[FATAL] Batch already submitted: $RECEIPT_DIR" >&2; exit 2; }
cp "$CONFIG_FOLDER/manifest.csv" "$CONFIG_FOLDER/config_audit.json" "$CONFIG_FOLDER/dataset_inventory.json" "$RECEIPT_DIR/"
printf 'task_id\tlabel\tconfig_path\tresult_path\n' > "$RECEIPT_DIR/submissions.tsv"
exec > >(tee -a "$RECEIPT_DIR/launch.log") 2>&1
submit() {
  local label="$1" config="$2" result="$3" task_id
  [[ ! -e "$result" ]] || { echo "[FATAL] Result path exists: $result" >&2; return 2; }
  task_id=$(qsub_local train.sh "$label" "$config")
  [[ "$task_id" =~ ^[0-9]+$ ]] || { echo "[FATAL] Unexpected queue reply: $task_id" >&2; return 2; }
  printf '%s\t%s\t%s\t%s\n' "$task_id" "$label" "$config" "$result" >> "$RECEIPT_DIR/submissions.tsv"
  printf '[Submitted] id=%s label=%s result=%s\n' "$task_id" "$label" "$result"
}
submit Q0929_Battery_F_AWI_T0000_EP0000_20261003_051753 /home/exouser/StormSurge/configs/CMIP6_Battery_QuickCheck_0929/future_year/train_config_CMIP6_AWI_Battery_future_year_G0_T0000_EP0000.sh /home/exouser/media/share/PACT/FormalRuns_0925/CMIP6_Battery_QuickCheck_0929/future_year/CMIP6_AWI_Battery_future_year_G0_T0000_EP0000__20261003_051753
submit Q0929_Battery_F_AWI_T0500_EP0000_20261003_051753 /home/exouser/StormSurge/configs/CMIP6_Battery_QuickCheck_0929/future_year/train_config_CMIP6_AWI_Battery_future_year_G0_T0500_EP0000.sh /home/exouser/media/share/PACT/FormalRuns_0925/CMIP6_Battery_QuickCheck_0929/future_year/CMIP6_AWI_Battery_future_year_G0_T0500_EP0000__20261003_051753
submit Q0929_Battery_F_AWI_T0000_EP0100_20261003_051753 /home/exouser/StormSurge/configs/CMIP6_Battery_QuickCheck_0929/future_year/train_config_CMIP6_AWI_Battery_future_year_G0_T0000_EP0100.sh /home/exouser/media/share/PACT/FormalRuns_0925/CMIP6_Battery_QuickCheck_0929/future_year/CMIP6_AWI_Battery_future_year_G0_T0000_EP0100__20261003_051753
submit Q0929_Battery_F_AWI_T0500_EP0100_20261003_051753 /home/exouser/StormSurge/configs/CMIP6_Battery_QuickCheck_0929/future_year/train_config_CMIP6_AWI_Battery_future_year_G0_T0500_EP0100.sh /home/exouser/media/share/PACT/FormalRuns_0925/CMIP6_Battery_QuickCheck_0929/future_year/CMIP6_AWI_Battery_future_year_G0_T0500_EP0100__20261003_051753
submit Q0929_Battery_F_CNRM_T0000_EP0000_20261003_051753 /home/exouser/StormSurge/configs/CMIP6_Battery_QuickCheck_0929/future_year/train_config_CMIP6_CNRM_Battery_future_year_G0_T0000_EP0000.sh /home/exouser/media/share/PACT/FormalRuns_0925/CMIP6_Battery_QuickCheck_0929/future_year/CMIP6_CNRM_Battery_future_year_G0_T0000_EP0000__20261003_051753
submit Q0929_Battery_F_CNRM_T0500_EP0000_20261003_051753 /home/exouser/StormSurge/configs/CMIP6_Battery_QuickCheck_0929/future_year/train_config_CMIP6_CNRM_Battery_future_year_G0_T0500_EP0000.sh /home/exouser/media/share/PACT/FormalRuns_0925/CMIP6_Battery_QuickCheck_0929/future_year/CMIP6_CNRM_Battery_future_year_G0_T0500_EP0000__20261003_051753
submit Q0929_Battery_F_CNRM_T0000_EP0100_20261003_051753 /home/exouser/StormSurge/configs/CMIP6_Battery_QuickCheck_0929/future_year/train_config_CMIP6_CNRM_Battery_future_year_G0_T0000_EP0100.sh /home/exouser/media/share/PACT/FormalRuns_0925/CMIP6_Battery_QuickCheck_0929/future_year/CMIP6_CNRM_Battery_future_year_G0_T0000_EP0100__20261003_051753
submit Q0929_Battery_F_CNRM_T0500_EP0100_20261003_051753 /home/exouser/StormSurge/configs/CMIP6_Battery_QuickCheck_0929/future_year/train_config_CMIP6_CNRM_Battery_future_year_G0_T0500_EP0100.sh /home/exouser/media/share/PACT/FormalRuns_0925/CMIP6_Battery_QuickCheck_0929/future_year/CMIP6_CNRM_Battery_future_year_G0_T0500_EP0100__20261003_051753
submit Q0929_Battery_F_EC_EARTH_T0000_EP0000_20261003_051753 /home/exouser/StormSurge/configs/CMIP6_Battery_QuickCheck_0929/future_year/train_config_CMIP6_EC_EARTH_Battery_future_year_G0_T0000_EP0000.sh /home/exouser/media/share/PACT/FormalRuns_0925/CMIP6_Battery_QuickCheck_0929/future_year/CMIP6_EC_EARTH_Battery_future_year_G0_T0000_EP0000__20261003_051753
submit Q0929_Battery_F_EC_EARTH_T0500_EP0000_20261003_051753 /home/exouser/StormSurge/configs/CMIP6_Battery_QuickCheck_0929/future_year/train_config_CMIP6_EC_EARTH_Battery_future_year_G0_T0500_EP0000.sh /home/exouser/media/share/PACT/FormalRuns_0925/CMIP6_Battery_QuickCheck_0929/future_year/CMIP6_EC_EARTH_Battery_future_year_G0_T0500_EP0000__20261003_051753
submit Q0929_Battery_F_EC_EARTH_T0000_EP0100_20261003_051753 /home/exouser/StormSurge/configs/CMIP6_Battery_QuickCheck_0929/future_year/train_config_CMIP6_EC_EARTH_Battery_future_year_G0_T0000_EP0100.sh /home/exouser/media/share/PACT/FormalRuns_0925/CMIP6_Battery_QuickCheck_0929/future_year/CMIP6_EC_EARTH_Battery_future_year_G0_T0000_EP0100__20261003_051753
submit Q0929_Battery_F_EC_EARTH_T0500_EP0100_20261003_051753 /home/exouser/StormSurge/configs/CMIP6_Battery_QuickCheck_0929/future_year/train_config_CMIP6_EC_EARTH_Battery_future_year_G0_T0500_EP0100.sh /home/exouser/media/share/PACT/FormalRuns_0925/CMIP6_Battery_QuickCheck_0929/future_year/CMIP6_EC_EARTH_Battery_future_year_G0_T0500_EP0100__20261003_051753
submit Q0929_Battery_F_MPI_T0000_EP0000_20261003_051753 /home/exouser/StormSurge/configs/CMIP6_Battery_QuickCheck_0929/future_year/train_config_CMIP6_MPI_Battery_future_year_G0_T0000_EP0000.sh /home/exouser/media/share/PACT/FormalRuns_0925/CMIP6_Battery_QuickCheck_0929/future_year/CMIP6_MPI_Battery_future_year_G0_T0000_EP0000__20261003_051753
submit Q0929_Battery_F_MPI_T0500_EP0000_20261003_051753 /home/exouser/StormSurge/configs/CMIP6_Battery_QuickCheck_0929/future_year/train_config_CMIP6_MPI_Battery_future_year_G0_T0500_EP0000.sh /home/exouser/media/share/PACT/FormalRuns_0925/CMIP6_Battery_QuickCheck_0929/future_year/CMIP6_MPI_Battery_future_year_G0_T0500_EP0000__20261003_051753
submit Q0929_Battery_F_MPI_T0000_EP0100_20261003_051753 /home/exouser/StormSurge/configs/CMIP6_Battery_QuickCheck_0929/future_year/train_config_CMIP6_MPI_Battery_future_year_G0_T0000_EP0100.sh /home/exouser/media/share/PACT/FormalRuns_0925/CMIP6_Battery_QuickCheck_0929/future_year/CMIP6_MPI_Battery_future_year_G0_T0000_EP0100__20261003_051753
submit Q0929_Battery_F_MPI_T0500_EP0100_20261003_051753 /home/exouser/StormSurge/configs/CMIP6_Battery_QuickCheck_0929/future_year/train_config_CMIP6_MPI_Battery_future_year_G0_T0500_EP0100.sh /home/exouser/media/share/PACT/FormalRuns_0925/CMIP6_Battery_QuickCheck_0929/future_year/CMIP6_MPI_Battery_future_year_G0_T0500_EP0100__20261003_051753
submit Q0929_Battery_F_MRI_T0000_EP0000_20261003_051753 /home/exouser/StormSurge/configs/CMIP6_Battery_QuickCheck_0929/future_year/train_config_CMIP6_MRI_Battery_future_year_G0_T0000_EP0000.sh /home/exouser/media/share/PACT/FormalRuns_0925/CMIP6_Battery_QuickCheck_0929/future_year/CMIP6_MRI_Battery_future_year_G0_T0000_EP0000__20261003_051753
submit Q0929_Battery_F_MRI_T0500_EP0000_20261003_051753 /home/exouser/StormSurge/configs/CMIP6_Battery_QuickCheck_0929/future_year/train_config_CMIP6_MRI_Battery_future_year_G0_T0500_EP0000.sh /home/exouser/media/share/PACT/FormalRuns_0925/CMIP6_Battery_QuickCheck_0929/future_year/CMIP6_MRI_Battery_future_year_G0_T0500_EP0000__20261003_051753
submit Q0929_Battery_F_MRI_T0000_EP0100_20261003_051753 /home/exouser/StormSurge/configs/CMIP6_Battery_QuickCheck_0929/future_year/train_config_CMIP6_MRI_Battery_future_year_G0_T0000_EP0100.sh /home/exouser/media/share/PACT/FormalRuns_0925/CMIP6_Battery_QuickCheck_0929/future_year/CMIP6_MRI_Battery_future_year_G0_T0000_EP0100__20261003_051753
submit Q0929_Battery_F_MRI_T0500_EP0100_20261003_051753 /home/exouser/StormSurge/configs/CMIP6_Battery_QuickCheck_0929/future_year/train_config_CMIP6_MRI_Battery_future_year_G0_T0500_EP0100.sh /home/exouser/media/share/PACT/FormalRuns_0925/CMIP6_Battery_QuickCheck_0929/future_year/CMIP6_MRI_Battery_future_year_G0_T0500_EP0100__20261003_051753
