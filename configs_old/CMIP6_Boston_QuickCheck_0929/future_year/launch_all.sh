#!/usr/bin/env bash
if [[ "${DRY_RUN:-0}" != "1" ]] && ! command -v qsub_local >/dev/null 2>&1 && [[ $- != *i* ]]; then
  exec bash -i "$0" "$@"
fi
set -euo pipefail
cd /home/exouser/StormSurge
CONFIG_ROOT=/home/exouser/StormSurge/configs/CMIP6_Boston_QuickCheck_0929
CONFIG_FOLDER=/home/exouser/StormSurge/configs/CMIP6_Boston_QuickCheck_0929/future_year
RESULTS_ROOT=/home/exouser/media/share/PACT/FormalRuns_0925/CMIP6_Boston_QuickCheck_0929/future_year
PYTHON_BIN="${S0_PYTHON_BIN:-/home/exouser/.conda/envs/torchpyg-cu12x/bin/python}"
export PACT_RUNSTAMP=20260929_054511
export USE_TMUX=1
export QSUB_LOCAL_SLOTS=1
"$PYTHON_BIN" "$CONFIG_ROOT/audit_configs.py" future_year
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  qsub_local() { DRY_RUN=1 USE_TMUX=0 bash "$1" "$3"; }
else
  "$PYTHON_BIN" - <<'CHECK_RUNTIME'
import sys, torch, torch_geometric, train
assert torch.cuda.is_available(), 'CUDA is required'
assert torch.cuda.is_bf16_supported(), 'BF16 is required'
print(f'[Preflight OK] Python={sys.executable}; torch={torch.__version__}; GPU={torch.cuda.get_device_name(0)}; BF16 available')
CHECK_RUNTIME
  if [[ -e "$RESULTS_ROOT/_orchestration" ]]; then
    echo '[FATAL] This group was already submitted; inspect submissions.tsv before any retry.' >&2
    exit 2
  fi
  mkdir -p "$RESULTS_ROOT"
  mkdir "$RESULTS_ROOT/_orchestration"
  cp "$CONFIG_FOLDER/manifest.csv" "$CONFIG_FOLDER/config_audit.json" "$CONFIG_FOLDER/dataset_audit.json" "$RESULTS_ROOT/_orchestration/"
  printf 'task_id\tlabel\tconfig_path\tresult_path\n' > "$RESULTS_ROOT/_orchestration/submissions.tsv"
  exec > >(tee -a "$RESULTS_ROOT/_orchestration/launch.log") 2>&1
fi
submit() {
  local label="$1" config="$2" result="$3" task_id
  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    qsub_local train.sh "$label" "$config"
  else
    [[ ! -e "$result" ]] || { echo "[FATAL] Result path already exists: $result" >&2; return 2; }
    task_id=$(qsub_local train.sh "$label" "$config")
    [[ "$task_id" =~ ^[0-9]+$ ]] || { echo "[FATAL] Unexpected queue reply: $task_id" >&2; return 2; }
    printf '%s\t%s\t%s\t%s\n' "$task_id" "$label" "$config" "$result" >> "$RESULTS_ROOT/_orchestration/submissions.tsv"
    printf '[Submitted] id=%s label=%s result=%s\n' "$task_id" "$label" "$result"
  fi
}
submit QB0929F_AWI_T0000_EP0000 /home/exouser/StormSurge/configs/CMIP6_Boston_QuickCheck_0929/future_year/train_config_CMIP6_AWI_Boston_future_year_G0_T0000_EP0000.sh /home/exouser/media/share/PACT/FormalRuns_0925/CMIP6_Boston_QuickCheck_0929/future_year/CMIP6_AWI_Boston_future_year_G0_T0000_EP0000__20260929_054511
submit QB0929F_AWI_T0500_EP0100 /home/exouser/StormSurge/configs/CMIP6_Boston_QuickCheck_0929/future_year/train_config_CMIP6_AWI_Boston_future_year_G0_T0500_EP0100.sh /home/exouser/media/share/PACT/FormalRuns_0925/CMIP6_Boston_QuickCheck_0929/future_year/CMIP6_AWI_Boston_future_year_G0_T0500_EP0100__20260929_054511
submit QB0929F_CNRM_T0000_EP0000 /home/exouser/StormSurge/configs/CMIP6_Boston_QuickCheck_0929/future_year/train_config_CMIP6_CNRM_Boston_future_year_G0_T0000_EP0000.sh /home/exouser/media/share/PACT/FormalRuns_0925/CMIP6_Boston_QuickCheck_0929/future_year/CMIP6_CNRM_Boston_future_year_G0_T0000_EP0000__20260929_054511
submit QB0929F_CNRM_T0500_EP0100 /home/exouser/StormSurge/configs/CMIP6_Boston_QuickCheck_0929/future_year/train_config_CMIP6_CNRM_Boston_future_year_G0_T0500_EP0100.sh /home/exouser/media/share/PACT/FormalRuns_0925/CMIP6_Boston_QuickCheck_0929/future_year/CMIP6_CNRM_Boston_future_year_G0_T0500_EP0100__20260929_054511
submit QB0929F_EC_EARTH_T0000_EP0000 /home/exouser/StormSurge/configs/CMIP6_Boston_QuickCheck_0929/future_year/train_config_CMIP6_EC_EARTH_Boston_future_year_G0_T0000_EP0000.sh /home/exouser/media/share/PACT/FormalRuns_0925/CMIP6_Boston_QuickCheck_0929/future_year/CMIP6_EC_EARTH_Boston_future_year_G0_T0000_EP0000__20260929_054511
submit QB0929F_EC_EARTH_T0500_EP0100 /home/exouser/StormSurge/configs/CMIP6_Boston_QuickCheck_0929/future_year/train_config_CMIP6_EC_EARTH_Boston_future_year_G0_T0500_EP0100.sh /home/exouser/media/share/PACT/FormalRuns_0925/CMIP6_Boston_QuickCheck_0929/future_year/CMIP6_EC_EARTH_Boston_future_year_G0_T0500_EP0100__20260929_054511
submit QB0929F_MPI_T0000_EP0000 /home/exouser/StormSurge/configs/CMIP6_Boston_QuickCheck_0929/future_year/train_config_CMIP6_MPI_Boston_future_year_G0_T0000_EP0000.sh /home/exouser/media/share/PACT/FormalRuns_0925/CMIP6_Boston_QuickCheck_0929/future_year/CMIP6_MPI_Boston_future_year_G0_T0000_EP0000__20260929_054511
submit QB0929F_MPI_T0500_EP0100 /home/exouser/StormSurge/configs/CMIP6_Boston_QuickCheck_0929/future_year/train_config_CMIP6_MPI_Boston_future_year_G0_T0500_EP0100.sh /home/exouser/media/share/PACT/FormalRuns_0925/CMIP6_Boston_QuickCheck_0929/future_year/CMIP6_MPI_Boston_future_year_G0_T0500_EP0100__20260929_054511
submit QB0929F_MRI_T0000_EP0000 /home/exouser/StormSurge/configs/CMIP6_Boston_QuickCheck_0929/future_year/train_config_CMIP6_MRI_Boston_future_year_G0_T0000_EP0000.sh /home/exouser/media/share/PACT/FormalRuns_0925/CMIP6_Boston_QuickCheck_0929/future_year/CMIP6_MRI_Boston_future_year_G0_T0000_EP0000__20260929_054511
submit QB0929F_MRI_T0500_EP0100 /home/exouser/StormSurge/configs/CMIP6_Boston_QuickCheck_0929/future_year/train_config_CMIP6_MRI_Boston_future_year_G0_T0500_EP0100.sh /home/exouser/media/share/PACT/FormalRuns_0925/CMIP6_Boston_QuickCheck_0929/future_year/CMIP6_MRI_Boston_future_year_G0_T0500_EP0100__20260929_054511
