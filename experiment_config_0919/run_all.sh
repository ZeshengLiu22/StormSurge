#!/usr/bin/env bash
# Validate, then submit through the existing one-slot qsub_local/tmux queue.
set -euo pipefail
suite_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd "${suite_dir}/.." && pwd)"
station="all"
do_dry_run="${DRY_RUN:-0}"
for argument in "$@"; do
  case "${argument}" in
    --dry-run) do_dry_run=1 ;;
    all|CBBT|Lewes|Battery|Boston) station="${argument}" ;;
    *) echo "Usage: bash run_all.sh [all|CBBT|Lewes|Battery|Boston] [--dry-run]" >&2; exit 2 ;;
  esac
done
python_bin="/media/volume/PACT-Data/conda_envs/torchpyg-cu124/bin/python"
cd "${repo_dir}"
"${python_bin}" -B "${suite_dir}/validate_configs.py" --check-only
mapfile -t configs < <("${python_bin}" -B - "${suite_dir}/manifest.csv" "${station}" <<'PY'
import csv
import sys
with open(sys.argv[1]) as handle:
    for row in csv.DictReader(handle):
        if sys.argv[2] == 'all' or row['station'] == sys.argv[2]:
            print(row['config_path'])
PY
)
expected=11
if [[ "${station}" == "all" ]]; then expected=44; fi
[[ "${#configs[@]}" -eq "${expected}" ]]
if [[ "${do_dry_run}" == "1" ]]; then
  for config in "${configs[@]}"; do
    DRY_RUN=1 bash train.sh "${config}"
  done
  exit 0
fi
if [[ "${WORLD_SIZE:-1}" != "1" ]]; then
  echo "This suite requires a single-process environment; leave the DDP allocation first." >&2
  exit 2
fi
# Formal runs require GPU 0 and BF16; never silently start a CPU job.
CUDA_VISIBLE_DEVICES=0 "${python_bin}" -B - <<'PY'
import torch
if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    raise SystemExit('Expected exactly one visible GPU (CUDA_VISIBLE_DEVICES=0); no jobs submitted.')
if not torch.cuda.is_bf16_supported():
    raise SystemExit('Canonical BF16 is unsupported on GPU 0; no precision fallback or job submission.')
PY
# qsub_local is an interactive-shell function installed in the existing environment.
# No queue commands are executed by --dry-run or validate_configs.py.
bash -ic '
set -euo pipefail
type qsub_local >/dev/null
export QSUB_LOCAL_SLOTS=1 TS_SLOTS=1
unset QSUB_LOCAL_SESSION_NAME
cd "$1"
shift
for config in "$@"; do
  label="${config##*/}"
  label="${label#train_config_}"
  label="${label%.sh}"
  qsub_local train.sh "$label" "$config"
done
' bash "${repo_dir}" "${configs[@]}"
