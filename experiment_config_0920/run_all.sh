#!/usr/bin/env bash
# Dry-run all 16 entries, or queue only the 12 training runs. F1 is evaluated later.
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
mapfile -t configs < <("${python_bin}" -B - "${suite_dir}/manifest.csv" "${station}" train <<'PY'
import csv
import sys
with open(sys.argv[1]) as handle:
    for row in csv.DictReader(handle):
        if row['mode'] == sys.argv[3] and (sys.argv[2] == 'all' or row['station'] == sys.argv[2]):
            print(row['config_path'])
PY
)
expected=3
if [[ "${station}" == "all" ]]; then expected=12; fi
[[ "${#configs[@]}" -eq "${expected}" ]]
if [[ "${do_dry_run}" == "1" ]]; then
  for config in "${configs[@]}"; do
    DRY_RUN=1 bash train.sh "${config}"
  done
  mapfile -t evaluations < <("${python_bin}" -B - "${suite_dir}/manifest.csv" "${station}" <<'PY'
import csv
import sys
with open(sys.argv[1]) as handle:
    for row in csv.DictReader(handle):
        if row['mode'] == 'postprocess' and (sys.argv[2] == 'all' or row['station'] == sys.argv[2]):
            print(row['config_path'])
PY
  )
  for config in "${evaluations[@]}"; do
    "${python_bin}" -B "${suite_dir}/run_f1.py" "${config}" --dry-run
  done
  exit 0
fi
if [[ "${WORLD_SIZE:-1}" != "1" ]]; then
  echo "This suite requires the canonical single-process environment." >&2
  exit 2
fi
CUDA_VISIBLE_DEVICES=0 "${python_bin}" -B - <<'PY'
import torch
if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    raise SystemExit('Expected one visible GPU (CUDA_VISIBLE_DEVICES=0); no jobs submitted.')
if not torch.cuda.is_bf16_supported():
    raise SystemExit('Canonical BF16 is unsupported on GPU 0; no jobs submitted.')
PY
# Same one-slot qsub_local/tmux convention as 0919. Only mode=train rows enter here.
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
echo 'F1 was not queued. After F0 finishes, run run_f1.py with its corresponding selected checkpoint.'
