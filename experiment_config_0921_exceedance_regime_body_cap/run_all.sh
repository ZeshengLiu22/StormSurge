#!/usr/bin/env bash
# Validate, then submit the matched study through the existing one-slot queue.
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
expected=6
if [[ "${station}" == "all" ]]; then expected=24; fi
[[ "${#configs[@]}" -eq "${expected}" ]]
if [[ "${do_dry_run}" == "1" ]]; then
  for config in "${configs[@]}"; do
    DRY_RUN=1 PACT_RUNSTAMP=20000101_000000 bash train.sh "${config}"
  done
  exit 0
fi
if [[ "${WORLD_SIZE:-1}" != "1" ]]; then
  echo "This suite requires a single-process environment." >&2
  exit 2
fi
CUDA_VISIBLE_DEVICES=0 "${python_bin}" -B - <<'PY'
import torch
if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    raise SystemExit('Expected GPU 0; no CPU fallback or job submission.')
if not torch.cuda.is_bf16_supported():
    raise SystemExit('GPU 0 must support BF16; no precision fallback or job submission.')
PY
# qsub_local captures this worktree's cwd; one slot serializes all 24 jobs.
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
