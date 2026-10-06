#!/usr/bin/env bash
# Fresh sequential run. No resume, skip, or recovery behavior.
set -euo pipefail
[[ $# == 0 ]] || { echo "Usage: DRY_RUN=0|1 bash $0" >&2; exit 2; }
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RESULT_ROOT="/home/exouser/media/share/PACT/FormalRuns_0925/CrossDatasetInference_4Stations"
if [[ "${DRY_RUN:-0}" != 1 ]]; then
  # Audit and separately labelled smoke artifacts may precede production.
  # Refuse an existing production tree rather than mixing or resuming it.
  for station in Boston CBBT Lewes Battery; do
    if [[ -e "${RESULT_ROOT}/${station}" ]]; then
      echo "[FATAL] Existing production station directory: ${RESULT_ROOT}/${station}. Fresh run required; no pairs executed." >&2
      exit 2
    fi
  done
fi
for station in Boston CBBT Lewes Battery; do
  bash "${SCRIPT_DIR}/run_station.sh" "${station}"
done
echo "[ALL COMPLETE] 264 pair evaluations (DRY_RUN=${DRY_RUN:-0})"
