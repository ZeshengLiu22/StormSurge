#!/usr/bin/env bash
set -euo pipefail
[[ $# == 1 ]] || { echo "Usage: bash $0 <Boston|CBBT|Lewes|Battery>" >&2; exit 2; }
case "$1" in Boston|CBBT|Lewes|Battery) ;; *) echo "Unknown station: $1" >&2; exit 2 ;; esac
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
for group in past_only future_year; do
  for source_name in NCEP AWI CNRM EC_EARTH MPI MRI; do
    bash "${SCRIPT_DIR}/run_group.sh" "${SCRIPT_DIR}/$1/${group}/${source_name}"
  done
done
echo "[STATION COMPLETE] $1: 66 pairs (DRY_RUN=${DRY_RUN:-0})"
