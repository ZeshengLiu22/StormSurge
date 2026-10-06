#!/usr/bin/env bash
# One station, period, and source; every pair runs in the foreground.
set -euo pipefail
fail() { echo "[FATAL] $*" >&2; exit 2; }
[[ $# == 1 ]] || fail "Usage: bash $0 <station>/<past_only|future_year>/<source> directory"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
SOURCE_DIR="$(realpath -e -- "$1")"
SOURCE="$(basename -- "${SOURCE_DIR}")"
GROUP="$(basename -- "$(dirname -- "${SOURCE_DIR}")")"
STATION_TOKEN="$(basename -- "$(dirname -- "$(dirname -- "${SOURCE_DIR}")")")"
case "${STATION_TOKEN}" in Boston|CBBT|Lewes|Battery) ;; *) fail "Unknown station: ${STATION_TOKEN}" ;; esac
case "${SOURCE}" in NCEP|AWI|CNRM|EC_EARTH|MPI|MRI) ;; *) fail "Unknown source: ${SOURCE}" ;; esac
[[ "${SOURCE_DIR}" == "${SCRIPT_DIR}/${STATION_TOKEN}/${GROUP}/${SOURCE}" ]] || fail "Source group must be inside ${SCRIPT_DIR}"
case "${GROUP}" in
  past_only) TARGETS=(AWI CNRM EC_EARTH MPI MRI NCEP); FIRST=2008; LAST=2014 ;;
  future_year) TARGETS=(AWI CNRM EC_EARTH MPI MRI); FIRST=2070; LAST=2099 ;;
  *) fail "Unknown period: ${GROUP}" ;;
esac
EXPECTED_YEARS=""
for ((year=FIRST; year<=LAST; year++)); do
  EXPECTED_YEARS+="${EXPECTED_YEARS:+,}${year}_$((year+1))"
done
shopt -s nullglob
mapfile -t CONFIGS < <(printf '%s\n' "${SOURCE_DIR}"/*.sh | LC_ALL=C sort)
[[ "${#CONFIGS[@]}" == "${#TARGETS[@]}" ]] || fail "Expected ${#TARGETS[@]} pair configs; found ${#CONFIGS[@]}"
for i in "${!CONFIGS[@]}"; do
  config="${CONFIGS[i]}"
  [[ "$(basename -- "${config}")" == "${SOURCE}_to_${TARGETS[i]}.sh" ]] || fail "Unexpected pair: ${config}"
  (
    source "${config}"
    [[ "${STATION}" == "${STATION_TOKEN}" && "${SOURCE_NAME}" == "${SOURCE}" && "${TARGET_NAME}" == "${TARGETS[i]}" ]] || fail "Pair identity mismatch: ${config}"
    [[ -n "${TARGET_ROOT}" && "${TEST_ROOT_DIR}" == "${TARGET_ROOT}" && "${ROOT_DIR}" == "${SOURCE_ROOT}" ]] || fail "Explicit source/target roots required: ${config}"
    [[ "${STRICT_YEARS}" == 1 && "${EXPERIMENT_GROUP}" == "${GROUP}" && "${YEARS}" == "${EXPECTED_YEARS}" ]] || fail "Wrong strict year population: ${config}"
    [[ "${MODEL}" == perceiver3 && "${ENCODER_TYPE}" == GraphSAGE && "${TEMPORAL_BLOCK}" == Transformer && "${HEAD_TYPE}" == single && "${HISTORY_HOURS}" == 24 ]] || fail "Wrong model configuration: ${config}"
    [[ "${INFERENCE_RESULTS_ROOT}" == "/home/exouser/media/share/PACT/FormalRuns_0925/CrossDatasetInference_4Stations/${STATION_TOKEN}/${GROUP}" ]] || fail "Wrong production output root: ${config}"
    [[ -f "${CKPT_PATH}" && "${CKPT_PATH}" == /*G0_T0500_EP0100__*/best_overall.pt && "${CKPT_PATH}" != *\** ]] || fail "Missing exact final checkpoint: ${config}"
  )
done
cd -- "${REPO_DIR}"
TOTAL_START=$SECONDS
for i in "${!CONFIGS[@]}"; do
  printf '[%d/%d] %s / %s / %s -> %s\n' "$((i+1))" "${#CONFIGS[@]}" "${STATION_TOKEN}" "${GROUP}" "${SOURCE}" "${TARGETS[i]}"
  PAIR_START=$SECONDS
  if DRY_RUN="${DRY_RUN:-0}" USE_TMUX=0 bash infer.sh "${CONFIGS[i]}"; then
    echo "Pair elapsed: $((SECONDS-PAIR_START)) s"
  else
    code=$?
    echo "[FAILED PAIR] ${STATION_TOKEN}/${GROUP}/${SOURCE}->${TARGETS[i]} (exit ${code}); config=${CONFIGS[i]}" >&2
    exit "${code}"
  fi
done
echo "Source group elapsed: $((SECONDS-TOTAL_START)) s"
