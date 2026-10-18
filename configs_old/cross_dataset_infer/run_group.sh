#!/usr/bin/env bash
# One source, one GPU, one pair at a time. This runner never reads manifest.csv.
set -euo pipefail
fail() { echo "[FATAL] $*" >&2; exit 2; }
[[ $# == 1 ]] || fail "Usage: bash $0 configs/cross_dataset_infer/<past_only|future_year>/<source>"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
SOURCE_DIR="$(realpath -e -- "$1")"
[[ -d "${SOURCE_DIR}" ]] || fail "Not a source directory: $1"
GROUP="$(basename -- "$(dirname -- "${SOURCE_DIR}")")"
SOURCE="$(basename -- "${SOURCE_DIR}")"
case "${GROUP}" in
  past_only) TARGETS=(AWI CNRM EC_EARTH MPI MRI NCEP); EXPECTED_YEARS=7 ;;
  future_year) TARGETS=(AWI CNRM EC_EARTH MPI MRI); EXPECTED_YEARS=30 ;;
  *) fail "Expected a past_only or future_year source directory" ;;
esac
case "${SOURCE}" in NCEP|AWI|CNRM|EC_EARTH|MPI|MRI) ;; *) fail "Unknown source: ${SOURCE}" ;; esac
mapfile -t CONFIGS < <(find "${SOURCE_DIR}" -maxdepth 1 -type f -name '*.sh' | LC_ALL=C sort)
[[ "${#CONFIGS[@]}" == "${#TARGETS[@]}" ]] || fail "Expected ${#TARGETS[@]} pair configs; found ${#CONFIGS[@]}"
# Validate every pair before executing the first. Each config gets an isolated shell.
for i in "${!CONFIGS[@]}"; do
  config="${CONFIGS[i]}"
  [[ "$(basename -- "${config}")" == "${SOURCE}_to_${TARGETS[i]}.sh" ]] || fail "Unexpected pair file: ${config}"
  (
    source "${config}"
    [[ "${SOURCE_NAME}" == "${SOURCE}" && "${TARGET_NAME}" == "${TARGETS[i]}" ]] || fail "Pair identity mismatch: ${config}"
    [[ -n "${TARGET_ROOT}" && "${TEST_ROOT_DIR}" == "${TARGET_ROOT}" && "${ROOT_DIR}" == "${SOURCE_ROOT}" ]] || fail "Explicit roots required: ${config}"
    [[ "${STRICT_YEARS}" == 1 && "${EXPERIMENT_GROUP}" == "${GROUP}" ]] || fail "Strict group settings required: ${config}"
    IFS=',' read -r -a year_tags <<< "${YEARS}"
    [[ "${#year_tags[@]}" == "${EXPECTED_YEARS}" ]] || fail "Unexpected year count: ${config}"
    [[ -f "${CKPT_PATH}" && "${CKPT_PATH}" == /*/best_overall.pt ]] || fail "Missing pinned checkpoint: ${config}"
  )
done
cd -- "${REPO_DIR}"
TOTAL_START=$SECONDS
for i in "${!CONFIGS[@]}"; do
  printf '[%d/%d] %s -> %s\n' "$((i + 1))" "${#CONFIGS[@]}" "${SOURCE}" "${TARGETS[i]}"
  echo "Config: ${CONFIGS[i]}"
  PAIR_START=$SECONDS
  # DRY_RUN is implemented before artifact creation or any Python/tmux invocation.
  DRY_RUN="${DRY_RUN:-0}" USE_TMUX=0 bash infer.sh "${CONFIGS[i]}"
  echo "Pair elapsed: $((SECONDS - PAIR_START)) s"
done
echo "Total elapsed: $((SECONDS - TOTAL_START)) s"
