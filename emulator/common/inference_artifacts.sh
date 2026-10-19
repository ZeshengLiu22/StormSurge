#!/usr/bin/env bash
# Checkpoint resolution and replay artifacts for the inference launcher.

# Resolve a checkpoint file or choose the most recently modified glob match.
# Args: $1 is a literal file path or shell glob pattern.
# Output: Prints the selected file's absolute path to standard output.
# Status: Zero on successful resolution; nonzero for an empty pattern, no file
# match, or a failed realpath invocation. Does not modify checkpoint files.
resolve_ckpt() {
  local pattern="$1" candidate newest=""
  [[ -n "${pattern}" ]] || return 1
  if [[ -f "${pattern}" ]]; then
    realpath -- "${pattern}"
    return
  fi
  while IFS= read -r candidate; do
    [[ -f "${candidate}" ]] || continue
    if [[ -z "${newest}" || "${candidate}" -nt "${newest}" ]]; then
      newest="${candidate}"
    fi
  done < <(compgen -G "${pattern}" || true)
  [[ -n "${newest}" ]] || return 1
  realpath -- "${newest}"
}

# Write a shell-readable snapshot of the launcher's resolved configuration.
# Args: $1 is the destination path; its parent directory must already exist.
# Inputs: Reads the listed launcher variables and CONFIG_PATH, WORKDIR,
# CKPT_PATH, and CKPT_RESOLVED. Relative data/script paths are rooted at WORKDIR.
# Output: Overwrites $1 with shell-quoted assignments, an absolute resolved
# checkpoint path, and a UTC generation timestamp. No standard output.
# Status: The redirected write command's exit status.
write_infer_config_snapshot() {
  local output_path="$1" name value
  local config_vars=(
    INFER_PY PYTHON_BIN USE_TMUX ROOT_DIR TEST_ROOT_DIR STATION STATION_JSON_DIR MODEL MODEL_LABEL
    SOURCE_NAME TARGET_NAME SOURCE_ROOT TARGET_ROOT STRICT_YEARS EXPERIMENT_GROUP EXPECTED_YEAR_COUNT
    USE_SITE_ELEVATION USE_BATHYMETRY
    ENCODER_TYPE TEMPORAL_BLOCK HISTORY_HOURS BATCH_SIZE YEARS
    CNN_INTERMEDIATE_CHANNEL
    USE_AMP AMP_DTYPE USE_TF32 TORCH_THREADS NUM_WORKERS PIN_MEMORY
    PERSISTENT_WORKERS PREFETCH_FACTOR MP_CONTEXT
    CUDA_LAUNCH_BLOCKING_FLAG TORCH_GPU_PROBE FAIL_IF_NO_SLURM
    INFERENCE_RESULTS_ROOT WORKDIR RUN_DIR OUT_DIR RUNSTAMP TEST_TAG
  )
  {
    printf '#!/usr/bin/env bash\n# Fully resolved inference configuration; no source-config dependencies.\n'
    printf '# Generated UTC: %s\n' "$(date -u +'%Y-%m-%d %H:%M:%S')"
    printf 'ORIGINAL_CONFIG_PATH=%q\n' "${CONFIG_PATH}"
    for name in "${config_vars[@]}"; do
      if [[ -v "${name}" ]]; then
        value="${!name}"
        if [[ "${name}" == "TARGET_ROOT" ]]; then value="${TEST_ROOT_DIR:-}"; fi
        case "${name}" in
          INFER_PY|ROOT_DIR|TEST_ROOT_DIR|SOURCE_ROOT|TARGET_ROOT|STATION_JSON_DIR|INFERENCE_RESULTS_ROOT)
            if [[ -n "${value}" && "${value}" != /* ]]; then
              value="${WORKDIR}/${value#./}"
            fi ;;
        esac
        printf '%s=%q\n' "${name}" "${value}"
      fi
    done
    printf 'CKPT_PATTERN=%q\n' "${CKPT_PATH}"
    printf 'CKPT_PATH=%q\nCKPT_RESOLVED=%q\n' "${CKPT_RESOLVED}" "${CKPT_RESOLVED}"
  } > "${output_path}"
}

# Write a replay script containing the exact inference executable and arguments.
# Args: $1 is the destination path; remaining arguments form the command vector.
# Inputs: Reads WORKDIR and optional CUDA_VISIBLE_DEVICES/CUDA_LAUNCH_BLOCKING.
# Output: Overwrites $1 with a shell-quoted command, working directory, and
# relevant environment settings. Does not execute the command or set file mode.
# Status: The redirected write command's exit status.
write_infer_command() {
  local output_path="$1" name
  shift
  {
    printf '#!/usr/bin/env bash\nset -euo pipefail\n'
    printf '# Exact interpreter and arguments used for this run.\ncd -- %q\n' "${WORKDIR}"
    printf 'export PYTHONUNBUFFERED=1\n'
    for name in CUDA_VISIBLE_DEVICES CUDA_LAUNCH_BLOCKING; do
      if [[ -v "${name}" ]]; then printf 'export %s=%q\n' "${name}" "${!name}"; fi
    done
    printf 'exec'
    printf ' %q' "$@"
    printf '\n'
  } > "${output_path}"
}
