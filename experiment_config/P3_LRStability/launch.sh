#!/usr/bin/env bash
# Source this wrapper to use qsub_local from the current interactive shell.
# Default: validate and print commands. Only --submit calls the existing queue.
# A subshell preserves the caller's directory, options and environment.
_p3_lr_stability_main() (
    local p3_dir p3_repo p3_python p3_records p3_kind p3_label p3_config p3_mode
    p3_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)" || return 1
    p3_repo="${P3_REPO_ROOT:-$(cd -- "${p3_dir}/../.." && pwd)}"
    p3_python="/media/volume/PACT-Data/conda_envs/torchpyg-cu124/bin/python"
    p3_records="$("${p3_python}" -B "${p3_dir}/launch.py" "$@" --repo "${p3_repo}" --queue-records)" || return $?
    # --help returns ordinary help text, not queue records.
    if [[ "${p3_records}" != MODE$'\t'* ]]; then
        printf '%s\n' "${p3_records}"
        return 0
    fi
    IFS=$'\t' read -r p3_kind p3_mode <<< "${p3_records%%$'\n'*}"
    if [[ "${p3_mode}" == SUBMIT ]] && ! command -v qsub_local >/dev/null 2>&1; then
        printf '%s\n' 'qsub_local is unavailable. Run this in your normal local shell with: source experiment_config/P3_LRStability/launch.sh --submit [filters]' >&2
        return 2
    fi
    cd -- "${p3_repo}" || return 1
    if [[ "${p3_mode}" == PREVIEW ]]; then
        printf 'cd %q\n' "${p3_repo}"
    fi
    while IFS=$'\t' read -r p3_kind p3_label p3_config; do
        [[ "${p3_kind}" == RUN ]] || continue
        if [[ "${p3_mode}" == SUBMIT ]]; then
            qsub_local train.sh "${p3_label}" "${p3_config}" || return $?
        else
            printf 'qsub_local train.sh %q %q\n' "${p3_label}" "${p3_config}"
        fi
    done <<< "${p3_records}"
)
_p3_lr_stability_main "$@"
