"""Render self-contained CMIP6 PBS failure notifications; no mail is sent here."""
import re
import shlex

MAIL_TO = 'zel220@lehigh.edu'
BEGIN = '# BEGIN CMIP6 failure notification'
END = '# END CMIP6 failure notification'


def add_failure_mail(text, log_dir):
    """Add scheduler abort mail and a status-preserving shell EXIT handler."""
    assert BEGIN not in text, 'Failure notifications already installed'
    assert not re.search(r'^#PBS -[mM] ', text, re.M), 'Unexpected existing mail directives'
    assert text.count('#PBS -j oe\n') == 1
    assert text.count('set -euo pipefail\n') == 1
    assert text.count('exec bash train.sh "${config}"\n') == 1
    block = BEGIN + '\n' + 'CMIP6_MAIL_TO=' + shlex.quote(MAIL_TO) + '\n'
    block += 'CMIP6_PBS_LOG_DIR=' + shlex.quote(str(log_dir)) + '\n'
    block += r'''# PBS -m a covers scheduler aborts; this catches ordinary nonzero exits.
# Send only inside a real PBS job, never from a CPU DRY_RUN.
cmip6_notify_failure() {
  local status="$1"
  trap - EXIT
  if (( status != 0 )) && [[ "${DRY_RUN:-0}" != 1 && -n "${PBS_JOBID:-}" ]]; then
    if command -v mail >/dev/null 2>&1 && command -v timeout >/dev/null 2>&1; then
      if ! {
        printf 'StormSurge CMIP6 job failed.\n\n'
        printf 'Job ID: %s\n' "${PBS_JOBID}"
        printf 'Job name: %s\n' "${PBS_JOBNAME:-unknown}"
        printf 'Host: %s\n' "${HOSTNAME:-unknown}"
        printf 'Exit status: %s\n' "${status}"
        printf 'Config: %s\n' "${config:-not reached}"
        printf 'PBS log directory: %s\n' "${CMIP6_PBS_LOG_DIR}"
      } | timeout --kill-after=5s 30s mail -S sendwait \
          -s "[StormSurge FAILED] ${PBS_JOBNAME:-CMIP6} ${PBS_JOBID} exit=${status}" "${CMIP6_MAIL_TO}"; then
        printf '[WARN] Failure email could not be sent to %s; original exit status=%s.\n' "${CMIP6_MAIL_TO}" "${status}" >&2
      fi
    else
      printf '[WARN] Failure email requires mail and timeout on the execution node; original exit status=%s.\n' "${status}" >&2
    fi
  fi
  exit "${status}"
}
trap 'cmip6_notify_failure "$?"' EXIT
'''
    block += END + '\n'
    text = text.replace('#PBS -j oe\n', '#PBS -j oe\n#PBS -m a\n#PBS -M ' + MAIL_TO + '\n')
    text = text.replace('set -euo pipefail\n', 'set -euo pipefail\n\n' + block + '\n')
    return text.replace('exec bash train.sh "${config}"\n',
                        '# Keep this shell alive so its EXIT handler observes the training status.\n'
                        'bash train.sh "${config}"\n')
