#!/usr/bin/env python3
"""Exercise a PBS launcher with local stubs; never train, submit, or send mail."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import tempfile

from pbs_failure_mail import MAIL_TO

REPO = Path('/glade/u/home/zel/scratch/StormSurge')


def executable(path, text):
    path.write_text('#!/bin/bash\n' + text)
    path.chmod(0o700)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pbs', type=Path, default=REPO/'PBS-launcher/cmip6/CMIP6_AWI/Battery/seed42/run_PACT_MSE.pbs')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    original = args.pbs.read_text()
    results = []
    cases = [
        ('success', 0, 0, '0', True, 'ok', False, 0, False),
        ('training_failure', 17, 0, '0', True, 'ok', False, 17, True),
        ('preflight_failure', 0, 23, '0', True, 'ok', False, 23, True),
        ('early_cd_failure', 0, 0, '0', True, 'ok', True, 1, True),
        ('cpu_preview_failure', 17, 0, '1', True, 'ok', False, 17, False),
        ('outside_pbs', 0, 0, '0', False, 'ok', False, 2, False),
        ('mail_transport_failure', 17, 0, '0', True, 'fail', False, 17, True),
        ('missing_mail', 17, 0, '0', True, 'missing', False, 17, False),
        ('invalid_dry_run', 0, 0, 'invalid', True, 'ok', False, 2, True),
    ]
    with tempfile.TemporaryDirectory(prefix='cmip6-mail-test-') as scratch:
        for name, train_status, preflight_status, dry_run, allocated, mail_mode, early, expected, expect_mail in cases:
            root = Path(scratch)/name
            root.mkdir()
            bin_dir = root/'bin'
            bin_dir.mkdir()
            # Deliberately exclude system mail from PATH in every test.
            for command in ['bash', 'mkdir', 'timeout']:
                (bin_dir/command).symlink_to('/usr/bin/'+command)
            if mail_mode != 'missing':
                executable(bin_dir/'mail', 'printf "%s\\n" "$@" > "${MAIL_TEST_ARGS}"\n'
                           '/bin/cat > "${MAIL_TEST_BODY}"\nexit "${MAIL_TEST_SEND_STATUS}"\n')
            executable(root/'python', 'exit "${MAIL_TEST_PREFLIGHT_STATUS}"\n')
            executable(root/'train.sh', 'printf "called\\n" > "${MAIL_TEST_TRAIN_CALLS}"\nexit "${MAIL_TEST_TRAIN_STATUS}"\n')
            (root/'train.py').write_text('# Placeholder; never executed.\n')
            (root/'bash_env').write_text('module() { return 0; }\n')
            text = original.replace('cd -- '+str(REPO)+'\n', 'cd -- '+str(root/'missing' if early else root)+'\n')
            text = text.replace('export PYTHON_BIN="/glade/work/zel/conda-envs/derecho-ml/bin/python"',
                                'export PYTHON_BIN="'+str(root/'python')+'"')
            text = text.replace('/glade/derecho/scratch/zel/tmp/stormsurge-cmip6-', str(root/'runtime-'))
            launcher = root/'launcher.sh'
            launcher.write_text(text)
            env = dict(os.environ, PATH=str(bin_dir), BASH_ENV=str(root/'bash_env'), DRY_RUN=dry_run,
                       MAIL_TEST_ARGS=str(root/'mail_args'), MAIL_TEST_BODY=str(root/'mail_body'),
                       MAIL_TEST_SEND_STATUS='75' if mail_mode == 'fail' else '0',
                       MAIL_TEST_TRAIN_CALLS=str(root/'train_calls'), MAIL_TEST_TRAIN_STATUS=str(train_status),
                       MAIL_TEST_PREFLIGHT_STATUS=str(preflight_status), PBS_JOBNAME='cmip6-mail-test')
            env.pop('PBS_JOBID', None)
            if allocated:
                env['PBS_JOBID'] = 'local-test.not-submitted'
            proc = subprocess.run(['/bin/bash', str(launcher)], env=env, capture_output=True, text=True, timeout=10)
            assert proc.returncode == expected, (name, proc.returncode, proc.stderr)
            assert (root/'mail_body').exists() == expect_mail, (name, proc.stderr)
            if expect_mail:
                sent_args = (root/'mail_args').read_text().splitlines()
                body = (root/'mail_body').read_text()
                assert sent_args[:3] == ['-S', 'sendwait', '-s'], sent_args
                assert sent_args[-1] == MAIL_TO and len(sent_args) == 5, sent_args
                assert f'exit={expected}' in sent_args[3]
                assert 'Job ID: local-test.not-submitted' in body
                assert f'Exit status: {expected}\n' in body
                assert 'Config: ' in body and 'PBS log directory: ' in body
            if mail_mode in ['fail', 'missing']:
                assert '[WARN] Failure email' in proc.stderr
            if preflight_status or early or not allocated or dry_run == 'invalid':
                assert not (root/'train_calls').exists()
            results.append(dict(case=name, status='PASS', exit_status=proc.returncode, mock_mail_called=expect_mail))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(dict(status='PASS', recipient=MAIL_TO, actual_emails_sent=0,
                                       actual_training_launches=0, pbs_submissions=0, cases=results), indent=2)+'\n')
    print(f'PASS: {len(results)} stubbed PBS scenarios; no emails, submissions, or training.')


if __name__ == '__main__':
    main()
