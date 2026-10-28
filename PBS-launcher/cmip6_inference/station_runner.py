#!/usr/bin/env python3
"""One already-isolated GPU; twelve independent, sequential infer.py invocations."""
import argparse
import json
import os
from pathlib import Path
import shlex
import signal
import socket
import subprocess
import sys
import time
import traceback
import uuid

from common import (LAUNCHER, SOURCES, STATIONS, YEARS, atomic_json, digest_json,
                    graph_inventory, inference_command, inspect_checkpoint, load_manifest,
                    memory_estimate, require, run_lock, runtime_from_env, sha256,
                    validate_outputs, verified_completion)


class StopRequested(Exception):
    pass


class ProcessControl:
    def __init__(self):
        self.signum = None
        self.child = None

    def handle(self, signum, frame):
        self.signum = signum
        if self.child is not None and self.child.poll() is None:
            try:
                os.killpg(self.child.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass

    def check(self):
        if self.signum:
            raise StopRequested(f'Received signal {self.signum}')

    def run(self, command, log, lock_fd, cwd):
        self.check()
        self.child = subprocess.Popen(command, cwd=cwd, stdout=log, stderr=subprocess.STDOUT,
                                      start_new_session=True, pass_fds=(lock_fd,))
        try:
            while True:
                if self.signum:
                    try:
                        os.killpg(self.child.pid, signal.SIGTERM)
                        self.child.wait(timeout=20)
                    except subprocess.TimeoutExpired:
                        os.killpg(self.child.pid, signal.SIGKILL)
                        self.child.wait()
                    except ProcessLookupError:
                        pass
                    self.check()
                try:
                    code = self.child.wait(timeout=0.25)
                    self.check()
                    return code
                except subprocess.TimeoutExpired:
                    pass
        finally:
            if self.child.poll() is None:
                os.killpg(self.child.pid, signal.SIGKILL)
                self.child.wait()
            self.child = None


def gpu_check(cluster, runtime):
    import torch
    import torch_geometric
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1,
            'Each station process must see exactly one usable CUDA device')
    device = torch.cuda.get_device_properties(0)
    if cluster == 'casper':
        require('A100' in device.name and device.total_memory >= 75 * 2**30,
                'Casper job must be assigned an A100 80GB GPU')
    if runtime['amp'] and runtime['amp_dtype'] == 'bf16':
        require(torch.cuda.is_bf16_supported(), 'Assigned GPU/runtime does not support BF16')
    # Force a tiny kernel to detect driver/runtime mismatch before loading 30 years.
    (torch.ones(8, device='cuda:0') + 1).sum().item()
    torch.cuda.synchronize()
    return dict(python=sys.executable, torch=torch.__version__, cuda_runtime=torch.version.cuda,
                pyg=torch_geometric.__version__, gpu=device.name, gpu_memory=device.total_memory,
                visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'), local_device='cuda:0', host=socket.gethostname())


def execute_run(row, manifest, runtime, control, environment):
    """Lock, validate, stage outputs, rename, and publish the completion marker last."""
    directory = Path(row['output_dir'])
    with run_lock(directory) as lock_fd:
        control.check()
        try:
            inventory = graph_inventory(row['test_root_dir'], row['station'])
            require(inventory == manifest['graph_inventory'][row['source'] + '/' + row['station']],
                    f'{row["id"]}: graph inventory changed; regenerate and review manifest')
            metadata = inspect_checkpoint(row['checkpoint'], row['source'], row['station'], row['seed'], row['model'])
            require(metadata == row['checkpoint_metadata'], f'{row["id"]}: checkpoint changed; regenerate manifest')
        except Exception as error:
            atomic_json(directory / 'status.json', dict(state='failed_validation', id=row['id'],
                        checkpoint=row['checkpoint'], error=str(error), finished_at=time.time(),
                        pbs_jobid=os.environ.get('PBS_JOBID')))
            raise
        fingerprint = digest_json(dict(row=row, graphs=inventory, runtime=runtime))
        if verified_completion(directory, row, fingerprint):
            print(f'[SKIP verified] {row["id"]}', flush=True)
            return 'skipped'
        control.check()
        attempt = uuid.uuid4().hex
        staging = directory / f'.attempt-{attempt}'
        output = staging / 'outputs'
        output.mkdir(parents=True)
        logs = directory / 'logs'
        logs.mkdir(exist_ok=True)
        command = inference_command(row, output, runtime, sys.executable, manifest['repo'])
        provenance = dict(run=row, graph_inventory=inventory, runtime=runtime, command=command,
                          fingerprint=fingerprint, environment=environment, pbs_jobid=os.environ.get('PBS_JOBID'),
                          attempt=attempt)
        atomic_json(staging / 'provenance.json', provenance)
        status = dict(state='running', id=row['id'], attempt=attempt, started_at=time.time(),
                      log=str(logs / f'{attempt}.log'), staging=str(staging), pbs_jobid=os.environ.get('PBS_JOBID'))
        atomic_json(directory / 'status.json', status)
        (directory / 'COMPLETE.json').unlink(missing_ok=True)
        try:
            print(f'[RUN] {row["id"]} -> {directory}', flush=True)
            with (logs / f'{attempt}.log').open('w') as log:
                log.write(shlex.join(command) + '\n')
                log.flush()
                code = control.run(command, log, lock_fd, manifest['repo'])
            require(code == 0, f'infer.py exited {code}; see {status["log"]}')
            control.check()
            files = validate_outputs(output, row)
            # Do not bless outputs if the source checkpoint/data changed during inference.
            require(sha256(row['checkpoint']) == metadata['sha256'], 'Checkpoint changed during inference')
            require(graph_inventory(row['test_root_dir'], row['station']) == inventory, 'Graphs changed during inference')
            control.check()
            final = directory / 'outputs'
            if final.exists():
                archive = directory / 'previous_outputs'
                archive.mkdir(exist_ok=True)
                os.replace(final, archive / attempt)
            os.replace(output, final)
            os.replace(staging / 'provenance.json', directory / 'provenance.json')
            staging.rmdir()
            status.update(state='success', finished_at=time.time(), exit_code=0)
            atomic_json(directory / 'status.json', status)
            control.check()
            atomic_json(directory / 'COMPLETE.json', dict(fingerprint=fingerprint, files=files, attempt=attempt))
            print(f'[SUCCESS] {row["id"]}', flush=True)
            return 'success'
        except BaseException as error:
            status.update(state='interrupted' if isinstance(error, StopRequested) else 'failed',
                          error=str(error), finished_at=time.time())
            atomic_json(directory / 'status.json', status)
            raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', choices=SOURCES)
    parser.add_argument('station', choices=STATIONS)
    parser.add_argument('--cluster', choices=('derecho', 'casper'), required=True)
    parser.add_argument('--manifest', type=Path, default=LAUNCHER / 'manifest.json')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)
    manifest = load_manifest(args.manifest)
    runtime = runtime_from_env()
    rows = [r for r in manifest['runs'] if (r['source'], r['station']) == (args.source, args.station)]
    require(len(rows) == 12, 'Expected twelve station combinations')
    budget = 487 / 4 if args.cluster == 'derecho' else 240
    estimate = memory_estimate(manifest['graph_inventory'][args.source + '/' + args.station], runtime['num_workers'])
    require(estimate < budget * 0.8, f'Estimated RAM {estimate:.1f} GiB exceeds station safety budget; reduce INFER_NUM_WORKERS')
    if args.dry_run:
        for row in rows:
            print(shlex.join(inference_command(row, Path(row['output_dir']) / 'outputs', runtime, sys.executable, manifest['repo'])))
        return 0
    require(os.environ.get('PBS_JOBID'), 'Real inference requires a PBS allocation; use --dry-run')
    control = ProcessControl()
    for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(signum, control.handle)
    environment = gpu_check(args.cluster, runtime)
    print(json.dumps(environment, sort_keys=True), flush=True)
    failed = []
    for row in rows:
        try:
            execute_run(row, manifest, runtime, control, environment)
        except StopRequested:
            return 128 + (control.signum or signal.SIGTERM)
        except Exception as error:
            failed.append(row['id'])
            print(f'[FAILED] {row["id"]}: {error}', file=sys.stderr, flush=True)
            traceback.print_exc()
    print(json.dumps(dict(source=args.source, station=args.station, failed=failed)), flush=True)
    return int(bool(failed))


if __name__ == '__main__':
    try:
        sys.exit(main())
    except Exception as error:
        print(f'[FATAL] {error}', file=sys.stderr)
        sys.exit(2)
