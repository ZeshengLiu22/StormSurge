#!/usr/bin/env python3
"""Publish existing EC_EARTH/MRI Lewes outputs without running inference or reading inputs."""
import argparse
import json
import os
from pathlib import Path
import re
import shlex
import time

from common import (LAUNCHER, MODELS, SEEDS, atomic_json, digest_json, identity,
                    inference_command, load_manifest, require, run_lock,
                    validate_outputs, verified_completion)


RECOVERY_IDS = {identity(source, 'Lewes', seed, model)
                for source in ('CMIP6_EC_EARTH', 'CMIP6_MRI')
                for seed in SEEDS for model in MODELS}


def recovery_rows(manifest):
    rows = [row for row in manifest['runs'] if row['id'] in RECOVERY_IDS]
    require(len(rows) == 24 and {row['id'] for row in rows} == RECOVERY_IDS,
            'Manifest must contain all 24 recovery targets exactly once')
    return rows


def recover_run(row, manifest):
    """Validate a recorded attempt, then publish under its original lock and fingerprint."""
    require(row['id'] in RECOVERY_IDS, 'Run is outside the fixed 24-run recovery scope')
    require(row['id'] == identity(row['source'], row['station'], row['seed'], row['model']),
            'Manifest identity mismatch')
    directory = Path(row['output_dir'])
    require(directory.resolve() == (Path(manifest['result_root']) / row['id']).resolve(),
            'Output directory does not match manifest identity')
    require(directory.is_dir(), f'Missing run directory: {directory}')
    with run_lock(directory):
        status = json.loads((directory / 'status.json').read_text())
        attempt = status['attempt']
        require(isinstance(attempt, str) and re.fullmatch(r'[0-9a-f]{32}', attempt), 'Invalid attempt ID')
        staging = directory / f'.attempt-{attempt}'
        require(status['id'] == row['id'] and Path(status['staging']).resolve() == staging.resolve(),
                'Status identifies another run or attempt')
        log = directory / 'logs' / f'{attempt}.log'
        require(Path(status['log']).resolve() == log.resolve() and log.is_file(), 'Missing/mismatched inference log')
        final = directory / 'outputs'
        marker = directory / 'COMPLETE.json'
        receipt = status.get('recovery')
        staged_provenance = staging / 'provenance.json'
        final_provenance = directory / 'provenance.json'
        require(not (staged_provenance.exists() and final_provenance.exists()), 'Ambiguous provenance files')
        provenance_path = staged_provenance if staged_provenance.is_file() else final_provenance
        provenance = json.loads(provenance_path.read_text())
        require(provenance['run'] == row, 'Staging provenance differs from manifest run')
        inventory = manifest['graph_inventory'][row['source'] + '/' + row['station']]
        require(provenance['graph_inventory'] == inventory, 'Saved graph inventory differs from manifest')
        require(provenance['attempt'] == attempt and provenance['pbs_jobid'] == status.get('pbs_jobid'),
                'Provenance identifies another attempt or job')
        fingerprint = digest_json(dict(row=row, graphs=inventory, runtime=provenance['runtime']))
        require(provenance['fingerprint'] == fingerprint, 'Provenance fingerprint mismatch')
        command = inference_command(row, staging / 'outputs', provenance['runtime'],
                                    manifest['python'], manifest['repo'])
        require(provenance['command'] == command, 'Saved inference command mismatch')
        with log.open() as stream:
            require(stream.readline().strip() == shlex.join(command), 'Inference log belongs to another command')
        if marker.exists():
            require(verified_completion(directory, row, fingerprint), 'Existing completion is invalid; refusing overwrite')
            return 'skipped'

        if receipt is None:
            require(status['state'] == 'failed' and status.get('error') == 'Nonfinite physical predictions',
                    'Attempt did not fail at the expected output validation step')
            require(not final.exists() and not final_provenance.exists(), 'Untracked published files; refusing overwrite')
            require(staging / 'outputs' in directory.glob('.attempt-*/outputs'), 'Recorded staging outputs not found')
        else:
            require(status['state'] in ('recovering', 'success') and receipt['attempt'] == attempt and
                    receipt['fingerprint'] == fingerprint, 'Recovery receipt mismatch')
        staged_outputs = staging / 'outputs'
        require(staged_outputs.is_dir() != final.is_dir(), 'Expected exactly one staged or published output directory')
        outputs = staged_outputs if staged_outputs.is_dir() else final
        files = validate_outputs(outputs, row)
        if receipt is None:
            # Persist checksums before any rename so an interrupted publication can resume safely.
            receipt = dict(attempt=attempt, fingerprint=fingerprint, files=files,
                           previous_state=status['state'], previous_error=status['error'], started_at=time.time())
            status.update(state='recovering', recovery=receipt)
            atomic_json(directory / 'status.json', status)
        else:
            require(receipt['files'] == files, 'Outputs changed since recovery started')
        if staged_outputs.is_dir():
            require(not final.exists(), 'Refusing to overwrite existing outputs')
            os.rename(staged_outputs, final)
        if staged_provenance.is_file():
            require(not final_provenance.exists(), 'Refusing to overwrite existing provenance')
            os.rename(staged_provenance, final_provenance)
        if staging.is_dir() and not any(staging.iterdir()):
            staging.rmdir()
        status.pop('error', None)
        status.update(state='success', exit_code=0, recovered_at=time.time())
        atomic_json(directory / 'status.json', status)
        atomic_json(marker, dict(fingerprint=fingerprint, files=files, attempt=attempt))
        require(verified_completion(directory, row, fingerprint), 'Published result failed completion verification')
        return 'recovered'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=LAUNCHER / 'manifest.json')
    args = parser.parse_args(argv)
    manifest = load_manifest(args.manifest)
    counts = dict(recovered=0, skipped=0, failed=0)
    for row in recovery_rows(manifest):
        try:
            outcome = recover_run(row, manifest)
            counts[outcome] += 1
            print(f'[{outcome.upper()}] {row["id"]}', flush=True)
        except Exception as error:
            counts['failed'] += 1
            print(f'[FAILED] {row["id"]}: {error}', flush=True)
    print(json.dumps(counts), flush=True)
    return int(counts['failed'] > 0)


if __name__ == '__main__':
    raise SystemExit(main())
