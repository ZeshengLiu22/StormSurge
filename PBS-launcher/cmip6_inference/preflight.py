#!/usr/bin/env python3
"""CPU-only, read-only validation; no GPU probes, job submissions, or inference."""
import argparse
import importlib
import json
from pathlib import Path
import subprocess
import sys

from common import (LAUNCHER, MODELS, SOURCES, STATIONS, YEARS, atomic_json, digest_json,
                    graph_inventory, inspect_checkpoint, load_manifest, memory_estimate,
                    require, resolve_checkpoint, runtime_from_env, verified_completion)
from generate import config_text, pbs_text


def validate(manifest_path, *, metadata=True, completed=False):
    manifest_path = Path(manifest_path).absolute()
    manifest = load_manifest(manifest_path)
    require(manifest['sources'] == list(SOURCES) and manifest['stations'] == list(STATIONS), 'Wrong sources/stations')
    require(len(SOURCES) == 9 and len(STATIONS) == 4 and len(YEARS) == 30, 'Invalid experiment constants')
    repo = Path(manifest['repo'])
    launcher = repo / 'PBS-launcher/cmip6_inference'
    configs = sorted((repo / 'configs/cmip6_inference').glob('*/*/seed*/infer_config_*.sh'))
    derecho = sorted((launcher / 'derecho').glob('*/run_inference.pbs'))
    casper = sorted((launcher / 'casper').glob('*/*/run_inference.pbs'))
    require(len(configs) == 432 and len(derecho) == 9 and len(casper) == 36,
            f'File count mismatch: configs={len(configs)}, Derecho={len(derecho)}, Casper={len(casper)}')
    require(set(configs) == {Path(r['config']) for r in manifest['runs']}, 'Config files do not match manifest')
    require(set(derecho) == {Path(r['derecho_pbs']) for r in manifest['runs']}, 'Derecho assignments mismatch')
    require(set(casper) == {Path(r['casper_pbs']) for r in manifest['runs']}, 'Casper assignments mismatch')
    runtime = runtime_from_env()
    memory = {}
    for source in SOURCES:
        for station in STATIONS:
            key = source + '/' + station
            inventory = graph_inventory(Path(manifest['data_root']) / source / 'graphs', station)
            require(inventory == manifest['graph_inventory'][key], f'{key}: graph files changed since generation')
            memory[key] = memory_estimate(inventory, runtime['num_workers'])
            require(memory[key] < 240 * 0.8, f'{key}: Casper RAM budget exceeded')
            require(memory[key] < 487 / 4 * 0.8, f'{key}: Derecho station RAM budget exceeded')
    pinned = {row['id']: row['checkpoint'] for row in manifest['runs']}
    counts = dict(verified_completed=0, pending=0)
    for index, row in enumerate(manifest['runs'], 1):
        source, station, seed, model = (row[key] for key in ('source', 'station', 'seed', 'model'))
        key = row['id']
        require(row['output_dir'] == str(Path(manifest['result_root']) / key), f'{key}: incorrect output path')
        require(row['test_root_dir'] == str(Path(manifest['data_root']) / source / 'graphs'), f'{key}: incorrect target path')
        require(row['source_root'] == row['checkpoint_metadata']['source_root'], f'{key}: original source root changed')
        require(row['config'] == str(repo / 'configs/cmip6_inference' / source / station / f'seed{seed}' / f'infer_config_{model}.sh'), f'{key}: incorrect config path')
        generated_manifest = repo / 'PBS-launcher/cmip6_inference/manifest.json'
        require(Path(row['config']).read_text() == config_text(row, generated_manifest), f'{key}: generated config modified; regenerate')
        checkpoint = resolve_checkpoint(manifest['checkpoint_root'], source, station, seed, model, pinned)
        require(str(checkpoint) == row['checkpoint'], f'{key}: checkpoint not pinned absolutely')
        if metadata:
            actual = inspect_checkpoint(checkpoint, source, station, seed, model)
            require(actual == row['checkpoint_metadata'], f'{key}: checkpoint metadata/hash changed')
        if completed:
            inventory = manifest['graph_inventory'][source + '/' + station]
            fingerprint = digest_json(dict(row=row, graphs=inventory, runtime=runtime))
            if (Path(row['output_dir']) / 'COMPLETE.json').exists():
                require(verified_completion(row['output_dir'], row, fingerprint), f'{key}: invalid/stale completion marker or outputs')
                counts['verified_completed'] += 1
            else:
                counts['pending'] += 1
        if index % 48 == 0:
            print(f'Validated {index}/432 combinations', file=sys.stderr, flush=True)
    for cluster, scripts in (('derecho', derecho), ('casper', casper)):
        for path in scripts:
            source = path.parent.name if cluster == 'derecho' else path.parent.parent.name
            station = None if cluster == 'derecho' else path.parent.name
            generated_launcher = repo / 'PBS-launcher/cmip6_inference'
            require(path.read_text() == pbs_text(cluster, source, station, generated_launcher, manifest['walltimes'][cluster]),
                    f'{path}: resource, GPU, environment, or failure-propagation template changed')
    for path in [*configs, *derecho, *casper, *launcher.glob('*.sh')]:
        subprocess.run(['bash', '-n', str(path)], check=True, capture_output=True, text=True)
    # Import and compile components without writing bytecode or touching CUDA.
    for path in launcher.glob('*.py'):
        compile(path.read_text(), str(path), 'exec')
        importlib.import_module(path.stem)
    import infer
    from launch import derecho_devices
    require(derecho_devices('3,2,1,0') == ['3', '2', '1', '0'], 'GPU mask reordering regression')
    import torch
    import torch_geometric
    return dict(status='passed', sources=9, stations=4, runs=432, configs=432, derecho_pbs=9, casper_pbs=36,
                years=YEARS, graph_files=sum(len(v) for v in manifest['graph_inventory'].values()),
                checkpoint_metadata_validated=metadata, memory_estimate_gib=memory,
                max_station_gib=max(memory.values()),
                max_derecho_node_gib=max(sum(memory[s + '/' + st] for st in STATIONS) for s in SOURCES),
                runtime=runtime, python=sys.executable, torch=torch.__version__, cuda_runtime=torch.version.cuda,
                pyg=torch_geometric.__version__, completion=counts if completed else 'not_checked',
                gpu_runtime='not tested: requires allocation')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=LAUNCHER / 'manifest.json')
    parser.add_argument('--structural-only', action='store_true', help='Skip deserializing/hash-checking checkpoints; still checks input paths and completion evidence')
    parser.add_argument('--completed', action='store_true', help='Also verify all existing completion markers and their NPZ/JSON payloads')
    parser.add_argument('--report', type=Path, help='Explicitly save this validation report')
    args = parser.parse_args(argv)
    report = validate(args.manifest, metadata=not args.structural_only, completed=args.completed)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        atomic_json(args.report, report)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except Exception as error:
        print(f'[PREFLIGHT FAILED] {error}', file=sys.stderr)
        sys.exit(1)
