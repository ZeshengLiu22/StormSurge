#!/usr/bin/env python3
"""Validate the 44 configs and dry-run commands; never launch training or queue jobs."""

import argparse
from collections import Counter
import csv
from dataclasses import fields
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shlex
import subprocess
import sys

sys.dont_write_bytecode = True
from generate_configs import (HERE, REPO, RESULTS, PYTHON, STATIONS, VARIANTS, POOLS, expected_files,
                              frozen_hashes, reference_hashes, reference_path)

sys.path.insert(0, str(REPO))
import torch
import torch_geometric
from torch_geometric.data import Batch, Data
from emulator.common.runtime import configure_runtime
from emulator.data import build_loader, load_station_json, station_features_from_json
from emulator.models import ModelConfig, build_model
from emulator.training.arguments import parse_args


STAMP = '20000101_000000'  # Dry-run placeholder only; no run directory is created.
BASE_ENV = dict(PATH=os.environ.get('PATH', '/usr/bin:/bin'), PYTHONDONTWRITEBYTECODE='1', LANG='C.UTF-8')
IDENTITY = {'SESSION_NAME', 'PACT_RUN_NAME', 'PYTHON_RUN_TAG_BASE', 'ALL_RESULTS_ROOT'}
FACTORS = {'EXCEEDANCE_HEAD_EXPERIMENT', 'EXCEEDANCE_GATE_POOLING'}
LIMITATIONS = [
    'The 0916 reference used DETERMINISTIC=0; all 0919 configs explicitly use 1 as requested.',
    'Canonical Dual weights are body=1, excess=2, gate=0.5, not three unit weights.',
    'Under mse the frozen launcher omits tail/slope flags: parser defaults 0.1/0.01 are inactive. '
    'Single branch flags are likewise omitted; dual_loss=0 disables their inactive parser defaults.',
    'Frozen train.py creates the model before a shuffled DataLoader with no dedicated generator. '
    'Different head construction consumes different global RNG draws. Same-config repeatability is checked; '
    'identical cross-variant minibatch order is NOT guaranteed. Enforcing it needs a separately authorized '
    'trainer change, which is outside this configuration-only task.',
]


def run(command, **kwargs):
    result = subprocess.run(command, cwd=REPO, env=kwargs.pop('env', BASE_ENV), text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=60, **kwargs)
    if result.returncode:
        raise RuntimeError(f'{shlex.join(map(str, command))}\n{result.stdout}')
    return result.stdout


def source_values(path):
    names = []
    for line in path.read_text().splitlines():
        if not line.strip() or line.startswith('#'):
            continue
        match = re.fullmatch(r'(?:export )?([A-Za-z_]\w*)=(.*)', line)
        assert match and match[1] not in names, (path, line)
        assert '$(' not in line and '`' not in line and ';' not in line, (path, line)
        names.append(match[1])
    script = '''set -euo pipefail
source "$1"
shift
for setting in "$@"; do
    declare -n setting_value="$setting"
    values=("${setting_value[@]}")
    printf '%s\\0' "$setting" "${#values[@]}" "${values[@]}"
    unset -n setting_value
done
'''
    words = iter(run(['bash', '--noprofile', '--norc', '-c', script, 'bash', str(path), *names]).rstrip('\0').split('\0'))
    values = {}
    for name in words:
        assert int(next(words)) == 1, f'{path}: arrays must select exactly one run'
        values[name] = next(words)
    return values


def dry_run(path):
    output = run(['bash', 'train.sh', str(path.relative_to(REPO))],
                 env=dict(BASE_ENV, DRY_RUN='1', PACT_RUNSTAMP=STAMP))
    commands = [shlex.split(line[5:]) for line in output.splitlines() if line.startswith('CMD: ')]
    assert len(commands) == 1 and commands[0][0] == 'train.py', path
    launcher = next(shlex.split(line.split(':', 1)[1]) for line in output.splitlines() if line.startswith('LAUNCHER:'))
    assert launcher == [PYTHON, '-u'], 'Expected one Python process, not torchrun/DDP'
    assert re.search(r'^num_gpus:\s+1$', output, re.MULTILINE), path
    assert re.search(r'^CUDA_VISIBLE_DEVICES:\s+0$', output, re.MULTILINE), path
    return output, commands[0], parse_args(commands[0][1:]), shlex.join(launcher + commands[0])


def tree_snapshot(root):
    return {str(path.relative_to(root)): (path.stat().st_size, path.stat().st_mtime_ns)
            for path in root.rglob('*') if path.is_file()} if root.exists() else {}


def smoke_model(args):
    values = {field.name: getattr(args, field.name) for field in fields(ModelConfig) if hasattr(args, field.name)}
    values.update(model='pact', in_channels=5, out_channels=6, station_feat_dim=6,
                  temporal_layers=args.transformer_layers, temporal_ff_mult=args.transformer_ff_mult,
                  temporal_dropout=args.transformer_dropout, history_steps=args.history_hours // 6,
                  peak_threshold_norm=[1.] * 6, peak_prior=.17)
    return build_model(ModelConfig(**values))


def reproducibility_check(representative, dual_args):
    configure_runtime(representative.seed, representative.torch_threads, bool(representative.deterministic),
                      bool(representative.tf32))
    settings = dict(seed=torch.initial_seed(), deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
        warn_only=torch.is_deterministic_algorithms_warn_only_enabled(),
        cublas_workspace_config=os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
        cudnn_deterministic=torch.backends.cudnn.deterministic, cudnn_benchmark=torch.backends.cudnn.benchmark,
        matmul_tf32=torch.backends.cuda.matmul.allow_tf32, cudnn_tf32=torch.backends.cudnn.allow_tf32)
    assert settings == dict(seed=42, deterministic_algorithms=True, warn_only=False,
        cublas_workspace_config=':4096:8', cudnn_deterministic=True, cudnn_benchmark=False,
        matmul_tf32=True, cudnn_tf32=True), settings
    cuda = torch.cuda.is_available()
    device = torch.device('cuda:0' if cuda else 'cpu')
    records = []
    for repeat in range(2):
        configure_runtime(42, representative.torch_threads, True, bool(representative.tf32))
        model = smoke_model(representative).to(device).train()
        generator = torch.Generator().manual_seed(17)
        batch = Batch.from_data_list([Data(x=torch.randn(6, 5, generator=generator),
            x_hist=torch.randn(6, 5, 5, generator=generator),
            edge_index=torch.tensor([[0, 1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 0]])) for _ in range(2)]).to(device)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=cuda and bool(representative.amp)):
            output = model(batch, torch.zeros(6, device=device))
        assert all(torch.isfinite(value).all() for value in output if value is not None)
        output.prediction.square().mean().backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        records.append((output.prediction.detach().cpu(), {k: p.grad.detach().cpu().clone() for k, p in model.named_parameters()}))
    torch.testing.assert_close(records[0][0], records[1][0], rtol=0, atol=0)
    for key in records[0][1]:
        torch.testing.assert_close(records[0][1][key], records[1][1][key], rtol=0, atol=0)
    order_hashes = {}
    for args in dual_args:
        hashes = []
        for repeat in range(2):
            configure_runtime(42, 1, True, True)
            smoke_model(args)  # Same CPU construction order as the frozen trainer.
            loader = build_loader(list(range(64)), None, batch_size=16, num_workers=0,
                pin_memory=False, persistent_workers=False, prefetch_factor=0, mp_context='fork', shuffle=True)
            order = [int(index) for batch in loader for index in batch]
            hashes.append(hashlib.sha256(json.dumps(order).encode()).hexdigest())
        assert hashes[0] == hashes[1], 'Same-config sample order was not reproducible'
        order_hashes[f'{args.exceedance_head_experiment}/{args.exceedance_gate_pooling}'] = hashes[0]
    return dict(representative='0919_CBBT_C3_Learned', runtime=settings, smoke_device=str(device),
        cuda_available=cuda, gpu_name=torch.cuda.get_device_name(0) if cuda else None,
        gpu_amp_determinism_verified=cuda, exact_repeat_predictions_and_gradients=True,
        note='Two synthetic forward/backward passes, no optimizer or dataset loading. '
             'CPU FP32 is a plumbing probe only when CUDA is unavailable; formal BF16 GPU execution is then unverified.',
        data_order_probe=dict(method='Frozen model construction + existing shuffled loader on synthetic indices 0..63.',
            same_config_repeat_equal=True, cross_variant_equal=len(set(order_hashes.values())) == 1,
            order_sha256=order_hashes), optimizer_steps=0, real_training_jobs=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-only', action='store_true', help='Validate without rewriting evidence files.')
    options = parser.parse_args()
    assert Path(sys.executable).resolve() == Path(PYTHON).resolve(), 'Use the canonical Python interpreter.'
    provenance = json.loads((HERE / 'provenance.json').read_text())
    assert reference_hashes() == provenance['reference_tree_sha256'], '0916 reference files changed'
    assert frozen_hashes() == provenance['frozen_source_sha256'], 'Frozen source changed'
    roots = (RESULTS, REPO / 'All_Results', REPO / 'All_results_0916')
    before = {str(root): tree_snapshot(root) for root in roots}
    for relative, expected in expected_files().items():
        assert (HERE / relative).read_text() == expected, f'Generated artifact differs: {relative}'
    rows = list(csv.DictReader((HERE / 'manifest.csv').open()))
    assert len(rows) == len(list((HERE / 'configs').glob('*.sh'))) == 44
    assert len({row['run_name'] for row in rows}) == len({row['config_path'] for row in rows}) == 44
    assert Counter(row['station'] for row in rows) == Counter({station: 11 for station in STATIONS})
    assert Counter(row['exceedance_gate_pooling'] for row in rows) == Counter(NA=4, mean=20, learned=20)
    for station in STATIONS:
        station_rows = [row for row in rows if row['station'] == station]
        assert Counter(row['head_type'] for row in station_rows) == Counter(single=1, dual=10)
        assert {(row['exceedance_head_experiment'], row['exceedance_gate_pooling']) for row in station_rows} == {
            ('NA', 'NA'), *[(variant, pool) for variant in VARIANTS for pool in POOLS]}
    for path in [*HERE.glob('run_*.sh'), *(HERE / 'configs').glob('*.sh')]:
        run(['bash', '-n', str(path)])
    evidence = HERE / 'validation'
    if not options.check_only:
        evidence.mkdir(exist_ok=True)
    reference_values, reference_args = {}, {}
    for station in STATIONS:
        for variant in (None, 'legacy'):
            path = reference_path(station, variant)
            reference_values[station, variant is None] = source_values(path)
            reference_args[station, variant is None] = dry_run(path)[2]
    records, duals, commands = [], {}, []
    for row in rows:
        path = REPO / row['config_path']
        values = source_values(path)
        single = row['head_type'] == 'single'
        reference = reference_values[row['station'], single]
        assert set(values) == set(reference) | FACTORS
        assert {k: v for k, v in values.items() if k not in IDENTITY | FACTORS | {'DETERMINISTIC'}} == {
            k: v for k, v in reference.items() if k not in IDENTITY | {'DETERMINISTIC'}}
        assert values['SEED'] == '42' and values['DETERMINISTIC'] == '1'
        assert values['num_gpus'] == '1' and values['CUDA_VISIBLE_DEVICES'] == '0'
        assert values['TAIL_LAMBDA_LIST'] == values['SLOPE_LAMBDA_LIST'] == '0'
        assert values['ALL_RESULTS_ROOT'] == str(RESULTS)
        output, argv, args, command = dry_run(path)
        allowed = {'deterministic', 'run_tag', 'output_dir', 'exceedance_head_experiment', 'exceedance_gate_pooling'}
        old = vars(reference_args[row['station'], single])
        assert {key for key in old if old[key] != vars(args)[key]} <= allowed, row['run_name']
        assert (args.station, args.head_type, args.loss_mode, args.excess_formulation) == (row['station'], row['head_type'], 'mse', 'direct')
        assert (args.seed, args.deterministic, args.dual_loss) == (42, 1, 0 if single else 1)
        assert (args.excess_amp_loss_weight, args.shape_loss_weight, args.peak_loss_weight) == (0, 0, 0)
        assert (args.lr, args.history_hours, args.hidden_channels, args.epochs) == (.005, 24, 128, 300)
        assert args.amp and args.amp_dtype == 'bf16' and args.tf32
        assert (args.num_workers, args.pin_memory, args.persistent_workers, args.prefetch_factor) == (0, False, False, 0)
        assert args.dual_ablation == 'none' and args.checkpoint_selection == 'overall' and args.save_aux_checkpoints == 0
        assert Path(args.output_dir).resolve().is_relative_to(RESULTS.resolve())
        assert Path(args.output_dir).resolve() == Path(row['output_dir'].replace('<TIMESTAMP>', STAMP)).resolve()
        assert row['run_name'] in args.run_tag
        if single:
            assert args.exceedance_head_experiment is None and '--exceedance_head_experiment' not in argv
            assert values['BODY_LOSS_WEIGHT'] == values['EXCESS_LOSS_WEIGHT'] == values['GATE_LOSS_WEIGHT'] == '0'
        else:
            assert (args.exceedance_head_experiment, args.exceedance_gate_pooling) == (
                row['exceedance_head_experiment'], row['exceedance_gate_pooling'])
            assert (args.body_loss_weight, args.excess_loss_weight, args.gate_loss_weight) == (1., 2., .5)
            duals.setdefault(row['station'], []).append(args)
        commands.append(f'# {row["run_name"]}\n{command}\n')
        records.append(dict(manifest=row, shell_values=values, resolved_args=vars(args), resolved_command=command))
        if not options.check_only:
            (evidence / f'{row["run_name"]}.dry_run.txt').write_text(output.rstrip() + '\n')
    ignored = {'station', 'run_tag', 'output_dir', 'exceedance_head_experiment', 'exceedance_gate_pooling'}
    matched = [{k: v for k, v in vars(args).items() if k not in ignored} for group in duals.values() for args in group]
    assert all(value == matched[0] for value in matched), 'Non-factor difference between Dual commands'
    data_paths = {}
    for station in STATIONS:
        args = duals[station][0]
        files = sorted(Path(args.root_dir).glob(f'*_{station}_*graphs.pt'))
        assert files, f'Missing canonical graph paths for {station}'
        features = station_features_from_json(load_station_json(args.station_json_dir, station),
            use_site_elevation=bool(args.use_site_elevation), use_bathymetry=bool(args.use_bathymetry))
        assert features.numel() == 6 and args.use_station_meta == 1
        data_paths[station] = dict(graph_root=args.root_dir, station_json_dir=str(args.station_json_dir),
                                  graph_files=len(files), station_features=6)
    representative = next(args for args in duals['CBBT'] if args.exceedance_head_experiment == 'c3'
                          and args.exceedance_gate_pooling == 'learned')
    repro = reproducibility_check(representative, duals['CBBT'])
    assert frozen_hashes() == provenance['frozen_source_sha256'] and reference_hashes() == provenance['reference_tree_sha256']
    assert before == {str(root): tree_snapshot(root) for root in roots}, 'Validation changed training result files'
    report = dict(configuration_status='PASS', strict_cross_variant_data_order='NOT_GUARANTEED_BY_FROZEN_TRAINER',
        validated_utc=datetime.now(timezone.utc).isoformat(), configs=44, per_station={s: 11 for s in STATIONS},
        single=4, dual_mean=20, dual_learned=20, syntax_passed=44, dry_runs_passed=44, parser_passed=44,
        matched_dual_non_factor_settings=True, reference_and_frozen_source_unchanged=True,
        result_trees_unchanged=True, real_training_jobs=0, queue_submissions=0,
        environment=dict(python=sys.version, executable=sys.executable, torch=torch.__version__,
                         torch_geometric=torch_geometric.__version__, cuda_build=torch.version.cuda,
                         platform=platform.platform()),
        limitations=LIMITATIONS, data_paths=data_paths, reproducibility=repro, records=records)
    if not options.check_only:
        (HERE / 'dry_run_commands.txt').write_text('\n'.join(commands))
        (HERE / 'validation_report.json').write_text(json.dumps(report, indent=2, default=str) + '\n')
        (evidence / 'reproducibility.json').write_text(json.dumps(repro, indent=2) + '\n')
        (HERE / 'validation_report.md').write_text(
            '# 0919 configuration validation: PASS\n\n'
            '44 configs and resolved commands passed: 11 per station; 4 Single, 20 Dual mean, 20 Dual learned.\n'
            'Every config explicitly uses seed 42 and deterministic execution. All non-factor Dual settings match.\n'
            'Source/reference hashes and training result trees are unchanged. No training jobs or queue submissions.\n\n'
            f'C3 + Learned synthetic repeatability: exact predictions and gradients on **{repro["smoke_device"]}**.\n'
            f'GPU BF16 determinism verified: **{repro["gpu_amp_determinism_verified"]}**.\n\n'
            '## Protocol notes and limits\n\n' + ''.join(f'- {note}\n' for note in LIMITATIONS) + '\n'
            'The sampler probe records the limitation in [reproducibility.json](validation/reproducibility.json).\n'
            'This suite preserves the trainer; it does not claim identical cross-variant minibatch order.\n\n'
            'See [manifest](manifest.csv), [resolved commands](dry_run_commands.txt), and '
            '[full report](validation_report.json). Individual launcher transcripts are in `validation/`.\n')
    print('PASS: 44 configs / 44 dry runs / 44 parsed commands; 4 Single + 20 mean + 20 learned. No training launched.')
    print('LIMIT: identical cross-variant minibatch order is not guaranteed by the frozen trainer.')
    if not repro['gpu_amp_determinism_verified']:
        print('LIMIT: CUDA unavailable; GPU BF16 determinism remains unverified. CPU plumbing smoke passed.')


if __name__ == '__main__':
    main()
