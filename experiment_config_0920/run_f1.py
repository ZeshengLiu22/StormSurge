#!/usr/bin/env python3
"""Evaluate an explicitly supplied F0 checkpoint with existing offline F1 diagnostics."""

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys

from generate_configs import REPO, RESULTS, STATIONS, config_path, run_name


def source_values(path, *, env=None):
    """Read standalone scalar/one-value-array shell configs without launching anything."""
    names = []
    for line in Path(path).read_text().splitlines():
        if not line.strip() or line.startswith('#'):
            continue
        match = re.fullmatch(r'(?:export )?([A-Za-z_]\w*)=(.*)', line)
        if not match or match[1] in names or any(token in line for token in ('$(', '`', ';')):
            raise ValueError(f'Expected standalone setting: {line}')
        names.append(match[1])
    script = '''set -euo pipefail
source "$1"
shift
for key in "$@"; do
    declare -n config_value="$key"
    values=("${config_value[@]}")
    printf '%s\\0' "$key" "${#values[@]}" "${values[@]}"
    unset -n config_value
done
'''
    result = subprocess.run(['bash', '--noprofile', '--norc', '-c', script, 'bash', str(Path(path).resolve()), *names],
                            cwd=REPO, env=env, check=True, capture_output=True, text=True)
    tokens = iter(result.stdout[:-1].split('\0'))
    values = {}
    for name in tokens:
        if int(next(tokens)) != 1:
            raise ValueError(f'{name} must select exactly one value')
        values[name] = next(tokens)
    return values


def validate_config(config):
    station = config['STATION']
    if station not in STATIONS or config['MODE'] != 'postprocess' or config['FORMULATION'] != 'F1_Hard':
        raise ValueError('F1 requires one of the four station postprocess configs')
    if float(config['HARD_GATE_THRESHOLD']) != .5:
        raise ValueError('F1 has a fixed hard-gate threshold of 0.5')
    if (config['SOURCE_RUN_NAME'] != run_name(station, 'F0_Soft')
            or config['SOURCE_CONFIG'] != config_path(station, 'F0_Soft')):
        raise ValueError('F1 must reference the corresponding station F0 config')
    if (config['ALL_RESULTS_ROOT'] != str(RESULTS) or config['PACT_RUN_NAME'] != run_name(station, 'F1_Hard')
            or config['EVALUATION_SCOPE'] != 'test' or config['DUAL_DIAGNOSTICS'] != '1'):
        raise ValueError('F1 must use the 0920 results root and saved F0 test split with dual diagnostics')


def validate_checkpoint(config, path):
    import torch

    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    model = checkpoint['model_config']
    training = checkpoint['training_config']
    expected_model = dict(model='pact', head_type='dual', encoder_type='GraphSAGE', temporal_block='Transformer',
                          hidden_channels=128, history_steps=4)
    if checkpoint['station'] != config['STATION'] or any(model.get(k) != v for k, v in expected_model.items()):
        raise ValueError('Checkpoint station/architecture does not match the corresponding F0')
    if (model.get('excess_formulation', 'direct') != 'direct'
            or model.get('direct_dual_reconstruction', 'soft_gate') != 'soft_gate'
            or model.get('exceedance_head_experiment') is not None
            or model.get('exceedance_gate_pooling', 'mean') != 'mean'
            or model.get('dual_ablation', 'none') != 'none'):
        raise ValueError('F1 requires production soft-gated direct F0, without capacity experiments or ablations')
    expected_training = dict(loss_mode='mse', dual_loss=1, body_loss_weight=1., excess_loss_weight=2.,
                             gate_loss_weight=.5, checkpoint_selection='overall', seed=42, deterministic=0,
                             lr=.005, epochs=300, batch_size=256, grad_accum_steps=4,
                             excess_amp_loss_weight=0., shape_loss_weight=0., peak_loss_weight=0.)
    if (training.get('excess_supervision_scope', 'event') != 'event'
            or any(training.get(k) != v for k, v in expected_training.items())):
        raise ValueError('Checkpoint does not use the canonical F0 training protocol')
    if not training.get('run_tag', '').startswith(config['SOURCE_RUN_NAME'] + '_'):
        raise ValueError('Checkpoint run_tag must identify the corresponding 0920 F0 run')
    if not checkpoint.get('dual_metadata'):
        raise ValueError('F0 checkpoint must retain its TRAIN event metadata')
    return checkpoint


def inference_command(config, checkpoint, output):
    command = [config['PYTHON_BIN'], '-B', config['INFER_PY'], '--ckpt', str(checkpoint),
               '--root_dir', config['ROOT_DIR'], '--station', config['STATION'],
               '--station_json_dir', config['STATION_JSON_DIR'], '--out_dir', str(output),
               '--model_label', config['PACT_RUN_NAME'], '--scope', config['EVALUATION_SCOPE']]
    for key in ('MODEL', 'HEAD_TYPE', 'ENCODER_TYPE', 'TEMPORAL_BLOCK', 'HISTORY_HOURS', 'USE_SITE_ELEVATION',
                'USE_BATHYMETRY', 'BATCH_SIZE', 'TORCH_THREADS', 'NUM_WORKERS', 'PREFETCH_FACTOR', 'MP_CONTEXT', 'DEVICE'):
        command.extend(['--' + key.lower(), config[key]])
    for key in ('PIN_MEMORY', 'PERSISTENT_WORKERS', 'DUAL_DIAGNOSTICS', 'SAVE_NPZ'):
        if config[key] == '1':
            command.append('--' + key.lower())
    if config['USE_AMP'] == '1':
        command.extend(['--amp', '--amp_dtype', config['AMP_DTYPE']])
    if config['USE_TF32'] == '1':
        command.append('--tf32')
    return command


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config', type=Path)
    parser.add_argument('--checkpoint', type=Path, help='Exact, already-selected F0 checkpoint; no glob or latest-run selection')
    parser.add_argument('--dry-run', action='store_true', help='Print the command; no inference, training, queue, or result directory')
    args = parser.parse_args(argv)
    config = source_values(args.config)
    validate_config(config)
    supplied = args.checkpoint or config.get('F0_CHECKPOINT')
    if not supplied and not args.dry_run:
        parser.error('Supply the corresponding selected F0 .pth with --checkpoint or F0_CHECKPOINT. F1 never selects a checkpoint.')
    if supplied:
        checkpoint = Path(supplied).expanduser().resolve()
        if not checkpoint.is_file():
            parser.error(f'F0 checkpoint does not exist: {checkpoint}')
        validate_checkpoint(config, checkpoint)
    else:
        checkpoint = config['SOURCE_CHECKPOINT']  # Unresolved, visibly labeled dry-run placeholder only.
    stamp = '<TIMESTAMP>' if args.dry_run else datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    output = RESULTS / f'{config["PACT_RUN_NAME"]}__{stamp}'
    command = inference_command(config, checkpoint, output)
    sys.path.insert(0, str(REPO))
    from infer import parse_args as parse_infer_args
    parse_infer_args(command[3:])
    print('SOURCE_F0: ' + config['SOURCE_RUN_NAME'], flush=True)
    print('CMD: ' + shlex.join(command), flush=True)
    if args.dry_run:
        return
    digest = hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest()
    output.mkdir(parents=True, exist_ok=False)
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES=config['CUDA_VISIBLE_DEVICES'], PYTHONDONTWRITEBYTECODE='1')
    subprocess.run(command, cwd=REPO, env=environment, check=True)
    report = json.loads((output / 'dual_diagnostics.json').read_text())
    hard = report['hard_gate_0p5']
    if hard['threshold'] != .5 or hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest() != digest:
        raise RuntimeError('F1 threshold or source checkpoint changed during evaluation')
    result = dict(formulation='F1_Hard', mode='postprocess', source_run=config['SOURCE_RUN_NAME'],
                  source_checkpoint=str(checkpoint), source_checkpoint_sha256=digest,
                  scope='held_out_years', **hard)
    (output / 'f1_metrics.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(f'F1 metrics: {output / "f1_metrics.json"}; standard y_pred remains the F0 soft prediction.')


if __name__ == '__main__':
    main()
