#!/usr/bin/env python3
"""Generate only the 44 requested configs from the frozen 0916 references."""

import argparse
import csv
import hashlib
import io
import itertools
import json
from pathlib import Path
import re
import shlex
import subprocess


HERE = Path(__file__).resolve().parent
REPO = HERE.parent
REFERENCE = REPO / 'experiment_config_0916/A0_HeadCapacity'
RESULTS = Path('/home/exouser/media/volume/PACT-Data/StormSurge/All_results_0919')
PYTHON = '/media/volume/PACT-Data/conda_envs/torchpyg-cu124/bin/python'
STATIONS = ('CBBT', 'Lewes', 'Battery', 'Boston')
VARIANTS = ('legacy', 'c1', 'c2', 'c2r', 'c3')
POOLS = ('mean', 'learned')
LABELS = dict(legacy='Legacy', c1='C1', c2='C2', c2r='C2R', c3='C3')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def matrix():
    for station in STATIONS:
        yield station, None, None
        for variant, pooling in itertools.product(VARIANTS, POOLS):
            yield station, variant, pooling


def reference_path(station, variant):
    return REFERENCE / f'train_config_NCEP_{station}_24h_{"single" if variant is None else "dual_direct"}_base.sh'


def run_name(station, variant, pooling):
    suffix = 'Single_MSE' if variant is None else f'{LABELS[variant]}_{pooling.title()}'
    return f'0919_{station}_{suffix}'


def render_config(station, variant, pooling):
    name = run_name(station, variant, pooling)
    changes = dict(SEED='42', DETERMINISTIC='1', SESSION_NAME=f'"${{SESSION_NAME:-{name}}}"',
                   PACT_RUN_NAME=f'"{name}"', ALL_RESULTS_ROOT=f'"{RESULTS}"')
    source = reference_path(station, variant).read_text()
    for key, value in changes.items():
        source, count = re.subn(rf'^{key}=.*$', lambda _: f'{key}={value}', source, flags=re.MULTILINE)
        assert count == 1, (key, count)
    source = source.replace('# A0 prediction-head / excess-branch capacity study.',
                            '# 0919 direct-dual excess-capacity / gate-pooling study.')
    source = source.replace('# Matched direct/severity supervision:', '# Fixed existing direct-dual supervision:')
    source = source.replace('# All peak-aware auxiliary losses are OFF, including for severity_shape.\n'
                            '# Severity/shape factorization learns through prediction and ordinary dual supervision.',
                            '# All optional peak-aware losses are OFF; the formulation remains direct.')
    source = source.replace('EXCESS_FORMULATION="direct"\n',
        f'EXCESS_FORMULATION="direct"\nEXCEEDANCE_HEAD_EXPERIMENT="{variant or ""}"\n'
        f'EXCEEDANCE_GATE_POOLING="{pooling or "mean"}"\n')
    return source


def manifest_rows():
    for station, variant, pooling in matrix():
        name = run_name(station, variant, pooling)
        single = variant is None
        path = f'experiment_config_0919/configs/train_config_{name}.sh'
        yield dict(station=station, run_name=name, head_type='single' if single else 'dual',
            exceedance_head_experiment=variant or 'NA', exceedance_gate_pooling=pooling or 'NA',
            loss_mode='mse', excess_formulation='direct', dual_loss=0 if single else 1,
            body_loss_weight=0 if single else 1, excess_loss_weight=0 if single else 2,
            gate_loss_weight=0 if single else .5, seed=42, deterministic=1, lr='5e-3', history_hours=24,
            hidden_channels=128, temporal_block='Transformer', encoder_type='GraphSAGE',
            batch_size=256, grad_accum_steps=4, epochs=300, num_gpus=1, amp_dtype='bf16',
            use_amp=1, use_tf32=1, num_workers=0, pin_memory=0, persistent_workers=0, prefetch_factor=0,
            checkpoint_selection='overall', save_aux_checkpoints=0,
            config_path=path, output_dir=str(RESULTS / f'{name}__<TIMESTAMP>'),
            reference_config=str(reference_path(station, variant).relative_to(REPO)),
            qsub_local_command=shlex.join(['qsub_local', 'train.sh', name, path]))


def expected_files():
    files = {f'configs/train_config_{run_name(*item)}.sh': render_config(*item) for item in matrix()}
    rows = list(manifest_rows())
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
    writer.writeheader()
    writer.writerows(rows)
    files['manifest.csv'] = stream.getvalue()
    files['commands_all.txt'] = ''.join(row['qsub_local_command'] + '\n' for row in rows)
    return files


def frozen_hashes():
    sources = [*sorted((REPO / 'emulator').rglob('*.py')), *sorted((REPO / 'station_json').glob('*.json')),
               *[REPO / name for name in ('train.py', 'infer.py', 'train.sh', 'infer.sh', 'infer_multi.sh',
                                          'environment_training.yml')]]
    return {str(path.relative_to(REPO)): digest(path) for path in sources}


def reference_hashes():
    return {str(path.relative_to(REPO)): digest(path) for path in sorted(REFERENCE.rglob('*')) if path.is_file()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='Check generated files without writing.')
    args = parser.parse_args()
    files = expected_files()
    expected = {name for name in files if name.startswith('configs/')}
    actual = {str(path.relative_to(HERE)) for path in (HERE / 'configs').glob('*.sh')}
    assert not actual - expected, f'Unexpected configs: {actual - expected}'
    provenance_path = HERE / 'provenance.json'
    provenance = dict(implementation_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'],
        cwd=REPO, text=True).strip(), reference_tree_sha256=reference_hashes(),
        frozen_source_sha256=frozen_hashes(),
        protocol_change='DETERMINISTIC=0 -> 1 as explicitly requested; seed remains 42.',
        reference_use='Read-only generation/validation input; generated configs never source 0916 at runtime.')
    if provenance_path.exists():
        saved = json.loads(provenance_path.read_text())
        assert saved['reference_tree_sha256'] == provenance['reference_tree_sha256'], '0916 references changed.'
        assert saved['frozen_source_sha256'] == provenance['frozen_source_sha256'], 'Frozen implementation changed.'
    elif not args.check:
        provenance_path.write_text(json.dumps(provenance, indent=2) + '\n')
    else:
        raise AssertionError('Missing provenance.json')
    for relative, content in files.items():
        path = HERE / relative
        if args.check:
            assert path.read_text() == content, f'Generated file differs: {relative}'
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)
    print(f'{"Checked" if args.check else "Generated"}: 44 configs, 11 per station; no jobs launched.')


if __name__ == '__main__':
    main()
