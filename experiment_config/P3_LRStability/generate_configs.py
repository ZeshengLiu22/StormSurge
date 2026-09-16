#!/usr/bin/env python3
"""Copy the 31 P3 treatments from their exact P2 scripts; never launch training."""

import argparse
import csv
import io
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
from p3_common import (P2_RELATIVE, command, matrix, matrix_text, read_manifest,
                       render_config, require, sha256, source_values)


def build(repo):
    rows = matrix()  # Cardinality checked before any writes.
    p2_manifest = repo / P2_RELATIVE / 'manifest.csv'
    p2_rows = read_manifest(p2_manifest)
    sources = {}
    contents = {}
    for row in rows:
        matches = [p for p in p2_rows if p['run_name'] == row['source_P2_semantic_config']]
        require(len(matches) == 1, f'P2 semantic source is missing/ambiguous: {row["run_name"]}')
        p2 = matches[0]
        for field in ('station', 'formulation'):
            require(p2[field] == row[field], f'P2 manifest mismatch: {field}')
        for field in ('tail', 'amp', 'shape', 'peak'):
            require(p2[field + '_enabled'] == row[field], f'P2 factor mismatch: {field}')
        require(p2['config_path'] == row['source_P2_config_path'], 'P2 source path mismatch')
        source = repo / row['source_P2_config_path']
        values = source_values(source, repo)
        require(values['LR_LIST'] == ['5e-3'], f'P2 LR is no longer 5e-3: {source}')
        require(values['HEAD_TYPE'] == ['dual'] and values['DUAL_MODE'] == ['exceedance']
                and values['EXCESS_FORMULATION'] == ['severity_shape'], f'Wrong P2 formulation: {source}')
        row['source_P2_sha256'] = sha256(source)
        sources[row['source_P2_config_path']] = row['source_P2_sha256']
        contents[Path(row['config_path']).name] = render_config(source, row)
    buffer = io.StringIO(newline='')
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    contents['manifest.csv'] = buffer.getvalue()
    # Lock the shared launcher, implementation, normalization and station metadata.
    # Config-vs-P2 comparison alone would miss a simultaneous shared-code change.
    paths = [repo / 'train.sh', repo / 'train.py', p2_manifest]
    paths += sorted((repo / 'emulator').rglob('*.py'))
    paths += sorted((repo / 'station_json').glob('*.json'))
    provenance = dict(
        repository_commit=command(['git', 'rev-parse', 'HEAD'], repo).strip(),
        source_P2_sha256=sources,
        shared_files_sha256={str(p.relative_to(repo)): sha256(p) for p in paths},
        python_bin=values['PYTHON_BIN'][0],
        severity_mode_note='P2 uses DUAL_MODE=exceedance and EXCESS_FORMULATION=severity_shape.',
        reference_note='P2 degradation observations were supplied by the user; never success thresholds.',
    )
    contents['provenance.json'] = json.dumps(provenance, indent=2) + '\n'
    contents['matrix.txt'] = matrix_text(rows)
    return contents


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument('--config-dir', type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    repo, directory = args.repo.resolve(), args.config_dir.resolve()
    contents = build(repo)
    # Refuse edited/extra treatments, including partial manifests, before writing anything.
    require(not any(p.is_symlink() for p in directory.glob('*')), 'Config directory contains a symlink')
    existing = {p.name for p in directory.glob('train_config_*.sh')}
    expected = {name for name in contents if name.startswith('train_config_')}
    require(existing <= expected, f'Unexpected existing configs: {sorted(existing - expected)}')
    for name, content in contents.items():
        path = directory / name
        require(not path.exists() or path.read_bytes() == content.encode(),
                f'Refusing to overwrite changed generated content: {path}')
    directory.mkdir(parents=True, exist_ok=True)
    for name, content in contents.items():
        path = directory / name
        if not path.exists():
            path.write_bytes(content.encode())
        if name.endswith('.sh'):
            path.chmod(0o755)
    print(contents['matrix.txt'], end='')
    print('\nGenerated exactly 31 standalone configs. No training or queue submission occurred.')


if __name__ == '__main__':
    main()
