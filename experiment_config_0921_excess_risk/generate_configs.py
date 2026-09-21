#!/usr/bin/env python3
"""Generate four-station 2x2 configs from the read-only, pinned 0920 CBBT F0 protocol."""

import argparse
import csv
import hashlib
import io
import itertools
from pathlib import Path
import re
import shlex
import subprocess

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
START = 'd665a4b0f54700a2564e54187f6c067c470d1df0'
REFERENCE_COMMIT = '7516f9c5148d3068ad86b984028d417fd226f782'
REFERENCE = f'{REFERENCE_COMMIT}:experiment_config_0920/configs/train_config_0920_CBBT_F0_Soft.sh'
RESULTS = '/media/share/PACT/Results/All_results_0921_excess_risk'
PYTHON = '/media/volume/PACT-Data/conda_envs/torchpyg-cu124/bin/python'
STATIONS = ('CBBT', 'Lewes', 'Battery', 'Boston')
MODES = (
    ('E0_Uniform', 'none', 'uniform'),
    ('E1_PriorNorm', 'train_prior', 'uniform'),
    ('E2_Magnitude', 'none', 'relative_magnitude'),
    ('E3_PriorNorm_Magnitude', 'train_prior', 'relative_magnitude'),
)


def reference_text():
    return subprocess.check_output(['git', 'show', REFERENCE], cwd=REPO, text=True)


def expected_files():
    reference = reference_text()
    files, rows = {}, []
    for station, (mode, normalization, weighting) in itertools.product(STATIONS, MODES):
        name = f'0921_{station}_{mode}'
        text = reference.replace('0920_CBBT_F0_Soft', name)
        text = text.replace('STATION="CBBT"', f'STATION="{station}"')
        text = text.replace('# 0920 direct-dual reconstruction / supervision-scope study.',
                            '# 0921 excess-risk aggregation study.')
        text = text.replace('DIRECT_DUAL_RECONSTRUCTION="soft_gate"\n', '')
        text = text.replace('EXCESS_SUPERVISION_SCOPE="event"\n', '')
        text = text.replace('EXCESS_FORMULATION="direct"\n',
                            'EXCESS_FORMULATION="direct"\nEXCEEDANCE_HEAD_EXPERIMENT=""\n')
        text = re.sub(r'^ALL_RESULTS_ROOT=.*$', f'ALL_RESULTS_ROOT="{RESULTS}"', text, flags=re.MULTILINE)
        text += ('\n# Independent factors; excess loss weight stays 2 in every run.\n'
                 f'EXCESS_EVENT_NORMALIZATION="{normalization}"\n'
                 f'EXCESS_HORIZON_WEIGHTING="{weighting}"\nEXCESS_MAGNITUDE_ALPHA="1.0"\n')
        path = f'configs/train_config_{name}.sh'
        files[path] = text
        command = shlex.join(['bash', 'train.sh', f'{HERE.name}/{path}'])
        rows.append(dict(mode=mode, excess_event_normalization=normalization,
            excess_horizon_weighting=weighting, excess_magnitude_alpha=1.0, excess_loss_weight=2,
            station=station, model='perceiver3', head_type='dual', excess_formulation='direct',
            checkpoint_selection='overall', config_path=f'{HERE.name}/{path}',
            output_dir=f'{RESULTS}/{name}__<TIMESTAMP>', command=command))
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
    writer.writeheader()
    writer.writerows(rows)
    files['manifest.csv'] = stream.getvalue()
    files['commands_all.txt'] = ''.join(row['command'] + '\n' for row in rows)
    return files, hashlib.sha256(reference.encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='Verify generated files without writing.')
    args = parser.parse_args()
    files, digest = expected_files()
    for relative, content in files.items():
        path = HERE / relative
        if args.check:
            assert path.read_text() == content, f'Generated file differs: {relative}'
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)
    assert len(list((HERE / 'configs').glob('*.sh'))) == len(STATIONS) * len(MODES)
    print(f'{"Checked" if args.check else "Generated"} {len(STATIONS) * len(MODES)} configs '
          f'({len(STATIONS)} stations x {len(MODES)} modes); reference SHA256={digest}; no jobs launched.')


if __name__ == '__main__':
    main()
