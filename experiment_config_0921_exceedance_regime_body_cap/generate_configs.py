#!/usr/bin/env python3
"""Generate the six matched TRAIN-regime/body-cap conditions for four stations."""

import argparse
import csv
import io
from pathlib import Path
import re
import shlex

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
BASE_SHA = 'd665a4b0f54700a2564e54187f6c067c470d1df0'
PRIMARY = Path('/media/volume/PACT-Data/StormSurge')
RESULTS = Path('/media/share/PACT/Results/All_results_0921_exceedance_regime_body_cap')
PYTHON = '/media/volume/PACT-Data/conda_envs/torchpyg-cu124/bin/python'
STATIONS = ('CBBT', 'Lewes', 'Battery', 'Boston')
CONDITIONS = (('T0', 95, 2, 'soft'), ('T1', 95, 2, 'exact'),
              ('T2', 90, 1, 'soft'), ('T3', 90, 1, 'exact'),
              ('T4', 90, 2, 'soft'), ('T5', 90, 2, 'exact'))


def reference_path(station):
    return REPO / 'experiment_config_0916/A0_HeadCapacity' / f'train_config_NCEP_{station}_24h_dual_direct_base.sh'


def run_name(station, condition, percentile, weight, cap):
    return f'0921_{station}_{condition}_Q{percentile}_W{weight}_{cap}'


def render_config(station, condition, percentile, weight, cap):
    name = run_name(station, condition, percentile, weight, cap)
    source = reference_path(station).read_text()
    changes = dict(EXCEEDANCE_PERCENTILE=str(percentile), EXCESS_LOSS_WEIGHT=str(weight),
                   SESSION_NAME=f'"${{SESSION_NAME:-{name}}}"', PACT_RUN_NAME=f'"{name}"',
                   ALL_RESULTS_ROOT=f'"{RESULTS}"')
    for key, value in changes.items():
        source, count = re.subn(rf'^{key}=.*$', lambda _: f'{key}={value}', source, flags=re.MULTILINE)
        assert count == 1, (key, count)
    source = source.replace('DUAL_MODE="exceedance"\n', f'DUAL_MODE="exceedance"\nDUAL_BODY_CAP="{cap}"\n')
    source = source.replace('# A0 prediction-head / excess-branch capacity study.',
                            '# 0921 TRAIN exceedance regime / body-cap study.')
    source = source.replace('# Matched direct/severity supervision: BODY=1, trajectory EXCESS=2, GATE=0.5.',
                            '# Matched direct supervision: BODY=1, GATE=0.5; EXCESS is the study factor.')
    source = source.replace('# All peak-aware auxiliary losses are OFF, including for severity_shape.\n'
                            '# Severity/shape factorization learns through prediction and ordinary dual supervision.',
                            '# All optional auxiliary losses are OFF; production direct excess is unchanged.')
    return source


def manifest_rows():
    for station in STATIONS:
        for condition, percentile, weight, cap in CONDITIONS:
            name = run_name(station, condition, percentile, weight, cap)
            path = f'{HERE.name}/configs/train_config_{name}.sh'
            yield dict(station=station, condition=condition, train_percentile=percentile,
                       excess_loss_weight=weight, dual_body_cap=cap, run_name=name,
                       config_path=path, output_dir=str(RESULTS / f'{name}__<TIMESTAMP>'),
                       command=shlex.join(['qsub_local', 'train.sh', name, path]))


def expected_files():
    files = {f'configs/train_config_{run_name(station, *condition)}.sh': render_config(station, *condition)
             for station in STATIONS for condition in CONDITIONS}
    rows = list(manifest_rows())
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
    writer.writeheader()
    writer.writerows(rows)
    files['manifest.csv'] = stream.getvalue()
    files['commands_all.txt'] = ''.join(row['command'] + '\n' for row in rows)
    return files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    files = expected_files()
    actual = {str(path.relative_to(HERE)) for path in (HERE / 'configs').glob('*.sh')}
    assert not actual - files.keys(), f'Unexpected configs: {actual - files.keys()}'
    for relative, content in files.items():
        path = HERE / relative
        if args.check:
            assert path.read_text() == content, f'Generated file differs: {relative}'
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)
    print(f'{"Checked" if args.check else "Generated"}: 24 configs, six per station. No jobs launched.')


if __name__ == '__main__':
    main()
