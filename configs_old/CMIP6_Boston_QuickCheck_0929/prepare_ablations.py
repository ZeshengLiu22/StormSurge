#!/usr/bin/env python3
"""Append Tail Only and Peak Only to the existing Boston quick-check batch."""
import csv
from datetime import datetime, timezone
from pathlib import Path
import re
import shlex

from prepare_configs import GROUPS, REPO, RESULTS, ROOT, write_manifest

VARIANTS = [('T0500_EP0000', '0.05', '0.0'), ('T0000_EP0100', '0.0', '0.01')]
DATASETS = ['CMIP6_AWI', 'CMIP6_CNRM', 'CMIP6_EC_EARTH', 'CMIP6_MPI', 'CMIP6_MRI']


def read_manifest(path):
    with path.open() as handle:
        return list(csv.DictReader(handle))


def assignments(text):
    return dict(re.findall(r'^([A-Za-z_][A-Za-z_0-9]*)=(.*)$', text, re.M))


def main():
    stamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
    original = read_manifest(ROOT / 'manifest.csv')
    expected = {(group, dataset, variant) for group in GROUPS for dataset in DATASETS
                for variant in ('T0000_EP0000', 'T0500_EP0100')}
    assert len(original) == len(expected) and {
        (row['group'], row['dataset'], row['variant']) for row in original
    } == expected, 'Expected only the original 20 runs; refuse to append twice'
    commands_path = ROOT / 'qsub_ablations.txt'
    assert not commands_path.exists(), 'Manual submission commands already exist'

    pending, added, groups = {}, [], {}
    for group, (_, _, _, short) in GROUPS.items():
        folder = ROOT / group
        rows = read_manifest(folder / 'manifest.csv')
        assert rows == [row for row in original if row['group'] == group]
        assert {str(path) for path in folder.glob('train_config_*.sh')} == {
            row['config_path'] for row in rows
        }, f'{group}: config inventory differs from manifest'
        groups[group] = list(rows)
        for dataset in DATASETS:
            combined = next(row for row in rows if row['dataset'] == dataset
                            and row['variant'] == 'T0500_EP0100')
            source = Path(combined['config_path']).read_text()
            for variant, tail, peak in VARIANTS:
                run_name = f'{dataset}_Boston_{group}_G0_{variant}'
                config = folder / f'train_config_{run_name}.sh'
                result = RESULTS / group / f'{run_name}__{stamp}'
                template = REPO / 'configs/single_tail_episodepeak_4x4' / f'train_config_NCEP_Boston_G0_{variant}.sh'
                assert template.is_file() and not config.exists() and not result.exists()
                text = source.replace('T0500_EP0100', variant)
                text = text.replace('T=0.05, EP=0.01.', f'T={tail}, EP={peak}.')
                for key, value in [('EXCEEDANCE_LOSS_WEIGHT', tail),
                                   ('EPISODE_GT_ALIGNED_PEAK_WEIGHT', peak)]:
                    text, count = re.subn(rf'^{key}=.*$', f'{key}={value}', text, flags=re.M)
                    assert count == 1
                before, after = assignments(source), assignments(text)
                changed_weight = ('EPISODE_GT_ALIGNED_PEAK_WEIGHT' if peak == '0.0'
                                  else 'EXCEEDANCE_LOSS_WEIGHT')
                assert before.keys() == after.keys()
                assert {key for key in before if before[key] != after[key]} == {
                    changed_weight, 'PACT_RUN_NAME', 'PYTHON_RUN_TAG_BASE'
                }, f'{config}: unexpected change from the existing combined run'
                row = dict(combined, variant=variant, tail_mse_weight=tail,
                           episode_gt_aligned_peak_mse_weight=peak,
                           config_path=str(config), result_path=str(result),
                           template_path=str(template), runstamp=stamp,
                           queue_label=f'QB0929{short}_{dataset.removeprefix("CMIP6_")}_{variant}')
                pending[config] = text
                groups[group].append(row)
                added.append(row)

    # Finish all collision and comparison checks before writing any files.
    assert len(pending) == len(added) == 20
    assert len({row['queue_label'] for row in original + added}) == 40
    for path, text in pending.items():
        path.write_text(text)
    for group, rows in groups.items():
        write_manifest(ROOT / group / 'manifest.csv', rows)
    write_manifest(ROOT / 'manifest.csv', original + added)

    commands = ['# Copy these commands into an interactive shell; submit each qsub_local line once.',
                f'cd {shlex.quote(str(REPO))}',
                f'export PACT_RUNSTAMP={stamp} USE_TMUX=1 QSUB_LOCAL_SLOTS=1']
    for group in GROUPS:
        commands.extend(['', f'# {group}: Tail Only / Peak Only for each dataset'])
        for row in added:
            if row['group'] == group:
                config = Path(row['config_path']).relative_to(REPO)
                commands.append(f'qsub_local train.sh {row["queue_label"]} {shlex.quote(str(config))}')
    commands_path.write_text('\n'.join(commands) + '\n')

    table = ['# Run table', '', f'Config paths are relative to `{ROOT}`.', '',
             f'Result paths are relative to `{RESULTS}`.', '',
             'The original 20 runs are followed by 20 additional Tail Only / Peak Only configs.',
             'Use `qsub_ablations.txt` to submit the additional runs manually.', '',
             '| Group | Dataset | Loss variant | Config path | Result path |',
             '| --- | --- | --- | --- | --- |']
    for row in original + added:
        table.append('| ' + ' | '.join([row['group'], row['dataset'], row['variant'],
                     str(Path(row['config_path']).relative_to(ROOT)),
                     str(Path(row['result_path']).relative_to(RESULTS))]) + ' |')
    (ROOT / 'RUN_TABLE.md').write_text('\n'.join(table) + '\n')
    print(f'Added {len(added)} configs in the existing folders; runstamp={stamp}')
    print(f'Manual submission commands: {commands_path}')


if __name__ == '__main__':
    main()
