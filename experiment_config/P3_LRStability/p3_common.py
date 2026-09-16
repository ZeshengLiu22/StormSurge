#!/usr/bin/env python3
"""P3 matrix and source-preserving config utilities (no training entry point)."""

from collections import Counter
import csv
import hashlib
import os
from pathlib import Path
import re
import shlex
import subprocess

GROUP = 'P3_LRStability'
RELATIVE = f'experiment_config/{GROUP}'
P2_RELATIVE = 'experiment_config/P2_PeakAwareFactorial'
RESULTS = f'All_Results/{GROUP}'
STATIONS = ('Lewes', 'Battery', 'Boston')
LRS = {'5e-3': '0.005', '2e-3': '0.002', '1e-3': '0.001'}
CORE = ('T1A1S0P1', 'T1A1S0P0', 'T0A1S0P1', 'T0A0S0P0')
# User-supplied historical observations, for documentation only. Never thresholds.
ALERTS = {
    ('Lewes', 'T0A0S0P1'): ('283', '31.933', ''),
    ('Battery', 'T0A0S1P1'): ('271', '43.668', ''),
    ('Battery', 'T1A0S1P0'): ('255', '43.769', ''),
    ('Battery', 'T1A1S0P0'): ('3', '95.713', '193'),
    ('Boston', 'T0A1S0P1'): ('260', '38.393', ''),
    ('Boston', 'T1A0S0P0'): ('286', '30.366', ''),
    ('Boston', 'T1A1S0P0'): ('27', '40.035', ''),
}
IDENTITY_SHELL = {'SESSION_NAME', 'PACT_RUN_NAME', 'ALL_RESULTS_ROOT', 'PYTHON_RUN_TAG_BASE'}
IDENTITY_CLI = {'run_tag', 'output_dir'}
STAMP = '20000101_000000'  # Validation only; never placed in runnable configs.


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def command(argv, repo, env=None):
    result = subprocess.run(argv, cwd=repo, env=env, text=True,
                            capture_output=True, timeout=30)
    require(result.returncode == 0,
            f'Command failed: {shlex.join(map(str, argv))}\n{result.stdout}\n{result.stderr}')
    return result.stdout


def matrix():
    treatments = {}
    for station in ('Battery', 'Boston'):
        for semantic in CORE:
            for lr in ('2e-3', '1e-3'):
                treatments.setdefault((station, semantic, lr), set()).add('core_probe')
    for station, semantic in ALERTS:
        for lr in LRS:
            treatments.setdefault((station, semantic, lr), set()).add('p2_degradation_alert')
    require(len(treatments) == 31, f'Expected exactly 31 unique runs, found {len(treatments)}; STOP.')
    rows = []
    order = lambda key: (STATIONS.index(key[0]), key[1], list(LRS).index(key[2]))
    for (station, semantic, lr), reasons in sorted(treatments.items(), key=lambda item: order(item[0])):
        tail, amp, shape, peak = (semantic[i] for i in (1, 3, 5, 7))
        factors = f'T{tail}_A{amp}_S{shape}_P{peak}'
        source_name = f'NCEP_{station}_24h_dual_severity_{factors}'
        name = f'P3_{station}_SS_{factors}_LR{lr}'
        epoch, rmse, late = ALERTS.get((station, semantic), ('', '', ''))
        rows.append(dict(
            station=station, formulation='severity_shape', tail=tail, amp=amp,
            shape=shape, peak=peak, learning_rate=LRS[lr], run_name=name,
            source_P2_semantic_config=source_name, reason='+'.join(sorted(reasons)),
            semantic_config=semantic, lr_label=f'LR{lr}',
            config_path=f'{RELATIVE}/train_config_{name}.sh',
            source_P2_config_path=f'{P2_RELATIVE}/train_config_{source_name}.sh',
            results_root=RESULTS, python_run_tag_base=name,
            p2_reference_selected_epoch=epoch, p2_reference_val_rmse_all_mm=rmse,
            p2_reference_late_val_rmse_all_mm=late,
            p2_reference_note=('User-supplied approximate P2 degradation reference; not a success threshold'
                               if epoch else ''),
        ))
    return rows


def read_manifest(path):
    with Path(path).open(newline='') as handle:
        return list(csv.DictReader(handle))


def declarations(text):
    """Restrict configs to P2's assignment-only shell format before sourcing."""
    result = {}
    for line in text.splitlines():
        if not line.strip() or line.lstrip().startswith('#'):
            continue
        match = re.fullmatch(r'(?:export )?([A-Za-z_]\w*)=(.*)', line)
        require(match is not None, f'Non-assignment config line: {line}')
        name, value = match.groups()
        require(name not in result, f'Duplicate config assignment: {name}')
        code = ' '.join(shlex.split(value, comments=True))
        require(not any(token in code for token in ('$(', '`', ';', '\\', '\n')),
                f'Unsupported shell expression: {line}')
        result[name] = value
    return result


def render_config(source, row):
    text = source.read_text()
    declarations(text)
    overrides = {
        'LR_LIST': f'("{row["lr_label"][2:]}")',
        'SESSION_NAME': '"${SESSION_NAME:-' + row['run_name'] + '}"',
        'PACT_RUN_NAME': f'"{row["run_name"]}"',
        'ALL_RESULTS_ROOT': f'"./{RESULTS}"',
    }
    for name, value in overrides.items():
        text, count = re.subn(rf'^{name}=.*$', lambda _: f'{name}={value}', text, flags=re.M)
        require(count == 1, f'{source}: expected one {name} assignment, found {count}')
    return text.replace('# P2 peak-aware factorial screen. Standalone; no config inheritance.',
                        '# P3 learning-rate stability probe. Standalone exact P2 copy except LR and identity.')


def source_values(path, repo):
    names = list(declarations(path.read_text()))
    script = r'''set -euo pipefail
source "$1"
shift
for p3_key in "$@"; do
    declare -n p3_value="$p3_key"
    p3_items=("${p3_value[@]}")
    printf '%s\0' "$p3_key" "${#p3_items[@]}" "${p3_items[@]}"
    unset -n p3_value
done
'''
    output = command(['bash', '--noprofile', '--norc', '-c', script, 'bash', str(path), *names],
                     repo, {'PATH': '/usr/bin:/bin'})
    words = iter(output.removesuffix('\0').split('\0'))
    values = {}
    for name in words:
        count = int(next(words))
        values[name] = [next(words) for _ in range(count)]
    require(all(len(v) == 1 for v in values.values()), f'{path}: expected singleton settings')
    return values


def differences(before, after):
    return {key: {'P2': before.get(key), 'P3': after.get(key)}
            for key in sorted(before.keys() | after.keys()) if before.get(key) != after.get(key)}


def dry_run(config, repo, parse_args):
    environment = dict(PATH='/usr/bin:/bin', DRY_RUN='1', PACT_RUNSTAMP=STAMP,
                       PYTHONDONTWRITEBYTECODE='1')
    log = command(['bash', str(repo / 'train.sh'), str(config)], repo, environment)
    lines = [line[5:] for line in log.splitlines() if line.startswith('CMD: ')]
    require(len(lines) == 1, f'{config}: expected exactly one dry-run command')
    require('DRY_RUN=1' in log and 'launching inside tmux' not in log,
            f'{config}: invalid dry-run evidence')
    argv = shlex.split(lines[0])
    require(argv[0] == 'train.py', f'{config}: unexpected entry point {argv[0]}')
    args = {key: str(value) if isinstance(value, Path) else value
            for key, value in vars(parse_args(argv[1:])).items()}
    return dict(args=args, command=argv, stdout=log)


def audit(rows):
    by_lr = Counter(row['learning_rate'] for row in rows)
    by_station = Counter(row['station'] for row in rows)
    core = sum('core_probe' in row['reason'].split('+') for row in rows)
    alerts = sum('p2_degradation_alert' in row['reason'].split('+') for row in rows)
    overlap = sum('+' in row['reason'] for row in rows)
    return dict(unique_runs=len(rows), by_lr={k: by_lr[v] for k, v in LRS.items()},
                by_station={s: by_station[s] for s in ('CBBT', *STATIONS)},
                core_probe=core, degradation_alert_treatments=alerts, overlap_removed=overlap)


def matrix_text(rows):
    counts = audit(rows)
    lines = [GROUP, '', f'Unique runs: {counts["unique_runs"]}', '', 'By LR:']
    lines += [f'    {lr}: {count}' for lr, count in counts['by_lr'].items()]
    lines += ['', 'By station:']
    lines += [f'    {station}: {count}' for station, count in counts['by_station'].items()]
    lines += ['', f'Core probe: {counts["core_probe"]}',
              f'Degradation-alert treatments: {counts["degradation_alert_treatments"]} nominal',
              f'Overlap removed: {counts["overlap_removed"]}',
              f'Final unique: {counts["unique_runs"]}', '', 'Station | T A S P | LR | reason']
    lines += [f'{r["station"]} | {r["tail"]} {r["amp"]} {r["shape"]} {r["peak"]} | '
              f'{r["lr_label"][2:]} | {r["reason"]}' for r in rows]
    return '\n'.join(lines) + '\n'


def results_snapshot(repo):
    entries = {}
    for directory, subdirs, files in os.walk(repo / 'All_Results', followlinks=False):
        for name in ['.', *files]:
            path = Path(directory) / name
            stat = path.lstat()
            entries[str(path.relative_to(repo))] = [stat.st_mode, stat.st_size, stat.st_mtime_ns]
    return entries
