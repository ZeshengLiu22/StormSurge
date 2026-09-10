"""Generate a source inventory and patch against an original Emulator checkout.

Usage: python docs/audit/compare_original.py --original /path/to/Emulator
Generated audit files and the comparison report are excluded to avoid self-reference.
"""

import argparse
import ast
from collections import Counter
import csv
from datetime import datetime, timezone
import difflib
import hashlib
import json
from pathlib import Path
import subprocess


def source_files(root):
    paths = subprocess.check_output(
        ['git', 'ls-files', '--cached', '--others', '--exclude-standard', '-z'], cwd=root,
    ).decode().split('\0')
    return {path for path in paths if path and (root / path).is_file()
            and not path.startswith('docs/audit/') and path != 'docs/ORIGINAL_COMPARISON.md'}


def declared_arguments(root):
    """Read parser declarations without importing or running either trainer."""
    path = root / 'emulator/training/arguments.py'
    if not path.exists():
        path = root / 'train.py'
    constants = {}
    shared = root / 'emulator/common/dual.py'
    if shared.exists():
        for node in ast.parse(shared.read_text()).body:
            if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'DUAL_ABLATIONS' for t in node.targets):
                constants['DUAL_ABLATIONS'] = ast.literal_eval(node.value)
    arguments = {}
    for node in ast.walk(ast.parse(path.read_text())):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name) and node.func.value.id == 'parser'
                and node.func.attr == 'add_argument'):
            continue
        flags = [ast.literal_eval(value) for value in node.args]
        kwargs = {item.arg: item.value for item in node.keywords}
        action = ast.literal_eval(kwargs['action']) if 'action' in kwargs else 'store'
        default = ast.literal_eval(kwargs['default']) if 'default' in kwargs else False if action == 'store_true' else None
        choices = kwargs.get('choices')
        if isinstance(choices, ast.Call) and ast.unparse(choices) == 'tuple(DUAL_ABLATIONS)':
            choices = list(constants['DUAL_ABLATIONS'])
        else:
            choices = ast.literal_eval(choices) if choices is not None else None
        dest = ast.literal_eval(kwargs['dest']) if 'dest' in kwargs else flags[0].lstrip('-').replace('-', '_')
        arguments[dest] = dict(flags=flags, default=default, choices=choices,
                               action={'store': '_StoreAction', 'store_true': '_StoreTrueAction'}[action],
                               type=ast.unparse(kwargs['type']) if 'type' in kwargs else None)
    return arguments


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--original', type=Path, required=True)
    parser.add_argument('--current', type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    original, current = args.original.resolve(), args.current.resolve()
    out = current / 'docs' / 'audit'
    out.mkdir(parents=True, exist_ok=True)
    rows, patches = [], []
    for path in sorted(source_files(original) | source_files(current)):
        before = (original / path).read_bytes() if (original / path).is_file() else None
        after = (current / path).read_bytes() if (current / path).is_file() else None
        status = 'unchanged' if before == after else 'added' if before is None else 'deleted' if after is None else 'modified'
        rows.append(dict(path=path, status=status,
                         original_sha256=hashlib.sha256(before).hexdigest() if before is not None else None,
                         current_sha256=hashlib.sha256(after).hexdigest() if after is not None else None,
                         original_lines=before.count(b'\n') if before else 0,
                         current_lines=after.count(b'\n') if after else 0))
        if status != 'unchanged':
            patches.extend(difflib.unified_diff(
                (before or b'').decode().splitlines(keepends=True),
                (after or b'').decode().splitlines(keepends=True),
                fromfile='original/' + path, tofile='StormSurge/' + path))
    (out / 'source_inventory.json').write_text(json.dumps(rows, indent=2))
    with (out / 'source_inventory.csv').open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    (out / 'original_to_current.diff').write_text(''.join(patches))
    before_args, after_args = declared_arguments(original), declared_arguments(current)
    argument_comparison = dict(
        removed={key: before_args[key] for key in sorted(before_args.keys() - after_args.keys())},
        added={key: after_args[key] for key in sorted(after_args.keys() - before_args.keys())},
        changed={key: dict(original=before_args[key], current=after_args[key])
                 for key in sorted(before_args.keys() & after_args.keys()) if before_args[key] != after_args[key]},
        unchanged_count=sum(before_args[key] == after_args[key] for key in before_args.keys() & after_args.keys()),
        scope='Training parser declarations only; post-parse validation and ablation resolution are described in ORIGINAL_COMPARISON.md.',
    )
    (out / 'argparse_comparison.json').write_text(json.dumps(argument_comparison, indent=2) + '\n')
    summary = dict(generated_utc=datetime.now(timezone.utc).isoformat(),
                   original_repository='https://github.com/BinaLab/PACT_Storm_Surge_Emulator',
                   original_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=original).decode().strip(),
                   original_worktree_status=subprocess.check_output(['git', 'status', '--short'], cwd=original).decode().splitlines(),
                   current_repository='https://github.com/ZeshengLiu22/StormSurge',
                   counts=dict(Counter(row['status'] for row in rows)), total_paths=len(rows),
                   excluded=['docs/ORIGINAL_COMPARISON.md', 'docs/audit/**', 'ignored data/results/checkpoints/caches', '.git/**'])
    (out / 'comparison_summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
