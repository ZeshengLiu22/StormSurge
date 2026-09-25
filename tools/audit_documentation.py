#!/usr/bin/env python3
"""Check current documentation against parsers, metric keys, model choices and links."""

import argparse
import ast
from dataclasses import fields
import json
from pathlib import Path
import re
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from emulator.training.arguments import parse_args
from emulator.training.losses import LossConfig
from emulator.training.metrics import METRIC_KEYS, METRIC_LABELS


DOCUMENTS = ('README.md', 'FORMULATION.md', 'BACKBONE.md', 'DUAL_EXCEEDANCE.md',
             'SEVERITY_SHAPE.md', 'LOSSES.md', 'CHECKPOINT_SELECTION.md', 'METRICS.md')
BANNED = ('rmse_peak5', 'mae_peak5', 'true_peak_rmse_top5', 'Top5RMSE', 'Top5MAE',
          'constrained_peak5', 'constrained_true_peak', 'checkpoint_score_refs',
          'window-max Q95', 'TRAIN window maxima')


def training_parser():
    captured = []
    original = argparse.ArgumentParser.parse_args

    def record(parser, args=None, namespace=None):
        captured.append(parser)
        return original(parser, args, namespace)

    with patch.object(argparse.ArgumentParser, 'parse_args', record):
        parse_args([])
    return captured[0]


def audit(root=ROOT):
    root = Path(root)
    errors = []
    actual = sorted(p.relative_to(root / 'docs').as_posix() for p in (root / 'docs').rglob('*') if p.is_file())
    if actual != sorted(DOCUMENTS):
        errors.append(f'Active docs set differs: {actual}')
    files = [root / 'README.md', *(root / 'docs' / name for name in DOCUMENTS)]
    text_by_file = {p: p.read_text() for p in files if p.exists()}
    parser = training_parser()
    actions = {a.dest: a for a in parser._actions}
    cli = {option for action in parser._actions for option in action.option_strings}
    # Include the inference and offline utility CLIs, without executing them.
    for source in [root / 'infer.py', *sorted((root / 'tools').glob('*.py'))]:
        tree = ast.parse(source.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == 'add_argument':
                cli.update(arg.value for arg in node.args if isinstance(arg, ast.Constant)
                           and isinstance(arg.value, str) and arg.value.startswith('--'))
    shell = (root / 'train.sh').read_text()
    # Uppercase assignment names declared in the current launcher and generator.
    declared = shell + '\n' + '\n'.join((root / 'tools' / 'generate_configs.py').read_text().splitlines())
    env_names = set(re.findall(r'\b([A-Z][A-Z0-9_]*)\b(?=[=}:])', declared))
    env_names.update({'PACT_PYTHON', 'PYTHONPATH'})
    env_names.update(a.dest.upper() for a in parser._actions)
    flags_checked, configs_checked, links_checked = set(), set(), 0
    for path, content in text_by_file.items():
        for term in BANNED:
            if term.lower() in content.lower():
                errors.append(f'{path.name}: forbidden obsolete term {term}')
        for option in set(re.findall(r'(?<![\w-])--[a-z][a-z0-9_-]*', content)):
            flags_checked.add(option)
            if option not in cli:
                errors.append(f'{path.name}: unknown CLI option {option}')
        config_tokens = set(re.findall(r'`([A-Z][A-Z0-9]*_[A-Z0-9_]+)`', content))
        # Configuration assignments occur in code, whereas prose equations may
        # use uppercase variables such as TP for true positives.
        code_fragments = re.findall(r'`([^`\n]+)`', content)
        code_fragments.extend(re.findall(r'```(?:bash|sh|shell)\n(.*?)```', content, re.S))
        config_tokens.update(re.findall(r'\b([A-Z][A-Z0-9_]+)=', '\n'.join(code_fragments)))
        for token in config_tokens:
            configs_checked.add(token)
            if token not in env_names:
                errors.append(f'{path.name}: unknown config variable {token}')
        for target in re.findall(r'\[[^\]]*\]\(([^)]+)\)', content):
            if '://' in target or target.startswith('#'):
                continue
            target = target.split('#')[0].strip('<>')
            if not (path.parent / target).exists():
                errors.append(f'{path.name}: broken local link {target}')
            links_checked += 1
    metric_doc = text_by_file.get(root / 'docs' / 'METRICS.md', '')
    documented_metrics = set(re.findall(r'`([a-z][a-z0-9_]+)`', metric_doc)) & set(METRIC_KEYS)
    for key in METRIC_KEYS:
        if key not in documented_metrics or METRIC_LABELS[key] not in metric_doc:
            errors.append(f'METRICS.md: missing canonical key/label {key}')
    for line in metric_doc.splitlines():
        if line.startswith('|') and any(label in line for label in METRIC_LABELS.values()):
            for key in re.findall(r'`([a-z][a-z0-9_]+)`', line):
                if key not in METRIC_KEYS and not key.endswith('_lead_0'):
                    errors.append(f'METRICS.md: unknown metric table key {key}')
    loss_doc = text_by_file.get(root / 'docs' / 'LOSSES.md', '')
    for field in fields(LossConfig):
        if field.name not in loss_doc and field.name.upper() not in loss_doc:
            errors.append(f'LOSSES.md: missing supported loss setting {field.name}')
        if field.name not in actions:
            errors.append(f'LossConfig field missing from CLI: {field.name}')
    # Enum declarations in the relevant documents are checked exactly against argparse.
    mappings = {'checkpoint_selection': 'CHECKPOINT_SELECTION.md', 'encoder_type': 'BACKBONE.md',
                'temporal_block': 'BACKBONE.md', 'excess_formulation': 'SEVERITY_SHAPE.md',
                'loss_mode': 'LOSSES.md', 'excess_loss_mode': 'LOSSES.md',
                'exceedance_loss_mode': 'LOSSES.md'}
    for name, doc in mappings.items():
        pattern = rf'<!-- choices {name}: ([^>]+) -->'
        matches = re.findall(pattern, text_by_file.get(root / 'docs' / doc, ''))
        expected = set(actions[name].choices)
        if len(matches) != 1 or set(matches[0].split(',')) != expected:
            errors.append(f'{doc}: choices declaration for {name} must equal {sorted(expected)}')
    return dict(passed=not errors, errors=errors, documents=len(text_by_file),
                canonical_metric_keys=len(documented_metrics), cli_options=sorted(flags_checked),
                config_variables=sorted(configs_checked), local_links=links_checked,
                loss_fields=len(fields(LossConfig)), supported_enums=len(mappings), banned_term_matches=0 if not any('forbidden' in e for e in errors) else None)


def main():
    result = audit()
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result['passed'] else 1)


if __name__ == '__main__':
    main()
