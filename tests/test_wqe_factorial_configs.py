"""Matched Single and Dual factorials, resolved dry runs, and safe launch order."""

import contextlib
import csv
import io
import itertools
import json
import os
from pathlib import Path
import shlex
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from emulator.training.arguments import parse_args
from emulator.training.checkpoint_selection import ROLES
from tools.generate_configs import (FACTORIAL_CONFIG_DIR, FACTORIAL_RESULTS_ROOT, FACTORIAL_PYTHON,
                                    S0_CONFIG_DIR, S0_RESULTS_ROOT, generate, main)
from test_wqe_configs import PARAMETERS, REPO, dry_run


STATIONS = ("CBBT", "Boston", "Battery", "Lewes")


class FactorialConfigChecks:
    """Shared contracts, exercised separately for each experiment family."""

    @property
    def config_dir(self):
        return REPO / "configs" / self.family

    def test_all_configs_reproduce_and_resolve_matched_settings(self):
        controls = {station: vars(dry_run(REPO / "configs" / self.control_directory /
                    f"train_config_NCEP_{station}_{self.control_suffix}.sh")[0][0]) for station in STATIONS}
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            rows = generate(output, family=self.family)
            count = len(self.cells) * len(STATIONS)
            self.assertEqual(len(rows), count)
            self.assertEqual(len(list(output.glob('train_config_*.sh'))), count)
            self.assertEqual(len(list(self.config_dir.glob('train_config_*.sh'))), count)
            self.assertEqual([(row['cell'], row['station']) for row in rows],
                             list(itertools.product(self.cells, STATIONS)))
            self.assertEqual((output / 'manifest.csv').read_text(), (self.config_dir / 'manifest.csv').read_text())
            with (output / 'manifest.csv').open() as handle:
                manifest = list(csv.DictReader(handle))
            run_names = set()
            for row, record in zip(rows, manifest):
                with self.subTest(station=row['station'], cell=row['cell']):
                    path = output / row['config']
                    self.assertEqual(path.read_text(), (self.config_dir / row['config']).read_text())
                    subprocess.run(['bash', '-n', str(path)], check=True)
                    commands, log = dry_run(path)
                    self.assertEqual(len(commands), 1)
                    args = commands[0]
                    factors = {part[0]: int(part[1:]) for part in row['cell'].split('_')}
                    g, e, t = factors['G'], factors.get('E', 0), factors['T']
                    expected_modes = (('mse', 'wqe')[g], ('mse', 'wqe')[e])
                    self.assertEqual((args.loss_mode, args.excess_loss_mode), expected_modes)
                    self.assertEqual((row['global_loss'], row['excess_loss']), expected_modes)
                    self.assertEqual(args.exceedance_loss_mode, 'wqe' if t == 2 else 'mse')
                    self.assertEqual(args.exceedance_loss_mode, record['exceedance_loss_mode'])
                    self.assertEqual((row['tail_enabled'], args.exceedance_loss_weight), (int(t > 0), .025 if t else 0.))
                    self.assertEqual(args.exceedance_loss_weight, float(record['exceedance_loss_weight']))
                    self.assertEqual((args.head_type, args.dual_loss), (self.head, int(self.head == 'dual')))
                    self.assertEqual(row['head_type'], self.head)
                    self.assertEqual((args.body_loss_weight, args.excess_loss_weight, args.gate_loss_weight), self.branch_weights)
                    self.assertEqual((args.excess_amp_loss_weight, args.shape_loss_weight), (0., 0.))
                    self.assertEqual(args.checkpoint_selection, 'overall')
                    for name, value in PARAMETERS.items():
                        self.assertEqual(getattr(args, name), value)
                    run_dir = Path(args.output_dir)
                    self.assertEqual(run_dir.parent, Path(self.results_root))
                    name = f'NCEP_{row["station"]}_{row["cell"]}'
                    self.assertEqual(args.run_tag, name + '_' + run_dir.name.split('__', 1)[1])
                    self.assertEqual(record['run_name'], name)
                    self.assertRegex(run_dir.name, rf'^{name}__\d{{8}}_\d{{6}}$')
                    self.assertFalse(run_dir.exists())
                    self.assertEqual(record['result_root'], self.results_root)
                    self.assertIn('Results root:  ' + self.results_root, log)
                    self.assertIn(f'Tail loss: {args.exceedance_loss_mode}', log)
                    self.assertIn(f"Checkpoint roles (VAL-only): {', '.join(ROLES)}; primary=overall", log)
                    run_names.add(name)
                    # Every resolved model/training option must equal the
                    # unchanged station control, apart from factors and identity.
                    changed = {'loss_mode', 'excess_loss_mode', 'exceedance_loss_mode',
                               'exceedance_loss_weight', 'output_dir', 'run_tag'}
                    self.assertEqual({k: v for k, v in vars(args).items() if k not in changed},
                                     {k: v for k, v in controls[row['station']].items() if k not in changed})
            self.assertEqual(len(run_names), count)

    def test_launchers_have_exact_order_and_execute_only_mock_queue(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / 'configs'
            rows = generate(output, family=self.family)
            # Never invoke the real scheduler in tests.
            queue = root / 'qsub_local'
            queue.write_text('#!/usr/bin/env python3\nimport json, os, sys\n'
                             'print(json.dumps(dict(args=sys.argv[1:], cwd=os.getcwd())))\n')
            queue.chmod(0o755)
            runtime = root / 'python'
            runtime.write_text('#!/bin/sh\ncat >/dev/null\nexit 0\n')
            runtime.chmod(0o755)
            environment = dict(os.environ, PATH=str(root) + os.pathsep + os.environ['PATH'], DRY_RUN='0',
                               **{self.runtime_variable: str(runtime)})
            for name, selected_rows in [('launch_all.sh', rows),
                    *((f'launch_{cell.replace("_", "")}.sh', [row for row in rows if row['cell'] == cell]) for cell in self.cells)]:
                script = output / name
                checked = self.config_dir / name
                self.assertTrue(os.access(script, os.X_OK))
                self.assertTrue(os.access(checked, os.X_OK))
                for path in (script, checked):
                    subprocess.run(['bash', '-n', str(path)], cwd=REPO, check=True)
                    commands = [shlex.split(line) for line in path.read_text().splitlines() if line.startswith('qsub_local ')]
                    self.assertEqual(len(commands), len(selected_rows))
                    for command, row in zip(commands, selected_rows):
                        self.assertEqual(command[:3], ['qsub_local', 'train.sh',
                                         f'{self.job_prefix}_{row["station"]}_{row["cell"].replace("_", "")}'])
                        config = Path(command[3])
                        self.assertEqual(config if config.is_absolute() else REPO / config, path.parent / row['config'])
                result = subprocess.run([str(script)], cwd=REPO, env=environment,
                                        capture_output=True, text=True, check=True)
                calls = [json.loads(line) for line in result.stdout.splitlines()]
                self.assertEqual(len(calls), len(selected_rows))
                for call, row in zip(calls, selected_rows):
                    self.assertEqual(call['cwd'], str(REPO))
                    self.assertEqual(call['args'], ['train.sh', f'{self.job_prefix}_{row["station"]}_{row["cell"].replace("_", "")}',
                                                  str(output / row['config'])])

    def test_launcher_dry_run_bypasses_runtime_and_queue_for_every_cell(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / 'configs'
            results = root / 'results'
            rows = generate(output, family=self.family, results_root=results)
            queue = root / 'qsub_local'
            queue.write_text('#!/bin/sh\necho SHOULD_NOT_SUBMIT >&2\nexit 99\n')
            queue.chmod(0o755)
            runtime = root / 'python'
            runtime.write_text('#!/bin/sh\necho SHOULD_NOT_RUN >&2\nexit 98\n')
            runtime.chmod(0o755)
            environment = dict(os.environ, PATH=str(root) + os.pathsep + os.environ['PATH'], DRY_RUN='1',
                               **{self.runtime_variable: str(runtime)})
            for script in sorted(output.glob('launch_*.sh')):
                with self.subTest(launcher=script.name):
                    result = subprocess.run(['bash', str(script)], cwd=REPO, env=environment,
                                            text=True, capture_output=True, check=True)
                    commands = [parse_args(shlex.split(line[5:])[1:]) for line in result.stdout.splitlines()
                                if line.startswith('CMD: ')]
                    selected = rows if script.name == 'launch_all.sh' else [
                        row for row in rows if script.stem == 'launch_' + row['cell'].replace('_', '')]
                    self.assertEqual([args.run_tag.rsplit('_', 2)[0] for args in commands],
                                     [row['run_name'] for row in selected])
                    self.assertNotIn('SHOULD_NOT_SUBMIT', result.stderr)
                    self.assertNotIn('SHOULD_NOT_RUN', result.stderr)
                    self.assertNotIn('[Preflight OK]', result.stdout)
                    self.assertFalse(results.exists())

    def test_configs_ignore_base_python_and_allow_explicit_runtime_override(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            rows = generate(output, family=self.family)
            environment = dict(os.environ, PYTHON_BIN='/wrong/base/python', CONDA_PREFIX='/wrong/base')
            environment.pop(self.runtime_variable, None)
            for row in rows:
                command = ['bash', '-c', 'source "$1"; printf "%s" "$PYTHON_BIN"', '_', str(output / row['config'])]
                result = subprocess.run(command, cwd=REPO, env=environment, text=True, capture_output=True, check=True)
                self.assertEqual(result.stdout, FACTORIAL_PYTHON)
            environment[self.runtime_variable] = '/explicit/training/python'
            result = subprocess.run(command, cwd=REPO, env=environment, text=True, capture_output=True, check=True)
            self.assertEqual(result.stdout, environment[self.runtime_variable])

    def test_failed_runtime_preflight_submits_no_jobs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / 'configs'
            generate(output, family=self.family)
            queue = root / 'qsub_local'
            queue.write_text('#!/bin/sh\necho SHOULD_NOT_SUBMIT\n')
            queue.chmod(0o755)
            runtime = root / 'broken_python'
            runtime.write_text('#!/bin/sh\ncat >/dev/null\necho missing_torch >&2\nexit 23\n')
            runtime.chmod(0o755)
            environment = dict(os.environ, PATH=str(root) + os.pathsep + os.environ['PATH'], DRY_RUN='0',
                               **{self.runtime_variable: str(runtime)})
            for script in output.glob('launch_*.sh'):
                result = subprocess.run([str(script)], cwd=REPO, env=environment, text=True, capture_output=True)
                self.assertEqual(result.returncode, 23)
                self.assertIn('missing_torch', result.stderr)
                self.assertEqual(result.stdout, '')

    def test_cli_default_directory_and_explicit_overrides(self):
        count = len(self.cells) * len(STATIONS)
        with patch('tools.generate_configs.generate', return_value=[{}] * count) as generated, contextlib.redirect_stdout(io.StringIO()):
            main(['--family', self.family])
        self.assertEqual(generated.call_args.args, (self.config_dir,))
        self.assertEqual(generated.call_args.kwargs['family'], self.family)
        with tempfile.TemporaryDirectory() as temporary, contextlib.redirect_stdout(io.StringIO()):
            output = Path(temporary) / 'configs'
            result_root = Path(temporary) / 'fresh_results'
            main(['--family', self.family, '--output', str(output), '--results-root', str(result_root)])
            self.assertEqual(len(list(output.glob('train_config_*.sh'))), count)
            with (output / 'manifest.csv').open() as handle:
                self.assertEqual({row['result_root'] for row in csv.DictReader(handle)}, {str(result_root)})
            self.assertFalse(result_root.exists())
            with self.assertRaises(ValueError):
                generate(output, family=self.family, include_severity_shape=True)
            with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
                main(['--family', self.family, '--include-severity-shape'])


class WQEFactorialConfigTests(FactorialConfigChecks, unittest.TestCase):
    family = FACTORIAL_CONFIG_DIR
    head = 'dual'
    cells = tuple(f'G{g}_E{e}_T{t}' for g, e, t in itertools.product((0, 1), (0, 1), (0, 1, 2)))
    results_root = FACTORIAL_RESULTS_ROOT
    branch_weights = (1., 2., .5)
    runtime_variable = 'WQEF_PYTHON_BIN'
    job_prefix = 'WQEF'
    control_directory = 'wqe'
    control_suffix = 'W1_GlobalWQE'


class SingleFactorialConfigTests(FactorialConfigChecks, unittest.TestCase):
    family = S0_CONFIG_DIR
    head = 'single'
    cells = tuple(f'G{g}_T{t}' for g, t in itertools.product((0, 1), (0, 1, 2)))
    results_root = S0_RESULTS_ROOT
    branch_weights = (1., 1., 1.)
    runtime_variable = 'S0_PYTHON_BIN'
    job_prefix = 'S0F'
    control_directory = 'baseline_ablation'
    control_suffix = 'S0_Single'

    def test_regeneration_removes_only_the_renamed_legacy_configs(self):
        mapping = {'S0_Single': 'G0_T0', 'S0_Single_WQE': 'G1_T0',
                   'S0_Single_Tail': 'G0_T1', 'S0_Single_Tail_WQE': 'G1_T1'}
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            for station, suffix in itertools.product(STATIONS, mapping):
                (output / f'train_config_NCEP_{station}_{suffix}.sh').touch()
            unrelated = output / 'user_notes.sh'
            unrelated.write_text('# preserve custom files\n')
            generate(output, family=self.family)
            self.assertEqual(unrelated.read_text(), '# preserve custom files\n')
            self.assertEqual(len(list(output.glob('train_config_*.sh'))), 24)
            for station, (suffix, cell) in itertools.product(STATIONS, mapping.items()):
                self.assertFalse((output / f'train_config_NCEP_{station}_{suffix}.sh').exists())
                args = dry_run(output / f'train_config_NCEP_{station}_{cell}.sh')[0][0]
                self.assertEqual(args.loss_mode, 'wqe' if cell.startswith('G1') else 'mse')
                self.assertEqual(args.exceedance_loss_mode, 'mse')
                self.assertEqual(args.exceedance_loss_weight, .025 if cell.endswith('T1') else 0.)


if __name__ == '__main__':
    unittest.main()
