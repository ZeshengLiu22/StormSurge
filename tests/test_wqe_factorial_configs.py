"""Fresh matched factorial generation, resolved dry runs, and safe launch order."""

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

from emulator.training.eventaware_checkpoints import ROLES
from tools.generate_configs import (FACTORIAL_CONFIG_DIR, FACTORIAL_RESULTS_ROOT,
                                    generate, main)
from test_wqe_configs import PARAMETERS, REPO, dry_run


FAMILY = "wqe_factorial_multickpt"
STATIONS = ("CBBT", "Boston", "Battery", "Lewes")
CELLS = tuple(f"G{g}_E{e}_T{t}" for g, e, t in itertools.product((0, 1), repeat=3))
CONFIG_DIR = REPO / "configs" / FACTORIAL_CONFIG_DIR


class WQEFactorialConfigTests(unittest.TestCase):
    def test_all_32_configs_reproduce_and_resolve_matched_settings(self):
        controls = {station: vars(dry_run(REPO / "configs/wqe" / f"train_config_NCEP_{station}_W1_GlobalWQE.sh")[0][0])
                    for station in STATIONS}
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            rows = generate(output, family=FAMILY)
            self.assertEqual(len(rows), 32)
            self.assertEqual(len(list(output.glob('train_config_*.sh'))), 32)
            self.assertEqual(len(list(CONFIG_DIR.glob('train_config_*.sh'))), 32)
            self.assertEqual([(row['cell'], row['station']) for row in rows],
                             list(itertools.product(CELLS, STATIONS)))
            self.assertEqual((output / 'manifest.csv').read_text(), (CONFIG_DIR / 'manifest.csv').read_text())
            with (output / 'manifest.csv').open() as handle:
                manifest = list(csv.DictReader(handle))
            for row, record in zip(rows, manifest):
                with self.subTest(station=row['station'], cell=row['cell']):
                    path = output / row['config']
                    self.assertEqual(path.read_text(), (CONFIG_DIR / row['config']).read_text())
                    commands, log = dry_run(path)
                    self.assertEqual(len(commands), 1)
                    args = commands[0]
                    g, e, t = (int(part[1:]) for part in row['cell'].split('_'))
                    expected_modes = (('mse', 'wqe')[g], ('mse', 'wqe')[e])
                    self.assertEqual((args.loss_mode, args.excess_loss_mode), expected_modes)
                    self.assertEqual((row['global_loss'], row['excess_loss']), expected_modes)
                    self.assertEqual((row['tail_enabled'], args.exceedance_loss_weight), (t, .025 if t else 0.))
                    self.assertEqual(args.exceedance_loss_weight, float(record['exceedance_loss_weight']))
                    self.assertEqual((args.body_loss_weight, args.excess_loss_weight, args.gate_loss_weight), (1., 2., .5))
                    self.assertEqual((args.excess_amp_loss_weight, args.shape_loss_weight), (0., 0.))
                    self.assertIn('EXCESS_AMP_LOSS_WEIGHT=0\n', path.read_text())
                    self.assertIn('SHAPE_LOSS_WEIGHT=0\n', path.read_text())
                    self.assertEqual((args.checkpoint_selection, args.ckpt_w_all, args.ckpt_w_exceedance, args.ckpt_w_peak),
                                     ('overall', .65, .20, .15))
                    for name, value in PARAMETERS.items():
                        self.assertEqual(getattr(args, name), value)
                    run_dir = Path(args.output_dir)
                    self.assertEqual(run_dir.parent, Path(FACTORIAL_RESULTS_ROOT))
                    self.assertRegex(run_dir.name, rf'^NCEP_{row["station"]}_{row["cell"]}__\d{{8}}_\d{{6}}$')
                    self.assertFalse(run_dir.exists())
                    self.assertEqual(record['result_root'], FACTORIAL_RESULTS_ROOT)
                    self.assertIn('Results root:  ' + FACTORIAL_RESULTS_ROOT, log)
                    self.assertIn(f"Checkpoint roles (VAL-only): {', '.join(ROLES)}; primary=overall", log)
                    # The entire resolved config must match the station's current
                    # WQE control except the three factors and output identity.
                    factors = {'loss_mode', 'excess_loss_mode', 'exceedance_loss_weight', 'output_dir', 'run_tag'}
                    self.assertEqual({k: v for k, v in vars(args).items() if k not in factors},
                                     {k: v for k, v in controls[row['station']].items() if k not in factors})

    def test_launchers_have_exact_order_and_execute_only_mock_queue(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rows = generate(root / 'configs', family=FAMILY)
            output = root / 'configs'
            # Never invoke the real scheduler in tests.
            queue = root / 'qsub_local'
            queue.write_text('#!/usr/bin/env python3\nimport json, os, sys\n'
                             'print(json.dumps(dict(args=sys.argv[1:], cwd=os.getcwd())))\n')
            queue.chmod(0o755)
            environment = dict(os.environ, PATH=str(root) + os.pathsep + os.environ['PATH'])
            for name, selected_rows in [('launch_all.sh', rows),
                    *((f'launch_{cell.replace("_", "")}.sh', [row for row in rows if row['cell'] == cell]) for cell in CELLS)]:
                script = output / name
                checked = CONFIG_DIR / name
                self.assertTrue(os.access(script, os.X_OK))
                self.assertTrue(os.access(checked, os.X_OK))
                for path in (script, checked):
                    subprocess.run(['bash', '-n', str(path)], cwd=REPO, check=True)
                    commands = [shlex.split(line) for line in path.read_text().splitlines() if line.startswith('qsub_local ')]
                    self.assertEqual(len(commands), len(selected_rows))
                    for command, row in zip(commands, selected_rows):
                        self.assertEqual(command[:3], ['qsub_local', 'train.sh',
                                         f'WQEF_{row["station"]}_{row["cell"].replace("_", "")}'])
                        config = Path(command[3])
                        self.assertEqual(config if config.is_absolute() else REPO / config, path.parent / row['config'])
                result = subprocess.run([str(script)], cwd=REPO, env=environment,
                                        capture_output=True, text=True, check=True)
                calls = [json.loads(line) for line in result.stdout.splitlines()]
                self.assertEqual(len(calls), len(selected_rows))
                for call, row in zip(calls, selected_rows):
                    self.assertEqual(call['cwd'], str(REPO))
                    self.assertEqual(call['args'], ['train.sh', f'WQEF_{row["station"]}_{row["cell"].replace("_", "")}',
                                                  str(output / row['config'])])

    def test_cli_default_directory_and_explicit_overrides(self):
        with patch('tools.generate_configs.generate', return_value=[{}] * 32) as generated, contextlib.redirect_stdout(io.StringIO()):
            main(['--family', FAMILY])
        self.assertEqual(generated.call_args.args, (CONFIG_DIR,))
        self.assertEqual(generated.call_args.kwargs['family'], FAMILY)
        with tempfile.TemporaryDirectory() as temporary, contextlib.redirect_stdout(io.StringIO()):
            output = Path(temporary) / 'configs'
            result_root = Path(temporary) / 'fresh_results'
            main(['--family', FAMILY, '--output', str(output), '--results-root', str(result_root)])
            self.assertEqual(len(list(output.glob('train_config_*.sh'))), 32)
            with (output / 'manifest.csv').open() as handle:
                self.assertEqual({row['result_root'] for row in csv.DictReader(handle)}, {str(result_root)})
            self.assertFalse(result_root.exists())
            with self.assertRaisesRegex(ValueError, 'direct D0'):
                generate(output, family=FAMILY, include_severity_shape=True)
            with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
                main(['--family', FAMILY, '--include-severity-shape'])


if __name__ == '__main__':
    unittest.main()
