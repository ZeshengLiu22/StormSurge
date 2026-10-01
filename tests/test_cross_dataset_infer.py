"""Pair config generation, dry runs, sequential execution, and replay provenance."""
import importlib.util
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile
import unittest

import infer

REPO = Path(__file__).resolve().parents[1]
TREE = REPO / 'configs/cross_dataset_infer'
spec = importlib.util.spec_from_file_location('cross_configs', TREE / 'generate_configs.py')
generator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(generator)


class CrossDatasetConfigTests(unittest.TestCase):
    def test_generated_pairs_manifest_and_exact_populations(self):
        checkpoints = {(group, source): (generator.FORMAL / run / 'best_overall.pt').resolve()
                       for group, runs in generator.RUNS.items() for source, run in runs.items()}
        with tempfile.TemporaryDirectory() as temporary:
            rows = generator.generate(temporary, checkpoints)
            self.assertEqual(len(rows), 66)
            self.assertEqual((Path(temporary) / 'manifest.csv').read_bytes(), (TREE / 'manifest.csv').read_bytes())
            for group, target_count, year_count in (('past_only', 6, 7), ('future_year', 5, 30)):
                selected = [row for row in rows if row['group'] == group]
                self.assertEqual(len(selected), 6 * target_count)
                for source in generator.SOURCES:
                    pairs = list((TREE / group / source).iterdir())
                    self.assertEqual(len(pairs), target_count)
                    self.assertTrue(all(p.suffix == '.sh' for p in pairs))
                for row in selected:
                    self.assertEqual(row['year_count'], year_count)
                    self.assertEqual(row['years'].split(','), generator.YEARS[group])
                    self.assertEqual(row['threshold_origin'], 'source_checkpoint_train')
                    self.assertTrue(row['target_root'])
                    config = REPO / row['config_path']
                    generated = Path(temporary) / config.relative_to(TREE)
                    self.assertEqual(config.read_bytes(), generated.read_bytes())
                    self.assertIn('TEST_ROOT_DIR="${TARGET_ROOT}"', config.read_text())
                    self.assertNotIn('BATCH_SIZE', config.read_text())
                    self.assertNotIn('*', row['checkpoint'])
                self.assertEqual((Path(temporary) / group / 'common.sh').read_bytes(), (TREE / group / 'common.sh').read_bytes())

    def launcher_fixture(self, root, *, external=True):
        checkpoint = root / 'best_overall.pt'
        checkpoint.touch()
        script = root / 'capture.py'
        script.write_text("import json, pathlib, sys\n"
                          "out=pathlib.Path(sys.argv[sys.argv.index('--out_dir')+1])\n"
                          "(out/'argv.json').write_text(json.dumps(sys.argv[1:]))\n")
        settings = dict(CKPT_PATH=checkpoint, PYTHON_BIN=sys.executable, INFER_PY=script,
                        INFERENCE_RESULTS_ROOT=root / 'results', ROOT_DIR='./source/graphs',
                        DO_CONDA=0, USE_TMUX=0, TORCH_GPU_PROBE=0, STATION='Boston',
                        MODEL='perceiver3', HEAD_TYPE='single', HISTORY_HOURS=24,
                        USE_SITE_ELEVATION=0, YEARS='2070_2071', STRICT_YEARS=1)
        if external:
            settings.update(SOURCE_NAME='NCEP', TARGET_NAME='AWI', SOURCE_ROOT='./source/graphs',
                            TARGET_ROOT='./target/graphs', EXPERIMENT_GROUP='future_year')
        config = root / 'pair.sh'
        config.write_text(''.join(f'{key}={shlex.quote(str(value))}\n' for key, value in settings.items()))
        return config

    def launch(self, config, dry=False):
        return subprocess.run(['bash', str(REPO / 'infer.sh'), str(config)], cwd=REPO,
                              env=dict(os.environ, DRY_RUN=str(int(dry))), text=True, capture_output=True)

    def test_launcher_dry_run_has_no_outputs_and_requires_explicit_target(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = self.launcher_fixture(root)
            result = self.launch(config, dry=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn('source_checkpoint_train', result.stdout)
            self.assertIn('Strict years:  1', result.stdout)
            self.assertFalse((root / 'results').exists())
            config.write_text(config.read_text() + 'TARGET_ROOT=""\n')
            result = self.launch(config, dry=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('require a nonempty TARGET_ROOT', result.stdout)

    def test_launcher_passes_pair_arguments_and_replays_internal_scope(self):
        for external in (True, False):
            with self.subTest(external=external), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                config = self.launcher_fixture(root, external=external)
                result = self.launch(config)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                captured = next((root / 'results').rglob('argv.json'))
                args = infer.parse_args(json.loads(captured.read_text()))
                self.assertTrue(args.strict_years)
                self.assertEqual(args.years, '2070_2071')
                self.assertEqual(bool(args.test_root_dir), external)
                if external:
                    self.assertEqual((args.source_name, args.target_name), ('NCEP', 'AWI'))
                    self.assertEqual(args.test_root_dir, str(REPO / 'target/graphs'))
                snapshot = next((root / 'results').rglob('infer_config_used.sh'))
                self.assertIn('STRICT_YEARS=1', snapshot.read_text())
                command = next((root / 'results').rglob('command_used.sh'))
                self.assertIn('--strict_years', command.read_text())
                replay = self.launch(snapshot)
                self.assertEqual(replay.returncode, 0, replay.stdout + replay.stderr)
                for path in (root / 'results').rglob('argv.json'):
                    replay_args = infer.parse_args(json.loads(path.read_text()))
                    self.assertEqual(bool(replay_args.test_root_dir), external)

    def test_group_order_foreground_failure_and_count_guard(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            tree = root / 'configs/cross_dataset_infer'
            source = tree / 'future_year/NCEP'
            source.mkdir(parents=True)
            shutil.copy(TREE / 'run_group.sh', tree / 'run_group.sh')
            checkpoint = root / 'best_overall.pt'
            checkpoint.touch()
            for target in ('MRI', 'MPI', 'EC_EARTH', 'CNRM', 'AWI'):
                (source / f'NCEP_to_{target}.sh').write_text(
                    f'SOURCE_NAME=NCEP\nTARGET_NAME={target}\nSOURCE_ROOT=source\nROOT_DIR=source\n'
                    f'TARGET_ROOT=target\nTEST_ROOT_DIR=target\nSTRICT_YEARS=1\nEXPERIMENT_GROUP=future_year\n'
                    f'YEARS={",".join(generator.YEARS["future_year"])}\nCKPT_PATH={shlex.quote(str(checkpoint))}\n')
            (root / 'infer.sh').write_text(
                '#!/usr/bin/env bash\nset -eu\nsource "$1"\n[[ "$USE_TMUX" == 0 ]]\n'
                'echo "$TARGET_NAME:$DRY_RUN" >> calls\n[[ "$TARGET_NAME" != CNRM ]]\n')
            result = subprocess.run(['bash', str(tree / 'run_group.sh'), str(source)],
                                    env=dict(os.environ, DRY_RUN='1'), text=True, capture_output=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual((root / 'calls').read_text().splitlines(), ['AWI:1', 'CNRM:1'])
            self.assertIn('[1/5] NCEP -> AWI', result.stdout)
            (root / 'calls').unlink()
            (source / 'NCEP_to_MRI.sh').unlink()
            result = subprocess.run(['bash', str(tree / 'run_group.sh'), str(source)], text=True, capture_output=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('Expected 5 pair configs; found 4', result.stderr)
            self.assertFalse((root / 'calls').exists())


if __name__ == '__main__':
    unittest.main()
