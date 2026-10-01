"""Year/group reports preserve evaluation populations and original file formats."""

import contextlib
from dataclasses import asdict
import io
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch_geometric.data import Data

import infer
import train
from emulator.data import ForcingGraphStore
from emulator.models import ModelConfig, build_model


class InferenceReportingTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(suffix='_dataset')
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.graphs = self.root / 'graphs'
        self.graphs.mkdir()
        self.truth = {}
        for index, (year, count) in enumerate(((1979, 1), (2014, 2), (2050, 3), (2070, 4), (2099, 1), (2100, 2))):
            values = [[index + 1., -(i + .25)] for i in range(count)]
            data = [Data(x=torch.ones(4, 3), y=torch.tensor(value),
                         edge_index=torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]]),
                         center_time=f"{year}-11-01 {j*2:02d}:00:00") for j, value in enumerate(values)]
            torch.save(data, self.graphs / f'{year}_{year + 1}_Battery_fixture_graphs.pt')
            self.truth[f'{year}_{year + 1}'] = np.array(values, dtype=np.float32)
        config = ModelConfig(3, 2, model='baseline', hidden_channels=8, head_type='single', history_steps=0)
        model = build_model(config)
        for parameter in model.parameters():
            parameter.data.zero_()
        training = vars(train.parse_args(['--model', 'baseline', '--history_hours', '0', '--station', 'Battery']))
        store = ForcingGraphStore(self.graphs, 'Battery')
        test_tags = [tag for tag in store.graph_tags if tag.startswith(('1979_', '2070_'))]
        self.ckpt = self.root / 'model.pth'
        torch.save(dict(model_config=asdict(config), model_state=model.state_dict(), training_config=training,
                        station='Battery', station_feat=None, split_tags=dict(test=test_tags),
                        threshold_metadata=dict(tau_physical=3., exceedance_percentile=95.,
                            metric_schema='hourly_q95_v1', threshold_schema='train_hourly_q95_v1'),
                        normalization=dict(x_center=torch.zeros(3), x_scale=torch.ones(3),
                                           y_mean=torch.tensor([.25, -.5]), y_std=torch.ones(2))), self.ckpt)

    def evaluate(self, name, options=()):
        out = self.root / name
        console = io.StringIO()
        with contextlib.redirect_stdout(console), patch.object(infer, 'run_epoch', wraps=infer.run_epoch) as evaluated:
            infer.main(['--ckpt', str(self.ckpt), '--root_dir', str(self.graphs), '--out_dir', str(out),
                        '--device', 'cpu', '--batch_size', '2', '--num_workers', '0', *options])
        report = json.loads(next(out.glob('metrics_per_year_*.json')).read_text())
        self.assertEqual(evaluated.call_count, len(report['years_evaluated']))
        self.assertTrue(all(call.kwargs['save_predictions'] for call in evaluated.call_args_list))
        for line in console.getvalue().splitlines():
            self.assertRegex(line, r'^\[\d{4}-\d{2}-\d{2}\|\d{2}:\d{2}:\d{2}\]')
        self.assertIn('Wall time:', console.getvalue())
        return report, out

    def check_metrics(self, metrics, years):
        error = np.array([.25, -.5]) - np.concatenate([self.truth[year] for year in years])
        self.assertAlmostEqual(metrics['rmse'], float(np.sqrt(np.mean(error ** 2))), places=6)
        self.assertAlmostEqual(metrics['mae'], float(np.mean(np.abs(error))), places=6)
        self.assertEqual(metrics['unit'], 'meters')

    def test_all_years_past_future_sample_weighting_and_original_files(self):
        options = ['--test_root_dir', str(self.graphs)]
        plain, plain_dir = self.evaluate('plain', options)
        saved, saved_dir = self.evaluate('saved', [*options, '--save_npz'])
        self.assertEqual(plain['evaluation_scope'], 'external_all_years')
        self.assertIsNone(plain['split_parameters'])
        self.assertEqual(plain['years_evaluated'], sorted(self.truth))
        for year, truth in self.truth.items():
            self.assertEqual(plain['results'][year]['samples'], len(truth))
            self.check_metrics(plain['results'][year], [year])
        for key, years in (('_overall', list(self.truth)), ('_overall_past', ['1979_1980', '2014_2015']),
                           ('_overall_future', ['2070_2071', '2099_2100'])):
            self.check_metrics(plain['results'][key], years)
            self.assertEqual(plain['results'][key], saved['results'][key])
        timing = plain['results']['_avg_time_per_year_excl_2014_2015']
        self.assertEqual(timing['n_years'], 5)
        expected_seconds = np.mean([plain['results'][year]['seconds'] for year in self.truth if year != '2014_2015'])
        self.assertEqual(timing['seconds'], expected_seconds)
        self.assertFalse(list(plain_dir.glob('*.npz')))
        exported = saved_dir / 'predictions.npz'
        self.assertTrue(exported.exists())
        with np.load(exported) as arrays:
            self.assertTrue({'y_true', 'y_pred', 'tags', 'target_timestamps', 'tau_physical'}.issubset(arrays.files))
            np.testing.assert_array_equal(arrays['y_true'], np.concatenate(list(self.truth.values())))
            self.assertEqual(len(arrays['tags']), 13)
            self.assertEqual(arrays['tau_physical'].item(), 3.)
        current = json.loads((plain_dir / 'metrics.json').read_text())
        self.assertEqual(current['samples'], 13)
        self.assertAlmostEqual(current['metrics']['all_rmse'], plain['results']['_overall']['rmse'])

    def test_saved_scope_year_filter_and_empty_group(self):
        report, out = self.evaluate('held_out')
        self.assertEqual(report['years_evaluated'], ['1979_1980', '2070_2071'])
        self.assertTrue((out / f'metrics_per_year_{self.root.name}_Battery_baseline.json').exists())
        self.assertEqual(report['evaluation_scope'], 'held_out_years')
        filtered, _ = self.evaluate('filtered', ['--years', '1979_1980'])
        self.assertEqual(filtered['years_evaluated'], ['1979_1980'])
        self.assertIsNone(filtered['results']['_overall_future']['rmse'])
        boundary, _ = self.evaluate('boundary', ['--scope', 'all', '--years', '2014_2015'])
        self.check_metrics(boundary['results']['_overall'], ['2014_2015'])
        timing = boundary['results']['_avg_time_per_year_excl_2014_2015']
        self.assertEqual(timing['n_years'], 0)
        self.assertIsNone(timing['seconds'])

    def test_original_automatic_directory_and_dataset_labels(self):
        from emulator.inference import infer_dataset_tag
        for path, expected in (('./Data/NCEP/graphs/', 'NCEP'), ('/root/CMIP6_MPI/GRAPHS/', 'CMIP6_MPI'),
                               ('/root/external', 'external'), (None, 'data')):
            self.assertEqual(infer_dataset_tag(path), expected)
        checkpoint = self.ckpt.with_name(f'{self.root.name}_Battery_P3_Best.pth')
        checkpoint.write_bytes(self.ckpt.read_bytes())
        outputs = self.root / 'automatic'
        with contextlib.redirect_stdout(io.StringIO()):
            infer.main(['--ckpt', str(checkpoint), '--root_dir', str(self.graphs), '--num_workers', '0',
                        '--device', 'cpu', '--inference_results_root', str(outputs)])
        directory = next(outputs.iterdir())
        self.assertRegex(directory.name, rf'^Battery_P3_Best_{self.root.name}_To_{self.root.name}_\d{{8}}_\d{{6}}$')
        report = json.loads(next((directory / 'outputs').glob('metrics_per_year_*.json')).read_text())
        self.assertEqual(report['model_label'], 'P3_Best')
        self.assertEqual(report['inference_args']['model_label'], 'P3_Best')

    def test_scope_all_keeps_saved_split_boundaries_and_external_uses_own_series(self):
        graphs = self.root / 'contiguous'
        graphs.mkdir()
        for year, center in ((2000, '2000-12-31 22:00:00'), (2001, '2001-01-01 00:00:00')):
            graph = Data(x=torch.ones(4, 3), y=torch.tensor([5., 5.]),
                         edge_index=torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]]), center_time=center)
            torch.save([graph], graphs / f'{year}_{year+1}_Battery_fixture_graphs.pt')
        checkpoint = torch.load(self.ckpt, weights_only=False)
        checkpoint['split_tags'] = dict(train=['2000_2001_Battery_fixture_0'],
                                        val=['2001_2002_Battery_fixture_0'], test=[])
        torch.save(checkpoint, self.ckpt)
        for external, expected in ((False, 2), (True, 1)):
            out = self.root / f'boundary_{external}'
            args = ['--ckpt', str(self.ckpt), '--root_dir', str(graphs), '--out_dir', str(out),
                    '--device', 'cpu', '--num_workers', '0', '--save_npz']
            args += ['--test_root_dir', str(graphs)] if external else ['--scope', 'all']
            with contextlib.redirect_stdout(io.StringIO()):
                infer.main(args)
            report = json.loads((out / 'metrics.json').read_text())
            self.assertEqual(report['scope'], 'external_all_years' if external else 'source_all_years')
            self.assertEqual(report['tau_physical'], 3.)
            with np.load(out / 'predictions.npz') as data:
                from emulator.training.metrics import gt_event_episodes, METRIC_KEYS
                self.assertEqual(set(report['metrics']), set(METRIC_KEYS))
                self.assertEqual(len(gt_event_episodes(data['y_true'], data['target_timestamps'], 3., split_ids=data['split_ids'])), expected)
                self.assertEqual(data['split'].item(), 'external' if external else 'mixed')
                self.assertEqual(set(data['split_ids']), {'external'} if external else {'train', 'val'})

    def test_empty_source_test_requires_external_or_explicit_all(self):
        checkpoint = torch.load(self.ckpt, weights_only=False)
        checkpoint['split_tags']['test'] = []
        torch.save(checkpoint, self.ckpt)
        for scope in ([], ['--scope', 'test']):
            with self.subTest(scope=scope), contextlib.redirect_stdout(io.StringIO()), \
                    patch.object(infer, 'run_epoch') as run, \
                    patch.object(infer, 'ForcingGraphStore') as store:
                with self.assertRaisesRegex(ValueError, 'Checkpoint has no held-out source TEST split.*--test_root_dir.*--scope all'):
                    infer.main(['--ckpt', str(self.ckpt), '--root_dir', str(self.graphs), '--device', 'cpu', *scope])
                run.assert_not_called()
                store.assert_not_called()
        for name, options in (('no_test_external', ['--test_root_dir', str(self.graphs)]),
                              ('no_test_all', ['--scope', 'all'])):
            report, _ = self.evaluate(name, options)
            self.assertEqual(report['evaluated_years'], sorted(self.truth))

    def test_strict_years_fail_before_forward_and_preserve_permissive_default(self):
        cases = [
            ('1979_1980,2012_2013', 'missing from target: 2012_2013', True),
            ('1979_1980,1979_1980', 'Duplicate requested', True),
            ('1979_1981', 'Invalid requested', True),
            (', ,', 'at least one', True),
            ('2014_2015', 'do not match requested years within the selected scope', False),
        ]
        for years, error, external in cases:
            out = self.root / 'strict_failure'
            options = ['--ckpt', str(self.ckpt), '--root_dir', str(self.graphs), '--out_dir', str(out),
                       '--device', 'cpu', '--strict_years', '--years', years]
            if external:
                options += ['--test_root_dir', str(self.graphs)]
            with self.subTest(years=years), contextlib.redirect_stdout(io.StringIO()), \
                    patch.object(infer, 'run_epoch') as run:
                with self.assertRaisesRegex(ValueError, error):
                    infer.main(options)
                run.assert_not_called()
                self.assertFalse(out.exists())
        report, _ = self.evaluate('permissive', ['--years', '1979_1980,2012_2013'])
        self.assertEqual(report['requested_year_count'], 2)
        self.assertEqual(report['evaluated_years'], ['1979_1980'])
        self.assertFalse(report['strict_years'])
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            infer.parse_args(['--ckpt', str(self.ckpt), '--root_dir', str(self.graphs), '--strict_years'])

    def test_strict_external_provenance_retains_source_stats_and_tau(self):
        checkpoint = torch.load(self.ckpt, weights_only=False)
        checkpoint['split_tags']['test'] = []
        torch.save(checkpoint, self.ckpt)
        source = self.root / 'unmounted_source'
        out = self.root / 'strict_external'
        with contextlib.redirect_stdout(io.StringIO()), \
                patch('emulator.data.fit_statistics', side_effect=AssertionError('target normalization fit')), \
                patch('emulator.data.fit_loss_thresholds', side_effect=AssertionError('target threshold fit')), \
                patch.object(infer, 'run_epoch', wraps=infer.run_epoch) as run:
            infer.main(['--ckpt', str(self.ckpt), '--root_dir', str(source), '--test_root_dir', str(self.graphs),
                        '--source_name', 'NCEP', '--target_name', 'AWI', '--out_dir', str(out),
                        '--years', '2099_2100, 2070_2071', '--strict_years', '--save_npz',
                        '--device', 'cpu', '--num_workers', '0', '--batch_size', '2'])
        self.assertEqual(run.call_count, 2)
        for call in run.call_args_list:
            self.assertEqual(call.kwargs['tau_physical'], checkpoint['threshold_metadata']['tau_physical'])
            for key, value in checkpoint['normalization'].items():
                torch.testing.assert_close(call.args[3][key], value, rtol=0, atol=0)
        expected = dict(source_name='NCEP', target_name='AWI', source_root=str(source),
                        target_root=str(self.graphs), checkpoint=str(self.ckpt),
                        requested_years=['2070_2071', '2099_2100'], evaluated_years=['2070_2071', '2099_2100'],
                        requested_year_count=2, evaluated_year_count=2, strict_years=True,
                        evaluation_threshold_origin='source_checkpoint_train', source_tau_physical=3.,
                        source_exceedance_percentile=95., tau_physical=3., exceedance_percentile=95.,
                        threshold_schema='train_hourly_q95_v1', normalization_origin='source_checkpoint_train')
        metrics = json.loads((out / 'metrics.json').read_text())
        report = json.loads(next(out.glob('metrics_per_year_*.json')).read_text())
        with np.load(out / 'predictions.npz', allow_pickle=False) as arrays:
            for document in (metrics, report, arrays):
                for key, value in expected.items():
                    np.testing.assert_equal(document[key], value, err_msg=key)
            np.testing.assert_allclose(arrays['y_pred'], np.tile([.25, -.5], (5, 1)))
            from emulator.training.metrics import evaluate_metrics
            reference = evaluate_metrics(arrays['y_pred'], arrays['y_true'], 3.,
                                         target_timestamps=arrays['target_timestamps'], split_ids=arrays['split_ids'])
            self.assertEqual(metrics['metrics'], reference)
            self.assertEqual(set(arrays['split_ids']), {'external'})
        self.assertEqual(metrics['threshold_metadata'], checkpoint['threshold_metadata'])
