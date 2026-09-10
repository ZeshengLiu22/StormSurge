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
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.graphs = self.root / 'graphs'
        self.graphs.mkdir()
        self.truth = {}
        for index, (year, count) in enumerate(((1979, 1), (2014, 2), (2050, 3), (2070, 4), (2099, 1), (2100, 2))):
            values = [[index + 1., -(i + .25)] for i in range(count)]
            data = [Data(x=torch.ones(4, 3), y=torch.tensor(value),
                         edge_index=torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]])) for value in values]
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
        self.assertEqual(metrics['unit'], 'physical')

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
        exported = saved_dir / f'preds_{self.root.name}_Battery_baseline_ALLYEARS.npz'
        self.assertTrue(exported.exists())
        with np.load(exported, allow_pickle=True) as arrays:
            self.assertEqual(set(arrays.files), {'y_true', 'y_pred', 'tags'})
            self.assertEqual(arrays['tags'].dtype, object)
            np.testing.assert_array_equal(arrays['y_true'], np.concatenate(list(self.truth.values())))
            self.assertEqual(len(arrays['tags']), 13)
        current = json.loads((plain_dir / 'metrics.json').read_text())
        self.assertEqual(current['samples'], 13)
        self.assertAlmostEqual(current['metrics']['rmse_all'], plain['results']['_overall']['rmse'])

    def test_saved_scope_year_filter_and_empty_group(self):
        report, out = self.evaluate('held_out')
        self.assertEqual(report['years_evaluated'], ['1979_1980', '2070_2071'])
        self.assertTrue((out / f'metrics_per_year_{self.root.name}_Battery_baseline.json').exists())
        self.assertEqual(report['evaluation_scope'], 'held_out_years')
        filtered, _ = self.evaluate('filtered', ['--years', '1979_1980'])
        self.assertEqual(filtered['years_evaluated'], ['1979_1980'])
        self.assertTrue(math.isnan(filtered['results']['_overall_future']['rmse']))
        boundary, _ = self.evaluate('boundary', ['--scope', 'all', '--years', '2014_2015'])
        self.check_metrics(boundary['results']['_overall'], ['2014_2015'])
        timing = boundary['results']['_avg_time_per_year_excl_2014_2015']
        self.assertEqual(timing['n_years'], 0)
        self.assertTrue(math.isnan(timing['seconds']))

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
