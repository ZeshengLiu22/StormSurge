"""Both checkpoint roles follow one trajectory; final TEST cannot select epochs."""

import contextlib
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

import train
from emulator.training.engine import EpochResult
from emulator.training.metrics import evaluate_metrics
from test_pipeline import make_fixture


VALUES = ((1., 2., 4.), (1.02, 1., 2.))


def arguments(graphs, stations, output, variant='single', mode='overall'):
    args = ['--root_dir', str(graphs), '--station', 'Battery', '--station_json_dir', str(stations),
            '--output_dir', str(output), '--model', 'perceiver3', '--head_type', 'single' if variant == 'single' else 'dual',
            '--hidden_channels', '16', '--node_read_heads', '2', '--time_read_heads', '2', '--history_hours', '12',
            '--epochs', '2', '--batch_size', '2', '--device', 'cpu', '--num_workers', '0', '--warmup_epochs', '0',
            '--checkpoint_selection', mode]
    if variant != 'single':
        args += ['--excess_formulation', variant, '--excess_amp_loss_weight', '.3']
    if variant == 'severity_shape':
        args += ['--shape_loss_weight', '.2']
    return args


def metric(values):
    result = evaluate_metrics(np.array([[0., 2., 1., 4.]]), np.array([[0., 1., 3., 4.]]), 2.)
    result.update(zip(('all_rmse', 'exceedance_rmse', 'gt_aligned_peak_rmse'), values))
    return result


def recorded_run(root, mode='overall', variant='single', test_score=999.):
    torch.manual_seed(824)
    graphs, stations = make_fixture(root)
    output = root / 'run'
    state = dict(train_calls=0, val_calls=0, final_calls=[], snapshots={})
    real_epoch = train.run_epoch

    def epoch(model, loader, **kwargs):
        if kwargs.get('save_predictions'):
            role = 'overall' if len(state['final_calls']) < 2 else 'eventaware'
            split = 'val' if len(state['final_calls']) % 2 == 0 else 'test'
            state['final_calls'].append((role, split))
            assert kwargs.get('optimizer') is None
            assert state['train_calls'] == state['val_calls'] == 2
            winner = 1 if role == 'overall' else 2
            for key, value in model.state_dict().items():
                torch.testing.assert_close(value, state['snapshots'][winner][key], rtol=0, atol=0)
            result = real_epoch(model, loader, **kwargs)
            if split == 'test':
                result.metrics.update(zip(('all_rmse', 'exceedance_rmse', 'gt_aligned_peak_rmse'), [test_score] * 3))
            return result
        if kwargs.get('optimizer') is not None:
            state['train_calls'] += 1
            with torch.no_grad():
                for parameter in model.parameters():
                    parameter.fill_(state['train_calls'] / 100.)
            state['snapshots'][state['train_calls']] = {key: value.detach().clone() for key, value in model.state_dict().items()}
        else:
            state['val_calls'] += 1
        return EpochResult(metric(VALUES[state['train_calls'] - 1]))

    console = io.StringIO()
    with contextlib.redirect_stdout(console), patch.object(train, 'run_epoch', side_effect=epoch), \
            patch.object(torch.optim.lr_scheduler.LambdaLR, 'step'), \
            patch.object(torch.optim.Adam, 'step', side_effect=AssertionError('mocked training')):
        train.main(arguments(graphs, stations, output, variant, mode))
    state['console'] = console.getvalue()
    return output, state


def tensor_digest(tensors):
    digest = hashlib.sha256()
    for name, tensor in sorted(tensors.items()):
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


class CheckpointPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_two_roles_all_formulations_both_final_splits_and_primary_alias(self):
        for variant in ('single', 'direct', 'severity_shape'):
            for mode in ('overall', 'eventaware'):
                with self.subTest(variant=variant, mode=mode), tempfile.TemporaryDirectory() as temporary:
                    output, state = recorded_run(Path(temporary), mode, variant)
                    self.assertEqual((state['train_calls'], state['val_calls']), (2, 2))
                    self.assertEqual(state['final_calls'], [('overall', 'val'), ('overall', 'test'), ('eventaware', 'val'), ('eventaware', 'test')])
                    summary = json.loads(next(output.glob('summary_*.json')).read_text())
                    self.assertEqual(summary['best_epoch'], 1 if mode == 'overall' else 2)
                    selected = torch.load(output / 'best.pt', weights_only=False)
                    self.assertEqual(selected['epoch'], summary['best_epoch'])
                    self.assertEqual(torch.load(next(output.glob('best_*.pth')), weights_only=False)['epoch'], selected['epoch'])
                    for role, epoch in (('overall', 1), ('eventaware', 2)):
                        checkpoint = torch.load(output / f'best_{role}.pt', weights_only=False)
                        self.assertEqual(checkpoint['epoch'], epoch)
                        self.assertEqual(checkpoint['val'], metric(VALUES[epoch - 1]))
                        for split in ('val', 'test'):
                            reported = summary['checkpoint_comparison'][role][split]
                            self.assertIn('episode_n', reported)
                            self.assertIn('exceedance_rmse_lead_0', reported)
                            with np.load(output / f'{split}_predictions_{role}.npz') as exported:
                                self.assertIn('target_timestamps', exported.files)
                                self.assertEqual(exported['tau_physical'], checkpoint['tau_physical'])
                    logs = [json.loads(line) for line in next(output.glob('metrics_*.jsonl')).read_text().splitlines()]
                    self.assertEqual(len(logs), 2)
                    self.assertAlmostEqual(logs[0]['eventaware_score'], 1.65)
                    self.assertAlmostEqual(logs[1]['eventaware_score'], 1.163)
                    report = (output / 'checkpoint_comparison.md').read_text()
                    comparison = json.loads((output / 'checkpoint_comparison.json').read_text())
                    self.assertEqual(comparison['threshold_metadata'], summary['threshold_metadata'])
                    self.assertIn('strict exceedance y > tau', comparison['method'])
                    self.assertIn('TRAIN hourly target Q95', comparison['method'])
                    for label in ('AllRMSE', 'ExceedanceRMSE', 'WindowPeakRMSE', 'GTAlignedPeakRMSE', 'EpisodePeakRMSE'):
                        self.assertIn(label, report)

    def test_test_values_cannot_change_either_selected_epoch(self):
        selected = []
        for score in (0., 1e12):
            with tempfile.TemporaryDirectory() as temporary:
                output, state = recorded_run(Path(temporary), 'eventaware', test_score=score)
                summary = json.loads(next(output.glob('summary_*.json')).read_text())
                selected.append({role: value['epoch'] for role, value in summary['checkpoint_selection']['checkpoints'].items()})
                self.assertEqual(summary['test']['all_rmse'], score)
        self.assertEqual(selected, [{'overall': 1, 'eventaware': 2}] * 2)

    def test_primary_role_does_not_change_real_cpu_training_trajectory(self):
        for variant in ('single', 'direct', 'severity_shape'):
            trajectories = []
            for mode in ('overall', 'eventaware'):
                with tempfile.TemporaryDirectory() as temporary:
                    torch.manual_seed(824)
                    root = Path(temporary)
                    graphs, stations = make_fixture(root)
                    history = []
                    real_epoch = train.run_epoch
                    def epoch(model, loader, **kwargs):
                        result = real_epoch(model, loader, **kwargs)
                        if kwargs.get('optimizer') is not None:
                            history.append(dict(metrics=result.metrics, weights=tensor_digest(model.state_dict()),
                                                rng=tensor_digest({'rng': torch.get_rng_state()})))
                        elif not kwargs.get('save_predictions'):
                            result.metrics.update(metric(VALUES[len(history) - 1]))
                        return result
                    with contextlib.redirect_stdout(io.StringIO()), patch.object(train, 'run_epoch', side_effect=epoch):
                        train.main(arguments(graphs, stations, root / 'run', variant, mode))
                    trajectories.append(history)
            self.assertEqual(trajectories[0], trajectories[1], variant)


if __name__ == '__main__':
    unittest.main()
