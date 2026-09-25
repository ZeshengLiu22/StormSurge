"""All six checkpoint roles follow one trajectory; final TEST cannot select epochs."""

import contextlib
import copy
import random
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
from emulator.training.eventaware_checkpoints import ROLES, SELECTION_METRICS
from emulator.training.metrics import METRIC_KEYS, METRIC_LABELS
from emulator.training.reporting import compact_comparison_report, display
from emulator.training.metrics import evaluate_metrics
from test_pipeline import make_fixture


VALUES = ((0., 10., 10.), (10., 0., 10.), (10., 10., 0.),
          (4., 4., 4.), (7., 4., 2.), (1., 5., 7.))
WINNERS = dict(zip(ROLES, range(1, 7)))


def arguments(graphs, stations, output, variant='single', mode='overall'):
    args = ['--root_dir', str(graphs), '--station', 'Battery', '--station_json_dir', str(stations),
            '--output_dir', str(output), '--model', 'perceiver3', '--head_type', 'single' if variant == 'single' else 'dual',
            '--hidden_channels', '16', '--node_read_heads', '2', '--time_read_heads', '2', '--history_hours', '12',
            '--epochs', str(len(VALUES)), '--batch_size', '2', '--device', 'cpu', '--num_workers', '0', '--warmup_epochs', '0',
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
            assert 'FINAL MULTI-CHECKPOINT RE-EVALUATION' in console.getvalue()
            assert 'Water-level errors and biases are displayed in mm; timing is displayed in hours.' in console.getvalue()
            role = ROLES[len(state['final_calls']) // 2]
            split = 'val' if len(state['final_calls']) % 2 == 0 else 'test'
            state['final_calls'].append((role, split))
            assert kwargs.get('optimizer') is None
            assert state['train_calls'] == state['val_calls'] == len(VALUES)
            winner = WINNERS[role]
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

    def test_six_roles_all_formulations_both_final_splits_and_primary_alias(self):
        for variant in ('single', 'direct', 'severity_shape'):
            for mode in ROLES:
                with self.subTest(variant=variant, mode=mode), tempfile.TemporaryDirectory() as temporary:
                    output, state = recorded_run(Path(temporary), mode, variant)
                    self.assertEqual((state['train_calls'], state['val_calls']), (len(VALUES), len(VALUES)))
                    self.assertEqual(state['final_calls'], [(role, split) for role in ROLES for split in ('val', 'test')])
                    summary = json.loads(next(output.glob('summary_*.json')).read_text())
                    self.assertEqual(summary['best_epoch'], WINNERS[mode])
                    selected = torch.load(output / 'best.pt', weights_only=False)
                    self.assertEqual(selected['epoch'], summary['best_epoch'])
                    self.assertEqual(selected['checkpoint_role'], mode)
                    self.assertEqual(summary['checkpoint_selection']['checkpoint_selection_metric_key'], SELECTION_METRICS[mode])
                    self.assertEqual(summary['val'], summary['checkpoint_comparison'][mode]['val'])
                    self.assertEqual(summary['test'], summary['checkpoint_comparison'][mode]['test'])
                    with np.load(next(output.glob('test_preds_*.npz'))) as primary, np.load(output / f'test_predictions_{mode}.npz') as canonical:
                        self.assertEqual(primary.files, canonical.files)
                        for key in primary.files:
                            np.testing.assert_array_equal(primary[key], canonical[key])
                    self.assertEqual(torch.load(next(output.glob('best_*.pth')), weights_only=False)['epoch'], selected['epoch'])
                    for role, epoch in WINNERS.items():
                        checkpoint = torch.load(output / f'best_{role}.pt', weights_only=False)
                        self.assertEqual(checkpoint['epoch'], epoch)
                        self.assertEqual(checkpoint['val'], metric(VALUES[epoch - 1]))
                        for split in ('val', 'test'):
                            reported = summary['checkpoint_comparison'][role][split]
                            self.assertTrue(set(METRIC_KEYS).issubset(reported))
                            self.assertIn('exceedance_rmse_lead_0', reported)
                            with np.load(output / f'{split}_predictions_{role}.npz') as exported:
                                self.assertIn('target_timestamps', exported.files)
                                self.assertEqual(exported['tau_physical'], checkpoint['tau_physical'])
                    logs = [json.loads(line) for line in next(output.glob('metrics_*.jsonl')).read_text().splitlines()]
                    self.assertEqual(len(logs), len(VALUES))
                    for log, (a, e, p) in zip(logs, VALUES):
                        self.assertEqual(log['eventaware_score'], .65 * a + .20 * e + .15 * p)
                        self.assertEqual(log['equal_score'], (a + e + p) / 3)
                        self.assertEqual(log['peak_priority_score'], (a + 2 * e + 4 * p) / 7)
                    report = (output / 'checkpoint_comparison.md').read_text()
                    comparison = json.loads((output / 'checkpoint_comparison.json').read_text())
                    self.assertEqual(comparison['threshold_metadata'], summary['threshold_metadata'])
                    self.assertEqual(set(summary['checkpoint_comparison']), set(ROLES))
                    for role in ROLES:
                        self.assertEqual(comparison[role], summary['checkpoint_comparison'][role])
                    self.assertEqual(state['console'].count('FINAL MULTI-CHECKPOINT RE-EVALUATION'), 1)
                    final_console = state['console'].split('FINAL MULTI-CHECKPOINT RE-EVALUATION', 1)[1]
                    self.assertNotIn('| Metric |', final_console)
                    final_lines = [line.split('] ', 1)[1] for line in final_console.splitlines()[1:]]
                    selected_index = next(i for i, line in enumerate(final_lines) if line.startswith('Selected epochs | '))
                    self.assertEqual(final_lines[selected_index:-1],
                                     compact_comparison_report(summary['checkpoint_comparison']).splitlines())
                    self.assertTrue(final_lines[-1].startswith('Wall time:'))
                    for label, role in zip(('Overall', 'Exceedance', 'Aligned-peak', 'Equal-composite', 'Peak-priority', 'Legacy event-aware'), ROLES):
                        self.assertIn(f'{label} epoch: {WINNERS[role]}', report)
                    for split in ('val', 'test'):
                        section = report.split(f'{split.upper()} — Overall', 1)[1].split('TEST — Overall')[0]
                        for key in comparison['overall'][split]:
                            row = '| ' + METRIC_LABELS.get(key, key) + ' | '
                            row += ' | '.join(display(comparison[role][split].get(key)) for role in ROLES) + ' |'
                            self.assertIn(row, section)
                    self.assertIn('strict exceedance y > tau', comparison['method'])
                    self.assertIn('TRAIN hourly target Q95', comparison['method'])
                    for label in ('AllRMSE', 'ExceedanceRMSE', 'WindowPeakRMSE', 'GTAlignedPeakRMSE', 'EpisodePeakRMSE'):
                        self.assertIn(label, report)

    def test_test_values_cannot_change_any_selected_epoch(self):
        selected = []
        for score in (0., 1e12):
            with tempfile.TemporaryDirectory() as temporary:
                output, state = recorded_run(Path(temporary), 'eventaware', test_score=score)
                summary = json.loads(next(output.glob('summary_*.json')).read_text())
                selected.append({role: value['epoch'] for role, value in summary['checkpoint_selection']['checkpoints'].items()})
                self.assertEqual(summary['test']['all_rmse'], score)
        self.assertEqual(selected, [WINNERS] * 2)

    def test_six_role_bookkeeping_and_primary_do_not_change_real_cpu_training(self):
        # Compare against retaining only the historical two roles, including
        # optimizer moments, scheduler state, losses, model weights, and RNG.
        for variant in ('single', 'direct', 'severity_shape'):
            trajectories = []
            for mode in ('legacy', *ROLES):
                with tempfile.TemporaryDirectory() as temporary, contextlib.ExitStack() as stack:
                    if mode == 'legacy':
                        for module in ('train', 'emulator.training.eventaware_checkpoints', 'emulator.training.reporting'):
                            stack.enter_context(patch(f'{module}.ROLES', ('overall', 'eventaware')))
                    torch.manual_seed(824)
                    root = Path(temporary)
                    graphs, stations = make_fixture(root)
                    history, schedules = [], []
                    real_epoch = train.run_epoch
                    real_step = torch.optim.lr_scheduler.LambdaLR.step

                    def step(scheduler, *args, **kwargs):
                        real_step(scheduler, *args, **kwargs)
                        schedules.append(copy.deepcopy(scheduler.state_dict()))

                    def epoch(model, loader, **kwargs):
                        result = real_epoch(model, loader, **kwargs)
                        if kwargs.get('optimizer') is not None:
                            optimizer = kwargs['optimizer'].state_dict()
                            moments = {f'{parameter}_{key}': value for parameter, values in optimizer['state'].items()
                                       for key, value in values.items()}
                            history.append(dict(metrics=result.metrics, weights=tensor_digest(model.state_dict()),
                                                rng=tensor_digest({'rng': torch.get_rng_state()}),
                                                python_rng=random.getstate(), numpy_rng=repr(np.random.get_state()),
                                                optimizer=tensor_digest(moments),
                                                param_groups=copy.deepcopy(optimizer['param_groups'])))
                        elif not kwargs.get('save_predictions'):
                            result.metrics.update(metric(VALUES[len(history) - 1]))
                        return result
                    with contextlib.redirect_stdout(io.StringIO()), patch.object(train, 'run_epoch', side_effect=epoch), \
                            patch.object(torch.optim.lr_scheduler.LambdaLR, 'step', new=step):
                        train.main(arguments(graphs, stations, root / 'run', variant,
                                             'overall' if mode == 'legacy' else mode))
                    trajectories.append((history, schedules))
            for actual in trajectories[1:]:
                self.assertEqual(trajectories[0], actual, variant)


if __name__ == '__main__':
    unittest.main()
