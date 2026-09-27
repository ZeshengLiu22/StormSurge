"""One physical TRAIN episode, one GT amplitude target, fixed J/N normalization."""

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler, RandomSampler

import train
from emulator.data import ForcingGraphStore, ForcingGraphView, build_loader, fit_loss_thresholds, supervised_targets
from emulator.models import ForecastOutput
from emulator.training import ForecastLoss, LossConfig, run_epoch
from emulator.training.episode_peaks import build_episode_peak_targets, validate_episode_peak_traversal
from emulator.training.losses import episode_gt_aligned_peak_mse
from emulator.training.metrics import METRIC_KEYS, evaluate_metrics, gt_event_episodes
from test_pipeline import make_fixture
from test_training import CountingModel
import test_training


class EpisodePeakTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.truth = np.array([[0., 3., 5.], [5., 0., 4.], [0., 7., 0.]])
        self.times = np.arange(9).reshape(3, 3) * 3600
        self.targets = build_episode_peak_targets(self.truth, self.times, 2.)

    def test_cross_window_single_hour_tied_peak_and_equal_episode_membership(self):
        targets = self.targets
        np.testing.assert_array_equal(targets.episode_id, [[-1, 0, 0], [0, -1, 1], [-1, 2, -1]])
        np.testing.assert_array_equal(np.flatnonzero(targets.is_episode_gt_peak), [2, 5, 7])
        self.assertEqual(targets.metadata['episode_count'], 3)
        self.assertEqual(targets.metadata['canonical_peak_target_count'], 3)
        self.assertEqual(targets.metadata['p_episode_peak'], 3/9)
        for episode in gt_event_episodes(self.truth, self.times, 2.):
            self.assertEqual(targets.is_episode_gt_peak.ravel()[episode].sum(), 1)
        order = [2, 1, 0]
        reordered = build_episode_peak_targets(self.truth[order], self.times[order], 2.)
        np.testing.assert_array_equal(reordered.is_episode_gt_peak, targets.is_episode_gt_peak[order])
        np.testing.assert_array_equal(reordered.episode_id, targets.episode_id[order])

    def test_strict_threshold_precision_missing_hours_and_datetime_parity(self):
        for times in (self.times, self.times.astype('datetime64[s]'), self.times.astype('datetime64[s]').astype(str)):
            np.testing.assert_array_equal(build_episode_peak_targets(self.truth, times, 2.).is_episode_gt_peak,
                                          self.targets.is_episode_gt_peak)
        truth = np.array([[1., 1., 2., 2.]])
        times = np.array([[0, 3600, 10800, 14400]])
        self.assertEqual(build_episode_peak_targets(truth, times, 1.).metadata['episode_count'], 1)
        self.assertEqual(build_episode_peak_targets(truth, times, 1.-1e-10).metadata['episode_count'], 2)
        self.assertEqual(build_episode_peak_targets(truth, times, 2.).metadata['episode_count'], 0)

    def test_duplicate_timestamps_cannot_reach_supervision(self):
        times = self.times.copy()
        times[1, 0] = times[0, 2]
        with self.assertRaisesRegex(ValueError, 'Duplicate target timestamp'):
            build_episode_peak_targets(self.truth, times, 2.)
        with tempfile.TemporaryDirectory() as temporary:
            graphs, _ = make_fixture(Path(temporary))
            store = ForcingGraphStore(graphs, 'Battery')
            indices = store.split()['train']
            store.graphs[indices[1]].center_time = store.graphs[indices[0]].center_time
            with self.assertRaisesRegex(ValueError, 'Duplicate supervised target timestamps'):
                ForcingGraphView(store, indices, 0)
            with self.assertRaisesRegex(ValueError, 'Duplicate supervised target timestamps'):
                fit_loss_thresholds(store, indices)

    def test_known_gt_peak_and_final_evaluator_parity_not_predicted_argmax(self):
        # A large prediction away from the GT maximum must never enter the loss.
        prediction = torch.tensor([[100., 200., 2.], [300., 0., 3.], [0., 4., 500.]], dtype=torch.float64, requires_grad=True)
        truth = torch.from_numpy(self.truth)
        mask = torch.from_numpy(self.targets.is_episode_gt_peak)
        loss = episode_gt_aligned_peak_mse((prediction - truth).square(), mask, 3/9)
        self.assertAlmostEqual(loss.item(), (9+1+9)/3)
        metrics = evaluate_metrics(prediction, truth, 2., target_timestamps=self.times)
        self.assertAlmostEqual(loss.item(), metrics['episode_gt_aligned_peak_rmse']**2)
        loss.backward()
        expected = torch.zeros_like(prediction)
        expected[mask] = 2*(prediction.detach()-truth)[mask]/3
        torch.testing.assert_close(prediction.grad, expected)

    def test_fixed_prevalence_partition_invariance_and_empty_mask_autograd(self):
        truth = torch.from_numpy(self.truth)
        prediction = torch.zeros_like(truth, requires_grad=True)
        mask = torch.from_numpy(self.targets.is_episode_gt_peak)
        whole = episode_gt_aligned_peak_mse((prediction-truth).square(), mask, 1/3)
        # Target-count weighting is essential for uneven minibatch reporting.
        split = sum(episode_gt_aligned_peak_mse((prediction[a:b]-truth[a:b]).square(), mask[a:b], 1/3)
                    * (b-a)/3 for a,b in ((0, 1), (1, 3)))
        torch.testing.assert_close(whole, split)
        torch.testing.assert_close(torch.autograd.grad(whole, prediction, retain_graph=True)[0],
                                   torch.autograd.grad(split, prediction)[0])
        empty_prediction = torch.ones((2, 3), requires_grad=True)
        empty = episode_gt_aligned_peak_mse(empty_prediction.square(), torch.zeros((2, 3), dtype=torch.bool), 1/3)
        self.assertEqual(empty.item(), 0.)
        empty.backward()
        torch.testing.assert_close(empty_prediction.grad, torch.zeros_like(empty_prediction))

    def test_weight_zero_matches_previous_global_plus_tail_value_and_gradient(self):
        generator = torch.Generator().manual_seed(183)
        for dtype in (torch.float32, torch.float64):
            for weight in (0., .025, .8):
                prediction = torch.randn(5, 6, generator=generator, dtype=dtype, requires_grad=True)
                truth = torch.randn(5, 6, generator=generator, dtype=dtype)
                stats = dict(y_mean=torch.zeros(6), y_std=torch.ones(6))
                criterion = ForecastLoss(LossConfig(exceedance_loss_weight=weight, episode_gt_aligned_peak_weight=0.),
                                         stats, .7, extreme_hour_prior=.05, p_episode_peak=.02)
                error = (prediction-truth).square()
                previous = error.mean()
                if weight:
                    previous = previous + weight * torch.where(truth.double() > .7, error, 0.).mean()/.05
                actual = criterion(ForecastOutput(prediction), prediction, truth,
                                   episode_gt_peak_mask=torch.arange(30).reshape(5, 6) == 0)
                torch.testing.assert_close(actual, previous, rtol=0, atol=0)
                torch.testing.assert_close(torch.autograd.grad(actual, prediction, retain_graph=True)[0],
                                           torch.autograd.grad(previous, prediction)[0], rtol=0, atol=0)

    def test_loss_validation_and_default_config(self):
        self.assertEqual(train.parse_args([]).episode_gt_aligned_peak_weight, 0.)
        self.assertEqual(LossConfig().episode_gt_aligned_peak_weight, 0.)
        stats = dict(y_mean=torch.zeros(3), y_std=torch.ones(3))
        for invalid in (None, 0., -1., float('nan'), float('inf'), 1.1):
            with self.subTest(prior=invalid), self.assertRaisesRegex(ValueError, 'p_episode_peak'):
                ForecastLoss(LossConfig(episode_gt_aligned_peak_weight=.1), stats, 2., p_episode_peak=invalid)
        for invalid in (-1., float('nan'), float('inf')):
            with self.assertRaisesRegex(ValueError, 'episode_gt_aligned_peak_weight'):
                ForecastLoss(LossConfig(episode_gt_aligned_peak_weight=invalid), stats, 2.)
        criterion = ForecastLoss(LossConfig(episode_gt_aligned_peak_weight=.1), stats, 2., p_episode_peak=.1)
        prediction = torch.zeros(2, 3)
        with self.assertRaisesRegex(ValueError, 'boolean canonical mask'):
            criterion(ForecastOutput(prediction), prediction, prediction)
        for flags in (['--head_type', 'dual'], ['--loss_mode', 'wqe'],
                      ['--exceedance_loss_mode', 'wqe'], ['--loss_mode', 'mse_slope'], ['--exceedance_percentile', '90']):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                train.parse_args(['--model', 'perceiver3', '--episode_gt_aligned_peak_weight', '.1', *flags])

    def test_sampler_padding_is_rejected_without_changing_sampling(self):
        data = list(range(5))
        for rank in (0, 1):
            sampler = DistributedSampler(data, num_replicas=2, rank=rank)
            with self.assertRaisesRegex(ValueError, 'padded/truncated'):
                validate_episode_peak_traversal(DataLoader(data, sampler=sampler))
        for rank in (0, 1):
            validate_episode_peak_traversal(DataLoader(data[:4], sampler=DistributedSampler(data[:4], num_replicas=2, rank=rank)))
        with self.assertRaisesRegex(ValueError, 'drop_last'):
            validate_episode_peak_traversal(DataLoader(data, batch_size=2, drop_last=True))
        with self.assertRaisesRegex(ValueError, 'one occurrence'):
            validate_episode_peak_traversal(DataLoader(data, sampler=RandomSampler(data, replacement=True)))

    def test_model_gradients_and_epoch_logging_count_once_with_short_batch(self):
        fixture = test_training.TrainingTests(); fixture.setUp()
        data = fixture.data[:9]
        truth = torch.cat([graph.y for graph in data]).numpy()
        times = torch.cat([graph.target_timestamps for graph in data]).numpy()
        targets = build_episode_peak_targets(truth, times, 4.)
        for graph, mask in zip(data, targets.is_episode_gt_peak):
            graph.is_episode_gt_peak = torch.from_numpy(mask).reshape(1, -1)
        criterion = ForecastLoss(LossConfig(exceedance_loss_weight=.1, episode_gt_aligned_peak_weight=.2),
                                 fixture.stats, 4., extreme_hour_prior=4/18, p_episode_peak=4/18)
        model = CountingModel()
        gradients = []
        model.weight.register_hook(lambda grad: gradients.append(grad.detach().clone()))
        result = run_epoch(model, fixture.loader(data, batch_size=4), torch.device('cpu'), fixture.stats,
                           criterion=criterion, optimizer=torch.optim.SGD(model.parameters(), lr=0.),
                           tau_physical=4., grad_accum_steps=2)
        self.assertEqual(len(gradients), 3)
        self.assertTrue(all(torch.isfinite(grad).all() and grad.abs().sum()>0 for grad in gradients))
        self.assertEqual(result.losses['episode_peak_target_count'], 4)
        self.assertEqual(result.losses['p_episode_peak'], 4/18)
        self.assertAlmostEqual(result.losses['total_loss'], result.losses['global_mse_raw']+
                               result.losses['tail_weighted']+result.losses['episode_gt_aligned_peak_weighted'], places=4)
        self.assertAlmostEqual(result.losses['episode_gt_aligned_peak_weighted'],
                               .2*result.losses['episode_gt_aligned_peak_mse_raw'], places=5)

    def test_fixture_training_exports_two_roles_eleven_metrics_and_train_only_metadata(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            graphs, stations = make_fixture(root)
            output = root/'run'
            with contextlib.redirect_stdout(io.StringIO()) as console:
                train.main(['--root_dir', str(graphs), '--station', 'Battery', '--station_json_dir', str(stations),
                            '--output_dir', str(output), '--head_type', 'single', '--model', 'perceiver3',
                            '--hidden_channels', '16', '--node_read_heads', '2', '--time_read_heads', '2',
                            '--history_hours', '0', '--epochs', '1', '--batch_size', '2', '--device', 'cpu',
                            '--num_workers', '0', '--warmup_epochs', '0', '--x_aug', '0',
                            '--exceedance_loss_weight', '.025', '--episode_gt_aligned_peak_weight', '.1'])
            store = ForcingGraphStore(graphs, 'Battery')
            labels, times = supervised_targets(store, store.split()['train'])
            threshold = json.loads((output/'threshold_metadata.json').read_text())
            targets = build_episode_peak_targets(labels, times, threshold['tau_physical'])
            metadata = json.loads((output/'episode_peak_metadata.json').read_text())
            self.assertEqual(metadata, targets.metadata)
            logs = json.loads(next(output.glob('metrics_*.jsonl')).read_text())
            self.assertEqual(logs['loss']['episode_peak_target_count'], metadata['episode_count'])
            summary = json.loads(next(output.glob('summary_*.json')).read_text())
            self.assertEqual(summary['checkpoint_selection']['checkpoint_selection_mode'], 'exceedance')
            self.assertEqual(set(summary['checkpoint_comparison']), {'overall', 'exceedance'})
            for role in ('overall', 'exceedance'):
                checkpoint = torch.load(output/f'best_{role}.pt', weights_only=False)
                self.assertEqual(checkpoint['episode_peak_metadata'], metadata)
                self.assertEqual(checkpoint['training_config']['episode_gt_aligned_peak_weight'], .1)
                for split in ('val', 'test'):
                    self.assertEqual(set(summary['checkpoint_comparison'][role][split]), set(METRIC_KEYS))
                    with np.load(output/f'{split}_predictions_{role}.npz') as saved:
                        metrics = evaluate_metrics(saved['y_pred'], saved['y_true'], threshold['tau_physical'], target_timestamps=saved['target_timestamps'])
                    self.assertEqual(summary['checkpoint_comparison'][role][split], metrics)
            self.assertEqual({path.name for path in output.glob('best_*.pt')}, {'best_overall.pt', 'best_exceedance.pt'})
            self.assertIn('EpisodeGTAlignedPeakMSE=', console.getvalue())
            self.assertNotRegex(console.getvalue(), r'(?<!Episode)GTAlignedPeakRMSE|BEAScore')


if __name__ == '__main__':
    unittest.main()
