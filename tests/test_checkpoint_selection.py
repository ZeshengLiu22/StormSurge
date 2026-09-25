"""Raw VAL score, independent role minima, durable state, and earliest ties."""

import random

import numpy as np
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import torch

from emulator.training.checkpoint_selection import (BEA_WEIGHTS, ROLES, SELECTION_METRICS, CheckpointTracker,
    checkpoint_scores, bea_score)


def metric(overall, exceedance, aligned):
    return dict(all_rmse=overall, exceedance_rmse=exceedance, gt_aligned_peak_rmse=aligned)


class CheckpointSelectionTests(unittest.TestCase):
    def test_raw_physical_weighted_sum_and_unrelated_fields_cannot_change_it(self):
        values = metric(1., 2., 4.)
        self.assertEqual(BEA_WEIGHTS, (.50, .25, .25))
        self.assertAlmostEqual(bea_score(values), 2.)
        self.assertEqual(bea_score(dict(values, test=1e99, exceedance_bias=-999.,
            gt_aligned_peak_under_pct=100., exceedance_f1=0.)), bea_score(values))
        for key in values:
            for value in (None, float('nan'), float('inf'), -1.):
                with self.subTest(key=key, value=value), self.assertRaisesRegex(ValueError, 'VAL'):
                    bea_score(dict(values, **{key: value}))
        with self.assertRaisesRegex(ValueError, 'VAL'):
            bea_score({})
        self.assertEqual(bea_score(metric(0., 0., 0.)), 0.)
        with self.assertRaises(TypeError):
            bea_score(values, (1., 0., 0.))

    def test_distinct_winners_and_earliest_ties_without_secondary_ranking(self):
        with tempfile.TemporaryDirectory() as temporary:
            tracker = CheckpointTracker(temporary)
            values = [metric(1., 2., 4.), metric(1.02, 1., 2.), metric(1.02, 1., 2.)]
            calls = []
            for epoch, val in enumerate(values, 1):
                def snapshot():
                    calls.append(epoch)
                    return dict(epoch=epoch, val=val, model_state={'weight': torch.tensor(epoch)})
                tracker.observe(epoch, val, snapshot)
            self.assertEqual(calls, [1, 2])
            self.assertEqual(tracker.best['overall']['epoch'], 1)
            self.assertEqual(tracker.best['bea']['epoch'], 2)
            for role, epoch in (('overall', 1), ('bea', 2)):
                checkpoint = torch.load(tracker.paths[role], weights_only=False)
                self.assertEqual(checkpoint['epoch'], epoch)
                self.assertEqual(checkpoint['model_state']['weight'].item(), epoch)
                self.assertEqual(checkpoint['selection_split'], 'val')
                self.assertEqual(checkpoint['bea_settings']['score_formula'], 'raw_weighted_sum')
                self.assertEqual(tracker.selection_summary(role)['selected_epoch'], epoch)
            # Metadata points to all retained artifacts regardless of the primary role.
            self.assertEqual(set(tracker.selection_summary()['checkpoints']), set(ROLES))

    def test_shared_epoch_factory_runs_once_and_tracker_does_not_draw_rng(self):
        with tempfile.TemporaryDirectory() as temporary:
            tracker = CheckpointTracker(temporary)
            val = metric(1., 2., 4.)
            snapshot = Mock(return_value=dict(epoch=1, val=val, model_state={'weight': torch.tensor(7.)}))
            rng = torch.get_rng_state().clone()
            python_rng, numpy_rng = random.getstate(), np.random.get_state()
            score, improved = tracker.observe(1, val, snapshot)
            snapshot.assert_called_once_with()
            self.assertEqual(random.getstate(), python_rng)
            actual_numpy_rng = np.random.get_state()
            self.assertEqual(actual_numpy_rng[0], numpy_rng[0])
            np.testing.assert_array_equal(actual_numpy_rng[1], numpy_rng[1])
            self.assertEqual(actual_numpy_rng[2:], numpy_rng[2:])
            torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
            self.assertEqual(improved, list(ROLES))
            self.assertAlmostEqual(score, 2.)
            for role in improved:
                checkpoint = torch.load(tracker.paths[role], weights_only=False)
                self.assertEqual(checkpoint['model_state']['weight'].item(), 7.)
            val['all_rmse'] = 999.
            self.assertEqual(tracker.best['overall']['val']['all_rmse'], 1.)

    def test_exactly_four_raw_scores_with_fixed_bea_weights(self):
        self.assertEqual(ROLES, ('overall', 'exceedance', 'aligned_peak', 'bea'))
        self.assertEqual(SELECTION_METRICS, dict(overall='all_rmse', exceedance='exceedance_rmse',
                                                aligned_peak='gt_aligned_peak_rmse', bea='bea_score'))
        values = metric(1., 2., 4.)
        self.assertEqual(checkpoint_scores(values), dict(overall=1., exceedance=2., aligned_peak=4., bea=2.))
        self.assertEqual(checkpoint_scores(dict(values, test=-1e12, reference_scale=999.)),
                         checkpoint_scores(values))
        with self.assertRaises(TypeError):
            checkpoint_scores(values, (1., 0., 0.))
        with tempfile.TemporaryDirectory() as temporary, self.assertRaises(TypeError):
            CheckpointTracker(temporary, (1., 0., 0.))

    def test_all_four_distinct_winners_and_earliest_ties(self):
        # Each row is the unique winner for its corresponding role in ROLES.
        trajectory = [(0., 10., 10.), (10., 0., 10.), (10., 10., 0.), (4., 4., 4.)]
        with tempfile.TemporaryDirectory() as temporary:
            tracker = CheckpointTracker(temporary)
            # Replaying the entire trajectory must not replace any tied winner.
            for epoch, values in enumerate(trajectory * 2, 1):
                val = metric(*values)
                factory = Mock(return_value=dict(epoch=epoch, val=val, model_state={'weight': torch.tensor(epoch)}))
                tracker.observe(epoch, val, factory)
                if epoch > len(trajectory):
                    factory.assert_not_called()
            summary = tracker.selection_summary()
            self.assertEqual(set(Path(temporary).glob('best_*.pt')), set(tracker.paths.values()))
            for expected_epoch, role in enumerate(ROLES, 1):
                selected = summary['checkpoints'][role]
                self.assertEqual(selected['epoch'], expected_epoch)
                self.assertEqual(selected['selection_metric_key'], SELECTION_METRICS[role])
                self.assertEqual(selected['selection_metric_value'], checkpoint_scores(metric(*trajectory[expected_epoch - 1]))[role])
                self.assertEqual(selected['path'], str(tracker.paths[role]))
                saved = torch.load(tracker.paths[role], weights_only=False)
                self.assertEqual(saved['epoch'], expected_epoch)
                self.assertEqual(saved['model_state']['weight'].item(), expected_epoch)
                self.assertEqual(saved['selection_split'], 'val')
                self.assertEqual(saved['selection_metric_value'], selected['selection_metric_value'])
                self.assertEqual(saved['bea_score'], selected['bea_score'])
                self.assertEqual(set(saved), {'epoch', 'val', 'model_state', 'checkpoint_role',
                    'checkpoint_roles', 'selection_metric_key', 'selection_metric_value',
                    'selection_split', 'bea_score', 'bea_settings'})
                self.assertEqual(saved['bea_settings']['weights'],
                                 dict(all_rmse=.50, exceedance_rmse=.25, gt_aligned_peak_rmse=.25))
                self.assertEqual(tracker.selection_summary(role)['selected_bea_score'], selected['bea_score'])
                self.assertEqual(tracker.selection_summary(role)['selected_epoch'], expected_epoch)

    def test_tied_selector_scores_with_different_components_keep_first_epoch(self):
        # Same role score with other components improving: no secondary ranking.
        pairs = dict(overall=((1., 4., 4.), (1., 2., 2.)),
                     exceedance=((4., 1., 4.), (2., 1., 2.)),
                     aligned_peak=((4., 4., 1.), (2., 2., 1.)),
                     bea=((1., 2., 3.), (2., 1., 2.)))
        for role, pair in pairs.items():
            with self.subTest(role=role), tempfile.TemporaryDirectory() as temporary:
                tracker = CheckpointTracker(temporary)
                self.assertEqual(*(checkpoint_scores(metric(*values))[role] for values in pair))
                for epoch, values in enumerate(pair, 1):
                    val = metric(*values)
                    tracker.observe(epoch, val, lambda: dict(epoch=epoch, val=val))
                self.assertEqual(tracker.best[role]['epoch'], 1)

    def test_invalid_epochs_missing_validation_and_wrong_snapshot_fail(self):
        with tempfile.TemporaryDirectory() as temporary:
            tracker = CheckpointTracker(temporary)
            with self.assertRaisesRegex(ValueError, 'No completed'):
                tracker.selection_summary()
            for epoch in (0, -1, True, 1.5):
                with self.assertRaisesRegex(ValueError, 'positive integers'):
                    tracker.observe(epoch, metric(1., 2., 3.), Mock())
            val = metric(1., 2., 3.)
            with self.assertRaisesRegex(ValueError, 'observed VAL epoch'):
                tracker.observe(1, val, lambda: dict(epoch=2, val=val))
            tracker.observe(1, val, lambda: dict(epoch=1, val=val))
            with self.assertRaisesRegex(ValueError, 'increase|increasing'):
                tracker.observe(1, val, Mock())
            with self.assertRaisesRegex(ValueError, 'fresh'):
                CheckpointTracker(temporary)
            with self.assertRaisesRegex(ValueError, 'Unknown'):
                tracker.selection_summary('anything')

    def test_save_failure_keeps_previous_role_bookkeeping(self):
        with tempfile.TemporaryDirectory() as temporary:
            tracker = CheckpointTracker(temporary)
            first, second = metric(2., 3., 4.), metric(1., 2., 3.)
            tracker.observe(1, first, lambda: dict(epoch=1, val=first))
            with patch('emulator.training.checkpoint_selection.atomic_save', side_effect=OSError('disk full')):
                with self.assertRaises(OSError):
                    tracker.observe(2, second, lambda: dict(epoch=2, val=second))
            self.assertEqual(tracker.last_epoch, 1)
            self.assertEqual(tracker.best['overall']['epoch'], 1)
            self.assertEqual(tracker.best['bea']['epoch'], 1)


if __name__ == '__main__':
    unittest.main()
