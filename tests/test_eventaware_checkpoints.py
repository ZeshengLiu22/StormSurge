"""Raw VAL score, independent role minima, durable state, and earliest ties."""

import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import torch

from emulator.training.eventaware_checkpoints import (DEFAULT_WEIGHTS, EventAwareTracker,
    eventaware_score, validate_score_weights)


def metric(overall, exceedance, aligned):
    return dict(all_rmse=overall, exceedance_rmse=exceedance, gt_aligned_peak_rmse=aligned)


class EventAwareCheckpointTests(unittest.TestCase):
    def test_raw_physical_weighted_sum_and_unrelated_fields_cannot_change_it(self):
        values = metric(1., 2., 4.)
        self.assertEqual(DEFAULT_WEIGHTS, (.65, .20, .15))
        self.assertAlmostEqual(eventaware_score(values), 1.65)
        self.assertEqual(eventaware_score(dict(values, test=1e99, exceedance_bias=-999.,
            gt_aligned_peak_under_pct=100., exceedance_f1=0.)), eventaware_score(values))
        self.assertEqual(eventaware_score(values, (1., 0., 0.)), 1.)
        for key in values:
            for value in (None, float('nan'), float('inf'), -1.):
                with self.subTest(key=key, value=value), self.assertRaisesRegex(ValueError, 'VAL'):
                    eventaware_score(dict(values, **{key: value}))
        with self.assertRaisesRegex(ValueError, 'VAL'):
            eventaware_score({}, (1., 0., 0.))
        for weights in ((.6, .2, .15), (-1., 1., 1.), (1.,), (math.nan, 0., 1.)):
            with self.assertRaises(ValueError):
                validate_score_weights(weights)

    def test_distinct_winners_and_earliest_ties_without_secondary_ranking(self):
        with tempfile.TemporaryDirectory() as temporary:
            tracker = EventAwareTracker(temporary)
            values = [metric(1., 2., 4.), metric(1.02, 1., 2.), metric(1.02, 1., 2.)]
            calls = []
            for epoch, val in enumerate(values, 1):
                def snapshot():
                    calls.append(epoch)
                    return dict(epoch=epoch, val=val, model_state={'weight': torch.tensor(epoch)})
                tracker.observe(epoch, val, snapshot)
            self.assertEqual(calls, [1, 2])
            self.assertEqual(tracker.best['overall']['epoch'], 1)
            self.assertEqual(tracker.best['eventaware']['epoch'], 2)
            for role, epoch in (('overall', 1), ('eventaware', 2)):
                checkpoint = torch.load(tracker.paths[role], weights_only=False)
                self.assertEqual(checkpoint['epoch'], epoch)
                self.assertEqual(checkpoint['model_state']['weight'].item(), epoch)
                self.assertEqual(checkpoint['selection_split'], 'val')
                self.assertEqual(checkpoint['eventaware_settings']['score_formula'], 'raw_weighted_sum')
                self.assertEqual(tracker.selection_summary(role)['selected_epoch'], epoch)
            # Metadata points to both retained artifacts regardless of the primary role.
            self.assertEqual(set(tracker.selection_summary()['checkpoints']), {'overall', 'eventaware'})

    def test_shared_epoch_factory_runs_once_and_tracker_does_not_draw_rng(self):
        with tempfile.TemporaryDirectory() as temporary:
            tracker = EventAwareTracker(temporary)
            val = metric(1., 2., 4.)
            snapshot = Mock(return_value=dict(epoch=1, val=val, model_state={'weight': torch.tensor(7.)}))
            rng = torch.get_rng_state().clone()
            score, improved = tracker.observe(1, val, snapshot)
            snapshot.assert_called_once_with()
            torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
            self.assertEqual(improved, ['overall', 'eventaware'])
            self.assertAlmostEqual(score, 1.65)
            a, b = [torch.load(tracker.paths[role], weights_only=False) for role in improved]
            torch.testing.assert_close(a['model_state']['weight'], b['model_state']['weight'])
            val['all_rmse'] = 999.
            self.assertEqual(tracker.best['overall']['val']['all_rmse'], 1.)

    def test_invalid_epochs_missing_validation_and_wrong_snapshot_fail(self):
        with tempfile.TemporaryDirectory() as temporary:
            tracker = EventAwareTracker(temporary)
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
                EventAwareTracker(temporary)
            with self.assertRaisesRegex(ValueError, 'Unknown'):
                tracker.selection_summary('anything')

    def test_save_failure_keeps_previous_role_bookkeeping(self):
        with tempfile.TemporaryDirectory() as temporary:
            tracker = EventAwareTracker(temporary)
            first, second = metric(2., 3., 4.), metric(1., 2., 3.)
            tracker.observe(1, first, lambda: dict(epoch=1, val=first))
            with patch('emulator.training.eventaware_checkpoints.atomic_save', side_effect=OSError('disk full')):
                with self.assertRaises(OSError):
                    tracker.observe(2, second, lambda: dict(epoch=2, val=second))
            self.assertEqual(tracker.last_epoch, 1)
            self.assertEqual(tracker.best['overall']['epoch'], 1)
            self.assertEqual(tracker.best['eventaware']['epoch'], 1)


if __name__ == '__main__':
    unittest.main()
