"""Same GT episode IDs/bins across models and exact physical branch accounting."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from event_diagnostics import align_exports, compare, main, shared_quantile_bins


def arrays():
    y = np.array([[0., .3, .4], [.5, .1, .6], [.1, .7, .8]])
    body = np.full(y.shape, .1)
    raw = np.maximum(y - .2, 0)
    return dict(y_true=y, y_pred=body + .5 * raw, tags=np.array(['a', 'b', 'c']),
                target_timestamps=np.arange(y.size).reshape(y.shape) * 3600,
                tau_physical=np.array(.2), exceedance_percentile=np.array(95.), metric_schema=np.array('hourly_q95_v1'),
                threshold_schema=np.array('train_hourly_q95_v1'), station=np.array('CBBT'), split=np.array('test'),
                body_phys=body, excess_phys=raw, gate_probability=np.full(3, .5), gate_logits=np.zeros(3))


class FixedEventTests(unittest.TestCase):
    def test_predictions_cannot_change_episode_ids_or_gt_bins(self):
        dual = arrays()
        single = {key: value.copy() for key, value in dual.items() if key not in ('body_phys', 'excess_phys', 'gate_probability', 'gate_logits')}
        single['y_pred'] = single['y_true'] * 10
        report = compare(dict(dual=dual, single=single))
        self.assertEqual(len(report['frozen_gt_episodes']), 3)
        self.assertEqual(report['frozen_gt_episodes'][0]['target_indices'], [1, 2, 3])
        for model in ('dual', 'single'):
            rows = [row for row in report['episode_rows'] if row['model'] == model]
            self.assertEqual([row['episode_id'] for row in rows], [0, 1, 2])
            self.assertEqual([row['gt_severity_bin'] for row in rows], [item['gt_severity_bin'] for item in report['frozen_gt_episodes']])
            self.assertEqual(report['metrics'][model]['episode_n'], 3)

    def test_exact_id_alignment_and_mismatched_populations_fail(self):
        original = arrays()
        reordered = {key: value[::-1] if key in ('y_true', 'y_pred', 'tags', 'target_timestamps', 'body_phys', 'excess_phys', 'gate_probability', 'gate_logits') else value for key, value in original.items()}
        aligned = align_exports(dict(a=original, b=reordered))
        np.testing.assert_array_equal(aligned['a']['y_pred'], aligned['b']['y_pred'])
        for key, value in (('tau_physical', np.array(.3)), ('exceedance_percentile', np.array(90.)), ('split', np.array('val')), ('metric_schema', np.array('wrong')), ('tags', np.array(['a', 'b', 'missing']))):
            bad = dict(original, **{key: value})
            with self.subTest(key=key), self.assertRaises(ValueError):
                align_exports(dict(a=original, b=bad))
        bad = copy.deepcopy(original)
        bad['y_true'][0, 0] += .01
        with self.assertRaisesRegex(ValueError, 'y_true differs'):
            align_exports(dict(a=original, b=bad))

    def test_branch_reconstruction_all_extreme_hours_and_loss_components(self):
        result = compare(dict(dual=arrays()))
        self.assertEqual(len(result['extreme_hour_rows']), 6)
        for row in result['extreme_hour_rows']:
            self.assertGreater(row['y_true'], row['tau'])
            self.assertAlmostEqual(row['reconstructed_final_prediction'], row['y_pred'])
            deficit = row['body_deficit'] - row['raw_excess_error'] + row['gate_attenuation']
            self.assertAlmostEqual(row['y_true'] - row['y_pred'], deficit)
        bad = arrays()
        bad['y_pred'][0, 0] += .01
        with self.assertRaisesRegex(ValueError, 'reconstruction failed'):
            compare(dict(bad=bad))

    def test_tied_gt_values_share_one_bin(self):
        membership, edges = shared_quantile_bins(np.ones(10), 4)
        np.testing.assert_array_equal(membership, np.zeros(10))
        self.assertEqual(edges, [1., 1.])

    def test_duplicate_timestamps_rejected_and_missing_hours_break_episodes(self):
        values = arrays()
        values['target_timestamps'][1:] += 3600
        self.assertEqual(len(compare(dict(a=values))['frozen_gt_episodes']), 4)
        values['target_timestamps'][1, 0] = values['target_timestamps'][0, 2]
        with self.assertRaisesRegex(ValueError, 'Duplicate supervised'):
            compare(dict(a=values))

    def test_optional_source_split_ids_break_episodes(self):
        values = arrays()
        values['split'] = np.array('all')
        values['split_ids'] = np.array(['train', 'test', 'test'])
        report = compare(dict(a=values))
        self.assertEqual(len(report['frozen_gt_episodes']), 4)
        self.assertEqual(report['metrics']['a']['episode_n'], 4)
        self.assertTrue(all(row['split'] in ('train', 'test') for row in report['extreme_hour_rows']))
        bad = dict(values, split_ids=np.array(['train', 'val', 'test']))
        with self.assertRaisesRegex(ValueError, 'split_ids differs'):
            align_exports(dict(a=values, b=bad))

    def test_strict_threshold_preserves_float64_quantile_against_float32_targets(self):
        values = {key: value for key, value in arrays().items() if key not in ('body_phys', 'excess_phys', 'gate_probability', 'gate_logits')}
        values['y_true'] = np.ones((3, 3), dtype=np.float32)
        values['y_pred'] = np.ones((3, 3), dtype=np.float32)
        values['tau_physical'] = np.array(np.nextafter(1., -np.inf))
        result = compare(dict(a=values))
        self.assertEqual(result['metrics']['a']['extreme_hour_n'], 9)
        self.assertEqual(len(result['extreme_hour_rows']), 9)

    def test_underprediction_tolerance_matches_canonical_metric(self):
        values = {key: value for key, value in arrays().items() if key not in ('body_phys', 'excess_phys', 'gate_probability', 'gate_logits')}
        values['y_pred'] = values['y_true'] - .01
        result = compare(dict(a=values))
        for row in result['episode_rows']:
            self.assertEqual(row['under_10mm'], row['prediction_at_gt_peak'] < row['gt_peak'] - .01)

    def test_cli_writes_canonical_frozen_episode_and_branch_tables(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            np.savez(root / 'predictions.npz', **arrays())
            main(['--predictions', f"dual={root / 'predictions.npz'}", '--output', str(root / 'out'), '--no-plots'])
            report = json.loads((root / 'out' / 'comparison.json').read_text())
            self.assertEqual(report['metrics']['dual']['episode_n'], 3)
            self.assertTrue((root / 'out' / 'extreme_hour_rows.csv').is_file())
            self.assertEqual(len(json.loads((root / 'out' / 'frozen_gt_episodes.json').read_text())), 3)


if __name__ == '__main__':
    unittest.main()
