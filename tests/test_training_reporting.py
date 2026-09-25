"""Compact console units and metric scope leave complete comparison artifacts intact."""

import copy
import unittest

from emulator.training.checkpoint_selection import ROLES
from emulator.training.metrics import METRIC_GROUPS, METRIC_KEYS, METRIC_LABELS
from emulator.training.reporting import compact_comparison_report, comparison_report


ROLE_LABELS = {
    'overall': 'Overall',
    'exceedance': 'Exceedance',
    'aligned_peak': 'Aligned-peak',
    'bea': 'BEA',
}
EXPECTED_LABELS = (
    'AllRMSE', 'AllMAE', 'ExceedanceRMSE', 'ExceedanceMAE',
    'EpisodePeakRMSE', 'EpisodePeakMAE', 'EpisodePeakBias',
    'EpisodeGTAlignedPeakRMSE', 'EpisodeGTAlignedPeakMAE', 'EpisodeGTAlignedPeakBias',
    'EpisodePeakTimingMAEHours',
)
VAL_METRICS = dict(
    all_rmse=.02412349, all_mae=.01781, exceedance_rmse=.0394444, exceedance_mae=.02892,
    episode_peak_rmse=.04431, episode_peak_mae=.031067, episode_peak_bias=-.0062,
    episode_gt_aligned_peak_rmse=.04862, episode_gt_aligned_peak_mae=.03451,
    episode_gt_aligned_peak_bias=-.0081, episode_peak_timing_mae_hours=.843,
)
TEST_METRICS = dict(
    all_rmse=.02503, all_mae=.01837, exceedance_rmse=.04102, exceedance_mae=.02971,
    episode_peak_rmse=.0452, episode_peak_mae=.03184, episode_peak_bias=-.007,
    episode_gt_aligned_peak_rmse=.04911, episode_gt_aligned_peak_mae=.03502,
    episode_gt_aligned_peak_bias=.0093, episode_peak_timing_mae_hours=.906,
)


def evaluations_for(roles=ROLES):
    metrics = dict.fromkeys(METRIC_KEYS, .123456789)
    metrics.update(extreme_hour_n_lead_0=3, exceedance_rmse_lead_0=.00456,
                   exceedance_rmse_lead_1=None)
    return {role: dict(epoch=epoch, val=dict(metrics, **VAL_METRICS),
                       test=dict(metrics, **TEST_METRICS))
            for epoch, role in enumerate(roles, 87)}


class CompactComparisonReportTests(unittest.TestCase):
    def test_mm_conversion_signed_bias_and_hours_with_two_decimals(self):
        report = compact_comparison_report(evaluations_for(('overall',)))
        self.assertEqual(report, """Selected epochs | Overall=87

[Overall | epoch 87]

VAL
AllRMSE=24.12 mm | AllMAE=17.81 mm
ExceedanceRMSE=39.44 mm | ExceedanceMAE=28.92 mm
EpisodePeakRMSE=44.31 mm | EpisodePeakMAE=31.07 mm | EpisodePeakBias=-6.20 mm
EpisodeGTAlignedPeakRMSE=48.62 mm | EpisodeGTAlignedPeakMAE=34.51 mm | EpisodeGTAlignedPeakBias=-8.10 mm
EpisodePeakTimingMAEHours=0.84 h

TEST
AllRMSE=25.03 mm | AllMAE=18.37 mm
ExceedanceRMSE=41.02 mm | ExceedanceMAE=29.71 mm
EpisodePeakRMSE=45.20 mm | EpisodePeakMAE=31.84 mm | EpisodePeakBias=-7.00 mm
EpisodeGTAlignedPeakRMSE=49.11 mm | EpisodeGTAlignedPeakMAE=35.02 mm | EpisodeGTAlignedPeakBias=9.30 mm
EpisodePeakTimingMAEHours=0.91 h
""")

    def test_every_retained_role_has_both_splits_and_exactly_eleven_metrics(self):
        evaluations = evaluations_for()
        report = compact_comparison_report(evaluations)
        selected = 'Selected epochs | ' + ' | '.join(
            f'{ROLE_LABELS[role]}={evaluations[role]["epoch"]}' for role in ROLES)
        self.assertEqual(report.splitlines()[0], selected)
        blocks = report.split('\n\n[')[1:]
        self.assertEqual(len(blocks), len(ROLES))
        for role, block in zip(ROLES, blocks):
            with self.subTest(role=role):
                self.assertTrue(block.startswith(f'{ROLE_LABELS[role]} | epoch {evaluations[role]["epoch"]}]'))
                for split in ('VAL', 'TEST'):
                    section = block.split(f'\n{split}\n', 1)[1].split('\n\n', 1)[0]
                    self.assertEqual(len(section.splitlines()), 5)
                    labels = tuple(item.split('=', 1)[0] for line in section.splitlines()
                                   for item in line.split(' | '))
                    self.assertEqual(labels, EXPECTED_LABELS)
        for excluded in ('EventWindow', 'WindowPeak', 'Under', 'Precision', 'Recall', 'F1',
                         'Rate', 'ExtremeHourN', 'EpisodeN', '_lead_', 'ExcessArea'):
            self.assertNotIn(excluded, report)
        self.assertNotRegex(report, r'(?m)(?:^| \| )(?:GTAlignedPeak|PeakTiming)\w*=')

    def test_none_and_missing_metrics_are_na(self):
        for missing in (False, True):
            with self.subTest(missing=missing):
                metrics = {} if missing else dict.fromkeys(VAL_METRICS)
                report = compact_comparison_report({'overall': dict(epoch=3, val=metrics, test=metrics)})
                for label in EXPECTED_LABELS:
                    self.assertEqual(report.count(f'{label}=NA'), 2)
                self.assertNotIn('None', report)
                self.assertNotIn('NA mm', report)
                self.assertNotIn('NA h', report)

    def test_zero_remains_zero(self):
        metrics = dict.fromkeys(VAL_METRICS, 0.)
        report = compact_comparison_report({'overall': dict(epoch=3, val=metrics, test=metrics)})
        self.assertEqual(report.count('=0.00 mm'), 20)
        self.assertEqual(report.count('=0.00 h'), 2)
        self.assertNotIn('NA', report)

    def test_role_subset_and_order_follow_evaluations(self):
        report = compact_comparison_report(evaluations_for(('bea', 'aligned_peak')))
        self.assertEqual(report.splitlines()[0], 'Selected epochs | BEA=87 | Aligned-peak=88')
        blocks = report.split('\n\n[')[1:]
        self.assertEqual(len(blocks), 2)
        self.assertTrue(blocks[0].startswith('BEA | epoch 87]'))
        self.assertTrue(blocks[1].startswith('Aligned-peak | epoch 88]'))

    def test_full_comparison_preserves_all_metrics_units_and_format(self):
        evaluations = evaluations_for()
        original = copy.deepcopy(evaluations)
        metadata = dict(exceedance_percentile=95, tau_physical=1.23456789)
        full = comparison_report(evaluations, metadata)
        compact_comparison_report(evaluations)
        self.assertEqual(evaluations, original)
        self.assertEqual(comparison_report(evaluations, metadata), full)
        self.assertTrue(full.startswith(
            'Extreme threshold: TRAIN hourly target Q95; tau = 1.234567890 meters; strict exceedance y > tau\n\n'))
        self.assertTrue(full.endswith('\n\n'))
        table_header = '| Metric | Overall | Exceedance | AlignedPeak | BEA |'
        self.assertEqual(full.count(table_header), 2 * (len(METRIC_GROUPS) + 1))
        for role in ROLES:
            self.assertIn(f'{ROLE_LABELS[role]} epoch: {evaluations[role]["epoch"]}', full)
        for split in ('val', 'test'):
            section = full.split(f'{split.upper()} — Overall', 1)[1].split('TEST — Overall')[0]
            for group in (*METRIC_GROUPS, 'Additional diagnostics'):
                self.assertIn(f'{split.upper()} — {group}', full)
            for key in evaluations['overall'][split]:
                values = [original[role][split][key] for role in ROLES]
                expected = ' | '.join('NA' if value is None else f'{value:.9g}' for value in values)
                self.assertIn(f'| {METRIC_LABELS.get(key, key)} | {expected} |', section)
        self.assertIn('| AllRMSE | 0.02412349 |', full)
        self.assertIn('| EpisodePeakBias | -0.0062 |', full)
        self.assertNotIn(' mm', full)
        self.assertNotIn('Selected epochs', full)


if __name__ == '__main__':
    unittest.main()
