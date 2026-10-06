"""Production station propagation and post-hoc provenance validation."""
import copy
import csv
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import unittest

REPO = Path(__file__).resolve().parents[1]
TREE = REPO / 'configs/cross_dataset_infer_4stations'
sys.path.insert(0, str(TREE))
spec = importlib.util.spec_from_file_location('four_station_generator', TREE / 'generate_configs.py')
generator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(generator)
from summarize_results import validate_metadata, matrix_rows


class FourStationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = list(csv.DictReader((TREE / 'manifest.csv').open()))
        cls.audits = json.loads((TREE / 'checkpoint_audit.json').read_text())['checkpoints']

    def test_every_pair_propagates_station_pinned_source_and_exact_years(self):
        self.assertEqual(len(self.rows), 264)
        self.assertEqual(len(self.audits), 48)
        keys = set()
        for row in self.rows:
            key = tuple(row[k] for k in ('station', 'group', 'source', 'target'))
            self.assertNotIn(key, keys)
            keys.add(key)
            config = Path(row['config_path'])
            self.assertTrue(config.is_relative_to(TREE))
            script = 'source "$1"; printf "%s\\n" "$STATION" "$EXPERIMENT_GROUP" "$SOURCE_NAME" "$TARGET_NAME" "$CKPT_PATH" "$ROOT_DIR" "$TEST_ROOT_DIR" "$YEARS" "$STRICT_YEARS" "$BATCH_SIZE" "$INFERENCE_RESULTS_ROOT"'
            result = subprocess.run(['bash', '-c', script, 'bash', str(config)], text=True, capture_output=True, check=True)
            self.assertEqual(result.stdout.splitlines(), [*key, row['checkpoint'], row['source_root'], row['target_root'],
                row['years'], '1', '256', str(generator.RESULTS / row['station'] / row['group'])])
            self.assertEqual(row['years'].split(','), generator.YEARS[row['group']])
            audit = next(a for a in self.audits if (a['station'], a['group'], a['source']) == key[:3])
            self.assertEqual(row['checkpoint'], audit['checkpoint'])
            self.assertEqual(Path(row['checkpoint']).name, 'best_overall.pt')
        for station in generator.STATIONS:
            self.assertEqual(sum(r['station'] == station for r in self.rows), 66)
            for group, targets in generator.TARGETS.items():
                for source in generator.SOURCES:
                    files = list((TREE / station / group / source).glob('*.sh'))
                    self.assertEqual(len(files), len(targets))
        self.assertEqual(sum(r['group'] == 'past_only' for r in self.rows), 144)
        self.assertEqual(sum(r['group'] == 'future_year' for r in self.rows), 120)

    def test_summary_rejects_wrong_station_years_threshold_and_checkpoint(self):
        row, audit = self.rows[0], self.audits[0]
        data = dict(station=row['station'], source_name=row['source'], target_name=row['target'],
            model='perceiver3', encoder_type='GraphSAGE', temporal_block='Transformer', head_type='single',
            history_hours=24, history_steps=4, strict_years=True, normalization_origin=generator.ORIGIN,
            normalization_type=audit['normalization_type'], evaluation_threshold_origin=generator.ORIGIN,
            tau_physical=audit['tau_physical'], source_tau_physical=audit['tau_physical'],
            source_exceedance_percentile=95., threshold_schema='train_hourly_q95_v1', metric_schema='hourly_q95_v1',
            checkpoint=row['checkpoint'], source_root=str(REPO / row['source_root']), target_root=str(REPO / row['target_root']),
            requested_years=row['years'].split(','), evaluated_years=row['years'].split(','),
            requested_year_count=7, evaluated_year_count=7, threshold_metadata=audit['threshold_metadata'],
            scope='external_all_years', metrics={'all_rmse': .123}, results={'_overall': {'all_rmse': .123}})
        validate_metadata(data, row, audit)
        for key, bad in [('station', 'Lewes'), ('evaluated_years', ['2008_2009']),
                         ('tau_physical', audit['tau_physical'] + .1),
                         ('normalization_origin', 'target'), ('checkpoint', '/wrong/best_overall.pt'),
                         ('scope', 'held_out_years')]:
            with self.subTest(key=key):
                changed = copy.deepcopy(data)
                changed[key] = bad
                with self.assertRaises(ValueError):
                    validate_metadata(changed, row, audit)

    def test_matrix_order_and_missing_cells_preserve_full_dimensions(self):
        values = {('Lewes', 'past_only', 'CNRM', 'EC_EARTH'): .123456789}
        rows = matrix_rows('Lewes', 'past_only', values)
        self.assertEqual([row['source'] for row in rows], list(generator.SOURCES))
        self.assertEqual(list(rows[0]), ['source', *generator.SOURCES])
        self.assertEqual(rows[2]['EC_EARTH'], .123456789)
        self.assertEqual(rows[3]['CNRM'], '')
        future = matrix_rows('Battery', 'future_year', {})
        self.assertEqual(len(future), 6)
        self.assertEqual(list(future[0]), ['source', *generator.SOURCES[1:]])


if __name__ == '__main__':
    unittest.main()
