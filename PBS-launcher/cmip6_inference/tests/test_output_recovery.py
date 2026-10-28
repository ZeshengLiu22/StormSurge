"""Small NumPy-only tests: no checkpoints, graph data, models, or inference."""
import json
from pathlib import Path
import shlex
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common
import recover_outputs as recovery


def fixture(root):
    key = 'CMIP6_EC_EARTH/Lewes/seed42/CNN_GRU_MSE'
    directory = root / 'results' / key
    attempt = 'a' * 32
    staging = directory / f'.attempt-{attempt}'
    output = staging / 'outputs'
    output.mkdir(parents=True)
    threshold = dict(tau_physical=0.8, exceedance_percentile=95.0, fitted_on='train',
                     metric_schema='hourly_q95_v1', threshold_schema='train_hourly_q95_v1')
    row = dict(id=key, source='CMIP6_EC_EARTH', station='Lewes', seed=42, model='CNN_GRU_MSE',
               model_key='baseline_cnn_gru', checkpoint=str(root / 'unread' / 'best_overall.pt'),
               source_root=str(root / 'unread_source'), test_root_dir=str(root / 'unread_graphs'),
               output_dir=str(directory), checkpoint_metadata=dict(pressure_mode='raw',
                   model_config=dict(out_channels=6), threshold_metadata=threshold))
    metadata = dict(requested_years=common.YEARS, evaluated_years=common.YEARS,
                    requested_year_count=30, evaluated_year_count=30, strict_years=True,
                    station='Lewes', model='baseline_cnn_gru', source_name=row['source'], target_name=row['source'],
                    checkpoint=row['checkpoint'], source_root=row['source_root'], target_root=row['test_root_dir'],
                    normalization_origin='source_checkpoint_train', evaluation_threshold_origin='source_checkpoint_train',
                    pressure_mode='raw', source_tau_physical=0.8, tau_physical=0.8,
                    exceedance_percentile=95.0, source_exceedance_percentile=95.0,
                    metric_schema='hourly_q95_v1', threshold_schema='train_hourly_q95_v1')
    document = dict(metadata, threshold_metadata=threshold, years=common.YEARS, years_evaluated=common.YEARS,
                    metrics=dict(all_rmse=0.0), samples=30,
                    results={year: dict(samples=1) for year in common.YEARS})
    for name in ('metrics.json', 'metrics_per_year_fixture.json'):
        (output / name).write_text(json.dumps(document))
    (output / 'metrics.md').write_text('Synthetic report\n')
    (output / 'config.json').write_text('{}\n')
    truth = np.zeros((30, 6), dtype=np.float32)
    truth[0, 0] = np.nan
    np.savez(output / 'predictions.npz', **metadata, y_pred=np.zeros_like(truth), y_true=truth,
             tags=np.array([year + '_Lewes_fixture_0' for year in common.YEARS]),
             target_timestamps=np.arange(180).reshape(30, 6), split_ids=np.array(['external'] * 30))
    inventory = [dict(path=str(root / 'unread_graph.pt'), size=123, mtime_ns=456)]
    manifest = dict(result_root=str(root / 'results'), repo=str(root / 'repo'), python=sys.executable,
                    graph_inventory={'CMIP6_EC_EARTH/Lewes': inventory})
    runtime = dict(common.DEFAULT_RUNTIME)
    fingerprint = common.digest_json(dict(row=row, graphs=inventory, runtime=runtime))
    command = common.inference_command(row, output, runtime, manifest['python'], manifest['repo'])
    provenance = dict(run=row, graph_inventory=inventory, runtime=runtime, fingerprint=fingerprint,
                      attempt=attempt, pbs_jobid='fixture', command=command)
    (staging / 'provenance.json').write_text(json.dumps(provenance))
    log = directory / 'logs' / f'{attempt}.log'
    log.parent.mkdir()
    log.write_text(shlex.join(command) + '\nSynthetic completed inference\n')
    status = dict(id=key, attempt=attempt, state='failed', error='Nonfinite physical predictions',
                  started_at=1.0, finished_at=2.0, log=str(log), staging=str(staging), pbs_jobid='fixture')
    (directory / 'status.json').write_text(json.dumps(status))
    return row, manifest, output, fingerprint


class OutputRecoveryTests(unittest.TestCase):
    def test_nan_truth_allowed_and_infinite_truth_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            row, _, output, _ = fixture(Path(temp))
            self.assertIn('predictions.npz', common.validate_outputs(output, row))
            with np.load(output / 'predictions.npz') as archive:
                arrays = {key: archive[key] for key in archive.files}
            for value in (np.inf, -np.inf):
                with self.subTest(value=value):
                    arrays['y_true'][0, 0] = value
                    np.savez(output / 'predictions.npz', **arrays)
                    with self.assertRaisesRegex(ValueError, 'Infinite physical ground truth'):
                        common.validate_outputs(output, row)

    def test_nonfinite_predictions_rejected_even_at_missing_truth(self):
        with tempfile.TemporaryDirectory() as temp:
            row, _, output, _ = fixture(Path(temp))
            with np.load(output / 'predictions.npz') as archive:
                arrays = {key: archive[key] for key in archive.files}
            for value in (np.nan, np.inf, -np.inf):
                with self.subTest(value=value):
                    arrays['y_pred'][0, 0] = value
                    np.savez(output / 'predictions.npz', **arrays)
                    with self.assertRaisesRegex(ValueError, 'Nonfinite physical predictions'):
                        common.validate_outputs(output, row)

    def test_publish_preserves_files_and_repeat_is_noop(self):
        with tempfile.TemporaryDirectory() as temp:
            row, manifest, output, fingerprint = fixture(Path(temp))
            directory = Path(row['output_dir'])
            before = {p.name: common.sha256(p) for p in output.iterdir()}
            provenance = (output.parent / 'provenance.json').read_bytes()
            log = next((directory / 'logs').iterdir())
            log_bytes = log.read_bytes()
            self.assertEqual(recovery.recover_run(row, manifest), 'recovered')
            self.assertTrue(common.verified_completion(directory, row, fingerprint))
            self.assertEqual(before, {p.name: common.sha256(p) for p in (directory / 'outputs').iterdir()})
            self.assertEqual(provenance, (directory / 'provenance.json').read_bytes())
            self.assertEqual(log_bytes, log.read_bytes())
            before_repeat = {str(p): (p.stat().st_mtime_ns, p.read_bytes()) for p in directory.rglob('*') if p.is_file()}
            self.assertEqual(recovery.recover_run(row, manifest), 'skipped')
            self.assertEqual(before_repeat, {str(p): (p.stat().st_mtime_ns, p.read_bytes())
                                            for p in directory.rglob('*') if p.is_file()})

    def test_interrupted_publication_resumes(self):
        for boundary in ('provenance', 'marker'):
            with self.subTest(boundary=boundary), tempfile.TemporaryDirectory() as temp:
                row, manifest, _, fingerprint = fixture(Path(temp))
                directory = Path(row['output_dir'])
                rename, atomic = recovery.os.rename, recovery.atomic_json

                def fail_rename(source, destination):
                    if Path(destination).name == 'provenance.json':
                        raise OSError('simulated interruption')
                    return rename(source, destination)

                def fail_marker(path, value):
                    if Path(path).name == 'COMPLETE.json':
                        raise OSError('simulated interruption')
                    return atomic(path, value)

                guard = (patch.object(recovery.os, 'rename', side_effect=fail_rename) if boundary == 'provenance'
                         else patch.object(recovery, 'atomic_json', side_effect=fail_marker))
                with guard, self.assertRaisesRegex(OSError, 'simulated interruption'):
                    recovery.recover_run(row, manifest)
                self.assertFalse((directory / 'COMPLETE.json').exists())
                self.assertEqual(recovery.recover_run(row, manifest), 'recovered')
                self.assertTrue(common.verified_completion(directory, row, fingerprint))

    def test_mismatched_provenance_refused_without_publication(self):
        with tempfile.TemporaryDirectory() as temp:
            row, manifest, output, _ = fixture(Path(temp))
            path = output.parent / 'provenance.json'
            provenance = json.loads(path.read_text())
            provenance['run']['source'] = 'CMIP6_MRI'
            path.write_text(json.dumps(provenance))
            with self.assertRaisesRegex(ValueError, 'differs from manifest run'):
                recovery.recover_run(row, manifest)
            self.assertTrue(output.is_dir())
            self.assertFalse((Path(row['output_dir']) / 'outputs').exists())
            self.assertFalse((Path(row['output_dir']) / 'COMPLETE.json').exists())

    def test_scope_is_exactly_24(self):
        manifest = dict(runs=[dict(id=common.identity(*combo)) for combo in common.combinations()])
        self.assertEqual(len(recovery.recovery_rows(manifest)), 24)
        with self.assertRaisesRegex(ValueError, 'outside'):
            recovery.recover_run(dict(id='CMIP6_AWI/Lewes/seed42/CNN_GRU_MSE'), manifest)


if __name__ == '__main__':
    unittest.main()
