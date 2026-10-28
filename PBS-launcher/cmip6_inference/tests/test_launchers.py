"""Synthetic CPU tests for source statistics, orchestration, restart, and failure paths."""
import contextlib
from dataclasses import asdict
import io
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common
import generate
import launch
import station_runner
import submit

import numpy as np
import torch
from torch_geometric.data import Data
import infer
from emulator.models import ModelConfig, build_model
from emulator.training.metrics import evaluate_metrics


def fixture(root, model='CNN_GRU_MSE'):
    """Thirty unequal winter populations; the historical root deliberately does not exist."""
    source, station, seed = 'CMIP6_AWI', 'Battery', 42
    key = common.identity(source, station, seed, model)
    graphs = root / 'data' / source / 'graphs'
    graphs.mkdir(parents=True, exist_ok=True)
    for index, year in enumerate(common.YEARS):
        entries = []
        for sample in range(1 + index % 3):
            history = torch.ones(2, 6, 5)
            entries.append(Data(x=history[-1], x_hist=history,
                                p_mean_hist=torch.full((2,), 1000.0), p_mean_curr=torch.tensor(1000.0),
                                y=torch.tensor([0.1 + index / 10, 0.9 + sample / 10]),
                                center_time=f'{year[:4]}-11-01 {sample * 6:02d}:00:00', grid_H=2, grid_W=3))
        torch.save(entries, graphs / f'{year}_{station}_fixture_graphs.pt')
    # A broken historical file proves filtering occurs before deserialization.
    (graphs / '1979_1980_Battery_unused_graphs.pt').write_bytes(b'not a graph')
    config = ModelConfig(5, 2, 8, model='baseline_cnn_gru' if model == 'CNN_GRU_MSE' else 'pact',
                         encoder_type='CNN', temporal_block='MLP', history_steps=1,
                         baseline_stem_channels=4, cnn_intermediate_channel=4,
                         node_read_heads=2, time_read_heads=2)
    network = build_model(config)
    for parameter in network.parameters():
        parameter.data.zero_()
    threshold = dict(tau_physical=0.8, exceedance_percentile=95.0, fitted_on='train',
                     threshold_schema='train_hourly_q95_v1', metric_schema='hourly_q95_v1')
    original = root / 'unmounted' / (source + '_PastOnly') / 'graphs'
    tail, peak = common.LOSSES[model]
    checkpoint = dict(model_config=asdict(config), model_state=network.state_dict(), station=station, station_feat=None,
                      training_config=dict(root_dir=str(original), station=station, seed=seed, model=common.MODELS[model],
                                           deterministic=False, use_site_elevation=0, use_bathymetry=0,
                                           x_norm='zscore', x_clip=0.0, pressure_mode='raw', checkpoint_selection='overall',
                                           exceedance_loss_weight=tail, episode_gt_aligned_peak_weight=peak),
                      normalization=dict(x_center=torch.zeros(5), x_scale=torch.ones(5),
                                         y_mean=torch.tensor([0.25, -0.5]), y_std=torch.ones(2)),
                      threshold_metadata=threshold, epoch=3, checkpoint_role='overall', selection_split='val',
                      selection_metric_key='all_rmse', split_tags=dict(train=[], val=[], test=[]))
    ckpt = root / 'checkpoints' / key / 'timestamped_training_run' / 'best_overall.pt'
    ckpt.parent.mkdir(parents=True)
    torch.save(checkpoint, ckpt)
    (ckpt.parent / 'exit_status').write_text('0\n')
    (ckpt.parent / 'summary_fixture.json').write_text(json.dumps(dict(checkpoint_selection=dict(
        checkpoint_selection_mode='overall', selected_epoch=3))))
    metadata = common.inspect_checkpoint(ckpt, source, station, seed, model)
    row = dict(id=key, source=source, station=station, seed=seed, model=model, model_key=common.MODELS[model],
               checkpoint=str(ckpt), checkpoint_metadata=metadata, source_root=str(original),
               test_root_dir=str(graphs), output_dir=str(root / 'results' / key))
    manifest = dict(repo=str(common.REPO), graph_inventory={source + '/' + station: common.graph_inventory(graphs, station)})
    return row, manifest, checkpoint


class CPUControl(station_runner.ProcessControl):
    """Execute the exact production command through infer.main with only device changed."""
    def run(self, command, log, lock_fd, cwd):
        args = command[3:]
        args[args.index('--device') + 1] = 'cpu'
        with (contextlib.redirect_stdout(log), patch('emulator.data.fit_statistics', side_effect=AssertionError('refit')),
              patch('emulator.data.fit_loss_thresholds', side_effect=AssertionError('refit'))):
            infer.main(args)
        return 0


class LauncherTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_exact_matrix_and_resources(self):
        self.assertEqual(len(list(common.combinations())), 432)
        self.assertEqual((len(common.YEARS), common.YEARS[0], common.YEARS[-1]), (30, '2070_2071', '2099_2100'))
        self.assertEqual(set(common.MODELS), {'CNN_GRU_MSE', 'PACT_MSE', 'PACT_Peak', 'PACT_Both'})
        for cluster in ('derecho', 'casper'):
            text = generate.pbs_text(cluster, 'CMIP6_AWI', 'Battery' if cluster == 'casper' else None,
                                     Path('/launchers'), '01:00:00')
            self.assertIn(common.RESOURCES[cluster], text)
            self.assertIn('#PBS -r n', text)
            self.assertIn('exec bash', text)
            self.assertNotIn('torchrun', text)

    def test_gpu_mask_order_uuid_and_invalid_masks(self):
        self.assertEqual(launch.derecho_devices('3,1,2,0'), ['3', '1', '2', '0'])
        self.assertEqual(launch.derecho_devices('GPU-a,GPU-b,GPU-c,GPU-d'), ['GPU-a', 'GPU-b', 'GPU-c', 'GPU-d'])
        self.assertEqual(launch.derecho_devices(None), ['0', '1', '2', '3'])
        for mask in ('', '0', '0,0,1,2', '0,1,2,-1'):
            with self.assertRaises(ValueError):
                launch.derecho_devices(mask)

    def test_checkpoint_ambiguity_role_and_metadata(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            row, manifest, checkpoint = fixture(root)
            path = Path(row['checkpoint'])
            resolve = lambda selection=None: common.resolve_checkpoint(root / 'checkpoints', 'CMIP6_AWI', 'Battery', 42, 'CNN_GRU_MSE', selection)
            self.assertEqual(resolve(), path)
            import shutil
            duplicate = path.parent.with_name('another_completed_run')
            shutil.copytree(path.parent, duplicate)
            with self.assertRaisesRegex(ValueError, 'found 2'):
                resolve()
            self.assertEqual(resolve({row['id']: str(path)}), path)
            (duplicate / 'exit_status').write_text('1')
            self.assertEqual(resolve(), path)
            with self.assertRaisesRegex(ValueError, 'station mismatch'):
                common.inspect_checkpoint(path, 'CMIP6_AWI', 'Boston', 42, 'CNN_GRU_MSE')
            with self.assertRaisesRegex(ValueError, 'seed mismatch'):
                common.inspect_checkpoint(path, 'CMIP6_AWI', 'Battery', 123, 'CNN_GRU_MSE')
            with self.assertRaisesRegex(ValueError, 'cross-GCM'):
                common.inspect_checkpoint(path, 'CMIP6_MPI', 'Battery', 42, 'CNN_GRU_MSE')
            checkpoint['normalization']['x_scale'][0] = 0
            torch.save(checkpoint, path)
            with self.assertRaisesRegex(ValueError, 'nonpositive'):
                common.inspect_checkpoint(path, 'CMIP6_AWI', 'Battery', 42, 'CNN_GRU_MSE')
            path.rename(path.with_name('best_exceedance.pt'))
            with self.assertRaisesRegex(ValueError, 'found 0'):
                resolve()

    def test_graph_filter_missing_duplicate_and_memory_guard(self):
        with tempfile.TemporaryDirectory() as temp:
            row, _, _ = fixture(Path(temp))
            root = Path(row['test_root_dir'])
            files = common.graph_inventory(root, 'Battery')
            self.assertEqual(len(files), 30)
            self.assertTrue(all('207' in p['path'] or '208' in p['path'] or '209' in p['path'] for p in files))
            original = Path(files[0]['path'])
            duplicate = original.with_name('2070_2071_Battery_duplicate_graphs.pt')
            duplicate.write_bytes(original.read_bytes())
            with self.assertRaisesRegex(ValueError, 'Duplicate'):
                common.graph_inventory(root, 'Battery')
            duplicate.unlink()
            original.unlink()
            with self.assertRaisesRegex(ValueError, '2070_2071'):
                common.graph_inventory(root, 'Battery')
            self.assertGreater(common.memory_estimate(files, 1), common.memory_estimate(files, 0))

    def test_real_thirty_year_inference_atomic_restart_and_corruption(self):
        with tempfile.TemporaryDirectory() as temp:
            row, manifest, checkpoint = fixture(Path(temp))
            runtime = dict(common.DEFAULT_RUNTIME, batch_size=2, amp=False)
            control = CPUControl()
            self.assertFalse(Path(row['source_root']).exists())
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(station_runner.execute_run(row, manifest, runtime, control, {}), 'success')
                self.assertEqual(station_runner.execute_run(row, manifest, runtime, control, {}), 'skipped')
            final = Path(row['output_dir'])
            self.assertEqual(json.loads((final / 'status.json').read_text())['state'], 'success')
            metrics = json.loads((final / 'outputs/metrics.json').read_text())
            with np.load(final / 'outputs/predictions.npz', allow_pickle=False) as values:
                reference = evaluate_metrics(values['y_pred'], values['y_true'], 0.8,
                                             target_timestamps=values['target_timestamps'], split_ids=values['split_ids'])
                self.assertEqual(metrics['metrics'], reference)
                np.testing.assert_array_equal(values['y_pred'], np.tile([0.25, -0.5], (60, 1)))
            self.assertNotAlmostEqual(metrics['metrics']['all_rmse'], np.mean([metrics['results'][y]['all_rmse'] for y in common.YEARS]))
            fingerprint = json.loads((final / 'COMPLETE.json').read_text())['fingerprint']
            (final / 'outputs/predictions.npz').write_bytes(b'partial zip')
            self.assertFalse(common.verified_completion(final, row, fingerprint))
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(station_runner.execute_run(row, manifest, runtime, control, {}), 'success')
            self.assertEqual(len(list((final / 'previous_outputs').iterdir())), 1)
            self.assertTrue(common.verified_completion(final, row, fingerprint))
            metrics['normalization_origin'] = 'future_fit'
            (final / 'outputs/metrics.json').write_text(json.dumps(metrics))
            self.assertFalse(common.verified_completion(final, row, fingerprint))

    def test_pact_checkpoint_architecture_and_stats(self):
        with tempfile.TemporaryDirectory() as temp:
            row, _, _ = fixture(Path(temp), 'PACT_Both')
            command = common.inference_command(row, '/out', common.DEFAULT_RUNTIME)
            self.assertIn('perceiver3', command)
            for option in ('--encoder_type', '--temporal_block', '--history_hours', '--pressure_mode'):
                self.assertNotIn(option, command)
            self.assertIn('--strict_years', command)
            self.assertIn('--save_npz', command)

    def test_partial_failure_and_interrupt_never_complete(self):
        with tempfile.TemporaryDirectory() as temp:
            row, manifest, _ = fixture(Path(temp))
            for exception in (None, station_runner.StopRequested('TERM')):
                control = station_runner.ProcessControl()
                with patch.object(control, 'run', side_effect=exception, return_value=3), self.assertRaises(Exception):
                    station_runner.execute_run(row, manifest, common.DEFAULT_RUNTIME, control, {})
                self.assertFalse((Path(row['output_dir']) / 'COMPLETE.json').exists())
                state = json.loads((Path(row['output_dir']) / 'status.json').read_text())['state']
                self.assertEqual(state, 'failed' if exception is None else 'interrupted')

    def test_lock_prevents_concurrent_writer_and_is_reusable(self):
        with tempfile.TemporaryDirectory() as temp:
            with common.run_lock(temp):
                with self.assertRaises(RuntimeError):
                    with common.run_lock(temp):
                        self.fail('Lock acquired twice')
            with common.run_lock(temp):
                pass

    def test_supervisor_propagates_failure_and_waits_for_all(self):
        with tempfile.TemporaryDirectory() as temp:
            marker = Path(temp) / 'finished'
            specs = [('bad', [sys.executable, '-c', 'raise SystemExit(7)'], os.environ.copy(), None),
                     ('good', [sys.executable, '-c', f'import pathlib,time; time.sleep(.2); pathlib.Path({str(marker)!r}).touch()'], os.environ.copy(), None)]
            self.assertEqual(launch.supervise(specs), 1)
            self.assertTrue(marker.exists())

    def test_launcher_gpu_assignment_and_casper_preservation(self):
        with tempfile.TemporaryDirectory() as temp:
            with (patch.object(launch, '__file__', str(Path(temp) / 'launch.py')),
                  patch.dict(os.environ, dict(PBS_JOBID='test', CUDA_VISIBLE_DEVICES='3,1,2,0', WORLD_SIZE='4')),
                  patch.object(launch, 'supervise', return_value=0) as supervisor):
                self.assertEqual(launch.main(['derecho', 'CMIP6_AWI']), 0)
                specs = supervisor.call_args.args[0]
                self.assertEqual([s[0] for s in specs], list(common.STATIONS))
                self.assertEqual([s[2]['CUDA_VISIBLE_DEVICES'] for s in specs], ['3', '1', '2', '0'])
                self.assertTrue(all('WORLD_SIZE' not in s[2] for s in specs))
                os.environ['CUDA_VISIBLE_DEVICES'] = 'GPU-casper-assigned'
                self.assertEqual(launch.main(['casper', 'CMIP6_AWI', 'Boston']), 0)
                self.assertEqual(supervisor.call_args.args[0][0][2]['CUDA_VISIBLE_DEVICES'], 'GPU-casper-assigned')

    def test_supervisor_sigterm_reaches_all_children(self):
        with tempfile.TemporaryDirectory() as temp:
            specs = []
            for index in range(2):
                path = Path(temp) / f'stopped{index}'
                code = ('import signal,time,pathlib; '
                        f'signal.signal(signal.SIGTERM, lambda *args: (pathlib.Path({str(path)!r}).touch(), exit(0))); '
                        'time.sleep(20)')
                specs.append((str(index), [sys.executable, '-c', code], os.environ.copy(), None))
            previous = signal.signal(signal.SIGALRM, lambda *args: signal.raise_signal(signal.SIGTERM))
            signal.setitimer(signal.ITIMER_REAL, 0.6)
            try:
                self.assertEqual(launch.supervise(specs), 143)
                self.assertEqual(len(list(Path(temp).glob('stopped*'))), 2)
            finally:
                signal.setitimer(signal.ITIMER_REAL, 0)
                signal.signal(signal.SIGALRM, previous)

    def test_signal_forwarding_terminates_inference_group(self):
        with tempfile.TemporaryDirectory() as temp:
            marker = Path(temp) / 'child_ready'
            program = ('import os,signal,time,pathlib; '
                       'signal.signal(signal.SIGTERM, lambda *args: exit(23)); '
                       f'pathlib.Path({str(marker)!r}).write_text(str(os.getpid())); '
                       'time.sleep(20)')
            control = station_runner.ProcessControl()
            previous = signal.signal(signal.SIGALRM, lambda *args: control.handle(signal.SIGTERM, None))
            signal.setitimer(signal.ITIMER_REAL, 0.5)
            try:
                with common.run_lock(temp) as fd, open(os.devnull, 'w') as log:
                    with self.assertRaises(station_runner.StopRequested):
                        control.run([sys.executable, '-c', program], log, fd, temp)
                if marker.exists():
                    with self.assertRaises(ProcessLookupError):
                        os.kill(int(marker.read_text()), 0)
            finally:
                signal.setitimer(signal.ITIMER_REAL, 0)
                signal.signal(signal.SIGALRM, previous)

    def test_submission_preview_never_calls_scheduler(self):
        mock_manifest = dict(runs=[dict(source='CMIP6_AWI', station='Battery', derecho_pbs='/tmp/job.pbs')])
        with (patch.object(submit, 'load_manifest', return_value=mock_manifest),
              patch.object(submit.subprocess, 'run') as scheduler, contextlib.redirect_stdout(io.StringIO())):
            self.assertEqual(submit.main(['derecho']), 0)
            scheduler.assert_not_called()

    def test_submission_refuses_other_active_strategy(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            ledger = root / '.submissions'
            ledger.mkdir()
            (ledger / 'jobs.json').write_text(json.dumps([dict(cluster='casper', jobid='123.server')]))
            manifest = dict(result_root=temp, runs=[dict(source='CMIP6_AWI', station='Battery', derecho_pbs='/tmp/job.pbs')])
            with (patch.object(submit, 'load_manifest', return_value=manifest),
                  patch('preflight.validate'), patch.object(submit, 'active_job', return_value=True),
                  patch.object(submit.subprocess, 'run') as scheduler):
                with self.assertRaisesRegex(ValueError, 'still active'):
                    submit.main(['derecho', '--submit'])
                scheduler.assert_not_called()

    def test_scheduler_status_uncertainty_fails_closed(self):
        result = subprocess.CompletedProcess([], 1, '', 'cannot connect to server')
        with patch.object(submit.subprocess, 'run', return_value=result):
            with self.assertRaisesRegex(ValueError, 'Cannot establish status'):
                submit.active_job('123.server')


if __name__ == '__main__':
    unittest.main()
