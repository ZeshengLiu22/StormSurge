"""Exercise the original shell sweep and compare the loss-mode definitions."""

import contextlib
import io
import itertools
import os
from pathlib import Path
import shlex
import subprocess
import tempfile
import unittest

import torch

import train
from emulator.models import ForecastOutput, ModelConfig, build_model
from emulator.training import ForecastLoss, LossConfig
from test_models import graph_batch

REPO = Path(__file__).resolve().parents[1]
MODES = ('mse', 'wmse', 'mse_tail', 'wmse_tail', 'mse_wtail',
         'mse_slope', 'wmse_slope', 'mse_tail_slope', 'wmse_tail_slope', 'mse_wtail_slope')


def dry_commands(config):
    environment = dict(os.environ, DRY_RUN='1')
    result = subprocess.run(['bash', 'train.sh', str(config)], cwd=REPO, env=environment,
                            capture_output=True, text=True, check=True)
    return [shlex.split(line.removeprefix('CMD: '))[1:]
            for line in result.stdout.splitlines() if line.startswith('CMD: ')]


class ConfigInterfaceTests(unittest.TestCase):
    def test_every_training_profile_generates_valid_commands(self):
        configs = sorted([*REPO.glob('configs/configs_train*/*/*.sh'),
                          *REPO.glob('configs/train_config_*_Stable_*.sh')])
        self.assertEqual(len(configs), 114)
        count = 0
        for config in configs:
            with self.subTest(config=config.relative_to(REPO)):
                commands = dry_commands(config)
                self.assertTrue(commands)
                for command in commands:
                    args = train.parse_args(command)
                    self.assertEqual(args.dual_mode, 'exceedance')
                    self.assertEqual(args.stable_arch, 1)
                    if 'configs_train_single' in config.parts:
                        self.assertEqual(args.grad_accum_steps, 4)
                    if args.model == 'perceiver3':
                        self.assertEqual(args.head_type, 'single' if 'Stable_Single' in config.name else 'dual')
                    count += 1
        self.assertGreaterEqual(count, 114)

    def test_original_conditional_cartesian_sweep_has_all_390_combinations(self):
        with tempfile.TemporaryDirectory() as temporary:
            config = Path(temporary) / 'sweep.sh'
            config.write_text('''MODEL=perceiver3
HEAD_TYPE=dual
LOSS_MODE_LIST=(mse wmse mse_tail wmse_tail mse_wtail mse_slope wmse_slope mse_tail_slope wmse_tail_slope mse_wtail_slope)
LR_LIST=(.001 .005)
HISTORY_HOURS_LIST=(0 12 48)
WMSE_Q_LIST=(90 95)
TAIL_LAMBDA_LIST=(.1 .2)
SLOPE_LAMBDA_LIST=(.01 .02)
SLOPE_MASK_S_LIST=(.1 .2)
ALL_RESULTS_ROOT="'''+str(Path(temporary) / 'results')+'"\n')
            commands = dry_commands(config)
            actual = set()
            for command in commands:
                args = train.parse_args(command)
                actual.add((args.loss_mode, args.lr, args.history_hours, args.wmse_q,
                            args.tail_lambda, args.slope_lambda, args.slope_mask_s))
            expected = set()
            for mode, lr, history in itertools.product(MODES, (.001, .005), (0, 12, 48)):
                q = (90., 95.) if mode.startswith('wmse') or 'wtail' in mode else (90.,)
                tail = (.1, .2) if 'tail' in mode else (.1,)
                slope = (.01, .02) if mode.endswith('_slope') else (.01,)
                mask = (.1, .2) if mode.endswith('_slope') else (.1,)
                expected.update((mode, lr, history, *values) for values in itertools.product(q, tail, slope, mask))
            self.assertEqual(len(commands), 390)
            self.assertEqual(actual, expected)
            self.assertFalse((Path(temporary) / 'results').exists())

    def test_loss_modes_match_physical_unit_reference_and_gradients(self):
        # Different pointwise and window thresholds catch the previous slope-threshold regression.
        target = torch.tensor([[-.8, .2, .3], [.1, 1.2, .7], [.3, .25, .2]], dtype=torch.float64)
        initial = torch.tensor([[-.6, .15, .6], [.2, 1., .5], [.4, .1, .35]], dtype=torch.float64)
        stats = dict(y_mean=torch.zeros(3), y_std=torch.ones(3))
        base_weighted = {'wmse', 'wmse_tail'}
        tail_weighted = {'wmse_tail', 'mse_wtail'}
        for mode, robust, use_abs, threshold in itertools.product(MODES, ('charb', 'huber'), (False, True), (.9, 2.)):
            with self.subTest(mode=mode, robust=robust, use_abs=use_abs, threshold=threshold):
                pred = initial.clone().requires_grad_()
                c = LossConfig(loss_mode=mode, slope_robust=robust, wmse_use_abs=use_abs,
                               tail_lambda=.2, slope_lambda=.3, wmse_s=.13, slope_mask_s=.21)
                actual = ForecastLoss(c, stats, threshold, .4)(ForecastOutput(pred), pred, target)
                reference_pred = initial.clone().requires_grad_()
                squared = (reference_pred - target).square()
                weight = 1 + 4 * torch.sigmoid(((target.abs() if use_abs else target) - .4) / .13)
                core = mode.removesuffix('_slope')
                expected = (squared * weight).mean() if core in base_weighted else squared.mean()
                selected = target.max(dim=1).values >= threshold
                if 'tail' in core and selected.any():
                    tail_error = (squared * weight) if core in tail_weighted else squared
                    expected += .2 * tail_error[selected].mean()
                if mode.endswith('_slope'):
                    difference = (reference_pred[:, 1:] - reference_pred[:, :-1]) - (target[:, 1:] - target[:, :-1])
                    if robust == 'charb':
                        penalty = torch.sqrt(difference**2 + .001**2)
                    else:
                        penalty = torch.where(difference.abs() <= .05, .5 * difference**2, .05 * (difference.abs() - .025))
                    mask = torch.sigmoid((.4 - target.abs().max(dim=1).values) / .21)
                    expected += .3 * (penalty * mask[:, None]).mean()
                torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
                actual.backward()
                expected.backward()
                torch.testing.assert_close(pred.grad, reference_pred.grad, rtol=1e-12, atol=1e-12)

    def test_lag_embedding_capacity_covers_the_history_window(self):
        torch.set_num_threads(1)
        for history, head in itertools.product((0, 8), ('single', 'dual')):
            c = ModelConfig(3, 4, head_type=head, hidden_channels=16, history_steps=history,
                            node_read_heads=2, time_read_heads=2, max_time_steps=20,
                            peak_threshold_norm=[1.] * 4)
            network = build_model(c).eval()
            output = network(graph_batch(history + 1))
            output.prediction.square().mean().backward()
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in network.parameters()))
            self.assertEqual(network.lag_embed.num_embeddings, 20)
            c.max_time_steps = history
            with self.assertRaisesRegex(ValueError, 'max_time_steps'):
                build_model(c)

    def test_rop_configuration_reaches_the_scheduler_and_snapshot(self):
        from unittest.mock import patch
        from test_pipeline import make_fixture
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            graphs, stations = make_fixture(root)
            config = root / 'rop.sh'
            config.write_text(f"ROOT_DIR={graphs}\nSTATION=Battery\nSTATION_JSON_DIR={stations}\n"
                              "SCHEDULER=rop\nROP_METRIC=val_rmse_peak\nROP_FACTOR=.3\nROP_PATIENCE=7\n"
                              "ROP_THRESHOLD=.02\nROP_COOLDOWN=4\nROP_MIN_LR=.00002\n")
            command = dry_commands(config)[0]
            # Do not create artifacts in the default sweep directory during this test.
            command[command.index('--output_dir') + 1] = str(root / 'run')
            command += ['--device', 'cpu', '--num_workers', '0']
            factory = torch.optim.lr_scheduler.ReduceLROnPlateau
            with patch.object(train.torch.optim.lr_scheduler, 'ReduceLROnPlateau', wraps=factory) as observed:
                with patch.object(train, 'run_epoch', side_effect=StopIteration), self.assertRaises(StopIteration):
                    train.main(command)
                self.assertEqual(observed.call_args.kwargs, dict(factor=.3, patience=7, threshold=.02,
                                                                  cooldown=4, min_lr=.00002))
            snapshot = (root / 'run' / 'config_used.sh').read_text()
            self.assertIn('ROP_COOLDOWN=4', snapshot)
            self.assertIn('ROP_THRESHOLD=0.02', snapshot)

    def test_removed_residual_head_flags_fail_instead_of_being_ignored(self):
        for arguments in (['--dual_mode', 'residual'], ['--alpha_init_logit', '-2'], ['--stable_arch', '0']):
            with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
                train.parse_args(arguments)
