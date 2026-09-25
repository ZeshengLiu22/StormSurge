"""Current shell/config interface and physical objective definitions."""

import contextlib
from dataclasses import asdict
import io
import itertools
import os
from pathlib import Path
import shlex
import subprocess
import tempfile
from types import SimpleNamespace
import unittest

import torch

import train
import infer
from emulator.data import fit_statistics
from emulator.models import ForecastOutput, ModelConfig, build_model
from emulator.training import ForecastLoss, LossConfig
from test_models import graph_batch

REPO = Path(__file__).resolve().parents[1]
MODES = ('mse', 'wmse', 'mse_slope', 'wmse_slope')


def dry_commands(config):
    environment = dict(os.environ, DRY_RUN='1')
    result = subprocess.run(['bash', 'train.sh', str(config)], cwd=REPO, env=environment,
                            capture_output=True, text=True, check=True)
    return [shlex.split(line.removeprefix('CMD: '))[1:]
            for line in result.stdout.splitlines() if line.startswith('CMD: ')]


class ConfigInterfaceTests(unittest.TestCase):
    def test_incomplete_loss_config_cannot_silently_disable_optional_objectives(self):
        stats = dict(y_mean=torch.zeros(2), y_std=torch.ones(2))
        for missing in ('excess_amp_loss_weight', 'shape_loss_weight', 'excess_formulation'):
            values = asdict(LossConfig())
            values.pop(missing)
            with self.subTest(missing=missing), self.assertRaisesRegex(AttributeError, missing):
                ForecastLoss(SimpleNamespace(**values), stats, 1.)
            # Every input uses the same explicit config-resolution step, without age/version checks.
            ForecastLoss(LossConfig(**values), stats, 1.)

    def test_every_loss_mode_keeps_explicit_amplitude_and_shape_controls(self):
        for mode in MODES:
            with self.subTest(mode=mode):
                args = train.parse_args(['--model', 'perceiver3', '--head_type', 'dual', '--loss_mode', mode,
                    '--excess_formulation', 'severity_shape', '--excess_amp_loss_weight', '.7',
                    '--shape_loss_weight', '.3'])
                self.assertEqual(args.loss_mode, mode)
                self.assertEqual(args.excess_formulation, 'severity_shape')
                self.assertEqual((args.excess_amp_loss_weight, args.shape_loss_weight),
                                 (.7, .3))

    def test_mag_ignores_unused_lower_percentile_and_keeps_used_range_checks(self):
        graph = SimpleNamespace(x=torch.tensor([[-4., 0.], [-1., 0.], [2., 0.], [9., 0.]]),
                                y=torch.tensor([.2, .7]))
        store = SimpleNamespace(graphs=[graph])
        reference = fit_statistics(store, [0], x_norm='mag', p_lo=1., p_hi=90.)
        for lower in (90., 95., -1., 101., float('nan')):
            with self.subTest(lower=lower):
                actual = fit_statistics(store, [0], x_norm='mag', p_lo=lower, p_hi=90.)
                for key in reference:
                    torch.testing.assert_close(actual[key], reference[key], rtol=0, atol=0)
                with self.assertRaisesRegex(ValueError, 'feature percentile'):
                    fit_statistics(store, [0], x_norm='robust', p_lo=lower, p_hi=90.)
        for mode, upper in itertools.product(('mag', 'robust'), (0., -1., 101., float('nan'), float('inf'))):
            with self.subTest(mode=mode, upper=upper), self.assertRaisesRegex(ValueError, 'feature percentile'):
                fit_statistics(store, [0], x_norm=mode, p_lo=1., p_hi=upper)
        maximum = fit_statistics(store, [0], x_norm='mag', p_lo=100., p_hi=100.)
        torch.testing.assert_close(maximum['x_center'], torch.zeros(2), rtol=0, atol=0)
        torch.testing.assert_close(maximum['x_scale'], torch.tensor([9., 1e-6]), rtol=0, atol=0)

    def test_train_and_infer_strip_head_and_temporal_names(self):
        blocks = dict(mlp='MLP', lstm='LSTM', gru='GRU', transformer='Transformer', attn='Transformer')
        for head, (alias, expected), whitespace in itertools.product(('single', 'dual'), blocks.items(), (' ', '\t\n')):
            with self.subTest(head=head, temporal=alias, whitespace=repr(whitespace)):
                options = ['--head_type', whitespace + head.upper() + whitespace,
                           '--temporal_block', whitespace + alias.upper() + whitespace]
                training = train.parse_args(['--model', 'perceiver3', *options])
                inference = infer.parse_args(['--ckpt', 'unused.pth', '--root_dir', '.', *options])
                self.assertEqual((training.head_type, training.temporal_block), (head, expected))
                self.assertEqual((inference.head_type, inference.temporal_block), (head, expected))
        for flag, invalid in (('--head_type', ' residual '), ('--head_type', '   '),
                              ('--temporal_block', ' ml p '), ('--temporal_block', '\t')):
            for parser, prefix in ((train.parse_args, ['--model', 'perceiver3']),
                                   (infer.parse_args, ['--ckpt', 'unused.pth', '--root_dir', '.'])):
                with self.subTest(flag=flag, invalid=invalid), self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
                    parser([*prefix, flag, invalid])

    def test_current_generator_emits_matched_valid_configs(self):
        from tools.generate_configs import generate, VARIANTS
        with tempfile.TemporaryDirectory() as temporary:
            generated = Path(temporary)
            rows = generate(generated)
            self.assertEqual(len(rows), 20)
            for row in rows:
                commands = dry_commands(generated / row['config'])
                self.assertEqual(len(commands), 1)
                args = train.parse_args(commands[0])
                self.assertEqual(args.head_type, row['head_type'])
                self.assertEqual(args.exceedance_loss_weight, row['exceedance_loss_weight'])
                self.assertEqual(args.excess_amp_loss_weight, row['excess_amp_loss_weight'])
                self.assertEqual((args.hidden_channels,args.lr,args.grad_accum_steps), (128,.005,4))
                self.assertEqual(args.exceedance_percentile,95.)
                self.assertEqual(args.checkpoint_selection, 'overall')
                checked_in = REPO / 'configs/baseline_ablation' / row['config']
                self.assertEqual((generated/row['config']).read_text(),checked_in.read_text())
            self.assertEqual(len(generate(generated,include_severity_shape=True)),36)
            for path in generated.glob('*SeverityShape.sh'):
                args = train.parse_args(dry_commands(path)[0])
                self.assertEqual(args.excess_formulation,'severity_shape')

    def test_conditional_cartesian_sweep(self):
        with tempfile.TemporaryDirectory() as temporary:
            config = Path(temporary) / 'sweep.sh'
            config.write_text('MODEL=perceiver3\nHEAD_TYPE=dual\n'
                'LOSS_MODE_LIST=(mse wmse mse_slope wmse_slope)\n'
                'LR_LIST=(.001 .005)\nHISTORY_HOURS_LIST=(0 12 48)\n'
                'EXCEEDANCE_LOSS_WEIGHT=.2\nSLOPE_LAMBDA_LIST=(.01 .02)\n'
                'SLOPE_MASK_S_LIST=(.1 .2)\nALL_RESULTS_ROOT="'+str(Path(temporary)/'results')+'"\n')
            commands=dry_commands(config)
            actual=set()
            for command in commands:
                args=train.parse_args(command)
                actual.add((args.loss_mode,args.lr,args.history_hours,args.slope_lambda,args.slope_mask_s))
                self.assertEqual(args.exceedance_loss_weight,.2)
            expected=set()
            for mode,lr,history in itertools.product(MODES,(.001,.005),(0,12,48)):
                slope=(.01,.02) if mode.endswith('_slope') else (.01,)
                mask=(.1,.2) if mode.endswith('_slope') else (.1,)
                expected.update((mode,lr,history,*values) for values in itertools.product(slope,mask))
            self.assertEqual(len(commands),60)
            self.assertEqual(actual,expected)
            self.assertFalse((Path(temporary)/'results').exists())

    def test_loss_modes_match_physical_reference_and_gradients(self):
        target=torch.tensor([[-.8,.2,.3],[.1,1.2,.7],[.3,.25,.2]],dtype=torch.float64)
        initial=torch.tensor([[-.6,.15,.6],[.2,1.,.5],[.4,.1,.35]],dtype=torch.float64)
        stats=dict(y_mean=torch.zeros(3),y_std=torch.ones(3))
        for mode,robust,tau in itertools.product(MODES,('charb','huber'),(.9,2.)):
            with self.subTest(mode=mode,robust=robust,tau=tau):
                pred=initial.clone().requires_grad_()
                config=LossConfig(loss_mode=mode,slope_robust=robust,exceedance_loss_weight=.2,
                                  slope_lambda=.3,wmse_s=.13,slope_mask_s=.21)
                actual=ForecastLoss(config,stats,tau,extreme_hour_prior=.2)(ForecastOutput(pred),pred,target)
                reference=initial.clone().requires_grad_()
                squared=(reference-target).square()
                weight=1+4*torch.sigmoid((target-tau)/.13)
                expected=(squared*weight).mean() if mode.startswith('wmse') else squared.mean()
                expected+=.2*(squared*(target>tau)).mean()/.2
                if mode.endswith('_slope'):
                    difference=reference.diff(dim=1)-target.diff(dim=1)
                    if robust=='charb':
                        penalty=torch.sqrt(difference.square()+.001**2)
                    else:
                        penalty=torch.where(difference.abs()<=.05,.5*difference.square(),.05*(difference.abs()-.025))
                    mask=torch.sigmoid((tau-target.max(dim=1).values)/.21)
                    expected+=.3*(penalty*mask[:,None]).mean()
                torch.testing.assert_close(actual,expected,rtol=1e-12,atol=1e-12)
                actual.backward(); expected.backward()
                torch.testing.assert_close(pred.grad,reference.grad,rtol=1e-12,atol=1e-12)

    def test_checkpoint_selection_accepts_only_four_roles_and_fixed_weights(self):
        self.assertEqual(train.parse_args([]).checkpoint_selection, 'overall')
        for role in ('overall', 'exceedance', 'aligned_peak', 'bea'):
            with self.subTest(role=role):
                self.assertEqual(train.parse_args(['--checkpoint_selection', role]).checkpoint_selection, role)
        for role in ('equal', 'peak_priority', 'eventaware', 'peakaware', 'BEA', 'balanced_event_aware'):
            with self.subTest(role=role), self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
                train.parse_args(['--checkpoint_selection', role])
        for option in ('ckpt_w_all', 'ckpt_w_exceedance', 'ckpt_w_peak'):
            self.assertNotIn(option, vars(train.parse_args([])))
            with self.subTest(option=option), self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
                train.parse_args([f'--{option}', '1'])
        for role in ('overall', 'exceedance', 'aligned_peak', 'bea'):
            with self.subTest(shell_role=role), tempfile.TemporaryDirectory() as temporary:
                config = Path(temporary) / 'selector.sh'
                config.write_text(f'CHECKPOINT_SELECTION={role}\nALL_RESULTS_ROOT={temporary}/results\n')
                commands = dry_commands(config)
                self.assertTrue(commands)
                for command in commands:
                    self.assertEqual(train.parse_args(command).checkpoint_selection, role)
                    self.assertFalse(any(option.startswith('--ckpt_w_') for option in command))

    def test_removed_scientific_options_are_rejected(self):
        for arguments in (['--tail_frac','.05'],['--tail_lambda','.1'],['--wmse_q','95'],
                          ['--wmse_use_abs','1'],['--loss_mode','mse_tail'],
                          ['--checkpoint_selection','peakaware'],['--save_aux_checkpoints','1'],
                          ['--track_peakaware','1'],['--checkpoint_score_refs','unused.json']):
            with self.subTest(arguments=arguments),self.assertRaises(SystemExit),contextlib.redirect_stderr(io.StringIO()):
                train.parse_args(arguments)

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
                              "SCHEDULER=rop\nROP_METRIC=val_exceedance_rmse\nROP_FACTOR=.3\nROP_PATIENCE=7\n"
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
        for arguments in (['--dual_mode', 'residual'], ['--alpha_init_logit', '-2'],
                          ['--stable_arch', '0'], ['--stable_arch', '1']):
            with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
                train.parse_args(arguments)
