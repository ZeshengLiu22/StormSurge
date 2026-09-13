"""Final physical peak risk: exact TRAIN events, gradients, and orthogonal controls."""

import contextlib
import copy
from dataclasses import asdict
from datetime import timedelta
import io
import itertools
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel

import train
from emulator.common.dual import DUAL_ABLATIONS
from emulator.data import fit_loss_thresholds
from emulator.models import ForecastOutput, ModelConfig, build_model
from emulator.models.heads import ExceedanceHead, SeverityShapeHead
from emulator.training import ForecastLoss, LossConfig
from emulator.training.excess_amplitude import peak_pool
from emulator.training.final_peak import final_peak_terms
from test_config_interfaces import MODES, dry_commands
from test_models import graph_batch


def peak_ddp_fixture():
    torch.manual_seed(716)
    model = torch.nn.Linear(2, 3)
    x = torch.arange(8, dtype=torch.float32).reshape(4, 2) / 8
    target = torch.tensor([[1., 0., 0.], [0., .5, 1.], [2., 0., 1.], [0., 3., 1.]])
    return model, x, target


def peak_ddp_worker(rank, rendezvous, destination):
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method=Path(rendezvous).as_uri(), rank=rank,
                            world_size=2, timeout=timedelta(seconds=30))
    try:
        model, x, target = peak_ddp_fixture()
        network = DistributedDataParallel(model)
        results = {}
        for pool in ('max', 'smoothmax'):
            network.zero_grad(set_to_none=True)
            terms = final_peak_terms(network(x[2 * rank:2 * rank + 2]),
                                     target[2 * rank:2 * rank + 2], 1., .17, pool, 2.)
            if rank == 0:
                assert terms.loss.item() == 0.
            terms.loss.backward()
            loss = terms.loss.detach().clone()
            dist.all_reduce(loss)
            results[pool] = dict(loss=loss / 2, grads={n: p.grad.clone() for n, p in model.named_parameters()})
        if rank == 0:
            torch.save(results, destination)
    finally:
        dist.destroy_process_group()


class FinalPeakTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_equation_is_final_physical_peak_and_ignores_normalized_output(self):
        prediction = torch.tensor([[1., 3.], [30., 50.]], requires_grad=True)
        target = torch.tensor([[5., 2.], [1., 2.]])
        stats = dict(y_mean=torch.tensor([10., -2.]), y_std=torch.tensor([.5, 4.]))
        # Unrelated normalized values must have no effect on this physical loss.
        output = ForecastOutput(torch.full_like(prediction, 1000.))
        terms = final_peak_terms(prediction, target, 2., .2)
        self.assertEqual(terms.event.tolist(), [True, False])
        self.assertEqual(terms.loss.item(), 10.)
        criterion = ForecastLoss(LossConfig(peak_loss_weight=.5), stats, 4., 1., .2, 2.)
        expected = (prediction - target).square().mean() + .5 * terms.loss
        torch.testing.assert_close(criterion(output, prediction, target), expected, rtol=0, atol=0)
        self.assertNotEqual(peak_pool(output.prediction).tolist(), terms.m_pred.tolist())

    def test_strict_event_threshold_is_separate_from_tail_threshold_mandatory(self):
        target = torch.tensor([[2., 1.], [3., 1.], [5., 1.]])
        prediction = torch.zeros_like(target)
        stats = dict(y_mean=torch.zeros(2), y_std=torch.ones(2))
        c = LossConfig(loss_mode='mse_tail', tail_frac=.5, tail_lambda=.3, peak_loss_weight=.4)
        legacy = copy.deepcopy(c)
        legacy.peak_loss_weight = 0
        # tau=2 selects 3 and 5, while the tail threshold=5 selects only 5.
        terms = final_peak_terms(prediction, target, 2., .25)
        self.assertEqual(terms.event.tolist(), [False, True, True])
        base = ForecastLoss(legacy, stats, 5., 1.)(ForecastOutput(prediction), prediction, target)
        total = ForecastLoss(c, stats, 5., 1., .25, 2.)(ForecastOutput(prediction), prediction, target)
        torch.testing.assert_close(total, base + .4 * (torch.tensor(34.) / 3 / .25), rtol=0, atol=0)

    def test_threshold_comparison_preserves_fitted_float64_precision(self):
        # This TRAIN percentile lies below 1 but rounds to 1 in FP32.
        tau = 1. - 1e-8
        terms = final_peak_terms(torch.zeros(1, 2), torch.tensor([[1., 0.]]), tau, .17)
        self.assertTrue(terms.event.item())
        self.assertFalse(final_peak_terms(torch.zeros(1, 2), torch.tensor([[1., 0.]]), 1., .17).event.item())

    def test_smooth_pool_never_changes_hard_truth_event_population(self):
        target = torch.tensor([[2.1, -20., -20.], [2., 2., 2.]])
        terms = final_peak_terms(torch.zeros_like(target), target, 2., .17, 'smoothmax', .01)
        self.assertLess(terms.m_true[0], 2.)
        self.assertEqual(terms.event.tolist(), [True, False])

    def test_exact_fitted_prior_not_tail_nominal_batch_or_clipped_prior(self):
        store = SimpleNamespace(graphs=[SimpleNamespace(y=torch.tensor([float(v), -1.]))
                                        for v in (0, 1, 2, 2, 2, 5, 10000)])
        fitted = fit_loss_thresholds(store, list(range(6)), tail_frac=.5, exceedance_percentile=75)
        self.assertEqual((fitted['tau_phys'], fitted['event_prior']), (2., 1 / 6))
        prediction = torch.tensor([[3., 1.], [40., 40.]])
        target = torch.tensor([[5., -1.], [2., -1.]])
        terms = final_peak_terms(prediction, target, fitted['tau_phys'], fitted['event_prior'])
        self.assertEqual(terms.loss.item(), 12.)
        for wrong in (.5, .25):
            self.assertNotEqual(terms.loss.item(), 2 / wrong)
        self.assertEqual(final_peak_terms(prediction, target, 2., 1e-8).loss.item(), 2e8)
        store.graphs[-1].y.fill_(-1e6)
        self.assertEqual(fitted, fit_loss_thresholds(store, list(range(6)), tail_frac=.5, exceedance_percentile=75))

    def test_each_event_weight_is_fixed_as_equal_size_batch_composition_changes(self):
        values = []
        for count in (1, 2):
            prediction = torch.tensor([[1., 3.]] * 4, requires_grad=True)
            target = torch.ones(4, 2)
            target[:count, 0] = 5.
            terms = final_peak_terms(prediction, target, 2., .2)
            terms.loss.backward()
            values.append((terms.loss.item(), prediction.grad[0].clone()))
            torch.testing.assert_close(prediction.grad[count:], torch.zeros_like(prediction.grad[count:]), rtol=0, atol=0)
        self.assertEqual(values[1][0], 2 * values[0][0])
        torch.testing.assert_close(values[0][1], values[1][1], rtol=0, atol=0)
        self.assertEqual(values[0][1].tolist(), [0., -5.])

    def test_no_event_batch_is_exact_zero_with_finite_zero_gradients(self):
        for pool, magnitude in itertools.product(('max', 'smoothmax'), (3., 1e20)):
            prediction = torch.tensor([[1., magnitude], [magnitude, 2.]], requires_grad=True)
            terms = final_peak_terms(prediction, torch.ones(2, 2), 1., .17, pool)
            self.assertEqual(terms.loss.item(), 0.)
            terms.loss.backward()
            torch.testing.assert_close(prediction.grad, torch.zeros_like(prediction), rtol=0, atol=0)

    def test_hard_max_gradient_reaches_only_unique_selected_predicted_peak(self):
        prediction = torch.tensor([[1., 3., 2.]], requires_grad=True)
        target = torch.tensor([[5., 1., 0.]])
        final_peak_terms(prediction, target, 2., .25).loss.backward()
        torch.testing.assert_close(prediction.grad, torch.tensor([[0., -16., 0.]]), rtol=0, atol=0)

    def test_both_pools_reuse_post2_implementation_and_identical_truth_is_zero(self):
        values = torch.tensor([[.1, .7, 1.3], [3., 2., .8]], requires_grad=True)
        for pool in ('max', 'smoothmax'):
            with patch('emulator.training.final_peak.peak_pool', wraps=peak_pool) as pooled:
                terms = final_peak_terms(values, values.detach().clone(), .5, .17, pool, 2.)
            self.assertEqual(terms.loss.item(), 0.)
            self.assertEqual(pooled.call_args_list[0].args[1:], (pool, 2.))
            self.assertEqual(pooled.call_args_list[1].args[1:], (pool, 2.))
            terms.loss.backward()
            self.assertTrue(torch.isfinite(values.grad).all())

    def test_smoothmax_is_finite_and_converges_to_hard_peak_loss(self):
        pred = torch.tensor([[.1, .7, 1.3], [3., 2., .8]], requires_grad=True)
        truth = torch.tensor([[.1, .6, 2.], [4., 1., 0.]])
        hard = final_peak_terms(pred, truth, .5, .17)
        for beta in (.1, 2., 100., 1e300):
            terms = final_peak_terms(pred, truth, .5, .17, 'smoothmax', beta)
            self.assertTrue(torch.isfinite(terms.loss))
            gradient, = torch.autograd.grad(terms.loss, pred)
            self.assertTrue(torch.isfinite(gradient).all())
            if beta >= 100:
                torch.testing.assert_close(terms.loss, hard.loss, rtol=0, atol=0)

    def test_config_defaults_validation_and_single_baseline_support(self):
        for model, head in itertools.product(('baseline', 'perceiver3'), ('single', 'dual')):
            args = train.parse_args(['--model', model, '--head_type', head])
            self.assertEqual((args.peak_loss_weight, args.peak_pool, args.peak_pool_beta), (0., 'max', 20.))
            args = train.parse_args(['--model', model, '--head_type', head, '--peak_loss_weight', '.4'])
            self.assertEqual(args.peak_loss_weight, .4)
        invalid = [['--peak_loss_weight', value] for value in ('-1', 'nan', 'inf')]
        invalid += [['--peak_pool', 'bad']]
        invalid += [['--peak_loss_weight', '1', '--peak_pool', 'smoothmax', '--peak_pool_beta', value]
                    for value in ('0', '-1', 'nan', 'inf')]
        for options in invalid:
            with self.subTest(options=options), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                train.parse_args(options)
        for weight, pool in ((0, 'smoothmax'), (1, 'max')):
            train.parse_args(['--peak_loss_weight', str(weight), '--peak_pool', pool, '--peak_pool_beta', 'nan'])
        stats = dict(y_mean=torch.zeros(2), y_std=torch.ones(2))
        for c in (LossConfig(peak_loss_weight=-1), LossConfig(peak_loss_weight=float('nan')),
                  LossConfig(peak_pool='bad'), LossConfig(peak_loss_weight=1, peak_pool='smoothmax', peak_pool_beta=0)):
            with self.assertRaises(ValueError):
                ForecastLoss(c, stats, 1., 1., .17, 1.)

    def test_prior_and_event_threshold_required_only_for_active_peak_objective(self):
        stats = dict(y_mean=torch.zeros(2), y_std=torch.ones(2))
        for prior in (None, 0., -1., float('nan'), float('inf')):
            ForecastLoss(LossConfig(), stats, 1., 1., prior)
            with self.assertRaisesRegex(ValueError, 'TRAIN event_prior'):
                ForecastLoss(LossConfig(peak_loss_weight=1), stats, 1., 1., prior, 1.)
        for tau in (None, float('nan'), float('inf')):
            ForecastLoss(LossConfig(), stats, 1., 1., .17, tau)
            with self.assertRaisesRegex(ValueError, 'TRAIN event_threshold'):
                ForecastLoss(LossConfig(peak_loss_weight=1), stats, 1., 1., .17, tau)

    def _dual(self, formulation, ablation='none'):
        torch.manual_seed(809)
        stats = dict(y_mean=torch.tensor([.1, -.2, .3]), y_std=torch.tensor([.5, 2., 4.]))
        threshold = ((1. - stats['y_mean']) / stats['y_std']).tolist()
        head = (ExceedanceHead(4, 0, threshold, .17, fixed_gate=ablation == 'fixed_gate') if formulation == 'direct' else
                SeverityShapeHead(4, 0, threshold, .17, stats['y_std'].tolist(), fixed_gate=ablation == 'fixed_gate'))
        output = head(torch.randn(3, 3, 4))
        pred = output.prediction * stats['y_std'] + stats['y_mean']
        truth = torch.tensor([[3., 2., 1.], [0., 1., .5], [2., 4., 1.]])
        return output, pred, truth, stats

    def test_peak_survives_all_branch_ablations_including_early_return(self):
        for formulation, ablation in itertools.product(('direct', 'severity_shape'), DUAL_ABLATIONS):
            with self.subTest(formulation=formulation, ablation=ablation):
                output, pred, truth, stats = self._dual(formulation, ablation)
                c = LossConfig(excess_formulation=formulation, dual_ablation=ablation)
                old = ForecastLoss(copy.deepcopy(c), stats, 3., 1., .17)(output, pred, truth)
                c.peak_loss_weight = .4
                total = ForecastLoss(c, stats, 3., 1., .17, 1.)(output, pred, truth)
                terms = final_peak_terms(pred, truth, 1., .17)
                torch.testing.assert_close(total, old + .4 * terms.loss)
                self.assertEqual(c.peak_loss_weight, .4)

    def test_same_physical_prediction_has_same_peak_risk_for_every_parameterization(self):
        values = []
        pred = torch.tensor([[1., 3., 2.], [2., 1., 0.]], requires_grad=True)
        truth = torch.tensor([[5., 2., 0.], [2., 0., 0.]])
        stats = dict(y_mean=torch.zeros(3), y_std=torch.ones(3))
        for formulation in ('single', 'direct', 'severity_shape'):
            output = ForecastOutput(torch.full_like(pred, 1e6)) if formulation == 'single' else self._dual(formulation)[0]
            c = LossConfig(excess_formulation='direct' if formulation == 'single' else formulation,
                           dual_ablation='no_branch_supervision', peak_loss_weight=.4)
            values.append(ForecastLoss(c, stats, 10., 1., .17, 2.)(output, pred, truth))
        for value in values[1:]:
            torch.testing.assert_close(value, values[0], rtol=0, atol=0)

    def test_peak_adds_orthogonally_to_all_six_post2_post3_combinations_and_loss_modes(self):
        combinations = [('direct', 0., 0.), ('direct', .7, 0.), ('severity_shape', 0., 0.),
                        ('severity_shape', .7, 0.), ('severity_shape', 0., .3), ('severity_shape', .7, .3)]
        for (formulation, amp, shape), mode in itertools.product(combinations, MODES):
            output, pred, truth, stats = self._dual(formulation)
            c = LossConfig(loss_mode=mode, excess_formulation=formulation, excess_amp_loss_weight=amp,
                           shape_loss_weight=shape, excess_loss_weight=2.)
            old = ForecastLoss(copy.deepcopy(c), stats, 3., 1., .17)(output, pred, truth)
            c.peak_loss_weight = .4
            actual = ForecastLoss(c, stats, 3., 1., .17, 1.)(output, pred, truth)
            torch.testing.assert_close(actual, old + .4 * final_peak_terms(pred, truth, 1., .17).loss)
            self.assertEqual((c.excess_amp_loss_weight, c.shape_loss_weight), (amp, shape))

    def test_disabled_path_skips_helper_after_config_resolution_and_preserves_rng(self):
        for formulation, mode in itertools.product(('direct', 'severity_shape'), MODES):
            output, pred, truth, stats = self._dual(formulation)
            c = LossConfig(loss_mode=mode, excess_formulation=formulation, excess_amp_loss_weight=.7,
                           shape_loss_weight=.3 if formulation == 'severity_shape' else 0.)
            legacy = asdict(c)
            for key in ('peak_loss_weight', 'peak_pool', 'peak_pool_beta'):
                legacy.pop(key)
            before = torch.get_rng_state().clone()
            with patch('emulator.training.losses.final_peak_terms', side_effect=AssertionError('disabled peak computation')):
                old = ForecastLoss(LossConfig(**legacy), stats, 3., 1., .17)(output, pred, truth)
                c.peak_pool, c.peak_pool_beta = 'smoothmax', float('nan')
                new = ForecastLoss(c, stats, 3., 1., .17)(output, pred, truth)
            torch.testing.assert_close(old, new, rtol=0, atol=0)
            torch.testing.assert_close(torch.get_rng_state(), before, rtol=0, atol=0)

    def test_peak_gradients_reach_single_dual_and_severity_branches_and_backbone(self):
        for formulation, pool in itertools.product(('single', 'direct', 'severity_shape'), ('max', 'smoothmax')):
            with self.subTest(formulation=formulation, pool=pool):
                torch.manual_seed(273)
                stats = dict(y_mean=torch.tensor([.1, -.2, .3, -.4]), y_std=torch.tensor([.5, 2., 1., 4.]))
                threshold = ((1. - stats['y_mean']) / stats['y_std']).tolist()
                config = ModelConfig(3, 4, hidden_channels=16, history_steps=2, node_read_heads=2, time_read_heads=2,
                    head_type='single' if formulation == 'single' else 'dual', peak_threshold_norm=threshold,
                    peak_prior=.17, excess_formulation='direct' if formulation == 'single' else formulation,
                    target_y_std=stats['y_std'].tolist() if formulation == 'severity_shape' else None)
                model = build_model(config)
                branches = (('body', 'excess', 'gate') if formulation == 'direct' else
                            ('body', 'severity', 'shape', 'gate') if formulation == 'severity_shape' else ())
                # Use a locally nonsaturated state beyond zero-initialized final layers.
                with torch.no_grad():
                    for name in branches:
                        getattr(model.head, name)[-1].weight.normal_(std=.2)
                output = model(graph_batch(count=3))
                pred = output.prediction * stats['y_std'] + stats['y_mean']
                final_peak_terms(pred, torch.full_like(pred, 5.), 1., .17, pool, 2.).loss.backward()
                for name in branches:
                    grad = getattr(model.head, name)[-1].weight.grad
                    self.assertTrue(torch.isfinite(grad).all())
                    self.assertGreater(grad.abs().sum().item(), 0., name)
                self.assertGreater(sum(p.grad.abs().sum().item() for n, p in model.named_parameters()
                                       if not n.startswith('head.') and p.grad is not None), 0.)
                self.assertGreater(sum(p.grad.abs().sum().item() for p in model.head.parameters() if p.grad is not None), 0.)

    def test_diagnostics_are_pure_and_explicitly_use_training_pool(self):
        pred = torch.tensor([[1., 3.], [4., 2.]])
        truth = torch.tensor([[5., 2.], [1., 2.]])
        terms = final_peak_terms(pred, truth, 2., .17, 'smoothmax', 2.)
        diagnostic = terms.diagnostics()
        self.assertEqual(set(diagnostic), {'peak_loss', 'pred_peak_mean', 'true_peak_mean', 'peak_bias',
                                          'event_count', 'event_fraction'})
        self.assertEqual((diagnostic['event_count'].item(), diagnostic['event_fraction'].item()), (1, .5))
        torch.testing.assert_close(diagnostic['peak_bias'], (terms.m_pred - terms.m_true).mean())

    def test_shell_forwards_independent_controls_without_new_loss_modes_or_configs(self):
        root = Path(__file__).resolve().parents[1]
        config = root / 'experiment_config/P0_QuickRun/train_config_NCEP_Lewes_24h_single_mse.sh'
        # Override only new knobs through the environment, using an EXISTING config.
        with patch.dict('os.environ', PEAK_LOSS_WEIGHT='.4', PEAK_POOL='smoothmax', PEAK_POOL_BETA='2.5',
                        EXCESS_AMP_POOL='max'):
            commands = dry_commands(config)
        self.assertEqual(len(commands), 1)
        args = train.parse_args(commands[0])
        self.assertEqual((args.peak_loss_weight, args.peak_pool, args.peak_pool_beta), (.4, 'smoothmax', 2.5))
        self.assertEqual((args.loss_mode, args.head_type, args.excess_formulation), ('mse', 'single', 'direct'))
        self.assertEqual(args.excess_amp_pool, 'max')

    def _amp(self, device, dtype):
        for pool in ('max', 'smoothmax'):
            torch.manual_seed(906)
            network = torch.nn.Linear(3, 3).to(device)
            with torch.autocast(device.type, dtype=dtype):
                normalized = network(torch.ones(2, 3, device=device))
                prediction = normalized.float() * torch.tensor([.5, 2., 4.], device=device) + .1
                terms = final_peak_terms(prediction, torch.tensor([[3., 1., 0.], [1., 0., 1.]], device=device), 1., .17, pool, 2.)
            terms.loss.backward()
            self.assertTrue(torch.isfinite(terms.loss))
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in network.parameters()))

    def test_cpu_bfloat16(self):
        self._amp(torch.device('cpu'), torch.bfloat16)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
    def test_cuda_fp16_and_bfloat16(self):
        for dtype in (torch.float16, torch.bfloat16):
            self._amp(torch.device('cuda'), dtype)

    def test_ddp_fixed_prior_matches_global_gradient_with_event_free_rank(self):
        with tempfile.TemporaryDirectory() as temporary:
            rendezvous, destination = Path(temporary) / 'init', Path(temporary) / 'result.pt'
            mp.spawn(peak_ddp_worker, args=(str(rendezvous), str(destination)), nprocs=2, join=True)
            actual = torch.load(destination, weights_only=False)
        model, x, target = peak_ddp_fixture()
        for pool in ('max', 'smoothmax'):
            model.zero_grad(set_to_none=True)
            terms = final_peak_terms(model(x), target, 1., .17, pool, 2.)
            terms.loss.backward()
            torch.testing.assert_close(actual[pool]['loss'], terms.loss)
            for name, p in model.named_parameters():
                torch.testing.assert_close(actual[pool]['grads'][name], p.grad)
