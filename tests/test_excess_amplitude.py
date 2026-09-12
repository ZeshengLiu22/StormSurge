"""Optional TRAIN-prior-normalized physical excess peak-amplitude supervision."""

import contextlib
import copy
from dataclasses import asdict
from datetime import timedelta
import io
import itertools
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel

import infer
import train
from emulator.common.dual import DUAL_ABLATIONS
from emulator.data import ForcingGraphStore, fit_loss_thresholds
from emulator.inference.dual_diagnostics import summarize_dual, summarize_excess_amplitude
from emulator.models import ForecastOutput
from emulator.models.heads import ExceedanceHead, SingleHead
from emulator.training import EpochResult, ForecastLoss, LossConfig, dual_loss_terms
from emulator.training.excess_amplitude import excess_amplitude_terms, peak_pool
from test_config_interfaces import MODES, dry_commands
from test_pipeline import make_fixture


def ddp_fixture():
    torch.manual_seed(821)
    head = ExceedanceHead(4, 0, [1., 1.], .25)
    # Avoid the production final layer's zero-init shared-context gradient.
    with torch.no_grad():
        head.excess[-1].weight.fill_(.05)
    context = torch.arange(32, dtype=torch.float32).reshape(4, 2, 4) / 32
    stats = dict(y_mean=torch.zeros(2), y_std=torch.tensor([.5, 2.]))
    target = torch.tensor([[1., 1.], [.5, 1.], [2., 1.], [1., 3.]]) * stats['y_std']
    return head, context, target, stats


def ddp_worker(rank, rendezvous, destination):
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method=Path(rendezvous).as_uri(), rank=rank,
                            world_size=2, timeout=timedelta(seconds=30))
    try:
        head, context, target, stats = ddp_fixture()
        model = DistributedDataParallel(head)
        records = {}
        # Consecutive backwards also exercise reducer state. Rank 0 has no events.
        for pool in ('max', 'smoothmax'):
            model.zero_grad(set_to_none=True)
            output = model(context[2 * rank:2 * rank + 2])
            criterion = ForecastLoss(LossConfig(excess_loss_weight=2., gate_loss_weight=.5,
                excess_amp_loss_weight=.7, excess_amp_pool=pool), stats, 1., 1., event_prior=.25)
            loss = criterion(output, output.prediction * stats['y_std'], target[2 * rank:2 * rank + 2])
            assert loss.ndim == 0 and torch.isfinite(loss)
            loss.backward()
            average = loss.detach().clone()
            dist.all_reduce(average)
            records[pool] = dict(loss=average / 2,
                                grads={name: p.grad.clone() for name, p in head.named_parameters()})
        if rank == 0:
            torch.save(records, destination)
    finally:
        dist.destroy_process_group()


class ExcessAmplitudeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def fixture(self):
        stats = dict(y_mean=torch.tensor([.5, -1.]), y_std=torch.tensor([.5, 4.]))
        tau = 2.
        threshold = ((tau - stats['y_mean']) / stats['y_std'])[None, :]
        target_norm = threshold + torch.tensor([[4., 2.], [0., -.5], [-1., -1.]])
        excess = torch.tensor([[6., 1.25], [50., 20.], [4., 3.]], requires_grad=True)
        body = torch.zeros_like(excess, requires_grad=True)
        gate = torch.zeros(3, 1, requires_grad=True)
        output = ForecastOutput(body + gate.sigmoid() * excess, body, excess, gate, threshold)
        return output, target_norm, stats, tau

    def test_physical_conversion_precedes_max_and_matches_train_tau(self):
        output, target, stats, tau = self.fixture()
        terms = excess_amplitude_terms(output.excess, target, output.threshold, stats['y_std'], .2)
        physical_target = target * stats['y_std'] + stats['y_mean']
        torch.testing.assert_close(terms.r_target_phys, (physical_target - tau).clamp_min(0))
        torch.testing.assert_close(terms.r_target_phys[0], torch.tensor([2., 8.]), rtol=0, atol=0)
        self.assertEqual((target - output.threshold)[0].argmax().item(), 0)
        self.assertEqual(terms.r_target_phys[0].argmax().item(), 1)
        self.assertEqual(output.excess[0].argmax().item(), 0)
        self.assertEqual(terms.r_pred_phys[0].argmax().item(), 1)
        self.assertEqual(terms.a_pred[0].item(), 5.)
        self.assertEqual(terms.a_target[0].item(), 8.)
        self.assertEqual(terms.loss.item(), 15.)  # 9 / whole batch of 3 / TRAIN prior .2

    def test_strict_event_mask_shared_with_trajectory_loss_excludes_threshold_ties(self):
        output, target, stats, _ = self.fixture()
        terms = excess_amplitude_terms(output.excess, target, output.threshold, stats['y_std'], .2)
        self.assertEqual(terms.event.flatten().tolist(), [True, False, False])
        _, trajectory, _ = dual_loss_terms(output, target, stats['y_std'])
        gradient, = torch.autograd.grad(trajectory, output.excess)
        torch.testing.assert_close(gradient[1:], torch.zeros(2, 2), rtol=0, atol=0)
        self.assertTrue(gradient[0].ne(0).any())
        # The tail convention would include the exact-threshold second sample.
        self.assertTrue((target >= output.threshold).any(dim=1)[1])

    def test_exact_fitted_train_prior_differs_from_tail_nominal_and_batch_fractions(self):
        store = SimpleNamespace(graphs=[SimpleNamespace(y=torch.tensor([float(y), -1.]))
                                        for y in (0, 1, 2, 2, 2, 5, 10000)])
        fitted = fit_loss_thresholds(store, list(range(6)), tail_frac=.5, exceedance_percentile=75)
        self.assertEqual((fitted['tau_phys'], fitted['event_prior']), (2., 1 / 6))
        stats = dict(y_mean=torch.zeros(2), y_std=torch.ones(2))
        head = ExceedanceHead(4, 0, [2., 2.], fitted['event_prior'])
        output = head(torch.ones(2, 2, 4))._replace(excess=torch.ones(2, 2, requires_grad=True))
        target = torch.tensor([[5., -1.], [2., -1.]])
        c = LossConfig(tail_frac=.5, excess_amp_loss_weight=.7)
        criterion = ForecastLoss(c, stats, fitted['tail_threshold'], fitted['wmse_threshold'],
                                 event_prior=fitted['event_prior'])
        terms = excess_amplitude_terms(output.excess, target, output.threshold, stats['y_std'],
                                       criterion.event_prior)
        self.assertEqual(terms.loss.item(), 12.)  # (1 - 3)^2 / 2 / (1/6)
        for wrong_prior in (c.tail_frac, .25, terms.event.float().mean().item()):
            self.assertNotEqual(terms.loss.item(), 2 / wrong_prior)
        store.graphs[-1].y.fill_(-1e6)  # Unseen VAL/TEST values cannot change TRAIN metadata.
        self.assertEqual(fitted, fit_loss_thresholds(store, list(range(6)), tail_frac=.5,
                                                    exceedance_percentile=75))
        # The auxiliary normalization must not use the gate's clipped initialization prior.
        tiny = excess_amplitude_terms(output.excess, target, output.threshold, stats['y_std'], 1e-8)
        self.assertEqual(tiny.loss.item(), 2e8)

    def test_equal_sized_batch_composition_preserves_each_event_weight(self):
        gradients, losses = [], []
        for count in (1, 2):
            excess = torch.full((4, 2), 3., requires_grad=True)
            target = torch.zeros(4, 2)
            target[:count, 0] = 2.
            terms = excess_amplitude_terms(excess, target, torch.ones(1, 2), torch.tensor([1., 2.]), .2)
            terms.loss.backward()
            gradients.append(excess.grad[0].clone())
            losses.append(terms.loss.item())
            torch.testing.assert_close(excess.grad[count:], torch.zeros_like(excess.grad[count:]), rtol=0, atol=0)
        torch.testing.assert_close(gradients[0], gradients[1], rtol=0, atol=0)
        # d[(6-1)^2 / (B*q)]/d(excess_norm_at_h1) = 2*5*2/(4*.2).
        self.assertEqual(gradients[0].tolist(), [0., 25.])
        self.assertEqual(losses[1], 2 * losses[0])

    def test_no_event_batch_has_exact_zero_loss_and_finite_zero_gradients(self):
        for pool in ('max', 'smoothmax'):
            with self.subTest(pool=pool):
                excess = torch.tensor([[1., 3.], [4., 2.]], requires_grad=True)
                terms = excess_amplitude_terms(excess, torch.ones(2, 2), torch.ones(1, 2),
                                               torch.tensor([.5, 4.]), .17, pool)
                self.assertEqual(terms.loss.ndim, 0)
                self.assertEqual(terms.loss.item(), 0.)
                terms.loss.backward()
                torch.testing.assert_close(excess.grad, torch.zeros_like(excess), rtol=0, atol=0)
                self.assertTrue(torch.isfinite(excess.grad).all())

    def test_gradient_routes_only_to_selected_excess_peak(self):
        output, target, stats, _ = self.fixture()
        loss = excess_amplitude_terms(output.excess, target, output.threshold, stats['y_std'], .2).loss
        excess_grad, body_grad, gate_grad = torch.autograd.grad(
            loss, (output.excess, output.body, output.gate_logits), allow_unused=True)
        self.assertIsNone(body_grad)
        self.assertIsNone(gate_grad)
        torch.testing.assert_close(excess_grad, torch.tensor([[0., -40.], [0., 0.], [0., 0.]]))

    def test_pool_max_is_exact_and_beta_is_inactive(self):
        values = torch.tensor([[.1, 2., .5], [3., 1., 0.]], dtype=torch.float64)
        for beta in (0., -1., float('nan'), float('inf'), 1e300):
            torch.testing.assert_close(peak_pool(values, 'max', beta), values.max(dim=1).values, rtol=0, atol=0)

    def test_smoothmax_is_bounded_stable_and_converges_toward_max(self):
        values = torch.tensor([[.1, .7, 1.3], [0., .2, .8]], requires_grad=True)
        previous = torch.zeros(2)
        for beta in (.1, 1., 10., 100.):
            pooled = peak_pool(values, 'smoothmax', beta)
            self.assertTrue(torch.isfinite(pooled).all())
            self.assertTrue((pooled >= values.min(dim=1).values).all())
            self.assertTrue((pooled <= values.max(dim=1).values).all())
            self.assertTrue((pooled >= previous).all())
            previous = pooled
        torch.testing.assert_close(previous, values.max(dim=1).values)
        previous.sum().backward()
        self.assertTrue(torch.isfinite(values.grad).all())
        for beta in (1e30, 1e300):
            large = torch.tensor([[1e10, 1e10 - 1024, 0.]], requires_grad=True)
            pooled = peak_pool(large, 'smoothmax', beta)
            self.assertTrue(torch.isfinite(pooled).all())
            pooled.sum().backward()
            self.assertTrue(torch.isfinite(large.grad).all())

    def test_identical_trajectories_have_zero_error_with_both_pools(self):
        output, target, stats, _ = self.fixture()
        correct = (target - output.threshold).clamp_min(0).requires_grad_()
        for pool in ('max', 'smoothmax'):
            terms = excess_amplitude_terms(correct, target, output.threshold, stats['y_std'], .2, pool, 2.)
            self.assertEqual(terms.loss.item(), 0.)

    def test_nonzero_excess_final_layer_propagates_to_shared_context(self):
        for pool in ('max', 'smoothmax'):
            head, context, target, stats = ddp_fixture()
            context.requires_grad_()
            output = head(context)
            terms = excess_amplitude_terms(output.excess, target / stats['y_std'], output.threshold,
                                           stats['y_std'], .25, pool, 2.)
            terms.loss.backward()
            self.assertTrue(torch.isfinite(context.grad).all())
            self.assertGreater(context.grad.abs().sum().item(), 0.)
            self.assertGreater(head.excess[-1].weight.grad.abs().sum().item(), 0.)
            self.assertTrue(all(p.grad is None for p in head.body.parameters()))
            self.assertTrue(all(p.grad is None for p in head.gate.parameters()))

    def test_orthogonal_weight_adds_exactly_to_each_existing_prediction_mode(self):
        output, target_norm, stats, _ = self.fixture()
        target = target_norm * stats['y_std'] + stats['y_mean']
        prediction = output.prediction * stats['y_std'] + stats['y_mean']
        amp = excess_amplitude_terms(output.excess, target_norm, output.threshold, stats['y_std'], .2).loss
        for mode, excess_weight, amp_weight in itertools.product(MODES, (2., 5.), (.3, .7)):
            config = LossConfig(loss_mode=mode, excess_loss_weight=excess_weight, gate_loss_weight=.5)
            old = ForecastLoss(config, stats, 2., 1.)(output, prediction, target)
            enabled = copy.deepcopy(config)
            enabled.excess_amp_loss_weight = amp_weight
            actual = ForecastLoss(enabled, stats, 2., 1., event_prior=.2)(output, prediction, target)
            torch.testing.assert_close(actual, old + amp_weight * amp, rtol=0, atol=0)
            self.assertEqual(enabled.excess_loss_weight, excess_weight)

    def test_zero_weight_preserves_old_objective_outputs_gradients_and_rng_exactly(self):
        for mode, single in itertools.product(MODES, (False, True)):
            with self.subTest(mode=mode, single=single):
                torch.manual_seed(35)
                head = SingleHead(4, .3) if single else ExceedanceHead(4, .3, [1., 1.], .25)
                initial_state = copy.deepcopy(head.state_dict())
                context = torch.randn(3, 2, 4)
                target = torch.tensor([[2., 5.], [0., 1.], [1., -1.]])
                stats = dict(y_mean=torch.tensor([.5, -1.]), y_std=torch.tensor([.5, 2.]))
                c = LossConfig(loss_mode=mode, excess_loss_weight=2., gate_loss_weight=.5)
                initial_rng = torch.get_rng_state()
                output = head(context)
                prediction = output.prediction * stats['y_std'] + stats['y_mean']
                # Independent original branch formula plus the existing single-head objective.
                reference = ForecastLoss(c, stats, 2., 1.)(ForecastOutput(prediction), prediction, target)
                if not single:
                    z = (target - stats['y_mean']) / stats['y_std']
                    event = (z > output.threshold).any(dim=1, keepdim=True)
                    body = ((output.body - torch.minimum(z, output.threshold)) * stats['y_std']).square().mean()
                    excess = (((output.excess - (z - output.threshold).clamp_min(0)) * stats['y_std']).square() * event).mean()
                    gate = torch.nn.functional.binary_cross_entropy_with_logits(output.gate_logits, event.float()) * stats['y_std'].square().mean()
                    reference = reference + (c.body_loss_weight * body + c.excess_loss_weight * excess + c.gate_loss_weight * gate)
                reference.backward()
                expected_rng = torch.get_rng_state()
                expected_grads = {name: p.grad.clone() for name, p in head.named_parameters()}
                head.zero_grad(set_to_none=True)
                torch.set_rng_state(initial_rng)
                repeat = head(context)
                legacy = asdict(c)
                for key in ('excess_amp_loss_weight', 'excess_amp_pool', 'excess_amp_beta'):
                    legacy.pop(key)
                for config in (SimpleNamespace(**legacy), LossConfig(**legacy, excess_amp_pool='smoothmax',
                                                                      excess_amp_beta=float('nan'))):
                    with patch('emulator.training.losses.excess_amplitude_terms', side_effect=AssertionError('disabled computation')):
                        actual = ForecastLoss(config, stats, 2., 1.)(repeat, repeat.prediction * stats['y_std'] + stats['y_mean'], target)
                    torch.testing.assert_close(actual, reference, rtol=0, atol=0)
                actual.backward()
                torch.testing.assert_close(torch.get_rng_state(), expected_rng, rtol=0, atol=0)
                for value, expected in zip(repeat, output):
                    if value is not None:
                        torch.testing.assert_close(value, expected, rtol=0, atol=0)
                for name, p in head.named_parameters():
                    torch.testing.assert_close(p.grad, expected_grads[name], rtol=0, atol=0)
                for name, value in head.state_dict().items():
                    torch.testing.assert_close(value, initial_state[name], rtol=0, atol=0)

    def test_missing_or_invalid_prior_fails_only_when_effective_weight_is_positive(self):
        stats = dict(y_mean=torch.zeros(2), y_std=torch.ones(2))
        for prior in (None, 0., -1., float('nan'), float('inf')):
            with self.subTest(prior=prior):
                ForecastLoss(LossConfig(), stats, 1., 1., event_prior=prior)
                with self.assertRaisesRegex(ValueError, 'TRAIN event_prior'):
                    ForecastLoss(LossConfig(excess_amp_loss_weight=1.), stats, 1., 1., event_prior=prior)
                for mode in ('no_excess_loss', 'no_branch_supervision'):
                    ForecastLoss(LossConfig(excess_amp_loss_weight=1., dual_ablation=mode), stats, 1., 1., event_prior=prior)

    def test_config_defaults_and_relevant_validation(self):
        args = train.parse_args(['--model', 'perceiver3'])
        self.assertEqual((args.excess_amp_loss_weight, args.excess_amp_pool, args.excess_amp_beta), (0., 'max', 20.))
        invalid = [(['--excess_amp_loss_weight', x], 'finite and nonnegative')
                   for x in ('-1', 'nan', 'inf')]
        invalid += [(['--excess_amp_pool', 'invalid'], 'invalid choice')]
        invalid += [(['--excess_amp_loss_weight', '1', '--excess_amp_pool', 'smoothmax',
                     '--excess_amp_beta', x], 'finite and positive') for x in ('0', '-1', 'nan', 'inf')]
        for options, message in invalid:
            stderr = io.StringIO()
            with self.subTest(options=options), contextlib.redirect_stderr(stderr), self.assertRaises(SystemExit):
                train.parse_args(['--model', 'perceiver3', *options])
            self.assertIn(message, stderr.getvalue())
        stats = dict(y_mean=torch.zeros(2), y_std=torch.ones(2))
        for c in (LossConfig(excess_amp_loss_weight=-1), LossConfig(excess_amp_loss_weight=float('nan')),
                  LossConfig(excess_amp_pool='invalid'),
                  LossConfig(excess_amp_loss_weight=1, excess_amp_pool='smoothmax', excess_amp_beta=0)):
            with self.assertRaises(ValueError):
                ForecastLoss(c, stats, 1., 1., event_prior=.2)
        for weight, pool in ((0, 'smoothmax'), (1, 'max')):
            train.parse_args(['--model', 'perceiver3', '--excess_amp_loss_weight', str(weight),
                              '--excess_amp_pool', pool, '--excess_amp_beta', 'nan'])

    def test_single_or_baseline_rejects_positive_weight_in_cli_and_direct_loss(self):
        for model in ('baseline', 'perceiver3'):
            self.assertEqual(train.parse_args(['--model', model, '--head_type', 'single']).excess_amp_loss_weight, 0.)
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr), self.assertRaises(SystemExit):
                train.parse_args(['--model', model, '--head_type', 'single', '--excess_amp_loss_weight', '1'])
            self.assertIn('supervised dual exceedance head', stderr.getvalue())
        stats = dict(y_mean=torch.zeros(2), y_std=torch.ones(2))
        criterion = ForecastLoss(LossConfig(excess_amp_loss_weight=1), stats, 1., 1., event_prior=.2)
        pred = torch.ones(2, 2, requires_grad=True)
        with self.assertRaisesRegex(ValueError, 'supervised dual exceedance head'):
            criterion(ForecastOutput(pred), pred, torch.zeros_like(pred))

    def test_all_ablations_resolve_amplitude_weight_and_skip_disabled_computation(self):
        for mode in DUAL_ABLATIONS:
            disabled = mode in ('no_excess_loss', 'no_branch_supervision')
            args = train.parse_args(['--model', 'perceiver3', '--dual_ablation', mode,
                                     '--excess_amp_loss_weight', '.7'])
            self.assertEqual(args.excess_amp_loss_weight, 0. if disabled else .7)
            output, z, stats, _ = self.fixture()
            if mode == 'fixed_gate':
                output = output._replace(gate_logits=None)
            c = LossConfig(dual_ablation=mode, excess_amp_loss_weight=.7)
            with patch('emulator.training.losses.excess_amplitude_terms', wraps=excess_amplitude_terms) as observed:
                loss = ForecastLoss(c, stats, 1., 1., event_prior=None if disabled else .2)(
                    output, output.prediction * stats['y_std'] + stats['y_mean'], z * stats['y_std'] + stats['y_mean'])
            self.assertEqual(observed.call_count, 0 if disabled else 1)
            self.assertTrue(torch.isfinite(loss))
            self.assertEqual(c.excess_amp_loss_weight, 0. if disabled else .7)
            if disabled:
                self.assertEqual(c.excess_loss_weight, 0.)
            if mode == 'no_branch_supervision':
                self.assertEqual((c.body_loss_weight, c.gate_loss_weight, c.dual_loss), (0., 0., 0))

    def test_cpu_bfloat16_autocast_and_low_precision_physical_products(self):
        self.check_runtime(torch.device('cpu'), torch.bfloat16)
        for dtype, pool in itertools.product((torch.bfloat16, torch.float16), ('max', 'smoothmax')):
            excess = torch.tensor([[300., 400.]], dtype=dtype, requires_grad=True)
            terms = excess_amplitude_terms(excess, torch.ones(1, 2, dtype=dtype),
                torch.zeros(1, 2, dtype=dtype), torch.tensor([300., 400.], dtype=dtype), .2, pool)
            self.assertEqual(terms.loss.dtype, torch.float32)
            self.assertTrue(torch.isfinite(terms.loss))

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA is unavailable')
    def test_cuda_float16_and_bfloat16_autocast(self):
        self.check_runtime(torch.device('cuda'), torch.float16)
        if torch.cuda.is_bf16_supported():
            self.check_runtime(torch.device('cuda'), torch.bfloat16)

    def check_runtime(self, device, dtype):
        for pool, mode in itertools.product(('max', 'smoothmax'), ('mse', 'mse_tail', 'mse_tail_slope')):
            head, context, target, stats = ddp_fixture()
            head = head.to(device)
            context, target = context.to(device), target.to(device)
            stats = {k: v.to(device) for k, v in stats.items()}
            criterion = ForecastLoss(LossConfig(loss_mode=mode, excess_loss_weight=5., gate_loss_weight=.5,
                excess_amp_loss_weight=.7, excess_amp_pool=pool), stats, 1., 1., event_prior=.25).to(device)
            with torch.autocast(device_type=device.type, dtype=dtype):
                output = head(context)
                loss = criterion(output, output.prediction * stats['y_std'], target)
            self.assertEqual(loss.ndim, 0)
            self.assertEqual(loss.dtype, torch.float32)
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in head.parameters()))

    @unittest.skipUnless(dist.is_available() and dist.is_gloo_available(), 'Gloo is unavailable')
    def test_two_rank_ddp_scalar_and_gradients_match_global_batch_with_event_free_rank(self):
        with tempfile.TemporaryDirectory() as temporary:
            rendezvous, destination = str(Path(temporary) / 'rendezvous'), str(Path(temporary) / 'result.pt')
            mp.spawn(ddp_worker, args=(rendezvous, destination), nprocs=2, join=True)
            records = torch.load(destination, weights_only=True)
        head, context, target, stats = ddp_fixture()
        for pool in ('max', 'smoothmax'):
            head.zero_grad(set_to_none=True)
            output = head(context)
            criterion = ForecastLoss(LossConfig(excess_loss_weight=2., gate_loss_weight=.5,
                excess_amp_loss_weight=.7, excess_amp_pool=pool), stats, 1., 1., event_prior=.25)
            loss = criterion(output, output.prediction * stats['y_std'], target)
            loss.backward()
            torch.testing.assert_close(records[pool]['loss'], loss)
            for name, parameter in head.named_parameters():
                torch.testing.assert_close(records[pool]['grads'][name], parameter.grad, rtol=2e-5, atol=2e-6)

    def test_shell_forwards_amplitude_knobs_even_for_single_head_validation(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'interface.sh'
            # Synthetic CLI fixture only: dry run never launches a training job.
            for head in ('dual', 'single'):
                path.write_text(f'MODEL=perceiver3\nHEAD_TYPE={head}\nEXCESS_AMP_LOSS_WEIGHT=.7\n'
                                'EXCESS_AMP_POOL=smoothmax\nEXCESS_AMP_BETA=3.5\n')
                command = dry_commands(path)[0]
                self.assertEqual(command[command.index('--excess_amp_loss_weight') + 1], '.7')
                if head == 'single':
                    with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                        train.parse_args(command)
                else:
                    args = train.parse_args(command)
                    self.assertEqual((args.excess_amp_loss_weight, args.excess_amp_pool, args.excess_amp_beta),
                                     (.7, 'smoothmax', 3.5))

    def test_train_passes_fitted_prior_and_snapshots_values_without_changing_selection_or_old_load(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            torch.manual_seed(83)
            graphs, stations = make_fixture(root)
            store = ForcingGraphStore(graphs, 'Battery')
            fitted = fit_loss_thresholds(store, store.split()['train'], tail_frac=.4, exceedance_percentile=75)
            output = root / 'run'
            metrics = dict(rmse_all=1., mae_all=1., rmse_peak5=1., mae_peak5=1.)
            val1 = dict(metrics, rmse_all=.5, rmse_peak5=2.)
            val2 = dict(metrics, rmse_all=.7, rmse_peak5=.1)
            # Exercise setup and checkpoint saving without any optimizer/model training.
            epochs = [EpochResult(metrics), EpochResult(val1), EpochResult(metrics), EpochResult(val2),
                      EpochResult(metrics, dict(y_true=np.zeros((0, 4)), y_pred=np.zeros((0, 4)), tags=np.array([], dtype=str)))]
            with patch.object(train, 'ForecastLoss', wraps=ForecastLoss) as constructor, \
                 patch.object(train, 'run_epoch', side_effect=epochs), \
                 patch.object(train.torch.optim.lr_scheduler.LambdaLR, 'step'), \
                 contextlib.redirect_stdout(io.StringIO()):
                train.main(['--root_dir', str(graphs), '--station', 'Battery', '--station_json_dir', str(stations),
                    '--output_dir', str(output), '--model', 'perceiver3', '--head_type', 'dual',
                    '--exceedance_percentile', '75', '--tail_frac', '.4', '--epochs', '2', '--device', 'cpu',
                    '--num_workers', '0', '--hidden_channels', '16', '--history_hours', '12',
                    '--excess_amp_loss_weight', '.7', '--excess_amp_pool', 'smoothmax', '--excess_amp_beta', '3.5'])
            self.assertEqual(constructor.call_args.kwargs, dict(event_prior=fitted['event_prior']))
            checkpoint = torch.load(next(output.glob('best_*.pth')), weights_only=False)
            self.assertEqual(checkpoint['loss_thresholds'], fitted)
            self.assertEqual(checkpoint['epoch'], 1)  # Production selection remains val rmse_all.
            snapshot = json.loads(next(output.glob('config_*.json')).read_text())
            shell_snapshot = (output / 'config_used.sh').read_text()
            for key, value in dict(excess_amp_loss_weight=.7, excess_amp_pool='smoothmax', excess_amp_beta=3.5).items():
                self.assertEqual(checkpoint['training_config'][key], value)
                self.assertEqual(snapshot[key], value)
                self.assertIn(f'{key.upper()}={value}', shell_snapshot)
                self.assertNotIn(key, checkpoint['model_config'])
            for legacy in (False, True):
                if legacy:
                    for key in ('excess_amp_loss_weight', 'excess_amp_pool', 'excess_amp_beta'):
                        checkpoint['training_config'].pop(key)
                ckpt = root / f'checkpoint_{legacy}.pth'
                torch.save(checkpoint, ckpt)
                with contextlib.redirect_stdout(io.StringIO()):
                    infer.main(['--ckpt', str(ckpt), '--root_dir', str(graphs), '--device', 'cpu',
                                '--num_workers', '0', '--out_dir', str(root / f'infer_{legacy}'), '--save_npz'])
            with np.load(root / 'infer_False/predictions.npz') as current, np.load(root / 'infer_True/predictions.npz') as old:
                np.testing.assert_array_equal(current['y_pred'], old['y_pred'])


class ExcessAmplitudeDiagnosticTests(unittest.TestCase):
    def test_true_event_metrics_use_ungated_physical_max_and_train_threshold(self):
        truth = np.array([[2., 5.], [4., 1.], [2., 2.], [0., 1.]])
        excess = np.array([[1., 2.], [5., 1.], [100., 100.], [100., 100.]])
        arrays = dict(y_true=truth, y_pred=np.zeros_like(truth), body_phys=np.zeros_like(truth),
                      excess_phys=excess, gate_probability=np.zeros(4))
        report = summarize_dual(arrays, tau_phys=2.)
        self.assertEqual(report['events'], 2)
        expected = dict(excess_amp_rmse=np.sqrt(5), excess_amp_mae=2., excess_amp_bias=1.,
                        pred_excess_amp_mean=3.5, target_excess_amp_mean=2.5)
        for key, value in expected.items():
            self.assertAlmostEqual(report[key], value)
        direct = summarize_excess_amplitude(excess, np.maximum(truth - 2., 0), (truth > 2.).any(axis=1))
        self.assertEqual(direct, expected)

    def test_no_events_and_empty_arrays_have_json_null_diagnostics(self):
        for count in (0, 3):
            report = summarize_excess_amplitude(np.ones((count, 2)), np.zeros((count, 2)), np.zeros(count, dtype=bool))
            self.assertEqual(len(report), 5)
            self.assertTrue(all(value is None for value in report.values()))
            json.dumps(report, allow_nan=False)


if __name__ == '__main__':
    unittest.main()
