"""Severity/shape factorization, fixed-TRAIN-prior shape loss and legacy loading."""

import contextlib
import copy
from dataclasses import asdict
from datetime import timedelta
import io
import itertools
import json
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel

import infer
import train
from emulator.common.dual import DUAL_ABLATIONS, initial_gate_prior
from emulator.data import ForcingGraphStore, fit_loss_thresholds
from emulator.models import ForecastOutput, ModelConfig, build_model
from emulator.models.heads import ExceedanceHead, SeverityShapeHead, head_mlp
from emulator.training import EpochResult, ForecastLoss, LossConfig, dual_loss_terms
from emulator.training.excess_amplitude import excess_amplitude_terms, peak_pool, physical_excess_target
from emulator.training.excess_shape import excess_shape_loss, excess_shape_target
from test_config_interfaces import MODES, dry_commands
import test_excess_amplitude as amplitude_tests
from test_models import graph_batch
from test_pipeline import make_fixture


class LegacyDirectHead(nn.Module):
    """Frozen direct construction/forward from post-#2 commit e2846dd."""
    def __init__(self, hidden, dropout, threshold, prior, fixed_gate=False):
        super().__init__()
        threshold = torch.as_tensor(threshold, dtype=torch.float32).reshape(1, -1)
        prior_init = initial_gate_prior(prior)
        self.register_buffer('threshold', threshold.clone())
        self.body = head_mlp(hidden, dropout)
        self.excess = head_mlp(hidden, dropout)
        self.gate = None if fixed_gate else head_mlp(hidden, dropout)
        if fixed_gate:
            self.register_buffer('fixed_gate_probability', torch.tensor(float(prior)))
        else:
            nn.init.zeros_(self.gate[-1].weight)
            nn.init.constant_(self.gate[-1].bias, math.log(prior_init / (1 - prior_init)))
        nn.init.zeros_(self.excess[-1].weight)
        nn.init.constant_(self.excess[-1].bias, math.log(math.expm1(.1)))

    def forward(self, context):
        raw = self.body(context).squeeze(-1).float()
        body = self.threshold - F.softplus(self.threshold - raw)
        excess = F.softplus(self.excess(context).squeeze(-1).float())
        gate = self.gate(context.mean(dim=1)).float() if self.gate is not None else None
        probability = gate.sigmoid() if gate is not None else self.fixed_gate_probability.expand(context.size(0), 1)
        return ForecastOutput(body + probability * excess, body, excess, gate, self.threshold, probability)


def severity_fixture(fixed_gate=False, nonzero=True):
    # Reuse #2's heterogeneous scales, strict event-free rank, and shared context.
    _, context, target, stats = amplitude_tests.ddp_fixture()
    head = SeverityShapeHead(4, 0, [1., 1.], .25, stats['y_std'].tolist(), fixed_gate=fixed_gate)
    if nonzero:
        with torch.no_grad():
            for branch in (head.severity, head.shape, head.gate):
                if branch is not None:
                    branch[-1].weight.copy_(torch.linspace(-.2, .3, branch[-1].in_features)[None, :])
    return head, context, target, stats


def shape_ddp_worker(rank, rendezvous, destination):
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method=Path(rendezvous).as_uri(), rank=rank,
                            world_size=2, timeout=timedelta(seconds=30))
    try:
        head, context, target, stats = severity_fixture()
        network = DistributedDataParallel(head)
        records = {}
        for pool in ('max', 'smoothmax'):
            network.zero_grad(set_to_none=True)
            output = network(context[2 * rank:2 * rank + 2])
            criterion = ForecastLoss(LossConfig(excess_formulation='severity_shape', excess_loss_weight=2.,
                excess_amp_loss_weight=.7, excess_amp_pool=pool, shape_loss_weight=.3), stats, 1., 1., event_prior=.25)
            loss = criterion(output, output.prediction * stats['y_std'], target[2 * rank:2 * rank + 2])
            assert loss.ndim == 0 and torch.isfinite(loss)
            loss.backward()
            average = loss.detach().clone()
            dist.all_reduce(average)
            records[pool] = dict(loss=average / 2, grads={name: p.grad.clone() for name, p in head.named_parameters()})
        if rank == 0:
            torch.save(records, destination)
    finally:
        dist.destroy_process_group()


class SeverityShapeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_direct_head_matches_frozen_post2_initialization_state_outputs_gradients_and_rng(self):
        for fixed in (False, True):
            torch.manual_seed(726)
            legacy = LegacyDirectHead(4, .3, [1., 2.], .17, fixed)
            legacy_rng = torch.get_rng_state()
            torch.manual_seed(726)
            direct = ExceedanceHead(4, .3, [1., 2.], .17, fixed)
            torch.testing.assert_close(torch.get_rng_state(), legacy_rng, rtol=0, atol=0)
            self.assertEqual(set(direct.state_dict()), set(legacy.state_dict()))
            self.assertEqual(set(direct._modules), {'body', 'excess'} if fixed else {'body', 'excess', 'gate'})
            self.assertFalse(hasattr(direct, 'severity'))
            self.assertFalse(hasattr(direct, 'shape'))
            self.assertFalse(hasattr(direct, 'target_y_std'))
            for name, value in direct.state_dict().items():
                torch.testing.assert_close(value, legacy.state_dict()[name], rtol=0, atol=0)
            direct.load_state_dict(legacy.state_dict(), strict=True)
            context = torch.randn(3, 2, 4)
            rng = torch.get_rng_state()
            expected = legacy(context)
            after_rng = torch.get_rng_state()
            torch.set_rng_state(rng)
            actual = direct(context)
            torch.testing.assert_close(torch.get_rng_state(), after_rng, rtol=0, atol=0)
            for before, after in zip(expected, actual):
                if before is not None:
                    torch.testing.assert_close(before, after, rtol=0, atol=0)
            expected.prediction.square().sum().backward()
            actual.prediction.square().sum().backward()
            for (_, before), (_, after) in zip(legacy.named_parameters(), direct.named_parameters()):
                torch.testing.assert_close(before.grad, after.grad, rtol=0, atol=0)
            self.assertIsNone(actual.severity_phys)
            self.assertIsNone(actual.excess_shape)

    def test_legacy_model_config_without_new_fields_loads_strictly_and_matches_explicit_direct(self):
        legacy = dict(in_channels=3, out_channels=4, hidden_channels=16, peak_threshold_norm=[1.] * 4,
                      peak_prior=.17, node_read_heads=2, time_read_heads=2)
        for key in ('excess_formulation', 'target_y_std', 'severity_shape_eps'):
            self.assertNotIn(key, legacy)
        for ablation in ('none', 'fixed_gate'):
            legacy['dual_ablation'] = ablation
            torch.manual_seed(302)
            old = build_model(ModelConfig(**legacy)).eval()
            rng = torch.get_rng_state()
            torch.manual_seed(302)
            explicit = build_model(ModelConfig(**legacy, excess_formulation='direct')).eval()
            self.assertIs(type(old.head), ExceedanceHead)
            torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
            self.assertEqual(list(old.state_dict()), list(explicit.state_dict()))
            explicit.load_state_dict(old.state_dict(), strict=True)
            batch = graph_batch()
            torch.testing.assert_close(old(batch).prediction, explicit(batch).prediction, rtol=0, atol=0)
        value = ForecastOutput(torch.zeros(2, 3))
        self.assertIsNone(value.severity_phys)
        self.assertIsNone(value.excess_shape)

    def test_direct_loss_keeps_post2_path_with_optional_shape_disabled(self):
        output, z, stats, _ = amplitude_tests.ExcessAmplitudeTests().fixture()
        target = z * stats['y_std'] + stats['y_mean']
        prediction = output.prediction * stats['y_std'] + stats['y_mean']
        for mode, weight in itertools.product(MODES, (0., .7)):
            config = LossConfig(loss_mode=mode, excess_loss_weight=2., excess_amp_loss_weight=weight)
            old = asdict(config)
            for key in ('excess_formulation', 'shape_loss_weight', 'severity_shape_eps'):
                old.pop(key)
            with patch('emulator.training.losses.excess_shape_loss', side_effect=AssertionError('disabled shape path')):
                expected = ForecastLoss(LossConfig(**old), stats, 1., 1., event_prior=.2)(output, prediction, target)
                actual = ForecastLoss(config, stats, 1., 1., event_prior=.2)(output, prediction, target)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_severity_initialization_is_small_finite_positive_and_preserves_gate_definition(self):
        for fixed, prior in itertools.product((False, True), (0., .17, 1.)):
            head = SeverityShapeHead(4, 0, [1., 2., 3.], prior, [.25, 2., 5.], fixed_gate=fixed, head_hidden=11)
            output = head(torch.randn(3, 3, 4))
            torch.testing.assert_close(output.severity_phys, torch.full((3,), .1))
            torch.testing.assert_close(output.excess_shape, torch.ones(3, 3), rtol=0, atol=0)
            torch.testing.assert_close(output.gate_probability, torch.full((3, 1), prior if fixed else initial_gate_prior(prior)))
            contribution = output.gate_probability * output.excess * head.target_y_std
            self.assertLessEqual(contribution.max().item(), .100001)
            self.assertEqual(head.target_y_std.shape, (1, 3))
            self.assertNotIn('excess', head._modules)
            for branch in (head.severity, head.shape):
                self.assertEqual(branch[0].out_features, 11)
                self.assertEqual(branch[-1].weight.count_nonzero().item(), 0)
            if fixed:
                self.assertIsNone(head.gate)
                self.assertIsNone(output.gate_logits)

    def test_physical_and_normalized_reconstruction_with_heterogeneous_horizon_scales(self):
        head, context, _, stats = severity_fixture()
        output = head(context)
        y_mean = torch.tensor([.3, -1.])
        physical = output.severity_phys[:, None] * output.excess_shape
        self.assertTrue((output.severity_phys >= 0).all())
        self.assertTrue((output.excess_shape >= 0).all())
        self.assertTrue((output.excess >= 0).all())
        torch.testing.assert_close(output.excess_shape.max(dim=1).values, torch.ones(4), rtol=0, atol=0)
        torch.testing.assert_close(output.excess, physical / stats['y_std'])
        torch.testing.assert_close(output.excess * stats['y_std'], physical)
        torch.testing.assert_close(physical.max(dim=1).values, output.severity_phys)
        torch.testing.assert_close(output.prediction, output.body + output.gate_probability * output.excess, rtol=0, atol=0)
        torch.testing.assert_close(output.prediction * stats['y_std'] + y_mean,
            output.body * stats['y_std'] + y_mean + output.gate_probability * physical)
        self.assertFalse(torch.allclose(output.excess, physical / stats['y_std'].mean()))

    def test_shape_normalizes_per_window_not_across_batch(self):
        head, context, _, _ = severity_fixture()
        with torch.no_grad():
            head.shape[-1].weight.fill_(.4)
        raw = F.softplus(head.shape(context).squeeze(-1).float())
        expected = raw / raw.max(dim=1, keepdim=True).values
        output = head(context)
        torch.testing.assert_close(output.excess_shape, expected, rtol=0, atol=0)
        for index in range(len(context)):
            torch.testing.assert_close(output.excess_shape[index:index + 1], head(context[index:index + 1]).excess_shape)

    def test_degenerate_underflow_shape_is_safe_under_epsilon_floor(self):
        head, context, target, stats = severity_fixture(nonzero=False)
        with torch.no_grad():
            head.shape[-1].bias.fill_(-1000)
        output = head(context)
        self.assertTrue(torch.isfinite(output.prediction).all())
        torch.testing.assert_close(output.excess_shape, torch.zeros_like(output.excess_shape), rtol=0, atol=0)
        criterion = ForecastLoss(LossConfig(excess_formulation='severity_shape', excess_amp_loss_weight=.7,
            shape_loss_weight=.3), stats, 1., 1., event_prior=.25)
        loss = criterion(output, output.prediction * stats['y_std'], target)
        loss.backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in head.parameters()))

    def test_severity_pact_reconstruction_checkpoint_and_label_free_inference(self):
        for encoder, history, fixed in itertools.product(('GraphSAGE', 'CNN'), (0, 2), (False, True)):
            config = ModelConfig(3, 4, hidden_channels=16, encoder_type=encoder, history_steps=history,
                temporal_block='MLP', node_read_heads=2, time_read_heads=2, peak_threshold_norm=[1.] * 4,
                excess_formulation='severity_shape', target_y_std=[.25, .5, 2., 5.], severity_shape_eps=2e-6,
                dual_ablation='fixed_gate' if fixed else 'none')
            network = build_model(config).eval()
            batch = graph_batch(history + 1)
            before = network(batch)
            batch.y.fill_(1e8)
            after = network(batch)
            del batch.y
            unlabeled = network(batch)
            restored = build_model(ModelConfig(**json.loads(json.dumps(asdict(config))))).eval()
            restored.load_state_dict(network.state_dict(), strict=True)
            for expected, modified, missing, actual in zip(before, after, unlabeled, restored(batch)):
                if expected is None:
                    self.assertIsNone(actual)
                else:
                    torch.testing.assert_close(expected, modified, rtol=0, atol=0)
                    torch.testing.assert_close(expected, missing, rtol=0, atol=0)
                    torch.testing.assert_close(expected, actual, rtol=0, atol=0)
            self.assertIn('head.target_y_std', network.state_dict())

    def test_target_reuses_post2_physical_excess_and_strict_event_mask(self):
        output, z, stats, tau = amplitude_tests.ExcessAmplitudeTests().fixture()
        with patch('emulator.training.excess_shape.physical_excess_target', wraps=physical_excess_target) as observed:
            target = excess_shape_target(z, output.threshold, stats['y_std'])
        observed.assert_called_once()
        physical_truth = z * stats['y_std'] + stats['y_mean']
        torch.testing.assert_close(target.r_target_phys, (physical_truth - tau).clamp_min(0))
        self.assertEqual(target.event.flatten().tolist(), [True, False, False])
        torch.testing.assert_close(target.a_target, torch.tensor([8., 0., 0.]), rtol=0, atol=0)
        torch.testing.assert_close(target.shape_target, torch.tensor([[.25, 1.], [0., 0.], [0., 0.]]), rtol=0, atol=0)
        output = output._replace(body=output.body + 1000)
        repeated = excess_shape_target(z, output.threshold, stats['y_std'])
        torch.testing.assert_close(repeated.shape_target, target.shape_target, rtol=0, atol=0)

    def test_near_zero_true_event_uses_safe_target_epsilon(self):
        z = torch.tensor([[1. + 1e-10, 1.]], dtype=torch.float64)
        target = excess_shape_target(z, torch.ones(1, 2, dtype=torch.float64), torch.tensor([.5, 4.]), eps=1e-6)
        self.assertTrue(target.event.item())
        self.assertGreater(target.a_target.item(), 0)
        self.assertLess(target.a_target.item(), 1e-6)
        self.assertTrue(torch.isfinite(target.shape_target).all())
        torch.testing.assert_close(target.shape_target, target.r_target_phys / 1e-6)

    def test_shape_loss_uses_empirical_train_prior_not_tail_or_batch_frequency(self):
        store = SimpleNamespace(graphs=[SimpleNamespace(y=torch.tensor([float(y), -1.]))
                                        for y in (0, 1, 2, 2, 2, 5, 10000)])
        fitted = fit_loss_thresholds(store, list(range(6)), tail_frac=.5, exceedance_percentile=75)
        self.assertEqual(fitted['event_prior'], 1 / 6)
        scale = torch.tensor([1., 2.])
        physical = torch.tensor([[5., -1.], [2., -1.]])
        shape = torch.tensor([[.5, 1.], [1., .25]], requires_grad=True)
        actual = excess_shape_loss(shape, physical / scale, fitted['tau_phys'] / scale, scale, fitted['event_prior'])
        self.assertEqual(actual.item(), 1.875)
        for wrong_prior in (.5, .25):
            self.assertNotEqual(actual.item(), .3125 / wrong_prior)
        # Magnifying event amplitude changes no dimensionless shape error.
        physical[0, 0] = 32.
        bigger = excess_shape_loss(shape, physical / scale, fitted['tau_phys'] / scale, scale, fitted['event_prior'])
        torch.testing.assert_close(bigger, actual, rtol=0, atol=0)

    def test_shape_per_event_weight_is_independent_of_batch_event_count(self):
        gradients, losses = [], []
        for count in (1, 2):
            shape = torch.tensor([[.5, 1.]] * 4, requires_grad=True)
            z = torch.ones(4, 2)
            z[:count, 0] = 2.
            loss = excess_shape_loss(shape, z, torch.ones(1, 2), torch.ones(2), .2)
            loss.backward()
            gradients.append(shape.grad[0].clone())
            losses.append(loss.item())
            torch.testing.assert_close(shape.grad[count:], torch.zeros_like(shape.grad[count:]), rtol=0, atol=0)
        torch.testing.assert_close(gradients[0], gradients[1], rtol=0, atol=0)
        torch.testing.assert_close(gradients[0], torch.tensor([-.625, 1.25]), rtol=0, atol=0)
        self.assertEqual(losses[1], 2 * losses[0])

    def test_event_free_shape_and_amplitude_losses_are_exact_zero_with_finite_gradients(self):
        head, context, _, stats = severity_fixture()
        output = head(context)
        z = torch.ones(4, 2)  # Exact threshold equality is not an event.
        shape_loss = excess_shape_loss(output.excess_shape, z, output.threshold, stats['y_std'], .25)
        amplitude = excess_amplitude_terms(output.excess, z, output.threshold, stats['y_std'], .25).loss
        self.assertEqual(shape_loss.item(), 0)
        self.assertEqual(amplitude.item(), 0)
        (shape_loss + amplitude).backward()
        for branch in (head.severity, head.shape):
            for p in branch.parameters():
                self.assertIsNotNone(p.grad)
                torch.testing.assert_close(p.grad, torch.zeros_like(p.grad), rtol=0, atol=0)

    def test_prediction_trajectory_amplitude_and_shape_gradient_routing(self):
        for term, active in [('prediction', {'body', 'gate', 'severity', 'shape'}),
                             ('trajectory', {'severity', 'shape'}), ('amplitude', {'severity'}), ('shape', {'shape'})]:
            with self.subTest(term=term):
                head, context, target, stats = severity_fixture()
                context.requires_grad_()
                output = head(context)
                z = target / stats['y_std']
                if term == 'prediction':
                    loss = (output.prediction * stats['y_std'] - target).square().mean()
                elif term == 'trajectory':
                    loss = dual_loss_terms(output, z, stats['y_std'])[1]
                elif term == 'amplitude':
                    loss = excess_amplitude_terms(output.excess, z, output.threshold, stats['y_std'], .25).loss
                else:
                    loss = excess_shape_loss(output.excess_shape, z, output.threshold, stats['y_std'], .25)
                loss.backward()
                for name in ('body', 'gate', 'severity', 'shape'):
                    gradients = [p.grad for p in getattr(head, name).parameters()]
                    norm = sum(g.abs().sum().item() for g in gradients if g is not None)
                    if name in active:
                        self.assertGreater(norm, 1e-8, (term, name))
                    elif term == 'amplitude' and name == 'shape':
                        self.assertLess(norm, 1e-5)  # Hard maximum of unit-peak shape is severity.
                    else:
                        self.assertTrue(all(g is None for g in gradients), (term, name))
                self.assertTrue(torch.isfinite(context.grad).all())
                self.assertGreater(context.grad.abs().sum().item(), 0.)

    def test_post2_hard_max_supervises_severity_and_smoothmax_keeps_same_trajectory_pool(self):
        head, context, target, stats = severity_fixture()
        output = head(context)
        z = target / stats['y_std']
        exact = excess_amplitude_terms(output.excess, z, output.threshold, stats['y_std'], .25)
        torch.testing.assert_close(exact.a_pred, output.severity_phys)
        expected = ((output.severity_phys - exact.a_target).square() * exact.event.squeeze(1)).mean() / .25
        torch.testing.assert_close(exact.loss, expected)
        smooth = excess_amplitude_terms(output.excess, z, output.threshold, stats['y_std'], .25, 'smoothmax', 2.)
        torch.testing.assert_close(smooth.a_pred, peak_pool(output.severity_phys[:, None] * output.excess_shape, 'smoothmax', 2.))
        self.assertTrue((smooth.a_pred < output.severity_phys).all())
        target_shape = excess_shape_target(z, output.threshold, stats['y_std'])
        torch.testing.assert_close(target_shape.a_target, exact.a_target, rtol=0, atol=0)

    def test_optional_objectives_are_orthogonal_and_trajectory_weight_is_retained(self):
        head, context, target, stats = severity_fixture()
        output = head(context)
        z = target / stats['y_std']
        amp = excess_amplitude_terms(output.excess, z, output.threshold, stats['y_std'], .25).loss
        shape = excess_shape_loss(output.excess_shape, z, output.threshold, stats['y_std'], .25)
        for mode, amp_weight, shape_weight in itertools.product(MODES, (0., .7), (0., .3)):
            c = LossConfig(loss_mode=mode, excess_formulation='severity_shape', excess_loss_weight=5.)
            prediction = output.prediction * stats['y_std']
            base = ForecastLoss(copy.deepcopy(c), stats, 1., 1.)(output, prediction, target)
            c.excess_amp_loss_weight, c.shape_loss_weight = amp_weight, shape_weight
            with patch('emulator.training.losses.excess_shape_loss', wraps=excess_shape_loss) as observed:
                actual = ForecastLoss(c, stats, 1., 1., event_prior=.25)(output, prediction, target)
            expected = base
            if amp_weight:
                expected = expected + amp_weight * amp
            if shape_weight:
                expected = expected + shape_weight * shape
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            self.assertEqual(observed.call_count, int(shape_weight > 0))
            self.assertEqual(c.excess_loss_weight, 5.)

    def test_ablation_disables_all_explicit_excess_supervision_and_keeps_optional_zeros(self):
        for mode in DUAL_ABLATIONS:
            head, context, target, stats = severity_fixture(fixed_gate=mode == 'fixed_gate')
            output = head(context)
            disabled = mode in ('no_excess_loss', 'no_branch_supervision')
            options = ['--model', 'perceiver3', '--excess_formulation', 'severity_shape', '--dual_ablation', mode]
            args = train.parse_args([*options, '--excess_amp_loss_weight', '.7', '--shape_loss_weight', '.3'])
            self.assertEqual((args.excess_amp_loss_weight, args.shape_loss_weight), (0., 0.) if disabled else (.7, .3))
            c = LossConfig(excess_formulation='severity_shape', dual_ablation=mode, excess_amp_loss_weight=.7, shape_loss_weight=.3)
            with patch('emulator.training.losses.excess_shape_loss', wraps=excess_shape_loss) as shape_fn, \
                 patch('emulator.training.losses.excess_amplitude_terms', wraps=excess_amplitude_terms) as amp_fn:
                loss = ForecastLoss(c, stats, 1., 1., event_prior=None if disabled else .25)(
                    output, output.prediction * stats['y_std'], target)
            self.assertTrue(torch.isfinite(loss))
            self.assertEqual((shape_fn.call_count, amp_fn.call_count), (0, 0) if disabled else (1, 1))
            for name in DUAL_ABLATIONS[mode]:
                self.assertEqual(getattr(c, name), 0.)
            zeros = train.parse_args(options)
            self.assertEqual((zeros.excess_amp_loss_weight, zeros.shape_loss_weight), (0., 0.))

    def test_shape_only_requires_valid_prior_and_rejects_direct_output(self):
        stats = dict(y_mean=torch.zeros(2), y_std=torch.ones(2))
        for prior in (None, 0., -1., float('nan'), float('inf')):
            with self.assertRaisesRegex(ValueError, 'TRAIN event_prior'):
                ForecastLoss(LossConfig(excess_formulation='severity_shape', shape_loss_weight=.3), stats, 1., 1., event_prior=prior)
        c = LossConfig(excess_formulation='severity_shape', shape_loss_weight=.3)
        criterion = ForecastLoss(c, stats, 1., 1., event_prior=.25)
        direct = ExceedanceHead(4, 0, [1., 1.], .25)(torch.ones(2, 2, 4))
        with self.assertRaisesRegex(ValueError, 'severity_shape dual head'):
            criterion(direct, direct.prediction, torch.zeros(2, 2))

    def test_parser_defaults_and_all_invalid_combinations(self):
        defaults = train.parse_args(['--model', 'perceiver3'])
        self.assertEqual((defaults.excess_formulation, defaults.shape_loss_weight, defaults.severity_shape_eps), ('direct', 0., 1e-6))
        invalid = [['--excess_formulation', 'unknown'], ['--shape_loss_weight', '.3'],
                   ['--head_type', 'single', '--excess_formulation', 'severity_shape'],
                   ['--model', 'baseline', '--excess_formulation', 'severity_shape'], ['--severity_loss_weight', '1']]
        invalid += [['--shape_loss_weight', x] for x in ('-1', 'nan', 'inf')]
        invalid += [['--severity_shape_eps', x] for x in ('0', '-1', 'nan', 'inf')]
        for options in invalid:
            with self.subTest(options=options), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                train.parse_args(['--model', 'perceiver3', *options])
        for amp_weight, shape_weight in itertools.product(('0', '.3'), ('0', '.7')):
            args = train.parse_args(['--model', 'perceiver3', '--excess_formulation', 'severity_shape',
                                    '--excess_amp_loss_weight', amp_weight, '--shape_loss_weight', shape_weight])
            self.assertEqual((args.excess_amp_loss_weight, args.shape_loss_weight), (float(amp_weight), float(shape_weight)))
        stats = dict(y_mean=torch.zeros(2), y_std=torch.ones(2))
        for c in (LossConfig(shape_loss_weight=.3), LossConfig(shape_loss_weight=-1),
                  LossConfig(excess_formulation='unknown'), LossConfig(severity_shape_eps=0)):
            with self.assertRaises(ValueError):
                ForecastLoss(c, stats, 1., 1., event_prior=.25)

    def test_model_and_target_scale_validation(self):
        options = dict(in_channels=3, out_channels=2, hidden_channels=8, node_read_heads=2,
                       time_read_heads=2, peak_threshold_norm=[1., 1.], excess_formulation='severity_shape')
        for scale in (None, [], [1.], [1., 2., 3.], [[1., 2.]], [0., 1.], [-1., 1.], [float('inf'), 1.], [float('nan'), 1.]):
            with self.subTest(scale=scale), self.assertRaisesRegex(ValueError, 'target_y_std'):
                build_model(ModelConfig(**options, target_y_std=scale))
        for model, head in (('baseline', 'single'), ('pact', 'single')):
            with self.assertRaisesRegex(ValueError, 'supervised dual exceedance head'):
                build_model(ModelConfig(**options, target_y_std=[1., 2.], model=model, head_type=head))
        for eps in (0., -1., float('nan'), float('inf')):
            with self.assertRaisesRegex(ValueError, 'severity_shape_eps'):
                build_model(ModelConfig(**options, target_y_std=[1., 2.], severity_shape_eps=eps))

    def check_runtime(self, device, dtype):
        for pool, fixed in itertools.product(('max', 'smoothmax'), (False, True)):
            head, context, target, stats = severity_fixture(fixed_gate=fixed)
            head, context, target = head.to(device), context.to(device), target.to(device)
            stats = {k: v.to(device) for k, v in stats.items()}
            c = LossConfig(loss_mode='mse_tail', excess_formulation='severity_shape', excess_loss_weight=5.,
                excess_amp_loss_weight=.7, excess_amp_pool=pool, shape_loss_weight=.3,
                dual_ablation='fixed_gate' if fixed else 'none')
            criterion = ForecastLoss(c, stats, 1., 1., event_prior=.25).to(device)
            with torch.autocast(device_type=device.type, dtype=dtype):
                output = head(context)
                loss = criterion(output, output.prediction * stats['y_std'], target)
            self.assertEqual(loss.shape, torch.Size([]))
            self.assertEqual(loss.dtype, torch.float32)
            self.assertTrue(torch.isfinite(loss))
            torch.testing.assert_close(output.excess_shape.max(dim=1).values, torch.ones(4, device=device), rtol=0, atol=0)
            loss.backward()
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in head.parameters()))

    def test_cpu_bfloat16_runtime(self):
        self.check_runtime(torch.device('cpu'), torch.bfloat16)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA is unavailable')
    def test_cuda_fp16_bfloat16_runtime(self):
        self.check_runtime(torch.device('cuda'), torch.float16)
        if torch.cuda.is_bf16_supported():
            self.check_runtime(torch.device('cuda'), torch.bfloat16)

    @unittest.skipUnless(dist.is_available() and dist.is_gloo_available(), 'Gloo is unavailable')
    def test_ddp_matches_global_loss_and_gradients_including_event_free_rank(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            mp.spawn(shape_ddp_worker, args=(str(path / 'rendezvous'), str(path / 'result.pt')), nprocs=2, join=True)
            records = torch.load(path / 'result.pt', weights_only=True)
        head, context, target, stats = severity_fixture()
        for pool in ('max', 'smoothmax'):
            head.zero_grad(set_to_none=True)
            output = head(context)
            c = LossConfig(excess_formulation='severity_shape', excess_loss_weight=2., excess_amp_loss_weight=.7,
                           excess_amp_pool=pool, shape_loss_weight=.3)
            loss = ForecastLoss(c, stats, 1., 1., event_prior=.25)(output, output.prediction * stats['y_std'], target)
            loss.backward()
            torch.testing.assert_close(records[pool]['loss'], loss)
            for name, p in head.named_parameters():
                torch.testing.assert_close(records[pool]['grads'][name], p.grad, rtol=2e-5, atol=2e-6)

    def test_shell_forwards_all_controls_and_preserves_direct_run_tag(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'interface.sh'
            for formulation in ('direct', 'severity_shape'):
                shape_weight = '.3' if formulation == 'severity_shape' else '0'
                path.write_text(f'MODEL=perceiver3\nHEAD_TYPE=dual\nEXCESS_FORMULATION={formulation}\n'
                                f'SHAPE_LOSS_WEIGHT={shape_weight}\nSEVERITY_SHAPE_EPS=2e-6\n'
                                'EXCESS_AMP_LOSS_WEIGHT=.7\nEXCESS_AMP_POOL=max\nEXCESS_AMP_BETA=20.0\n')
                command = dry_commands(path)[0]
                args = train.parse_args(command)
                self.assertEqual((args.excess_formulation, args.shape_loss_weight, args.severity_shape_eps),
                                 (formulation, float(shape_weight), 2e-6))
                self.assertEqual(args.excess_amp_loss_weight, .7)
                if formulation == 'severity_shape':
                    self.assertIn('_efseverity_shape', args.run_tag)
                else:
                    self.assertNotIn('_efdirect', args.run_tag)

    def test_training_metadata_and_optional_diagnostic_roundtrip_without_training(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            torch.manual_seed(83)
            graphs, stations = make_fixture(root)
            store = ForcingGraphStore(graphs, 'Battery')
            fitted = fit_loss_thresholds(store, store.split()['train'], tail_frac=.4, exceedance_percentile=75)
            metrics = dict(rmse_all=1., mae_all=1., rmse_peak5=1., mae_peak5=1.)
            for formulation, ablation in [('direct', 'none'), ('severity_shape', 'none'), ('severity_shape', 'fixed_gate')]:
                destination = root / f'{formulation}_{ablation}'
                shape_weight = '.3' if formulation == 'severity_shape' else '0'
                epochs = [EpochResult(metrics), EpochResult(metrics), EpochResult(metrics,
                    dict(y_true=np.zeros((0, 4)), y_pred=np.zeros((0, 4)), tags=np.array([], dtype=str)))]
                # Exercise persistence without executing an optimizer or training epoch.
                with patch.object(train, 'ForecastLoss', wraps=ForecastLoss) as constructor, \
                     patch.object(train, 'run_epoch', side_effect=epochs), \
                     patch.object(train.torch.optim.lr_scheduler.LambdaLR, 'step'), \
                     contextlib.redirect_stdout(io.StringIO()):
                    train.main(['--root_dir', str(graphs), '--station', 'Battery', '--station_json_dir', str(stations),
                        '--output_dir', str(destination), '--model', 'perceiver3', '--head_type', 'dual',
                        '--excess_formulation', formulation, '--dual_ablation', ablation, '--excess_amp_loss_weight', '.7',
                        '--shape_loss_weight', shape_weight, '--severity_shape_eps', '2e-6', '--tail_frac', '.4',
                        '--exceedance_percentile', '75', '--epochs', '1', '--device', 'cpu', '--num_workers', '0',
                        '--hidden_channels', '16', '--history_hours', '12'])
                self.assertEqual(constructor.call_args.kwargs, dict(event_prior=fitted['event_prior'], event_threshold=fitted['tau_phys']))
                ckpt = next(destination.glob('best_*.pth'))
                saved = torch.load(ckpt, weights_only=False)
                self.assertEqual(saved['loss_thresholds'], fitted)
                self.assertEqual(saved['model_config']['excess_formulation'], formulation)
                self.assertEqual(saved['dual_metadata']['excess_formulation'], formulation)
                config = ModelConfig(**saved['model_config'])
                restored = build_model(config)
                restored.load_state_dict(saved['model_state'], strict=True)
                if formulation == 'severity_shape':
                    self.assertEqual(config.target_y_std, saved['normalization']['y_std'].tolist())
                    torch.testing.assert_close(restored.head.target_y_std[0], saved['normalization']['y_std'], rtol=0, atol=0)
                    self.assertEqual(saved['dual_metadata']['severity_shape_eps'], 2e-6)
                else:
                    self.assertIsNone(config.target_y_std)
                    self.assertNotIn('head.target_y_std', saved['model_state'])
                    # Also exercise a real legacy-style inference reconstruction.
                    for key in ('excess_formulation', 'target_y_std', 'severity_shape_eps'):
                        saved['model_config'].pop(key)
                    for key in ('excess_formulation', 'shape_loss_weight', 'severity_shape_eps'):
                        saved['training_config'].pop(key)
                    torch.save(saved, ckpt)
                snapshot = json.loads(next(destination.glob('config_*.json')).read_text())
                shell = (destination / 'config_used.sh').read_text()
                for key, value in dict(excess_formulation=formulation, shape_loss_weight=float(shape_weight),
                                       excess_amp_loss_weight=.7, severity_shape_eps=2e-6).items():
                    self.assertEqual(snapshot[key], value)
                    self.assertIn(f'{key.upper()}={value}', shell)
                for diagnostic in (False, True):
                    with contextlib.redirect_stdout(io.StringIO()):
                        infer.main(['--ckpt', str(ckpt), '--root_dir', str(graphs), '--out_dir', str(destination / str(diagnostic)),
                                    '--device', 'cpu', '--num_workers', '0', '--batch_size', '2', '--save_npz',
                                    *(['--dual_diagnostics'] if diagnostic else [])])
                with np.load(destination / 'True/dual_diagnostics.npz') as arrays, np.load(destination / 'False/predictions.npz') as plain:
                    self.assertEqual(set(plain.files), {'y_true', 'y_pred', 'tags'})
                    np.testing.assert_array_equal(arrays['y_pred'], plain['y_pred'])
                    self.assertEqual(str(arrays['excess_formulation']), formulation)
                    if formulation == 'severity_shape':
                        np.testing.assert_allclose(arrays['excess_phys'], arrays['severity_phys'][:, None] * arrays['excess_shape'], rtol=1e-6, atol=1e-7)
                        np.testing.assert_allclose(arrays['y_pred'], arrays['body_phys'] + arrays['gate_probability'][:, None] * arrays['severity_phys'][:, None] * arrays['excess_shape'], rtol=1e-6, atol=1e-6)
                        self.assertEqual(float(arrays['severity_shape_eps']), 2e-6)
                    else:
                        self.assertNotIn('severity_phys', arrays.files)
                        self.assertNotIn('excess_shape', arrays.files)
                        self.assertNotIn('severity_shape_eps', arrays.files)
                report = json.loads((destination / 'True/dual_diagnostics.json').read_text())
                self.assertEqual(report['excess_formulation'], formulation)
                self.assertIn('excess_amp_rmse', report['overall'])
                if formulation == 'severity_shape':
                    self.assertEqual(report['severity_shape_eps'], 2e-6)


if __name__ == '__main__':
    unittest.main()
