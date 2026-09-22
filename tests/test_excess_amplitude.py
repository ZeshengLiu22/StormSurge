"""GT-aligned peak-time excess supervision with an exact fixed TRAIN event prior."""

import contextlib
import copy
from datetime import timedelta
import io
import itertools
import json
import os
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel

import train
from emulator.common.dual import DUAL_ABLATIONS
from emulator.data import fit_loss_thresholds
from emulator.inference.dual_diagnostics import summarize_dual, summarize_excess_amplitude
from emulator.models import ForecastOutput
from emulator.models.heads import ExceedanceHead, SingleHead
from emulator.training import ForecastLoss, LossConfig, dual_loss_terms
from emulator.training.excess_amplitude import excess_amplitude_terms
from test_config_interfaces import MODES, dry_commands


def ddp_fixture():
    torch.manual_seed(821)
    # A scalar tau_phys=2 with heterogeneous horizon scales.
    head = ExceedanceHead(4, 0, [4., 1.], .25)
    with torch.no_grad():
        head.excess[-1].weight.fill_(.05)
    context = torch.arange(32, dtype=torch.float32).reshape(4, 2, 4) / 32
    stats = dict(y_mean=torch.zeros(2), y_std=torch.tensor([.5, 2.]))
    target = torch.tensor([[4., 1.], [3., 1.], [6., 1.], [4., 3.]]) * stats['y_std']
    return head, context, target, stats


def ddp_worker(rank, rendezvous, destination):
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method=Path(rendezvous).as_uri(), rank=rank,
                            world_size=2, timeout=timedelta(seconds=30))
    try:
        head, context, target, stats = ddp_fixture()
        model = DistributedDataParallel(head)
        records = {}
        # Repeated backwards exercise reducer state with an event-free rank.
        for step in range(2):
            model.zero_grad(set_to_none=True)
            output = model(context[2 * rank:2 * rank + 2])
            criterion = ForecastLoss(LossConfig(excess_loss_weight=2., gate_loss_weight=.5,
                excess_amp_loss_weight=.7), stats, 2., event_prior=.25)
            loss = criterion(output, output.prediction * stats['y_std'], target[2 * rank:2 * rank + 2])
            loss.backward()
            average = loss.detach().clone()
            dist.all_reduce(average)
            records[step] = dict(loss=average / 2,
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

    def test_wrong_time_equal_maximum_has_error_and_only_gt_aligned_peak_gradient(self):
        target_excess = torch.tensor([[.02, .04, .15, .08]])
        target = target_excess + 1.
        predicted = torch.tensor([[.02, .15, .10, .08]], requires_grad=True)
        body = torch.zeros_like(predicted, requires_grad=True)
        gate = torch.zeros(1, 1, requires_grad=True)
        output = ForecastOutput(body + gate.sigmoid() * predicted, body, predicted, gate, torch.ones(1, 4))
        terms = excess_amplitude_terms(output.excess, target, output.threshold, torch.ones(4), 1.,
                                       target_phys=target, tau_physical=1.)
        self.assertAlmostEqual(terms.loss.item(), (.10 - .15) ** 2)
        excess_grad, body_grad, gate_grad = torch.autograd.grad(
            terms.loss, (predicted, body, gate), allow_unused=True)
        torch.testing.assert_close(excess_grad, torch.tensor([[0., 0., -.1, 0.]]))
        self.assertIsNone(body_grad)
        self.assertIsNone(gate_grad)

    def test_original_physical_target_peak_first_tie_and_target_consistency(self):
        mean, std, tau = torch.tensor([.5, -1., 3.]), torch.tensor([.5, 4., 2.]), 2.
        target = torch.tensor([[4., 10., 10.], [2., 2., 1.]])
        threshold = (tau - mean) / std
        normalized = (target - mean) / std
        predicted = torch.tensor([[12., 1.25, 8.], [50., 50., 50.]], requires_grad=True)
        terms = excess_amplitude_terms(predicted, normalized, threshold, std, .2,
                                       target_phys=target, tau_physical=tau)
        self.assertNotEqual(normalized[0].argmax(), target[0].argmax())
        torch.testing.assert_close(terms.r_target_phys, (target - tau).clamp_min(0))
        self.assertEqual(terms.event.flatten().tolist(), [True, False])
        self.assertEqual(terms.a_pred[0].item(), 5.)
        self.assertEqual(terms.a_target[0].item(), 8.)
        self.assertEqual(terms.loss.item(), 22.5)
        self.assertEqual(terms.r_target_phys[0].argmax().item(), target[0].argmax().item())
        torch.testing.assert_close(terms.a_target[0], terms.r_target_phys[0].max())
        terms.loss.backward()
        torch.testing.assert_close(predicted.grad, torch.tensor([[0., -60., 0.], [0., 0., 0.]]))

    def test_strict_physical_threshold_keeps_fitted_double_precision(self):
        # Rounding tau down to float32 would incorrectly classify this event as a tie.
        target = torch.tensor([[1., 0.]])
        tau = 1. - 1e-10
        predicted = torch.ones(1, 2, requires_grad=True)
        terms = excess_amplitude_terms(predicted, target, torch.ones(2), torch.ones(2), .2,
                                       target_phys=target, tau_physical=tau)
        self.assertTrue(terms.event.item())
        self.assertEqual(terms.loss.item(), 5.)
        terms.loss.backward()
        self.assertEqual(predicted.grad.tolist(), [[10., 0.]])

    def test_exact_train_prior_batch_additivity_and_gradient_accumulation(self):
        store = SimpleNamespace(graphs=[SimpleNamespace(y=torch.tensor([float(y), -1.]), center_time=f'2001-01-01T{i*2:02}:00:00')
                                        for i,y in enumerate((0, 1, 2, 2, 2, 5, 10000))])
        fitted = fit_loss_thresholds(store, list(range(6)), exceedance_percentile=75)
        self.assertEqual((fitted['tau_physical'], fitted['event_prior']), (2., 1 / 6))
        target = torch.tensor([[5., -1.], [2., -1.], [1., 2.], [3., -1.]])
        predicted = torch.ones(4, 2, requires_grad=True)
        def loss(excess, truth):
            return excess_amplitude_terms(excess, truth, torch.full((2,), 2.), torch.ones(2),
                fitted['event_prior'], target_phys=truth, tau_physical=fitted['tau_physical']).loss
        full = loss(predicted, target)
        full.backward()
        expected_gradient = predicted.grad.clone()
        self.assertEqual(full.item(), 6.)
        predicted.grad = None
        parts = [loss(predicted[start:start + 2], target[start:start + 2]) for start in (0, 2)]
        self.assertEqual(sum(part.item() for part in parts) / 2, full.item())
        for part in parts:
            (part / 2).backward()
        torch.testing.assert_close(predicted.grad, expected_gradient, rtol=0, atol=0)
        self.assertTrue(predicted.grad[1:3].eq(0).all())
        store.graphs[-1].y.fill_(-1e6)
        self.assertEqual(fitted, fit_loss_thresholds(store, list(range(6)), exceedance_percentile=75))

    def test_per_event_weight_is_independent_of_batch_event_count(self):
        gradients, losses = [], []
        for count in (1, 2):
            predicted = torch.full((4, 2), 3., requires_grad=True)
            target = torch.zeros(4, 2)
            target[:count, 0] = 2.
            loss = excess_amplitude_terms(predicted, target, torch.ones(2), torch.ones(2), .2,
                                          target_phys=target, tau_physical=1.).loss
            loss.backward()
            gradients.append(predicted.grad[0].clone())
            losses.append(loss.item())
            self.assertTrue(predicted.grad[count:].eq(0).all())
        torch.testing.assert_close(gradients[0], gradients[1], rtol=0, atol=0)
        self.assertEqual(gradients[0].tolist(), [5., 0.])
        self.assertEqual(losses[1], 2 * losses[0])

    def test_no_events_has_differentiable_exact_zero(self):
        predicted = torch.tensor([[1., 3.], [4., 2.]], requires_grad=True)
        target = torch.tensor([[1., 1.], [0., .5]])
        terms = excess_amplitude_terms(predicted, target, torch.ones(2), torch.ones(2), .17,
                                       target_phys=target, tau_physical=1.)
        self.assertEqual(terms.loss.item(), 0.)
        terms.loss.backward()
        torch.testing.assert_close(predicted.grad, torch.zeros_like(predicted), rtol=0, atol=0)

    def test_amp_reaches_shared_context_without_body_or_gate(self):
        head, context, target, stats = ddp_fixture()
        context.requires_grad_()
        output = head(context)
        excess_amplitude_terms(output.excess, target / stats['y_std'], output.threshold,
            stats['y_std'], .25, target_phys=target, tau_physical=2.).loss.backward()
        self.assertGreater(context.grad.abs().sum().item(), 0.)
        self.assertTrue(all(p.grad is None for p in head.body.parameters()))
        self.assertTrue(all(p.grad is None for p in head.gate.parameters()))

    def test_optional_weight_adds_exactly_for_all_existing_modes(self):
        head, context, target, stats = ddp_fixture()
        output = head(context)
        prediction = output.prediction * stats['y_std']
        amp = excess_amplitude_terms(output.excess, target / stats['y_std'], output.threshold,
            stats['y_std'], .25, target_phys=target, tau_physical=2.).loss
        for mode, weight in itertools.product(MODES, (.3, .7)):
            config = LossConfig(loss_mode=mode, excess_loss_weight=2., gate_loss_weight=.5)
            base = ForecastLoss(config, stats, 2.)(output, prediction, target)
            enabled = copy.deepcopy(config)
            enabled.excess_amp_loss_weight = weight
            actual = ForecastLoss(enabled, stats, 2., event_prior=.25)(output, prediction, target)
            torch.testing.assert_close(actual, base + weight * amp, rtol=0, atol=0)

    def test_zero_weight_skips_amp_and_preserves_outputs_gradients_rng(self):
        for mode, single in itertools.product(MODES, (False, True)):
            torch.manual_seed(35)
            head = SingleHead(4, .3) if single else ExceedanceHead(4, .3, [2., 2.], .25)
            context = torch.randn(3, 2, 4)
            target = torch.tensor([[2., 5.], [0., 1.], [1., -1.]])
            stats = dict(y_mean=torch.zeros(2), y_std=torch.ones(2))
            config = LossConfig(loss_mode=mode, excess_loss_weight=2., gate_loss_weight=.5)
            output = head(context)
            reference = ForecastLoss(config, stats, 2.)(ForecastOutput(output.prediction), output.prediction, target)
            if not single:
                body, excess, gate = dual_loss_terms(output, target, stats['y_std'])
                reference = reference + body + 2 * excess + .5 * gate
            expected_grads = torch.autograd.grad(reference, tuple(head.parameters()), retain_graph=True)
            rng = torch.get_rng_state()
            with patch('emulator.training.losses.excess_amplitude_terms', side_effect=AssertionError('disabled computation')):
                actual = ForecastLoss(config, stats, 2.)(output, output.prediction, target)
            torch.testing.assert_close(actual, reference, rtol=0, atol=0)
            actual_grads = torch.autograd.grad(actual, tuple(head.parameters()))
            for actual_grad, expected in zip(actual_grads, expected_grads):
                torch.testing.assert_close(actual_grad, expected, rtol=0, atol=0)
            torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)

    def test_validation_and_ablations(self):
        stats = dict(y_mean=torch.zeros(2), y_std=torch.ones(2))
        self.assertEqual(train.parse_args(['--model', 'perceiver3']).excess_amp_loss_weight, 0.)
        for value in ('-1', 'nan', 'inf'):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                train.parse_args(['--model', 'perceiver3', '--excess_amp_loss_weight', value])
        for prior in (None, 0., -1., float('nan'), float('inf')):
            ForecastLoss(LossConfig(), stats, 1., event_prior=prior)
            with self.assertRaisesRegex(ValueError, 'TRAIN event_prior'):
                ForecastLoss(LossConfig(excess_amp_loss_weight=1.), stats, 1., event_prior=prior)
        for tau in (None, float('nan'), float('inf')):
            with self.assertRaisesRegex(ValueError, 'TRAIN tau_physical'):
                ForecastLoss(LossConfig(excess_amp_loss_weight=1.), stats, tau, event_prior=.2)
        for mode in DUAL_ABLATIONS:
            disabled = mode in ('no_excess_loss', 'no_branch_supervision')
            args = train.parse_args(['--model', 'perceiver3', '--dual_ablation', mode, '--excess_amp_loss_weight', '.7'])
            self.assertEqual(args.excess_amp_loss_weight, 0. if disabled else .7)
            head, context, target, stats = ddp_fixture()
            output = head(context)
            if mode == 'fixed_gate':
                output = output._replace(gate_logits=None)
            config = LossConfig(dual_ablation=mode, excess_amp_loss_weight=.7)
            with patch('emulator.training.losses.excess_amplitude_terms', wraps=excess_amplitude_terms) as observed:
                ForecastLoss(config, stats, 2., event_prior=None if disabled else .25)(output, output.prediction * stats['y_std'], target)
            self.assertEqual(observed.call_count, 0 if disabled else 1)
        for model in ('baseline', 'perceiver3'):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                train.parse_args(['--model', model, '--head_type', 'single', '--excess_amp_loss_weight', '1'])

    def test_cpu_bfloat16_and_low_precision_products(self):
        self.check_runtime(torch.device('cpu'), torch.bfloat16)
        for dtype in (torch.bfloat16, torch.float16):
            predicted = torch.tensor([[300., 400.]], dtype=dtype, requires_grad=True)
            std = torch.tensor([300., 400.], dtype=dtype)
            target = torch.ones(1, 2, dtype=dtype)
            terms = excess_amplitude_terms(predicted, target, torch.zeros(2, dtype=dtype), std, .2,
                                           target_phys=target.float() * std.float(), tau_physical=0.)
            self.assertEqual(terms.loss.dtype, torch.float32)
            self.assertTrue(torch.isfinite(terms.loss))
            self.assertEqual(terms.r_pred_phys.dtype, torch.float32)
            self.assertEqual(terms.r_target_phys.dtype, torch.float32)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA is unavailable')
    def test_cuda_float16_and_bfloat16_autocast(self):
        self.check_runtime(torch.device('cuda'), torch.float16)
        if torch.cuda.is_bf16_supported():
            self.check_runtime(torch.device('cuda'), torch.bfloat16)

    def check_runtime(self, device, dtype):
        for mode in ('mse', 'wmse', 'mse_slope'):
            head, context, target, stats = ddp_fixture()
            head, context, target = head.to(device), context.to(device), target.to(device)
            stats = {key: value.to(device) for key, value in stats.items()}
            criterion = ForecastLoss(LossConfig(loss_mode=mode, excess_loss_weight=5., gate_loss_weight=.5,
                excess_amp_loss_weight=.7), stats, 2., event_prior=.25).to(device)
            with torch.autocast(device_type=device.type, dtype=dtype):
                output = head(context)
                loss = criterion(output, output.prediction * stats['y_std'], target)
            self.assertEqual(loss.dtype, torch.float32)
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in head.parameters()))

    @unittest.skipUnless(dist.is_available() and dist.is_gloo_available(), 'Gloo is unavailable')
    def test_ddp_matches_global_batch_with_event_free_rank(self):
        with tempfile.TemporaryDirectory() as temporary:
            rendezvous, destination = str(Path(temporary) / 'rendezvous'), str(Path(temporary) / 'result.pt')
            mp.spawn(ddp_worker, args=(rendezvous, destination), nprocs=2, join=True)
            records = torch.load(destination, weights_only=True)
        head, context, target, stats = ddp_fixture()
        output = head(context)
        criterion = ForecastLoss(LossConfig(excess_loss_weight=2., gate_loss_weight=.5,
            excess_amp_loss_weight=.7), stats, 2., event_prior=.25)
        loss = criterion(output, output.prediction * stats['y_std'], target)
        loss.backward()
        for record in records.values():
            torch.testing.assert_close(record['loss'], loss)
            for name, parameter in head.named_parameters():
                torch.testing.assert_close(record['grads'][name], parameter.grad, rtol=2e-5, atol=2e-6)

    def test_shell_forwards_amplitude_weight(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'interface.sh'
            path.write_text('MODEL=perceiver3\nHEAD_TYPE=dual\nEXCESS_AMP_LOSS_WEIGHT=.7\n')
            command = dry_commands(path)[0]
            self.assertEqual(command[command.index('--excess_amp_loss_weight') + 1], '.7')
            self.assertEqual(train.parse_args(command).excess_amp_loss_weight, .7)
            # Exercise the shell's actual resolved snapshot in a dry run.
            snapshot = Path(temporary) / 'resolved.sh'
            repo = Path(__file__).resolve().parents[1]
            subprocess.run(['bash', '-c', 'source "$1" "$2"; write_resolved_config "$3"',
                            'amp-config-test', str(repo / 'train.sh'), str(path), str(snapshot)],
                           cwd=repo, env=dict(os.environ, DRY_RUN='1'), check=True, capture_output=True, text=True)
            self.assertIn('EXCESS_AMP_LOSS_WEIGHT=".7"', snapshot.read_text())



class ExcessAmplitudeDiagnosticTests(unittest.TestCase):
    def test_gt_aligned_peak_diagnostic_ignores_larger_wrong_time_peak(self):
        truth = np.array([[2., 5.], [4., 1.], [2., 2.], [0., 1.]])
        excess = np.array([[30., 2.], [5., 20.], [100., 100.], [100., 100.]])
        arrays = dict(y_true=truth, y_pred=np.zeros_like(truth), body_phys=np.zeros_like(truth),
                      excess_phys=excess, gate_probability=np.zeros(4))
        report = summarize_dual(arrays, tau_phys=2.)
        expected = dict(gt_aligned_raw_excess_peak_rmse=np.sqrt(5), gt_aligned_raw_excess_peak_mae=2., gt_aligned_raw_excess_peak_bias=1.,
                        gt_aligned_raw_excess_peak_pred_mean=3.5, gt_aligned_raw_excess_peak_target_mean=2.5)
        for key, value in expected.items():
            self.assertAlmostEqual(report[key], value)
        self.assertEqual(summarize_excess_amplitude(excess, np.maximum(truth - 2., 0), truth,
                                                   (truth > 2.).any(axis=1)), expected)

    def test_no_events_and_empty_arrays_have_json_null_diagnostics(self):
        for count in (0, 3):
            truth = np.zeros((count, 2))
            report = summarize_excess_amplitude(np.ones((count, 2)), truth, truth, np.zeros(count, dtype=bool))
            self.assertEqual(len(report), 5)
            self.assertTrue(all(value is None for value in report.values()))
            json.dumps(report, allow_nan=False)


if __name__ == '__main__':
    unittest.main()
