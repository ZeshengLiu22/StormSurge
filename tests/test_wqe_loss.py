"""Scale-aware WQE and independent global/raw-excess objective switches."""

from dataclasses import replace
import itertools
import unittest

import torch
import torch.nn.functional as F

from emulator.models import ForecastOutput
from emulator.training.excess_amplitude import excess_amplitude_terms
from emulator.training.excess_shape import excess_shape_loss
from emulator.training.losses import ForecastLoss, LossConfig, dual_loss_terms, wqe_penalty


class WQEPenaltyTests(unittest.TestCase):
    def test_manual_quantile_expectile_and_mixture(self):
        # Normalized residuals are +2 (under), -2 (over), and exactly zero.
        target = torch.tensor([[2., -2., 0.]], dtype=torch.float64)
        prediction = torch.zeros_like(target, requires_grad=True)
        quantile = torch.tensor([[.5, 1.5, 0.]], dtype=torch.float64)
        expectile = torch.tensor([[3.28, .72, 0.]], dtype=torch.float64)
        for weights, expected in (((1., 0.), quantile), ((0., 1.), expectile),
                                  ((1. / 6., 5. / 6.), quantile / 6 + 5 * expectile / 6)):
            with self.subTest(weights=weights):
                actual = wqe_penalty(prediction, target, torch.ones(3),
                                     quantile_weight=weights[0], expectile_weight=weights[1])
                self.assertEqual(actual.shape, target.shape)
                torch.testing.assert_close(actual, expected, rtol=1e-14, atol=1e-14)
                self.assertEqual(actual[0, 2].item(), 0.)
                gradient, = torch.autograd.grad(actual.sum(), prediction)
                self.assertTrue(torch.isfinite(gradient).all())
                self.assertLess(gradient[0, 0].item(), 0.)
                self.assertGreater(gradient[0, 1].item(), 0.)

    def test_large_equal_errors_penalize_underprediction_more(self):
        penalty = wqe_penalty(torch.zeros(2), torch.tensor([4., -4.]), torch.ones(2))
        self.assertGreater(penalty[0].item(), penalty[1].item())

    def test_horizon_scales_broadcast_and_receive_no_gradient(self):
        # Each horizon has the same normalized error, but different physical units.
        scale = torch.tensor([2., 10., 100.], dtype=torch.float64, requires_grad=True)
        target = torch.tensor([[4., 20., 200.], [-4., -20., -200.]], dtype=torch.float64)
        prediction = torch.zeros_like(target, requires_grad=True)
        normalized_penalty = torch.tensor([[.5 / 6 + 5 * 3.28 / 6],
                                           [1.5 / 6 + 5 * .72 / 6]], dtype=torch.float64)
        expected = normalized_penalty * scale.detach().square()
        for broadcast_scale in (scale, scale.reshape(1, 3)):
            with self.subTest(shape=broadcast_scale.shape):
                torch.testing.assert_close(wqe_penalty(prediction, target, broadcast_scale), expected)
        actual = wqe_penalty(prediction, target, scale)
        actual.sum().backward()
        self.assertIsNone(scale.grad)
        self.assertTrue(torch.isfinite(prediction.grad).all())
        self.assertGreater(prediction.grad.abs().sum().item(), 0.)
        expected_gradient = torch.tensor([[-(.25 / 6 + 5 * 2 * .82 * 2 / 6)],
                                          [.75 / 6 + 5 * 2 * .18 * 2 / 6]], dtype=torch.float64) * scale.detach()
        torch.testing.assert_close(prediction.grad, expected_gradient)

    def test_scalar_scale_and_custom_parameters(self):
        prediction = torch.tensor([[0., 4.]], dtype=torch.float64)
        target = torch.tensor([[6., 0.]], dtype=torch.float64)
        # scale=2 => residuals +3,-2; q=.6,1.6; e=6.3,1.2.
        expected = torch.tensor([[4 * (.4 * .6 + .6 * 6.3),
                                  4 * (.4 * 1.6 + .6 * 1.2)]], dtype=torch.float64)
        actual = wqe_penalty(prediction, target, 2., quantile_tau=.2, expectile_tau=.7,
                             quantile_weight=.4, expectile_weight=.6)
        torch.testing.assert_close(actual, expected)

    def test_python_scalar_scale_preserves_double_precision(self):
        prediction = torch.tensor([[1.25, -3.5]], dtype=torch.float64)
        target = torch.tensor([[5.3, -6.7]], dtype=torch.float64)
        scale = 1.2345678901234567
        actual = wqe_penalty(prediction, target, scale)
        expected = wqe_penalty(prediction, target, torch.tensor(scale, dtype=torch.float64))
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_invalid_parameters_and_scales_are_rejected(self):
        prediction = torch.zeros(2, 3)
        for name in ("quantile_tau", "expectile_tau"):
            for value in (0., 1., -.1, 1.1, float("nan"), float("inf")):
                with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                    wqe_penalty(prediction, prediction, 1., **{name: value})
        for name in ("quantile_weight", "expectile_weight"):
            for value in (-.1, float("nan"), float("inf")):
                with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                    wqe_penalty(prediction, prediction, 1., **{name: value})
        for weights in ((0., 0.), (.2, .3), (1., 1.)):
            with self.subTest(weights=weights), self.assertRaises(ValueError):
                wqe_penalty(prediction, prediction, 1., quantile_weight=weights[0],
                            expectile_weight=weights[1])
        for invalid in (0., -1., float("nan"), float("inf")):
            for scale in (invalid, torch.tensor([1., invalid, 2.])):
                with self.subTest(scale=scale), self.assertRaises(ValueError):
                    wqe_penalty(prediction, prediction, scale)


class WQEObjectiveTests(unittest.TestCase):
    def setUp(self):
        self.stats = dict(y_mean=torch.tensor([-1., 0., 3.], dtype=torch.float64),
                          y_std=torch.tensor([2., 4., 8.], dtype=torch.float64))
        self.target = torch.tensor([[14., 8., 10.], [9., -2., 10.], [4., 26., 0.]], dtype=torch.float64)
        self.prediction = torch.tensor([[12., 6., 9.], [8., 0., 12.], [8., 22., 2.]], dtype=torch.float64)
        self.target_norm = (self.target - self.stats["y_mean"]) / self.stats["y_std"]
        self.tau = 10.
        self.event = torch.tensor([[True], [False], [True]])
        self.excess_target = torch.tensor([[2., 0., 0.], [0., 0., 0.], [0., 4., 0.]], dtype=torch.float64)
        self.output = ForecastOutput(
            self.prediction,
            torch.tensor([[2., 1., -.5], [.5, 1., .75], [-1., .5, 1.]], dtype=torch.float64),
            torch.tensor([[1., 2., .5], [2., 4., 6.], [3., 1., 2.]], dtype=torch.float64, requires_grad=True),
            torch.tensor([[-2.], [1.], [.5]], dtype=torch.float64, requires_grad=True),
            (self.tau - self.stats["y_mean"]) / self.stats["y_std"],
            excess_shape=torch.tensor([[.5, .4, 1.], [.2, .5, 1.], [.6, 1., .2]], dtype=torch.float64),
        )

    def criterion(self, config):
        return ForecastLoss(config, self.stats, self.tau, event_prior=.4, extreme_hour_prior=.2)

    def branches(self, mode="mse", *, physical=True, output=None):
        kwargs = dict(target_phys=self.target, tau_physical=self.tau) if physical else {}
        return dual_loss_terms(output or self.output, self.target_norm, self.stats["y_std"],
                               excess_loss_mode=mode, **kwargs)

    def test_defaults_and_old_mse_objective_are_exact(self):
        config = LossConfig(body_loss_weight=1., excess_loss_weight=2., gate_loss_weight=.5)
        self.assertEqual((config.loss_mode, config.excess_loss_mode), ("mse", "mse"))
        self.assertEqual((config.wqe_quantile_tau, config.wqe_expectile_tau,
                          config.wqe_quantile_weight, config.wqe_expectile_weight),
                         (.25, .82, 1. / 6., 5. / 6.))
        scale = self.stats["y_std"]
        body = ((self.output.body - (self.target_norm - self.excess_target)) * scale).square().mean()
        excess = (((self.output.excess - self.excess_target) * scale).square() * self.event).mean()
        gate = F.binary_cross_entropy_with_logits(self.output.gate_logits, self.event.float()) * scale.square().mean()
        for physical in (False, True):
            with self.subTest(physical=physical):
                for actual, expected in zip(self.branches(physical=physical), (body, excess, gate)):
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        expected = (self.prediction - self.target).square().mean() + (body + 2 * excess + .5 * gate)
        actual = self.criterion(config)(self.output, self.prediction, self.target)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_global_wqe_is_helper_mean_over_all_hours(self):
        output = ForecastOutput(self.prediction)
        for mode in ("mse", "wqe"):
            with self.subTest(mode=mode):
                expected = ((self.prediction - self.target).square().mean() if mode == "mse" else
                            wqe_penalty(self.prediction, self.target, self.stats["y_std"]).mean())
                config = LossConfig(loss_mode=mode)
                actual = self.criterion(config)(output, self.prediction, self.target)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                # Even a threshold above every target leaves global WQE unchanged.
                high_tau = ForecastLoss(config, self.stats, 1000.)(output, self.prediction, self.target)
                torch.testing.assert_close(high_tau, actual, rtol=0, atol=0)

    def test_excess_wqe_keeps_event_mask_all_horizons_and_full_denominator(self):
        scale = self.stats["y_std"]
        pointwise = wqe_penalty(self.output.excess * scale, self.excess_target * scale, scale)
        expected = (pointwise * self.event).mean()
        for physical in (False, True):
            with self.subTest(physical=physical):
                actual = self.branches("wqe", physical=physical)[1]
                torch.testing.assert_close(actual, expected)
        # Two event windows and three horizons: denominator remains B*K=9.
        torch.testing.assert_close(expected, pointwise[[0, 2]].sum() / 9)
        self.assertGreater(pointwise[0, 1:].sum().item(), 0.)
        expected.backward()
        torch.testing.assert_close(self.output.excess.grad[1], torch.zeros(3, dtype=torch.float64))
        self.assertTrue((self.output.excess.grad[0, 1:].abs() > 0).all())
        self.assertIsNone(self.output.gate_logits.grad)

    def test_excess_modes_use_raw_excess_before_gate(self):
        for mode in ("mse", "wqe"):
            with self.subTest(mode=mode):
                loss = self.branches(mode)[1]
                closed = self.output._replace(gate_logits=torch.full_like(self.output.gate_logits, -100.))
                opened = self.output._replace(gate_logits=torch.full_like(self.output.gate_logits, 100.))
                torch.testing.assert_close(self.branches(mode, output=closed)[1], loss, rtol=0, atol=0)
                torch.testing.assert_close(self.branches(mode, output=opened)[1], loss, rtol=0, atol=0)
                gradient, gate_gradient = torch.autograd.grad(
                    loss, (self.output.excess, self.output.gate_logits), allow_unused=True)
                self.assertIsNone(gate_gradient)
                torch.testing.assert_close(gradient[1], torch.zeros_like(gradient[1]), rtol=0, atol=0)
                self.assertTrue((gradient[0, 1:].abs() > 0).all())

    def test_excess_wqe_does_not_differentiate_train_scale_through_target_normalization(self):
        scale = self.stats["y_std"].clone().requires_grad_()
        excess = dual_loss_terms(self.output, self.target_norm, scale, target_phys=self.target,
                                tau_physical=self.tau, excess_loss_mode="wqe")[1]
        scale_gradient, excess_gradient = torch.autograd.grad(
            excess, (scale, self.output.excess), allow_unused=True)
        self.assertIsNone(scale_gradient)
        self.assertTrue(torch.isfinite(excess_gradient).all())
        self.assertGreater(excess_gradient.abs().sum().item(), 0.)

    def test_forecast_loss_treats_train_scale_as_fixed_in_all_combinations(self):
        for global_mode, excess_mode in itertools.product(("mse", "wqe"), repeat=2):
            with self.subTest(global_mode=global_mode, excess_mode=excess_mode):
                scale = self.stats["y_std"].clone().requires_grad_()
                stats = dict(self.stats, y_std=scale)
                prediction = self.prediction.clone().requires_grad_()
                config = LossConfig(loss_mode=global_mode, excess_loss_mode=excess_mode)
                actual = ForecastLoss(config, stats, self.tau)(self.output, prediction, self.target)
                scale_gradient, prediction_gradient = torch.autograd.grad(actual, (scale, prediction), allow_unused=True)
                self.assertIsNone(scale_gradient)
                self.assertTrue(torch.isfinite(prediction_gradient).all())
                self.assertGreater(prediction_gradient.abs().sum().item(), 0.)

    def test_event_free_excess_is_differentiable_zero(self):
        for mode in ("mse", "wqe"):
            with self.subTest(mode=mode):
                target = torch.zeros_like(self.target)
                target_norm = (target - self.stats["y_mean"]) / self.stats["y_std"]
                excess = dual_loss_terms(self.output, target_norm, self.stats["y_std"],
                    target_phys=target, tau_physical=self.tau, excess_loss_mode=mode)[1]
                self.assertEqual(excess.item(), 0.)
                gradient, = torch.autograd.grad(excess, self.output.excess)
                torch.testing.assert_close(gradient, torch.zeros_like(gradient), rtol=0, atol=0)

    def test_all_four_global_excess_combinations_are_independent(self):
        totals = {}
        body, mse_excess, gate = self.branches()
        wqe_body, wqe_excess, wqe_gate = self.branches("wqe")
        torch.testing.assert_close(wqe_body, body, rtol=0, atol=0)
        torch.testing.assert_close(wqe_gate, gate, rtol=0, atol=0)
        global_terms = dict(mse=(self.prediction - self.target).square().mean(),
                            wqe=wqe_penalty(self.prediction, self.target, self.stats["y_std"]).mean())
        excess_terms = dict(mse=mse_excess, wqe=wqe_excess)
        for global_mode, excess_mode in itertools.product(("mse", "wqe"), repeat=2):
            with self.subTest(global_mode=global_mode, excess_mode=excess_mode):
                config = LossConfig(loss_mode=global_mode, excess_loss_mode=excess_mode,
                                    body_loss_weight=1., excess_loss_weight=2., gate_loss_weight=.5)
                actual = self.criterion(config)(self.output, self.prediction, self.target)
                expected = global_terms[global_mode] + (body + 2 * excess_terms[excess_mode] + .5 * gate)
                torch.testing.assert_close(actual, expected)
                totals[global_mode, excess_mode] = actual
        for excess_mode in ("mse", "wqe"):
            torch.testing.assert_close(totals["wqe", excess_mode] - totals["mse", excess_mode],
                                       global_terms["wqe"] - global_terms["mse"])
        for global_mode in ("mse", "wqe"):
            torch.testing.assert_close(totals[global_mode, "wqe"] - totals[global_mode, "mse"],
                                       2 * (wqe_excess - mse_excess))

    def test_same_custom_wqe_parameters_reach_global_and_excess(self):
        config = LossConfig(loss_mode="wqe", excess_loss_mode="wqe", wqe_quantile_tau=.4,
            wqe_expectile_tau=.7, wqe_quantile_weight=.3, wqe_expectile_weight=.7)
        parameters = dict(quantile_tau=.4, expectile_tau=.7, quantile_weight=.3, expectile_weight=.7)
        scale = self.stats["y_std"]
        body, _, gate = self.branches()
        prediction = wqe_penalty(self.prediction, self.target, scale, **parameters).mean()
        excess = (wqe_penalty(self.output.excess * scale, self.excess_target * scale,
                              scale, **parameters) * self.event).mean()
        actual = self.criterion(config)(self.output, self.prediction, self.target)
        torch.testing.assert_close(actual, prediction + (body + excess + gate))

    def test_optional_objectives_keep_existing_definitions_in_all_combinations(self):
        scale = self.stats["y_std"]
        strict_mse = ((self.prediction - self.target).square() * (self.target > self.tau)).mean() / .2
        amplitude = excess_amplitude_terms(self.output.excess, self.target_norm, self.output.threshold,
            scale, .4, target_phys=self.target, tau_physical=self.tau).loss
        shape = excess_shape_loss(self.output.excess_shape, self.target_norm, self.output.threshold,
            scale, .4, target_phys=self.target, tau_physical=self.tau)
        additions = ((dict(exceedance_loss_weight=.3), .3 * strict_mse),
                     (dict(excess_amp_loss_weight=.7), .7 * amplitude),
                     (dict(excess_formulation="severity_shape", shape_loss_weight=.2), .2 * shape))
        for global_mode, excess_mode in itertools.product(("mse", "wqe"), repeat=2):
            config = LossConfig(loss_mode=global_mode, excess_loss_mode=excess_mode)
            baseline = self.criterion(config)(self.output, self.prediction, self.target)
            for changes, expected in additions:
                with self.subTest(global_mode=global_mode, excess_mode=excess_mode, changes=changes):
                    actual = self.criterion(replace(config, **changes))(self.output, self.prediction, self.target)
                    torch.testing.assert_close(actual - baseline, expected)

    def test_wqe_slope_adds_existing_slope_penalty(self):
        for robust in ("charb", "huber"):
            with self.subTest(robust=robust):
                config = LossConfig(loss_mode="wqe_slope", slope_lambda=.4, slope_mask_s=2., slope_robust=robust)
                output = ForecastOutput(self.prediction)
                actual = self.criterion(config)(output, self.prediction, self.target)
                difference = self.prediction.diff(dim=1) - self.target.diff(dim=1)
                penalty = ((difference.square() + config.slope_charb_eps ** 2).sqrt() if robust == "charb" else
                           F.huber_loss(difference, torch.zeros_like(difference), reduction="none",
                                        delta=config.slope_huber_delta))
                mask = torch.sigmoid((self.tau - self.target.amax(dim=1)) / config.slope_mask_s)
                expected = wqe_penalty(self.prediction, self.target, self.stats["y_std"]).mean()
                expected += .4 * (penalty * mask[:, None]).mean()
                torch.testing.assert_close(actual, expected)

    def test_invalid_excess_mode_is_rejected(self):
        with self.assertRaises(ValueError):
            self.criterion(LossConfig(excess_loss_mode="wmse"))
        with self.assertRaises(ValueError):
            self.branches("wmse")


if __name__ == "__main__":
    unittest.main()
