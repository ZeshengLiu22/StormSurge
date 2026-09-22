"""Strict hourly exceedance population and fixed TRAIN normalization."""

from dataclasses import replace
import unittest

import torch

from emulator.models import ForecastOutput
from emulator.training.losses import ForecastLoss, LossConfig, dual_loss_terms
from emulator.training.excess_amplitude import excess_amplitude_terms
from emulator.training.excess_shape import excess_shape_target


class ExceedanceLossTests(unittest.TestCase):
    def objective(self, prediction, target, *, prior=.25, tau=1., weight=.3):
        config = LossConfig(exceedance_loss_weight=weight)
        stats = dict(y_mean=torch.zeros(target.shape[1]), y_std=torch.ones(target.shape[1]))
        return ForecastLoss(config, stats, tau, extreme_hour_prior=prior)(ForecastOutput(prediction), prediction, target)

    def extra(self, prediction, target, **kwargs):
        return self.objective(prediction, target, **kwargs) - self.objective(prediction, target, weight=0.)

    def test_selects_only_strict_extreme_hours_and_gradient(self):
        target = torch.tensor([[2., 0., 1.], [.5, 3., -4.]], dtype=torch.float64)
        prediction = torch.zeros_like(target, requires_grad=True)
        extra = self.extra(prediction, target)
        mask = target > 1.
        expected = .3 * ((prediction-target).square()*mask).mean() / .25
        torch.testing.assert_close(extra, expected)
        extra.backward()
        torch.testing.assert_close(prediction.grad, .3 * 2 * (prediction-target)*mask / (target.numel()*.25))
        self.assertEqual(torch.count_nonzero(prediction.grad).item(), 2)

    def test_empty_event_batch_is_differentiable_exact_zero(self):
        target = torch.tensor([[0., 1.], [-1., .4]])
        prediction = torch.zeros_like(target, requires_grad=True)
        extra = self.extra(prediction,target)
        self.assertEqual(extra.item(), 0.)
        extra.backward()
        torch.testing.assert_close(prediction.grad, torch.zeros_like(prediction))

    def test_fixed_prior_preserves_batch_partition_and_event_free_rank(self):
        target = torch.tensor([[2., 0.], [3., 4.], [0., 1.], [-1., 0.]])
        initial = torch.tensor([[.2, .3], [.1, .2], [.3, .4], [.2, .5]])
        whole = initial.clone().requires_grad_()
        whole_loss = self.extra(whole,target)
        whole_loss.backward()
        split = initial.clone().requires_grad_()
        split_loss = sum(self.extra(split[i:i+2],target[i:i+2]) for i in (0,2))/2
        split_loss.backward()
        torch.testing.assert_close(whole_loss,split_loss)
        torch.testing.assert_close(whole.grad,split.grad)
        repeated = self.extra(initial.repeat(3,1),target.repeat(3,1))
        torch.testing.assert_close(repeated,whole_loss)

    def test_one_physical_tau_defines_every_branch_even_near_float_boundary(self):
        tau=1.-1e-10
        target=torch.tensor([[1.,0.], [0.,0.]])
        scales=torch.tensor([.5,4.])
        norm=target/scales
        rounded=torch.tensor([2.,.25])
        excess=torch.ones_like(target,requires_grad=True)
        body=torch.zeros_like(target,requires_grad=True)
        logits=torch.zeros(2,1,requires_grad=True)
        output=ForecastOutput(body+logits.sigmoid()*excess,body,excess,logits,rounded)
        branch=dual_loss_terms(output,norm,scales,target_phys=target,tau_physical=tau)
        branch[1].backward(retain_graph=True)
        self.assertGreater(excess.grad[0].abs().sum().item(),0.)
        self.assertEqual(excess.grad[1].abs().sum().item(),0.)
        amp=excess_amplitude_terms(excess,norm,rounded,scales,.5,target_phys=target,tau_physical=tau)
        shape=excess_shape_target(norm,rounded,scales,target_phys=target,tau_physical=tau)
        self.assertEqual(amp.event.flatten().tolist(),[True,False])
        self.assertEqual(shape.event.flatten().tolist(),[True,False])
        self.assertGreater(amp.r_target_phys[0,0].item(),0.)
        torch.testing.assert_close(shape.shape_target,torch.tensor([[1.,0.],[0.,0.]]))
        pred=torch.zeros_like(target,requires_grad=True)
        self.assertGreater(self.extra(pred,target,tau=tau).item(),0.)

    def test_fixed_train_prior_required_only_when_enabled(self):
        stats=dict(y_mean=torch.zeros(2),y_std=torch.ones(2))
        for prior in (None,0.,-1.,1.1,float('nan'),float('inf')):
            with self.subTest(prior=prior),self.assertRaisesRegex(ValueError,'extreme_hour_prior'):
                ForecastLoss(LossConfig(exceedance_loss_weight=.1),stats,1.,extreme_hour_prior=prior)
            ForecastLoss(LossConfig(),stats,1.,extreme_hour_prior=prior)
        for weight in (-1.,float('nan'),float('inf')):
            with self.assertRaisesRegex(ValueError,'exceedance_loss_weight'):
                ForecastLoss(LossConfig(exceedance_loss_weight=weight),stats,1.)


if __name__=='__main__':
    unittest.main()
