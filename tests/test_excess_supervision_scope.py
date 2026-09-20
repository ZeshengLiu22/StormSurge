"""Loss-only scope, exact legacy denominator, and non-event gradient contracts."""

from dataclasses import asdict, fields
from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F

from emulator.models import ForecastOutput, ModelConfig
from emulator.training import ForecastLoss, LossConfig, dual_loss_terms
from emulator.training.arguments import parse_args
from test_config_interfaces import dry_commands


def example():
    target = torch.tensor([[.5, .5, 4.], [-1., 1., 1.]], dtype=torch.float64)
    scale = torch.tensor([.25, 2., 5.], dtype=torch.float64)
    body = torch.tensor([[-.2, .4, 1.], [-.5, .8, .7]], dtype=torch.float64, requires_grad=True)
    excess = torch.tensor([[.3, .4, 1.4], [.2, .5, .7]], dtype=torch.float64, requires_grad=True)
    logits = torch.tensor([[-.7], [.3]], dtype=torch.float64, requires_grad=True)
    output = ForecastOutput(body + excess, body, excess, logits,
                            torch.tensor([[0., 1., 2.]], dtype=torch.float64), logits.sigmoid())
    return output, target, scale


def test_event_vs_all_changes_only_non_event_excess_auxiliary_gradients():
    output, target, scale = example()
    event_terms = dual_loss_terms(output, target, scale, "event")
    all_terms = dual_loss_terms(output, target, scale, "all")
    excess_target = torch.tensor([[.5, 0., 2.], [0., 0., 0.]], dtype=torch.float64)
    expected_all = 2 * (output.excess - excess_target) * scale.square() / target.numel()
    event_gradient = torch.autograd.grad(event_terms[1], output.excess, retain_graph=True)[0]
    all_gradient = torch.autograd.grad(all_terms[1], output.excess, retain_graph=True)[0]
    torch.testing.assert_close(all_gradient, expected_all, rtol=1e-15, atol=0)
    torch.testing.assert_close(event_gradient[0], all_gradient[0], rtol=0, atol=0)
    assert (event_gradient[0] != 0).all()
    assert torch.count_nonzero(event_gradient[1]) == 0
    assert (all_gradient[1] > 0).all()  # Gradient descent reduces non-event excess toward zero.
    # Below-threshold horizons within an event window still have a zero target.
    assert event_gradient[0, 1] > 0
    for index, branch in ((0, output.body), (2, output.gate_logits)):
        torch.testing.assert_close(event_terms[index], all_terms[index], rtol=0, atol=0)
        torch.testing.assert_close(torch.autograd.grad(event_terms[index], branch, retain_graph=True)[0],
                                   torch.autograd.grad(all_terms[index], branch, retain_graph=True)[0], rtol=0, atol=0)


def test_three_argument_call_preserves_frozen_event_formula_and_gradients():
    output, target, scale = example()
    event = (target > output.threshold).any(dim=1, keepdim=True)
    excess_target = (target - output.threshold).clamp_min(0)
    legacy = (
        ((output.body - torch.minimum(target, output.threshold)) * scale).square().mean(),
        (((output.excess - excess_target) * scale).square() * event).mean(),
        F.binary_cross_entropy_with_logits(output.gate_logits, event.float()) * scale.square().mean(),
    )
    default = dual_loss_terms(output, target, scale)
    explicit = dual_loss_terms(output, target, scale, "event")
    for expected, actual, named, branch in zip(legacy, default, explicit,
                                               (output.body, output.excess, output.gate_logits)):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(actual, named, rtol=0, atol=0)
        torch.testing.assert_close(torch.autograd.grad(actual, branch, retain_graph=True)[0],
                                   torch.autograd.grad(expected, branch, retain_graph=True)[0], rtol=0, atol=0)


@pytest.mark.parametrize("events", [False, True])
def test_event_free_and_all_event_batches_keep_whole_batch_denominator(events):
    output, target, scale = example()
    target = output.threshold.expand_as(target) + (1. if events else 0.)
    event = dual_loss_terms(output, target, scale, "event")[1]
    all_windows = dual_loss_terms(output, target, scale, "all")[1]
    if events:
        torch.testing.assert_close(event, all_windows, rtol=0, atol=0)
    else:
        assert event.item() == 0.
        torch.testing.assert_close(all_windows, (output.excess * scale).square().mean(), rtol=0, atol=0)


@pytest.mark.parametrize("scope", ["event", "all"])
def test_forecast_loss_passes_scope_locally_with_unchanged_weights(scope):
    output, target, scale = example()
    mean = torch.tensor([-.5, .2, .7], dtype=torch.float64)
    target_phys, pred_phys = target * scale + mean, output.prediction * scale + mean
    config = LossConfig(excess_supervision_scope=scope, body_loss_weight=1., excess_loss_weight=2., gate_loss_weight=.5)
    criterion = ForecastLoss(config, dict(y_mean=mean, y_std=scale), 1., 1.)
    with patch("emulator.training.losses.dual_loss_terms", wraps=dual_loss_terms) as called:
        loss = criterion(output, pred_phys, target_phys)
    assert called.call_count == 1
    assert called.call_args.args[3] == scope
    # Match the existing physical-to-normalized target conversion exactly.
    body, excess, gate = dual_loss_terms(output, (target_phys - mean) / scale, scale, scope)
    expected = (pred_phys - target_phys).square().mean() + (body + 2. * excess + .5 * gate)
    torch.testing.assert_close(loss, expected, rtol=0, atol=0)
    loss.backward()
    assert output.gate_logits.grad.abs().sum() > 0  # Gate BCE remains active.


def test_all_scope_is_inactive_for_single_head():
    prediction = torch.tensor([[.2, .3]], requires_grad=True)
    target = torch.tensor([[.1, .7]])
    criterion = ForecastLoss(LossConfig(excess_supervision_scope="all"),
                             dict(y_mean=torch.zeros(2), y_std=torch.ones(2)), 1., 1.)
    with patch("emulator.training.losses.dual_loss_terms", side_effect=AssertionError("inactive")):
        loss = criterion(ForecastOutput(prediction), prediction, target)
    torch.testing.assert_close(loss, (prediction - target).square().mean(), rtol=0, atol=0)


def test_invalid_scope_fails_for_cli_loss_and_direct_call():
    with pytest.raises(SystemExit):
        parse_args(["--excess_supervision_scope", "balanced"])
    with pytest.raises(ValueError, match="excess_supervision_scope"):
        dual_loss_terms(*example(), excess_supervision_scope="balanced")
    with pytest.raises(ValueError, match="excess_supervision_scope"):
        ForecastLoss(LossConfig(excess_supervision_scope="balanced"),
                     dict(y_mean=torch.zeros(3), y_std=torch.ones(3)), 1., 1.)


@pytest.mark.parametrize("scope", [None, "event", "all"])
def test_loss_only_cli_shell_and_snapshot_fields(tmp_path, scope):
    fixture = tmp_path / "config.sh"
    fixture.write_text("MODEL=perceiver3\nBODY_LOSS_WEIGHT=1\nEXCESS_LOSS_WEIGHT=2\nGATE_LOSS_WEIGHT=0.5\n"
                       + (f"EXCESS_SUPERVISION_SCOPE={scope}\n" if scope else ""))
    args = parse_args(dry_commands(fixture)[0])
    assert vars(args)["excess_supervision_scope"] == (scope or "event")
    config = LossConfig(**{field.name: getattr(args, field.name) for field in fields(LossConfig)})
    assert config.excess_supervision_scope == (scope or "event")
    assert (config.body_loss_weight, config.excess_loss_weight, config.gate_loss_weight) == (1., 2., .5)
    assert "excess_supervision_scope" not in asdict(ModelConfig(3, 4, 8))
    single = parse_args(["--head_type", "single", "--excess_supervision_scope", scope or "event"])
    assert single.dual_loss == 0
