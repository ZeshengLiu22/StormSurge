"""Reconstruction-only contracts; no fitting or training jobs."""

from dataclasses import asdict, fields

import pytest
import torch

from emulator.common.dual import initial_gate_prior
from emulator.models import ModelConfig, build_model
from emulator.models.heads import AdditiveExceedanceHead, ExceedanceHead
from emulator.training.arguments import parse_args
from test_config_interfaces import dry_commands
from test_models import graph_batch


@pytest.fixture(autouse=True)
def single_thread():
    torch.set_num_threads(1)


def test_default_and_historical_config_select_exact_production_head():
    config = ModelConfig(3, 4, 8, peak_threshold_norm=[1.] * 4)
    old = asdict(config)
    old.pop("direct_dual_reconstruction")
    for values in (old, asdict(config)):
        restored = ModelConfig(**values)
        assert restored.direct_dual_reconstruction == "soft_gate"
        assert type(build_model(restored).head) is ExceedanceHead


@pytest.mark.parametrize("width", [None, 13])
@pytest.mark.parametrize("training", [False, True])
def test_identical_initialization_rng_and_copied_branch_weights(width, training):
    context = torch.randn(3, 4, 8)
    options = dict(hidden=8, dropout=.3, threshold=[-1., .2, 1., 2.], prior=.17, head_hidden=width)
    torch.manual_seed(42)
    production = ExceedanceHead(**options).train(training)
    production_rng = torch.get_rng_state()
    torch.manual_seed(42)
    additive = AdditiveExceedanceHead(**options).train(training)
    assert torch.equal(torch.get_rng_state(), production_rng)
    assert production.state_dict().keys() == additive.state_dict().keys()
    for name, value in production.state_dict().items():
        torch.testing.assert_close(value, additive.state_dict()[name], rtol=0, atol=0)
    assert additive.body[0].out_features == (width or 16)
    assert additive.excess[0].out_features == (width or 16)
    assert additive.gate[0].out_features == (width or 16)
    # Exercise learned, nonconstant excess and probability, including dropout.
    with torch.no_grad():
        production.excess[-1].weight.normal_(std=.2)
        production.gate[-1].weight.normal_(std=.2)
    additive.load_state_dict(production.state_dict(), strict=True)
    torch.manual_seed(19)
    soft = production(context)
    production_rng = torch.get_rng_state()
    torch.manual_seed(19)
    add = additive(context)
    assert torch.equal(torch.get_rng_state(), production_rng)
    for name in ("body", "excess", "gate_logits", "gate_probability", "threshold"):
        torch.testing.assert_close(getattr(soft, name), getattr(add, name), rtol=0, atol=0)
    torch.testing.assert_close(soft.prediction, soft.body + soft.gate_probability * soft.excess, rtol=0, atol=0)
    torch.testing.assert_close(add.prediction, add.body + add.excess, rtol=0, atol=0)
    assert not torch.equal(soft.prediction, add.prediction)
    assert (add.body <= add.threshold).all() and (add.excess >= 0).all()


@pytest.mark.parametrize("prior", [0., .17, 1.])
def test_prior_clipping_and_initial_excess_are_production_values(prior):
    head = AdditiveExceedanceHead(8, 0., [1.] * 4, prior)
    output = head(torch.randn(3, 4, 8))
    torch.testing.assert_close(output.excess, torch.full((3, 4), .1))
    torch.testing.assert_close(output.gate_probability, torch.full((3, 1), initial_gate_prior(prior)))
    assert torch.count_nonzero(head.excess[-1].weight) == 0
    assert torch.count_nonzero(head.gate[-1].weight) == 0


def test_gate_is_auxiliary_and_keeps_mean_pooling():
    head = AdditiveExceedanceHead(8, 0., [1.] * 4, .17)
    with torch.no_grad():
        head.gate[-1].weight.normal_()
    context = torch.randn(2, 4, 8)
    output = head(context)
    torch.testing.assert_close(output.gate_logits, head.gate(context.mean(1)), rtol=0, atol=0)
    gradients = torch.autograd.grad(output.prediction.sum(), tuple(head.gate.parameters()), allow_unused=True)
    assert all(value is None for value in gradients)
    torch.nn.functional.binary_cross_entropy_with_logits(output.gate_logits, torch.ones(2, 1)).backward()
    assert head.gate[-1].bias.grad.abs().sum() > 0


def test_model_forward_uses_additive_head_with_identical_backbone():
    common = dict(in_channels=3, out_channels=4, hidden_channels=8, peak_threshold_norm=[1.] * 4)
    production = build_model(ModelConfig(**common)).eval()
    additive = build_model(ModelConfig(**common, direct_dual_reconstruction="additive")).eval()
    assert type(additive.head) is AdditiveExceedanceHead
    additive.load_state_dict(production.state_dict(), strict=True)
    batch = graph_batch()
    soft, add = production(batch), additive(batch)
    for expected, actual in zip(soft[1:], add[1:]):
        if expected is None:
            assert actual is None
        else:
            torch.testing.assert_close(expected, actual, rtol=0, atol=0)
    torch.testing.assert_close(add.prediction, add.body + add.excess, rtol=0, atol=0)


@pytest.mark.parametrize("extra", [
    {"head_type": "single"}, {"model": "baseline", "head_type": "single"},
    {"excess_formulation": "severity_shape", "target_y_std": [1.] * 4},
    *[{"exceedance_head_experiment": variant} for variant in ("legacy", "c1", "c2", "c2r", "c3")],
    {"direct_dual_reconstruction": "hard_gate"},
])
def test_model_rejects_invalid_combinations(extra):
    values = dict(in_channels=3, out_channels=4, hidden_channels=8, peak_threshold_norm=[1.] * 4,
                  direct_dual_reconstruction="additive")
    values.update(extra)
    with pytest.raises(ValueError):
        build_model(ModelConfig(**values))


@pytest.mark.parametrize("extra", [
    ["--head_type", "single"], ["--model", "baseline"],
    ["--excess_formulation", "severity_shape"],
    *[["--exceedance_head_experiment", variant] for variant in ("legacy", "c1", "c2", "c2r", "c3")],
    ["--direct_dual_reconstruction", "hard_gate"],
])
def test_cli_rejects_invalid_combinations(extra):
    with pytest.raises(SystemExit):
        parse_args(["--model", "perceiver3", "--direct_dual_reconstruction", "additive", *extra])


@pytest.mark.parametrize("mode", [None, "soft_gate", "additive"])
def test_shell_cli_and_natural_model_config_plumbing(tmp_path, mode):
    fixture = tmp_path / "config.sh"
    fixture.write_text("MODEL=perceiver3\nBODY_LOSS_WEIGHT=1\nEXCESS_LOSS_WEIGHT=2\nGATE_LOSS_WEIGHT=0.5\n"
                       + (f"DIRECT_DUAL_RECONSTRUCTION={mode}\n" if mode else ""))
    commands = dry_commands(fixture)
    assert len(commands) == 1
    args = parse_args(commands[0])
    assert args.direct_dual_reconstruction == (mode or "soft_gate")
    assert (args.body_loss_weight, args.excess_loss_weight, args.gate_loss_weight) == (1., 2., .5)
    assert args.checkpoint_selection == "overall"
    assert args.loss_mode == "mse"
    assert (args.excess_amp_loss_weight, args.shape_loss_weight, args.peak_loss_weight) == (0., 0., 0.)
    # This is the existing train.py field collection, with required dimensions supplied.
    values = {field.name: getattr(args, field.name) for field in fields(ModelConfig) if hasattr(args, field.name)}
    values.update(in_channels=3, out_channels=4, model="pact", peak_threshold_norm=[1.] * 4)
    assert asdict(ModelConfig(**values))["direct_dual_reconstruction"] == (mode or "soft_gate")
