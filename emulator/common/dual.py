"""Explicit dual experiments and finite initialization of learned event logits."""

import math


DUAL_ABLATIONS = {
    "none": (),
    "no_gate_bce": ("gate_loss_weight",),
    "no_excess_loss": ("excess_loss_weight", "excess_amp_loss_weight", "shape_loss_weight"),
    "no_branch_supervision": ("body_loss_weight", "excess_loss_weight", "excess_amp_loss_weight", "shape_loss_weight", "gate_loss_weight"),
    "fixed_gate": ("gate_loss_weight",),
}

EXCESS_FORMULATIONS = ("direct", "severity_shape")


def validate_excess_formulation(formulation, eps=1e-6, head_type="dual", model="pact"):
    if formulation not in EXCESS_FORMULATIONS:
        raise ValueError("excess_formulation must be direct or severity_shape.")
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError("severity_shape_eps must be finite and positive.")
    if formulation == "severity_shape" and (head_type != "dual" or model not in ("pact", "perceiver3")):
        raise ValueError("severity_shape requires the supervised dual exceedance head "
                         "(--model perceiver3 --head_type dual).")


def dual_excess_target(target_norm, threshold):
    """The production normalized excess target and strict TRAIN-threshold event."""
    excess_target = (target_norm - threshold).clamp_min(0)
    event = (target_norm > threshold).any(dim=1, keepdim=True)
    return event, excess_target


def initial_gate_prior(event_prior):
    if not math.isfinite(event_prior) or not 0 <= event_prior <= 1:
        raise ValueError("TRAIN event prior must be finite and in [0, 1].")
    # Preserve the empirical prevalence separately; clipping only makes logit finite.
    return min(max(event_prior, 1e-6), 1 - 1e-6)
