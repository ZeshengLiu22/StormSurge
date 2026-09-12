"""Physical-unit objectives. Metrics are computed separately from these losses."""

from dataclasses import dataclass
import math
import os

import torch
from torch import nn
import torch.nn.functional as F

from emulator.common.runtime import log_message
from emulator.common.dual import DUAL_ABLATIONS, dual_excess_target, validate_excess_formulation
from .excess_amplitude import excess_amplitude_terms, validate_event_prior
from .excess_shape import excess_shape_loss


def validate_shape_config(config):
    formulation = getattr(config, "excess_formulation", "direct")
    validate_excess_formulation(formulation, getattr(config, "severity_shape_eps", 1e-6),
                               getattr(config, "head_type", "dual"), getattr(config, "model", "pact"))
    weight = getattr(config, "shape_loss_weight", 0.0)
    if not math.isfinite(weight) or weight < 0:
        raise ValueError("--shape_loss_weight must be finite and nonnegative.")
    if weight > 0 and formulation != "severity_shape":
        raise ValueError("Positive --shape_loss_weight requires --excess_formulation severity_shape.")
    return 0.0 if "shape_loss_weight" in DUAL_ABLATIONS.get(config.dual_ablation, ()) else weight


def validate_excess_amp_config(config):
    """Validate the optional objective and return its effective ablation weight."""
    weight = getattr(config, "excess_amp_loss_weight", 0.0)
    pool = getattr(config, "excess_amp_pool", "max")
    if not math.isfinite(weight) or weight < 0:
        raise ValueError("--excess_amp_loss_weight must be finite and nonnegative.")
    if pool not in ("max", "smoothmax"):
        raise ValueError("--excess_amp_pool must be max or smoothmax.")
    if weight > 0 and getattr(config, "head_type", "dual") != "dual":
        raise ValueError("Excess-amplitude supervision requires the supervised dual exceedance head "
                         "(--model perceiver3 --head_type dual).")
    if "excess_amp_loss_weight" in DUAL_ABLATIONS.get(config.dual_ablation, ()):
        weight = 0.0
    if weight > 0 and pool == "smoothmax":
        beta = getattr(config, "excess_amp_beta", 20.0)
        if not math.isfinite(beta) or beta <= 0:
            raise ValueError("--excess_amp_beta must be finite and positive for smoothmax (inverse meters).")
    return weight


def enforce_dual_loss(config):
    """Enforce full supervision unless a named ablation disables specific terms."""
    if config.dual_ablation not in DUAL_ABLATIONS:
        raise ValueError(f"Unknown dual ablation: {config.dual_ablation}")
    disabled = DUAL_ABLATIONS[config.dual_ablation]
    for name in disabled:
        setattr(config, name, 0.0)
    if config.dual_ablation == "no_branch_supervision":
        config.dual_loss = 0
        return
    corrections = []
    if not config.dual_loss:
        config.dual_loss = 1
        corrections.append("dual_loss=1 (was 0)")
    for name in ("body_loss_weight", "excess_loss_weight", "gate_loss_weight"):
        if name not in disabled and getattr(config, name) == 0:
            setattr(config, name, 1.0)
            corrections.append(f"{name}=1 (was 0)")
    if corrections and int(os.environ.get("RANK", "0")) == 0:
        log_message(f"WARNING: head_type=dual requires dual loss for dual_ablation={config.dual_ablation}; "
                    "forcing " + ", ".join(corrections) + ".")


def dual_loss_terms(output, target_norm, y_std):
    """The three dual loss terms: body, conditional excess and event BCE.

    Excess risk uses the WHOLE batch denominator. Non-event windows have zero
    excess auxiliary gradient; rare-event batches never amplify it by 1/p.
    """
    event, excess_target = dual_excess_target(target_norm, output.threshold)
    body_target = torch.minimum(target_norm, output.threshold)
    body = ((output.body - body_target) * y_std).square().mean()
    excess = (((output.excess - excess_target) * y_std).square() * event).mean()
    gate = (F.binary_cross_entropy_with_logits(output.gate_logits, event.float()) * y_std.square().mean()
            if output.gate_logits is not None else body.new_zeros(()))
    return body, excess, gate


@dataclass
class LossConfig:
    loss_mode: str = "mse"
    wmse_alpha: float = 4.0
    wmse_s: float = 0.1
    wmse_use_abs: bool = True
    tail_lambda: float = 0.1
    tail_frac: float = 0.05
    slope_lambda: float = 0.01
    slope_mask_s: float = 0.1
    slope_robust: str = "charb"
    slope_charb_eps: float = 0.001
    slope_huber_delta: float = 0.05
    dual_loss: bool = True
    body_loss_weight: float = 1.0
    excess_loss_weight: float = 1.0
    gate_loss_weight: float = 1.0
    dual_ablation: str = "none"
    excess_amp_loss_weight: float = 0.0
    excess_amp_pool: str = "max"
    excess_amp_beta: float = 20.0
    excess_formulation: str = "direct"
    shape_loss_weight: float = 0.0
    severity_shape_eps: float = 1e-6


class ForecastLoss(nn.Module):
    def __init__(self, config, stats, peak_threshold, wmse_threshold, event_prior=None):
        super().__init__()
        self.config = config
        self.register_buffer("y_mean", stats["y_mean"])
        self.register_buffer("y_std", stats["y_std"])
        self.peak_threshold = peak_threshold
        self.wmse_threshold = wmse_threshold
        amp_weight = validate_excess_amp_config(config)
        shape_weight = validate_shape_config(config)
        if amp_weight > 0 or shape_weight > 0:
            validate_event_prior(event_prior)
        self.event_prior = event_prior

    def forward(self, output, prediction, target):
        c = self.config
        if output.body is None and getattr(c, "excess_amp_loss_weight", 0.0) > 0:
            raise ValueError("Excess-amplitude supervision requires the supervised dual exceedance head.")
        if output.body is None and getattr(c, "shape_loss_weight", 0.0) > 0:
            raise ValueError("Shape supervision requires the severity_shape dual head.")
        error = (prediction - target).square()
        core = c.loss_mode.removesuffix("_slope")
        if core in ("wmse", "wmse_tail", "mse_wtail"):
            magnitude = target.abs() if c.wmse_use_abs else target
            weight = 1 + c.wmse_alpha * torch.sigmoid((magnitude - self.wmse_threshold) / max(c.wmse_s, 1e-6))
            weighted_error = weight * error
        loss = weighted_error.mean() if core in ("wmse", "wmse_tail") else error.mean()
        if core in ("mse_tail", "wmse_tail", "mse_wtail") and c.tail_lambda:
            tail_error = weighted_error if core in ("wmse_tail", "mse_wtail") else error
            per_sample_tail_error = tail_error.mean(dim=1)
            event = (target.amax(dim=1) >= self.peak_threshold).to(tail_error.dtype)
            # Fixed TRAIN fraction keeps each event's weight independent of batch composition.
            tail_loss = (per_sample_tail_error * event).mean() / c.tail_frac
            loss = loss + c.tail_lambda * tail_loss
        if c.loss_mode.endswith("_slope") and c.slope_lambda and target.size(1) > 1:
            slope_error = prediction.diff(dim=1) - target.diff(dim=1)
            if c.slope_robust == "huber":
                penalty = F.huber_loss(slope_error, torch.zeros_like(slope_error), reduction="none", delta=max(c.slope_huber_delta, 1e-12))
            else:
                penalty = (slope_error.square() + max(c.slope_charb_eps, 1e-12) ** 2).sqrt()
            mask = torch.sigmoid((self.wmse_threshold - target.abs().amax(dim=1)) / max(c.slope_mask_s, 1e-6))
            loss = loss + c.slope_lambda * (penalty * mask[:, None]).mean()
        if output.body is not None:
            if (output.gate_logits is None) != (c.dual_ablation == "fixed_gate"):
                raise ValueError("The loss and model must select the same fixed_gate ablation.")
            # Also enforce this for callers that construct LossConfig directly.
            enforce_dual_loss(c)
            if not c.dual_loss:
                return loss
            target_norm = (target - self.y_mean) / self.y_std
            body, excess, gate = dual_loss_terms(output, target_norm, self.y_std)
            dual_loss = c.body_loss_weight * body + c.excess_loss_weight * excess + c.gate_loss_weight * gate
            loss = loss + dual_loss
            # Leave the historical numerical/RNG path untouched at weight zero.
            if getattr(c, "excess_amp_loss_weight", 0.0) > 0:
                amplitude = excess_amplitude_terms(output.excess, target_norm, output.threshold, self.y_std,
                    self.event_prior, getattr(c, "excess_amp_pool", "max"), getattr(c, "excess_amp_beta", 20.0))
                loss = loss + c.excess_amp_loss_weight * amplitude.loss
            if getattr(c, "shape_loss_weight", 0.0) > 0:
                if output.excess_shape is None:
                    raise ValueError("Shape supervision requires the severity_shape dual head.")
                shape_loss = excess_shape_loss(output.excess_shape, target_norm, output.threshold,
                    self.y_std, self.event_prior, getattr(c, "severity_shape_eps", 1e-6))
                loss = loss + c.shape_loss_weight * shape_loss
        return loss
