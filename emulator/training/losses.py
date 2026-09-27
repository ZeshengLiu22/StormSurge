"""Physical-unit objectives. Metrics are computed separately from these losses."""

from dataclasses import dataclass
import math
import os

import torch
from torch import nn
import torch.nn.functional as F

from emulator.common.runtime import log_message
from emulator.common.dual import DUAL_ABLATIONS, dual_excess_target, validate_excess_formulation
from .excess_amplitude import excess_amplitude_terms, physical_excess_target, validate_event_prior
from .excess_shape import excess_shape_loss


def _validate_wqe_parameters(quantile_tau, expectile_tau, quantile_weight, expectile_weight):
    for name, tau in (("quantile_tau", quantile_tau), ("expectile_tau", expectile_tau)):
        if not math.isfinite(tau) or not 0 < tau < 1:
            raise ValueError(f"WQE {name} must be finite and strictly between 0 and 1.")
    for name, weight in (("quantile_weight", quantile_weight), ("expectile_weight", expectile_weight)):
        if not math.isfinite(weight) or weight < 0:
            raise ValueError(f"WQE {name} must be finite and nonnegative.")
    if not math.isclose(quantile_weight + expectile_weight, 1.0, rel_tol=1e-6, abs_tol=1e-7):
        raise ValueError("WQE weights must sum to 1 within floating-point tolerance.")


def validate_wqe_config(config):
    if config.loss_mode not in ("mse", "wqe", "wmse", "mse_slope", "wqe_slope", "wmse_slope"):
        raise ValueError("loss_mode must be mse, wqe, wmse, mse_slope, wqe_slope or wmse_slope.")
    if config.excess_loss_mode not in ("mse", "wqe"):
        raise ValueError("excess_loss_mode must be mse or wqe.")
    if config.exceedance_loss_mode not in ("mse", "wqe"):
        raise ValueError("exceedance_loss_mode must be mse or wqe.")
    _validate_wqe_parameters(config.wqe_quantile_tau, config.wqe_expectile_tau,
                             config.wqe_quantile_weight, config.wqe_expectile_weight)


def wqe_penalty(prediction, target, scale, *, quantile_tau=0.25, expectile_tau=0.82,
                quantile_weight=1.0 / 6.0, expectile_weight=5.0 / 6.0):
    """Pointwise weighted quantile/expectile penalty in physical squared units.

    Positive residuals mean underprediction. A scalar or per-horizon TRAIN scale
    broadcasts over [B, K] and is always treated as a fixed statistic.
    """
    _validate_wqe_parameters(quantile_tau, expectile_tau, quantile_weight, expectile_weight)
    scale_dtype = scale.dtype if torch.is_tensor(scale) else torch.promote_types(prediction.dtype, target.dtype)
    scale = torch.as_tensor(scale, dtype=scale_dtype, device=prediction.device).detach()
    if not torch.isfinite(scale).all() or not (scale > 0).all():
        raise ValueError("WQE scale must be finite and positive.")
    # Standardize for the published WQE geometry, then rescale to retain the
    # physical-loss scale used by MSE and the existing branch coefficients.
    epsilon = (target - prediction) / scale
    quantile = torch.where(epsilon >= 0, quantile_tau * epsilon, (quantile_tau - 1.0) * epsilon)
    squared = epsilon.square()
    expectile = torch.where(epsilon >= 0, expectile_tau * squared, (1.0 - expectile_tau) * squared)
    return scale.square() * (quantile_weight * quantile + expectile_weight * expectile)


def validate_shape_config(config, *, head_type="dual", model="pact"):
    formulation = config.excess_formulation
    validate_excess_formulation(formulation, config.severity_shape_eps, head_type, model)
    weight = config.shape_loss_weight
    if not math.isfinite(weight) or weight < 0:
        raise ValueError("--shape_loss_weight must be finite and nonnegative.")
    if weight > 0 and formulation != "severity_shape":
        raise ValueError("Positive --shape_loss_weight requires --excess_formulation severity_shape.")
    return 0.0 if "shape_loss_weight" in DUAL_ABLATIONS.get(config.dual_ablation, ()) else weight


def validate_excess_amp_config(config, *, head_type="dual"):
    """Validate the optional objective and return its effective ablation weight."""
    weight = config.excess_amp_loss_weight
    if not math.isfinite(weight) or weight < 0:
        raise ValueError("--excess_amp_loss_weight must be finite and nonnegative.")
    if weight > 0 and head_type != "dual":
        raise ValueError("Excess-amplitude supervision requires the supervised dual exceedance head "
                         "(--model perceiver3 --head_type dual).")
    if "excess_amp_loss_weight" in DUAL_ABLATIONS.get(config.dual_ablation, ()):
        weight = 0.0
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


def dual_loss_terms(output, target_norm, y_std, *, target_phys=None, tau_physical=None,
                    excess_loss_mode="mse", wqe_quantile_tau=0.25, wqe_expectile_tau=0.82,
                    wqe_quantile_weight=1.0 / 6.0, wqe_expectile_weight=5.0 / 6.0):
    """The three dual loss terms: body, conditional excess and event BCE.

    Excess risk uses the WHOLE batch denominator. Non-event windows have zero
    excess auxiliary gradient; rare-event batches never amplify it by 1/p.
    """
    if excess_loss_mode == "wqe":
        y_std = y_std.detach()
    if target_phys is None:
        event, excess_target = dual_excess_target(target_norm, output.threshold)
    else:
        event, physical_excess = physical_excess_target(target_norm, output.threshold, y_std,
            target_phys=target_phys, tau_physical=tau_physical)
        excess_target = physical_excess / y_std
    body_target = target_norm - excess_target
    body = ((output.body - body_target) * y_std).square().mean()
    if excess_loss_mode == "mse":
        excess = (((output.excess - excess_target) * y_std).square() * event).mean()
    elif excess_loss_mode == "wqe":
        pointwise = wqe_penalty(output.excess * y_std, excess_target * y_std, y_std,
            quantile_tau=wqe_quantile_tau, expectile_tau=wqe_expectile_tau,
            quantile_weight=wqe_quantile_weight, expectile_weight=wqe_expectile_weight)
        excess = (pointwise * event).mean()
    else:
        raise ValueError("excess_loss_mode must be mse or wqe.")
    gate = (F.binary_cross_entropy_with_logits(output.gate_logits, event.float()) * y_std.square().mean()
            if output.gate_logits is not None else body.new_zeros(()))
    return body, excess, gate


def validate_episode_peak_config(config, *, head_type="single"):
    weight = config.episode_gt_aligned_peak_weight
    if not math.isfinite(weight) or weight < 0:
        raise ValueError("episode_gt_aligned_peak_weight must be finite and nonnegative.")
    if weight > 0 and (head_type != "single" or config.loss_mode != "mse"
                       or config.exceedance_loss_mode != "mse"):
        raise ValueError("Episode GT-aligned peak supervision requires Single with global MSE and Tail MSE.")


def episode_gt_aligned_peak_mse(squared_error, mask, p_episode_peak):
    """Mean over all batch targets (including zeros), divided by fixed TRAIN J/N."""
    if p_episode_peak is None or not math.isfinite(p_episode_peak) or not 0 < p_episode_peak <= 1:
        raise ValueError("Episode peak supervision requires positive fixed TRAIN p_episode_peak.")
    if mask is None or mask.dtype != torch.bool or mask.shape != squared_error.shape:
        raise ValueError("Episode peak supervision requires a boolean canonical mask matching targets.")
    return torch.where(mask, squared_error, 0.0).mean() / p_episode_peak


@dataclass
class LossConfig:
    """Complete objective settings for every run; no config-version dispatch.

    Resolve omitted options here or in the CLI before constructing ForecastLoss.
    Loss computation reads these fields directly and never infers a zero weight
    from a missing attribute. All loss modes share the same optional controls.
    """
    loss_mode: str = "mse"
    wmse_alpha: float = 4.0
    wmse_s: float = 0.1
    exceedance_loss_weight: float = 0.0
    exceedance_loss_mode: str = "mse"
    episode_gt_aligned_peak_weight: float = 0.0
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
    excess_formulation: str = "direct"
    shape_loss_weight: float = 0.0
    severity_shape_eps: float = 1e-6
    excess_loss_mode: str = "mse"
    wqe_quantile_tau: float = 0.25
    wqe_expectile_tau: float = 0.82
    wqe_quantile_weight: float = 1.0 / 6.0
    wqe_expectile_weight: float = 5.0 / 6.0


class ForecastLoss(nn.Module):
    def __init__(self, config: LossConfig, stats, tau_physical, event_prior=None, extreme_hour_prior=None, p_episode_peak=None):
        super().__init__()
        self.config = config
        self.register_buffer("y_mean", stats["y_mean"])
        self.register_buffer("y_std", stats["y_std"].detach())
        if tau_physical is None or not math.isfinite(tau_physical):
            raise ValueError("ForecastLoss requires finite TRAIN tau_physical.")
        validate_wqe_config(config)
        validate_episode_peak_config(config)
        if config.episode_gt_aligned_peak_weight > 0 and (p_episode_peak is None
                or not math.isfinite(p_episode_peak) or not 0 < p_episode_peak <= 1):
            raise ValueError("Episode peak supervision requires positive fixed TRAIN p_episode_peak.")
        self.p_episode_peak = p_episode_peak
        self.last_components = {}
        if (config.loss_mode.removesuffix("_slope") == "wqe" or config.excess_loss_mode == "wqe"
                or config.exceedance_loss_mode == "wqe"):
            if not torch.isfinite(self.y_std).all() or not (self.y_std > 0).all():
                raise ValueError("WQE scale must be finite and positive.")
        if not math.isfinite(config.exceedance_loss_weight) or config.exceedance_loss_weight < 0:
            raise ValueError("exceedance_loss_weight must be finite and nonnegative.")
        if config.exceedance_loss_weight > 0 and (extreme_hour_prior is None
                or not math.isfinite(extreme_hour_prior) or not 0 < extreme_hour_prior <= 1):
            raise ValueError("Exceedance supervision requires a positive TRAIN extreme_hour_prior.")
        self.tau_physical = float(tau_physical)
        self.extreme_hour_prior = extreme_hour_prior
        amp_weight = validate_excess_amp_config(config)
        shape_weight = validate_shape_config(config)
        if amp_weight > 0 or shape_weight > 0:
            validate_event_prior(event_prior)
        self.event_prior = event_prior

    def forward(self, output, prediction, target, *, episode_gt_peak_mask=None):
        c = self.config
        if c.episode_gt_aligned_peak_weight > 0 and output.body is not None:
            raise ValueError("Episode GT-aligned peak supervision requires Single.")
        if output.body is None and c.excess_amp_loss_weight > 0:
            raise ValueError("Excess-amplitude supervision requires the supervised dual exceedance head.")
        if output.body is None and c.shape_loss_weight > 0:
            raise ValueError("Shape supervision requires the severity_shape dual head.")
        error = (prediction - target).square()
        core = c.loss_mode.removesuffix("_slope")
        if core == "wmse":
            weight = 1 + c.wmse_alpha * torch.sigmoid((target - self.tau_physical) / max(c.wmse_s, 1e-6))
            weighted_error = weight * error
        if core == "wqe":
            loss = wqe_penalty(prediction, target, self.y_std,
                quantile_tau=c.wqe_quantile_tau, expectile_tau=c.wqe_expectile_tau,
                quantile_weight=c.wqe_quantile_weight, expectile_weight=c.wqe_expectile_weight).mean()
        else:
            loss = weighted_error.mean() if core == "wmse" else error.mean()
        exceedance = error.new_zeros(())
        if c.exceedance_loss_weight:
            # Compare in FP64: a fitted threshold must not round to a target tie.
            mask = target.double() > self.tau_physical
            tail_penalty = error
            if c.exceedance_loss_mode == "wqe":
                tail_penalty = wqe_penalty(prediction, target, self.y_std,
                    quantile_tau=c.wqe_quantile_tau, expectile_tau=c.wqe_expectile_tau,
                    quantile_weight=c.wqe_quantile_weight, expectile_weight=c.wqe_expectile_weight)
            exceedance = torch.where(mask, tail_penalty, 0.0).mean() / self.extreme_hour_prior
            loss = loss + c.exceedance_loss_weight * exceedance
        peak = error.sum() * 0.0
        if c.episode_gt_aligned_peak_weight > 0 or (episode_gt_peak_mask is not None and self.p_episode_peak):
            peak = episode_gt_aligned_peak_mse(error, episode_gt_peak_mask, self.p_episode_peak)
        if c.episode_gt_aligned_peak_weight:
            loss = loss + c.episode_gt_aligned_peak_weight * peak
        # Detached diagnostics never change the legacy objective or its gradients.
        tail_mse = (torch.where(target.double() > self.tau_physical, error, 0.0).mean()
                    / self.extreme_hour_prior if self.extreme_hour_prior else error.new_zeros(()))
        self.last_components = {
            "global_mse_raw": error.mean().detach(),
            "tail_mse_raw": tail_mse.detach(),
            "tail_weighted": (c.exceedance_loss_weight * exceedance).detach(),
            "episode_gt_aligned_peak_mse_raw": peak.detach(),
            "episode_gt_aligned_peak_weighted": (c.episode_gt_aligned_peak_weight * peak).detach(),
            "episode_peak_target_count": (episode_gt_peak_mask.sum().detach()
                if episode_gt_peak_mask is not None else error.new_zeros(())),
            "total_loss": loss.detach(),
        }
        if c.loss_mode.endswith("_slope") and c.slope_lambda and target.size(1) > 1:
            slope_error = prediction.diff(dim=1) - target.diff(dim=1)
            if c.slope_robust == "huber":
                penalty = F.huber_loss(slope_error, torch.zeros_like(slope_error), reduction="none", delta=max(c.slope_huber_delta, 1e-12))
            else:
                penalty = (slope_error.square() + max(c.slope_charb_eps, 1e-12) ** 2).sqrt()
            mask = torch.sigmoid((self.tau_physical - target.amax(dim=1)) / max(c.slope_mask_s, 1e-6))
            loss = loss + c.slope_lambda * (penalty * mask[:, None]).mean()
        if output.body is not None:
            if (output.gate_logits is None) != (c.dual_ablation == "fixed_gate"):
                raise ValueError("The loss and model must select the same fixed_gate ablation.")
            # Also enforce this for callers that construct LossConfig directly.
            enforce_dual_loss(c)
            if not c.dual_loss:
                self.last_components["total_loss"] = loss.detach()
                return loss
            target_norm = (target - self.y_mean) / self.y_std
            body, excess, gate = dual_loss_terms(output, target_norm, self.y_std,
                target_phys=target, tau_physical=self.tau_physical,
                excess_loss_mode=c.excess_loss_mode,
                wqe_quantile_tau=c.wqe_quantile_tau, wqe_expectile_tau=c.wqe_expectile_tau,
                wqe_quantile_weight=c.wqe_quantile_weight, wqe_expectile_weight=c.wqe_expectile_weight)
            dual_loss = c.body_loss_weight * body + c.excess_loss_weight * excess + c.gate_loss_weight * gate
            loss = loss + dual_loss
            if c.excess_amp_loss_weight > 0:
                amplitude = excess_amplitude_terms(output.excess, target_norm, output.threshold, self.y_std,
                    self.event_prior, target_phys=target, tau_physical=self.tau_physical)
                loss = loss + c.excess_amp_loss_weight * amplitude.loss
            if c.shape_loss_weight > 0:
                if output.excess_shape is None:
                    raise ValueError("Shape supervision requires the severity_shape dual head.")
                shape_loss = excess_shape_loss(output.excess_shape, target_norm, output.threshold,
                    self.y_std, self.event_prior, c.severity_shape_eps,
                    target_phys=target, tau_physical=self.tau_physical)
                loss = loss + c.shape_loss_weight * shape_loss
        self.last_components["total_loss"] = loss.detach()
        return loss
