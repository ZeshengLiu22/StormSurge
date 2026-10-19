"""Physical Global MSE + strict TRAIN-Q95 Tail MSE + episode peak MSE."""

from dataclasses import dataclass
import math

import torch
from torch import nn


def validate_episode_peak_config(config):
    """Validate the weight of canonical ground-truth episode-peak supervision.

    Args:
        config: Object exposing episode_gt_aligned_peak_weight as a finite, nonnegative
            scalar.

    Returns:
        None.

    Raises:
        ValueError: The episode-peak loss weight is negative or nonfinite.
    """
    weight = config.episode_gt_aligned_peak_weight
    if not math.isfinite(weight) or weight < 0:
        raise ValueError("episode_gt_aligned_peak_weight must be finite and nonnegative.")


def episode_gt_aligned_peak_mse(squared_error, mask, p_episode_peak):
    """Normalize masked peak squared errors by fixed training peak prevalence.

    Args:
        squared_error: Tensor of physical squared errors, normally [batch, horizons].
        mask: Boolean tensor of the identical shape, selecting canonical ground-truth
            episode-peak occurrences.
        p_episode_peak: Fixed source-training peak count divided by eligible target-hour
            count, strictly greater than zero and at most one.

    Returns:
        torch.Tensor: Scalar mean of zero-masked errors over every supplied target,
        divided by p_episode_peak. The caller handles incomplete-window denominator
        corrections.

    Raises:
        ValueError: The prevalence is invalid or the mask is absent, nonboolean, or
            shape-incompatible.
    """
    if p_episode_peak is None or not math.isfinite(p_episode_peak) or not 0 < p_episode_peak <= 1:
        raise ValueError("Episode peak supervision requires positive fixed TRAIN p_episode_peak.")
    if mask is None or mask.dtype != torch.bool or mask.shape != squared_error.shape:
        raise ValueError(
            "Episode peak supervision requires a boolean canonical mask matching targets."
        )
    return torch.where(mask, squared_error, 0.0).mean() / p_episode_peak


@dataclass
class LossConfig:
    """Nonnegative weights for optional physical-unit loss components.

    Attributes:
        exceedance_loss_weight: Coefficient for strict training-threshold tail MSE
            normalized by fixed training extreme-hour prevalence.
        episode_gt_aligned_peak_weight: Coefficient for canonical ground-truth
            episode-peak MSE normalized by fixed training peak prevalence.

    Global physical MSE always has coefficient one.
    """

    exceedance_loss_weight: float = 0.0
    episode_gt_aligned_peak_weight: float = 0.0


class ForecastLoss(nn.Module):
    """Physical MSE with optional training-threshold tail and episode-peak terms.
    """
    def __init__(
        self,
        config: LossConfig,
        stats,
        tau_physical,
        extreme_hour_prior=None,
        p_episode_peak=None,
    ):
        """Initialize fixed training statistics and optional tail/peak loss weights.

        Args:
            config: LossConfig containing finite, nonnegative optional term weights.
            stats: Training normalization mapping with y_mean and y_std tensors. Both are
                registered as buffers for checkpoint/device consistency; forward receives
                already physical predictions.
            tau_physical: Finite threshold fitted on training targets, in the same units as
                predictions and targets.
            extreme_hour_prior: Fixed training fraction of eligible target hours strictly
                above tau_physical. Must lie in (0, 1] when the tail loss weight is
                positive.
            p_episode_peak: Fixed training fraction of eligible target hours marked as
                canonical episode peaks. Must lie in (0, 1] when the peak loss weight is
                positive.

        Returns:
            None. Registers normalization buffers and initializes last_components.

        Raises:
            ValueError: The threshold, weights, or required population prevalences are
                invalid.
        """
        super().__init__()
        self.config = config
        self.register_buffer("y_mean", stats["y_mean"])
        self.register_buffer("y_std", stats["y_std"].detach())
        if tau_physical is None or not math.isfinite(tau_physical):
            raise ValueError("ForecastLoss requires finite TRAIN tau_physical.")
        validate_episode_peak_config(config)
        if config.episode_gt_aligned_peak_weight > 0 and (
            p_episode_peak is None
            or not math.isfinite(p_episode_peak)
            or not 0 < p_episode_peak <= 1
        ):
            raise ValueError(
                "Episode peak supervision requires positive fixed TRAIN p_episode_peak."
            )
        if not math.isfinite(config.exceedance_loss_weight) or config.exceedance_loss_weight < 0:
            raise ValueError("exceedance_loss_weight must be finite and nonnegative.")
        if config.exceedance_loss_weight > 0 and (
            extreme_hour_prior is None
            or not math.isfinite(extreme_hour_prior)
            or not 0 < extreme_hour_prior <= 1
        ):
            raise ValueError(
                "Exceedance supervision requires a positive TRAIN extreme_hour_prior."
            )
        self.tau_physical = float(tau_physical)
        self.extreme_hour_prior = extreme_hour_prior
        self.p_episode_peak = p_episode_peak
        self.last_components = {}

    def forward(self, output, prediction, target, *, episode_gt_peak_mask=None):
        """Compute physical regression loss and record detached component diagnostics.

        Global MSE divides by the number of finite individual targets. Tail and
        peak terms exclude every target in an incomplete window and divide by the
        remaining target-hour count and their fixed training prevalences. Tail
        membership uses the strict condition target > tau_physical in float64. Empty
        eligible populations use a denominator floor of one. Peak diagnostics can be
        computed at zero optimization weight when a valid mask and prevalence exist.

        Args:
            output: ForecastOutput accepted for the common epoch interface; its normalized
                predictions are not used by this objective.
            prediction: Physical prediction tensor [batch, horizons], in the same units as
                target and finite wherever targets are finite.
            target: Physical target tensor with the same shape. Nonfinite entries are
                excluded from the global MSE.
            episode_gt_peak_mask: Optional boolean tensor with the target shape, selecting
                canonical peaks built over the full training split. Required when the peak
                weight is positive.

        Returns:
            torch.Tensor: Differentiable scalar global MSE plus weighted tail and peak
            terms, in squared physical units. Updates last_components with detached
            raw/weighted losses, eligible peak count, and total loss.

        Raises:
            ValueError: Requested peak supervision lacks a valid mask or fixed training
                prevalence.
        """
        point_valid = torch.isfinite(target)
        window_valid = point_valid.all(dim=1)
        config = self.config

        # Mask before squaring so missing targets cannot poison backward with NaNs.
        safe_target = torch.where(point_valid, target, 0.0)
        error = torch.where(point_valid, prediction - safe_target, 0.0).square()
        global_mse = error.sum() / point_valid.sum().clamp_min(1)
        eligible_target_count = (window_valid.sum() * target.size(1)).clamp_min(1)
        # FP64 preserves strict threshold membership near a fitted Q95 boundary.
        tail_mask = window_valid[:, None] & (target.double() > self.tau_physical)

        loss = global_mse
        tail = error.new_zeros(())
        if config.exceedance_loss_weight:
            tail = (
                torch.where(tail_mask, error, 0.0).sum()
                / eligible_target_count
                / self.extreme_hour_prior
            )
            loss = loss + config.exceedance_loss_weight * tail

        peak = error.sum() * 0.0
        if (
            episode_gt_peak_mask is not None
            and episode_gt_peak_mask.dtype == torch.bool
            and episode_gt_peak_mask.shape == error.shape
        ):
            episode_gt_peak_mask = episode_gt_peak_mask & window_valid[:, None]
        if config.episode_gt_aligned_peak_weight > 0 or (
            episode_gt_peak_mask is not None and self.p_episode_peak
        ):
            peak = episode_gt_aligned_peak_mse(error, episode_gt_peak_mask, self.p_episode_peak)
            # Convert the all-target mean to a complete-window target mean.
            peak = peak * (error.numel() / eligible_target_count.to(error.dtype))
        if config.episode_gt_aligned_peak_weight:
            loss = loss + config.episode_gt_aligned_peak_weight * peak

        # Report raw terms even when their optimization weights are zero.
        tail_mse = (
            torch.where(tail_mask, error, 0.0).sum()
            / eligible_target_count
            / self.extreme_hour_prior
            if self.extreme_hour_prior
            else error.new_zeros(())
        )
        self.last_components = {
            "global_mse_raw": global_mse.detach(),
            "tail_mse_raw": tail_mse.detach(),
            "tail_weighted": (config.exceedance_loss_weight * tail).detach(),
            "episode_gt_aligned_peak_mse_raw": peak.detach(),
            "episode_gt_aligned_peak_weighted": (
                config.episode_gt_aligned_peak_weight * peak
            ).detach(),
            "episode_peak_target_count": (
                episode_gt_peak_mask.sum().detach()
                if episode_gt_peak_mask is not None
                else error.new_zeros(())
            ),
            "total_loss": loss.detach(),
        }
        return loss
