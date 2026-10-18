"""Physical Global MSE + strict TRAIN-Q95 Tail MSE + episode peak MSE."""

from dataclasses import dataclass
import math

import torch
from torch import nn


def validate_episode_peak_config(config):
    weight = config.episode_gt_aligned_peak_weight
    if not math.isfinite(weight) or weight < 0:
        raise ValueError("episode_gt_aligned_peak_weight must be finite and nonnegative.")


def episode_gt_aligned_peak_mse(squared_error, mask, p_episode_peak):
    """Average masked errors over all targets, normalized by fixed TRAIN J/N."""
    if p_episode_peak is None or not math.isfinite(p_episode_peak) or not 0 < p_episode_peak <= 1:
        raise ValueError("Episode peak supervision requires positive fixed TRAIN p_episode_peak.")
    if mask is None or mask.dtype != torch.bool or mask.shape != squared_error.shape:
        raise ValueError(
            "Episode peak supervision requires a boolean canonical mask matching targets."
        )
    return torch.where(mask, squared_error, 0.0).mean() / p_episode_peak


@dataclass
class LossConfig:
    """Weights for the two optional terms; global physical MSE always has weight 1.

    ``exceedance_loss_weight`` retains its established name for the strict
    TRAIN-Q95 Tail MSE coefficient.
    """

    exceedance_loss_weight: float = 0.0
    episode_gt_aligned_peak_weight: float = 0.0


class ForecastLoss(nn.Module):
    def __init__(
        self,
        config: LossConfig,
        stats,
        tau_physical,
        extreme_hour_prior=None,
        p_episode_peak=None,
    ):
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
        """Consume physical predictions/targets and canonical GT peak locations.

        Global MSE masks individual missing targets. Tail and peak terms exclude
        every target in an incomplete forecast window, including its denominator.
        ``output`` keeps the common epoch-call interface; only physical values
        enter this objective.
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
            # Preserve the original arithmetic while excluding invalid windows.
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
