"""Pure physical excess-amplitude supervision, independent of the gate and body."""

import math
from typing import NamedTuple

import torch

from emulator.common.dual import dual_excess_target


class ExcessAmplitudeTerms(NamedTuple):
    event: torch.Tensor
    r_pred_phys: torch.Tensor
    r_target_phys: torch.Tensor
    a_pred: torch.Tensor
    a_target: torch.Tensor
    loss: torch.Tensor


def validate_event_prior(event_prior):
    if event_prior is None or not math.isfinite(event_prior) or not 0 < event_prior <= 1:
        raise ValueError("Event-only supervision requires a finite, positive TRAIN event_prior "
                         "from fit_loss_thresholds (train_event_window_count / train_windows).")


def _full_precision(value):
    # Preserve float64 callers, but avoid low-precision physical products/squares.
    return value.float() if value.dtype in (torch.float16, torch.bfloat16) else value


def physical_excess_target(target_norm, threshold, y_std, *, target_phys=None, tau_physical=None):
    """Shared strict event and physical target for amplitude and shape supervision."""
    if target_phys is not None:
        if tau_physical is None or not math.isfinite(tau_physical):
            raise ValueError("Physical excess requires finite TRAIN tau_physical.")
        physical = _full_precision(target_phys)
        difference = physical.double() - tau_physical
        event = (difference > 0).any(dim=1, keepdim=True)
        return event, difference.clamp_min(0).to(physical.dtype)
    event, excess_target_norm = dual_excess_target(target_norm, threshold)
    return event, _full_precision(excess_target_norm) * _full_precision(y_std)


def excess_amplitude_terms(excess_pred_norm, target_norm, threshold, y_std,
                           event_prior, *, target_phys, tau_physical):
    """Supervise excess at the original physical target's first peak horizon.

    For a scalar physical threshold, event target excess has the same peak
    horizon and amplitude as the original target (up to ordinary ties). Gather
    BOTH branches at that target horizon, so only the selected predicted excess
    receives gradient. Body, gate and the predicted argmax never enter the loss.

    Normalize mean(E * amplitude_error**2) by the exact fixed TRAIN event prior,
    never batch prevalence. Event-free ranks retain differentiable
    exact zero. Physical products and squares remain FP32 or better.
    """
    validate_event_prior(event_prior)
    if tau_physical is None or not math.isfinite(tau_physical):
        raise ValueError("Excess-amplitude supervision requires a finite TRAIN tau_physical.")
    target_phys = _full_precision(target_phys)
    gt_peak_index = target_phys.argmax(dim=1, keepdim=True)
    event, r_target_phys = physical_excess_target(
        _full_precision(target_norm), _full_precision(threshold), y_std,
        target_phys=target_phys, tau_physical=tau_physical)
    r_pred_phys = _full_precision(excess_pred_norm) * _full_precision(y_std)
    a_pred = r_pred_phys.gather(1, gt_peak_index).squeeze(1)
    a_target = r_target_phys.gather(1, gt_peak_index).squeeze(1)
    error = torch.where(event.squeeze(1), a_pred - a_target, 0.0)
    loss = error.square().mean() / event_prior
    return ExcessAmplitudeTerms(event, r_pred_phys, r_target_phys, a_pred, a_target, loss)
