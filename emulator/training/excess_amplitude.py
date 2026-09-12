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
    if event_prior is None or not math.isfinite(event_prior) or event_prior <= 0:
        raise ValueError("Excess-amplitude supervision requires a finite, positive TRAIN event_prior "
                         "from fit_loss_thresholds (event_count / train_windows).")


def _full_precision(value):
    # Preserve float64 callers, but avoid low-precision physical products/squares.
    return value.float() if value.dtype in (torch.float16, torch.bfloat16) else value


def peak_pool(values, pool="max", beta=20.0):
    """Pool physical meters along the horizon axis; beta has inverse-meter units.

    Smoothmax is a bounded softmax-weighted mean. Center before scaling to
    avoid overflow in beta * values; max pooling never reads beta.
    """
    values = _full_precision(values)
    maximum = values.max(dim=-1, keepdim=True).values
    if pool == "max":
        return maximum.squeeze(-1)
    if pool != "smoothmax":
        raise ValueError("excess_amp_pool must be max or smoothmax.")
    if not math.isfinite(beta) or beta <= 0:
        raise ValueError("excess_amp_beta must be finite and positive for smoothmax (inverse meters).")
    centered = values - maximum
    # A finite Python beta can exceed FP32's range. Use FP64 logits only in
    # that case so the maximum's zero logit cannot become 0 * inf = NaN.
    logits = centered.double() * beta if beta > torch.finfo(values.dtype).max else centered * beta
    weights = torch.softmax(logits, dim=-1).to(values.dtype)
    return (weights * values).sum(dim=-1)


def excess_amplitude_terms(excess_pred_norm, target_norm, threshold, y_std,
                           event_prior, pool="max", beta=20.0):
    """Return reusable targets, amplitudes and a scalar loss in square meters.

    Convert EACH horizon to physical units BEFORE pooling. The event and
    normalized target are shared with the existing trajectory-wise dual loss.
    event_prior is the exact fixed empirical TRAIN prevalence, never a batch
    fraction or tail_frac. No host-side event branch or collective is needed,
    so a rank with no events retains a differentiable, exactly zero loss.
    """
    validate_event_prior(event_prior)
    event, excess_target_norm = dual_excess_target(target_norm, threshold)
    r_target_phys = _full_precision(excess_target_norm) * _full_precision(y_std)
    r_pred_phys = _full_precision(excess_pred_norm) * _full_precision(y_std)
    a_pred = peak_pool(r_pred_phys, pool, beta)
    a_target = peak_pool(r_target_phys, pool, beta)
    # Mask before squaring; this equals mean(E * amplitude_error**2) / q_E.
    error = torch.where(event.squeeze(1), a_pred - a_target, 0.0)
    loss = error.square().mean() / event_prior
    return ExcessAmplitudeTerms(event, r_pred_phys, r_target_phys, a_pred, a_target, loss)
