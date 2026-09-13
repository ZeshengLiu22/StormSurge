"""Dimensionless temporal shape supervision using the post-#2 physical target."""

from typing import NamedTuple

import torch

from emulator.common.dual import validate_excess_formulation
from .excess_amplitude import _full_precision, peak_pool, physical_excess_target, validate_event_prior


class ExcessShapeTarget(NamedTuple):
    event: torch.Tensor
    r_target_phys: torch.Tensor
    a_target: torch.Tensor
    shape_target: torch.Tensor


def excess_shape_target(target_norm, threshold, y_std, eps=1e-6):
    """Targets depend only on the existing TRAIN-threshold excess, never the body.

    a_target is always the hard maximum in meters, even when optional amplitude
    supervision uses smoothmax. Every event is normalized by its positive
    physical amplitude, giving a unit peak even below eps. Non-events stay zero.
    The shared eps setting is validated but does not floor target amplitudes.
    """
    validate_excess_formulation("severity_shape", eps)
    event, physical = physical_excess_target(target_norm, threshold, y_std)
    amplitude = peak_pool(physical, "max")
    # Strict events have positive physical amplitude. A unit denominator for
    # non-events avoids 0/0 while retaining their exactly zero physical target.
    denominator = torch.where(event.squeeze(1), amplitude, torch.ones_like(amplitude))
    shape = physical / denominator[:, None]
    return ExcessShapeTarget(event, physical, amplitude, shape)


def excess_shape_loss(shape_pred, target_norm, threshold, y_std, event_prior, eps=1e-6):
    """mean_i[E_i * mean_h((shape_pred - shape_target)**2)] / fixed TRAIN q_E."""
    validate_event_prior(event_prior)
    target = excess_shape_target(target_norm, threshold, y_std, eps)
    error = torch.where(target.event, _full_precision(shape_pred) - target.shape_target, 0.0)
    return error.square().mean(dim=1).mean() / event_prior
