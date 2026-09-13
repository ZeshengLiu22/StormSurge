"""Forecast heads and the one output contract used by training and inference."""

import math
from typing import NamedTuple

import torch
from torch import nn
import torch.nn.functional as F

from emulator.common.dual import initial_gate_prior, validate_excess_formulation


class ForecastOutput(NamedTuple):
    prediction: torch.Tensor
    body: torch.Tensor | None = None
    excess: torch.Tensor | None = None
    gate_logits: torch.Tensor | None = None
    threshold: torch.Tensor | None = None
    gate_probability: torch.Tensor | None = None
    severity_phys: torch.Tensor | None = None
    excess_shape: torch.Tensor | None = None


def head_mlp(hidden, dropout, head_hidden=None):
    # All forecast branches use this same small architecture.
    if head_hidden is None:
        head_hidden = 2 * hidden
    return nn.Sequential(nn.Linear(hidden, head_hidden), nn.LeakyReLU(0.1),
                         nn.Dropout(dropout), nn.Linear(head_hidden, 1))


class SingleHead(nn.Module):
    def __init__(self, hidden, dropout, head_hidden=None):
        super().__init__()
        self.regression = head_mlp(hidden, dropout, head_hidden)

    def forward(self, context):
        return ForecastOutput(self.regression(context).squeeze(-1))


class ExceedanceHead(nn.Module):
    """y = body + P(window exceeds threshold | X) * conditional excess.

    Labels enter the loss only. The TRAIN-derived threshold is fixed in the
    checkpoint; the window gate and both regression branches are learned.
    """
    def __init__(self, hidden, dropout, threshold, prior, fixed_gate=False, head_hidden=None):
        super().__init__()
        threshold = torch.as_tensor(threshold, dtype=torch.float32).reshape(1, -1)
        init_prior = initial_gate_prior(prior)
        if not torch.isfinite(threshold).all():
            raise ValueError("Dual head needs finite TRAIN thresholds.")
        self.register_buffer("threshold", threshold.clone())
        self.body = head_mlp(hidden, dropout, head_hidden)
        self.excess = head_mlp(hidden, dropout, head_hidden)
        self.gate = None if fixed_gate else head_mlp(hidden, dropout, head_hidden)
        if fixed_gate:
            self.register_buffer("fixed_gate_probability", torch.tensor(float(prior)))
        else:
            nn.init.zeros_(self.gate[-1].weight)
            nn.init.constant_(self.gate[-1].bias, math.log(init_prior / (1 - init_prior)))
        nn.init.zeros_(self.excess[-1].weight)
        nn.init.constant_(self.excess[-1].bias, math.log(math.expm1(0.1)))

    def forward(self, context):
        raw_body = self.body(context).squeeze(-1).float()
        body = self.threshold - F.softplus(self.threshold - raw_body)
        excess = F.softplus(self.excess(context).squeeze(-1).float())
        gate_logits = self.gate(context.mean(dim=1)).float() if self.gate is not None else None
        probability = (gate_logits.sigmoid() if gate_logits is not None
                       else self.fixed_gate_probability.expand(context.size(0), 1))
        prediction = body + probability * excess
        return ForecastOutput(prediction, body, excess, gate_logits, self.threshold, probability)


class SeverityShapeHead(nn.Module):
    """Physical excess = window severity (meters) * horizon shape (unitless).

    This separate opt-in head leaves the direct head's modules, initialization
    order and buffers untouched. Body and event gate retain their definitions.
    """
    def __init__(self, hidden, dropout, threshold, prior, target_y_std,
                 severity_shape_eps=1e-6, fixed_gate=False, head_hidden=None):
        super().__init__()
        validate_excess_formulation("severity_shape", severity_shape_eps)
        threshold = torch.as_tensor(threshold, dtype=torch.float32).reshape(1, -1)
        init_prior = initial_gate_prior(prior)
        if not torch.isfinite(threshold).all():
            raise ValueError("Dual head needs finite TRAIN thresholds.")
        if target_y_std is None:
            raise ValueError("severity_shape requires one finite, positive TRAIN target_y_std per horizon.")
        scale = torch.as_tensor(target_y_std, dtype=torch.float32)
        if (scale.ndim != 1 or scale.numel() != threshold.numel()
                or not torch.isfinite(scale).all() or not (scale > 0).all()):
            raise ValueError("target_y_std must contain one finite, positive TRAIN scale per output horizon.")
        self.register_buffer("threshold", threshold.clone())
        self.register_buffer("target_y_std", scale.reshape(1, -1).clone())
        self.severity_shape_eps = severity_shape_eps
        self.body = head_mlp(hidden, dropout, head_hidden)
        self.severity = head_mlp(hidden, dropout, head_hidden)
        self.shape = head_mlp(hidden, dropout, head_hidden)
        self.gate = None if fixed_gate else head_mlp(hidden, dropout, head_hidden)
        if fixed_gate:
            self.register_buffer("fixed_gate_probability", torch.tensor(float(prior)))
        else:
            nn.init.zeros_(self.gate[-1].weight)
            nn.init.constant_(self.gate[-1].bias, math.log(init_prior / (1 - init_prior)))
        # Start at severity = 0.1 meters, softplus shape = 1 (normalized shape = 1).
        # No fitted severity statistic or additional label-derived initialization.
        for branch, positive in ((self.severity, .1), (self.shape, 1.)):
            nn.init.zeros_(branch[-1].weight)
            nn.init.constant_(branch[-1].bias, math.log(math.expm1(positive)))

    def forward(self, context):
        raw_body = self.body(context).squeeze(-1).float()
        body = self.threshold - F.softplus(self.threshold - raw_body)
        pooled = context.mean(dim=1)
        severity = F.softplus(self.severity(pooled).squeeze(-1).float())
        # Add epsilon before normalization: even softplus underflow retains
        # a unit peak, so severity remains the physical peak excess amplitude.
        raw_shape = F.softplus(self.shape(context).squeeze(-1).float()) + self.severity_shape_eps
        shape = raw_shape / raw_shape.max(dim=1, keepdim=True).values
        excess = severity[:, None] * shape / self.target_y_std
        gate_logits = self.gate(pooled).float() if self.gate is not None else None
        probability = (gate_logits.sigmoid() if gate_logits is not None
                       else self.fixed_gate_probability.expand(context.size(0), 1))
        prediction = body + probability * excess
        return ForecastOutput(prediction, body, excess, gate_logits, self.threshold, probability,
                              severity, shape)
