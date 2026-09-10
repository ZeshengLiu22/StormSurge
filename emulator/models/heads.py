"""Forecast heads and the one output contract used by training and inference."""

import math
from typing import NamedTuple

import torch
from torch import nn
import torch.nn.functional as F


class ForecastOutput(NamedTuple):
    prediction: torch.Tensor
    body: torch.Tensor | None = None
    excess: torch.Tensor | None = None
    gate_logits: torch.Tensor | None = None
    threshold: torch.Tensor | None = None


def head_mlp(hidden, dropout, input_dim=None):
    # All three dual-head branches use this same small architecture.
    return nn.Sequential(nn.Linear(input_dim or hidden, 2 * hidden), nn.LeakyReLU(0.1),
                         nn.Dropout(dropout), nn.Linear(2 * hidden, 1))


class SingleHead(nn.Module):
    def __init__(self, hidden, dropout, input_dim=None):
        super().__init__()
        self.regression = head_mlp(hidden, dropout, input_dim)

    def forward(self, context):
        return ForecastOutput(self.regression(context).squeeze(-1))


class ExceedanceHead(nn.Module):
    """y = body + P(window exceeds threshold | X) * conditional excess.

    Labels enter the loss only. The TRAIN-derived threshold is fixed in the
    checkpoint; the window gate and both regression branches are learned.
    """
    def __init__(self, hidden, dropout, threshold, prior, input_dim=None):
        super().__init__()
        threshold = torch.as_tensor(threshold, dtype=torch.float32).reshape(1, -1)
        if not torch.isfinite(threshold).all() or not 0 < prior < 1:
            raise ValueError("Dual head needs finite TRAIN thresholds and a prior in (0, 1).")
        self.register_buffer("threshold", threshold.clone())
        self.body = head_mlp(hidden, dropout, input_dim)
        self.excess = head_mlp(hidden, dropout, input_dim)
        self.gate = head_mlp(hidden, dropout, input_dim)
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.constant_(self.gate[-1].bias, math.log(prior / (1 - prior)))
        nn.init.zeros_(self.excess[-1].weight)
        nn.init.constant_(self.excess[-1].bias, math.log(math.expm1(0.1)))

    def forward(self, context):
        raw_body = self.body(context).squeeze(-1).float()
        body = self.threshold - F.softplus(self.threshold - raw_body)
        excess = F.softplus(self.excess(context).squeeze(-1).float())
        gate_logits = self.gate(context.mean(dim=1)).float()
        prediction = body + gate_logits.sigmoid() * excess
        return ForecastOutput(prediction, body, excess, gate_logits, self.threshold)
