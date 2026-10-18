"""Single regression head shared by training and inference."""

from typing import NamedTuple

import torch
from torch import nn


class ForecastOutput(NamedTuple):
    prediction: torch.Tensor


def head_mlp(hidden, dropout, head_hidden=None):
    if head_hidden is None:
        head_hidden = 2 * hidden
    return nn.Sequential(
        nn.Linear(hidden, head_hidden),
        nn.LeakyReLU(0.1),
        nn.Dropout(dropout),
        nn.Linear(head_hidden, 1),
    )


class SingleHead(nn.Module):
    def __init__(self, hidden, dropout, head_hidden=None):
        super().__init__()
        self.regression = head_mlp(hidden, dropout, head_hidden)

    def forward(self, context):
        return ForecastOutput(self.regression(context).squeeze(-1))
