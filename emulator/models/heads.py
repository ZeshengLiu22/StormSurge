"""Single regression head shared by training and inference."""

from typing import NamedTuple

import torch
from torch import nn


class ForecastOutput(NamedTuple):
    """Model return container for training-standardized predictions.

    Attributes:
        prediction: Tensor [batch, horizons] in normalized target units.
            Physical values are recovered with training y_std and y_mean.
    """
    prediction: torch.Tensor


def head_mlp(hidden, dropout, head_hidden=None):
    """Build the shared scalar-regression MLP used for each horizon context.

    Args:
        hidden: Number of input context features.
        dropout: Dropout probability after the hidden activation.
        head_hidden: Intermediate feature width, or None for twice hidden.

    Returns:
        torch.nn.Sequential: A two-layer MLP mapping [..., hidden] to [..., 1].
    """
    if head_hidden is None:
        head_hidden = 2 * hidden
    return nn.Sequential(
        nn.Linear(hidden, head_hidden),
        nn.LeakyReLU(0.1),
        nn.Dropout(dropout),
        nn.Linear(head_hidden, 1),
    )


class SingleHead(nn.Module):
    """Shared horizon-wise MLP producing one normalized scalar per context.
    """
    def __init__(self, hidden, dropout, head_hidden=None):
        """Initialize the scalar regression head.

        Args:
            hidden: Number of features in each horizon context.
            dropout: Dropout probability inside the regression MLP.
            head_hidden: Intermediate MLP width, or None for twice hidden.

        Returns:
            None. Registers the regression MLP.
        """
        super().__init__()
        self.regression = head_mlp(hidden, dropout, head_hidden)

    def forward(self, context):
        """Regress one standardized target for each forecast-horizon context.

        Args:
            context: Tensor [batch, horizons, hidden] of horizon-specific features.

        Returns:
            ForecastOutput: prediction [batch, horizons], with the trailing scalar-output
            dimension removed.
        """
        return ForecastOutput(self.regression(context).squeeze(-1))
