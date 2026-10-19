"""Residual temporal MLP, self-attention, and recurrent feature blocks."""

import torch
from torch import nn


class TemporalMLP(nn.Module):
    """Pre-normalized, positionwise feedforward residual block with a learned gain.
    """
    def __init__(self, hidden, ff_dim, dropout):
        """Build the temporal feedforward residual transformation.

        Args:
            hidden: Input and output feature width.
            ff_dim: Intermediate feedforward feature width.
            dropout: Dropout probability after each feedforward linear transformation.

        Returns:
            None. Registers layer normalization, the MLP, and a positive residual gain
            initialized to 0.1.
        """
        super().__init__()
        self.norm = nn.LayerNorm(hidden)
        self.ff = nn.Sequential(
            nn.Linear(hidden, ff_dim), nn.LeakyReLU(0.1), nn.Dropout(dropout),
            nn.Linear(ff_dim, hidden), nn.Dropout(dropout),
        )
        # A log parameter keeps the learnable residual gain positive.
        self.log_gain = nn.Parameter(torch.tensor(0.1).log())

    def forward(self, x):
        """Apply a shared residual MLP independently at each time step.

        Args:
            x: Tensor [batch, steps, hidden] of temporal features.

        Returns:
            torch.Tensor: Features of the same shape, without mixing information across time
            steps.
        """
        return x + self.log_gain.exp() * self.ff(self.norm(x))


class TemporalTransformer(nn.Module):
    """Pre-normalized self-attention followed by a residual temporal MLP.
    """
    def __init__(self, hidden, heads, ff_dim, dropout):
        """Build a full-history self-attention block and its feedforward sublayer.

        Args:
            hidden: Input and output feature width, divisible by heads.
            heads: Number of self-attention heads.
            ff_dim: Intermediate feature width in the residual MLP.
            dropout: Dropout probability for attention, its residual output, and the MLP.

        Returns:
            None. Registers unmasked self-attention and learned positive residual gains.
        """
        super().__init__()
        self.norm = nn.LayerNorm(hidden)
        self.attention = nn.MultiheadAttention(hidden, heads, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.log_gain = nn.Parameter(torch.tensor(0.1).log())
        self.mlp = TemporalMLP(hidden, ff_dim, dropout)

    def forward(self, x):
        """Mix all supplied forcing-history steps through self-attention.

        Args:
            x: Tensor [batch, steps, hidden] containing the complete observed history.

        Returns:
            torch.Tensor: Updated temporal features with the same shape. No causal or
            padding mask is applied.
        """
        z = self.norm(x)
        residual, _ = self.attention(z, z, z, need_weights=False)
        x = x + self.log_gain.exp() * self.dropout(residual)
        return self.mlp(x)


class TemporalRNN(nn.Module):
    """Residual LSTM or GRU returning normalized features at every time step.
    """
    def __init__(self, hidden, depth, dropout, kind):
        """Build a recurrent temporal block with matching input and output widths.

        Args:
            hidden: Input, recurrent-state, and output feature width.
            depth: Positive number of stacked recurrent layers.
            dropout: Output dropout probability and, for depth greater than one, inter-layer
                recurrent dropout.
            kind: Recurrent operator key: LSTM or GRU.

        Returns:
            None. Registers the recurrent network, residual dropout, and layer
            normalization.

        Raises:
            KeyError: The recurrent operator key is unsupported.
        """
        super().__init__()
        rnn = {"LSTM": nn.LSTM, "GRU": nn.GRU}[kind]
        self.rnn = rnn(hidden, hidden, depth, batch_first=True, dropout=dropout if depth > 1 else 0)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden)

    def forward(self, x):
        """Add recurrent history features to the input and normalize the residual.

        Args:
            x: Chronologically ordered tensor [batch, steps, hidden].

        Returns:
            torch.Tensor: Updated features [batch, steps, hidden]. Initial recurrent states
            are zero and final hidden/cell states are discarded.
        """
        residual, _ = self.rnn(x)
        return self.norm(x + self.dropout(residual))
