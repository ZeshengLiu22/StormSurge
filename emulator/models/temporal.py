"""Current temporal blocks; no historical architecture switches."""

import torch
from torch import nn


class TemporalMLP(nn.Module):
    def __init__(self, hidden, ff_dim, dropout):
        super().__init__()
        self.norm = nn.LayerNorm(hidden)
        self.ff = nn.Sequential(
            nn.Linear(hidden, ff_dim), nn.LeakyReLU(0.1), nn.Dropout(dropout),
            nn.Linear(ff_dim, hidden), nn.Dropout(dropout),
        )
        # Positive learnable gain: 0.1 is an initialization, not a tuned constant.
        self.log_gain = nn.Parameter(torch.tensor(0.1).log())

    def forward(self, x):
        return x + self.log_gain.exp() * self.ff(self.norm(x))


class TemporalTransformer(nn.Module):
    def __init__(self, hidden, heads, ff_dim, dropout):
        super().__init__()
        self.norm = nn.LayerNorm(hidden)
        self.attention = nn.MultiheadAttention(hidden, heads, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.log_gain = nn.Parameter(torch.tensor(0.1).log())
        self.mlp = TemporalMLP(hidden, ff_dim, dropout)

    def forward(self, x):
        z = self.norm(x)
        residual, _ = self.attention(z, z, z, need_weights=False)
        x = x + self.log_gain.exp() * self.dropout(residual)
        return self.mlp(x)


class TemporalRNN(nn.Module):
    def __init__(self, hidden, depth, dropout, kind):
        super().__init__()
        rnn = {"LSTM": nn.LSTM, "GRU": nn.GRU}[kind]
        self.rnn = rnn(hidden, hidden, depth, batch_first=True, dropout=dropout if depth > 1 else 0)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden)

    def forward(self, x):
        residual, _ = self.rnn(x)
        return self.norm(x + self.dropout(residual))
