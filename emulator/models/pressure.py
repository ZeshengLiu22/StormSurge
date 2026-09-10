"""Optional pressure metadata encoders for the original p_mean ablations."""

from torch import nn
import torch.nn.functional as F


class PressureFeatures(nn.Module):
    def __init__(self, config, baseline=False):
        super().__init__()
        c = config
        self.window = c.history_steps + 1
        tokens = not baseline and c.perceiver_pmean_mode in ("tokens", "both")
        global_vector = baseline or c.perceiver_pmean_mode in ("global", "both")
        self.tokens = nn.Sequential(nn.Linear(1, c.hidden_channels), nn.LayerNorm(c.hidden_channels)) if tokens else None
        self.width = self.window if baseline else c.max_time_steps
        self.global_dim = c.pmean_dim if global_vector else 0
        self.global_encoder = None
        if global_vector:
            layers = [nn.Linear(self.width, c.pmean_dim), nn.LeakyReLU(0.1),
                      nn.Linear(c.pmean_dim, c.pmean_dim)]
            if not baseline:
                layers.append(nn.LayerNorm(c.pmean_dim))
            self.global_encoder = nn.Sequential(*layers)

    def forward(self, history):
        if history.size(1) != self.window:
            raise ValueError("p_mean history must match the forcing history window.")
        tokens = self.tokens(history.unsqueeze(-1)) if self.tokens is not None else None
        vector = self.global_encoder(F.pad(history, (self.width - self.window, 0))) if self.global_encoder is not None else None
        return tokens, vector
