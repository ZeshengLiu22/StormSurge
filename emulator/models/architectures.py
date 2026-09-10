"""Compose spatial encoding, station readout, temporal memory and forecast head."""

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool
from torch_geometric.utils import to_dense_batch

from emulator.common.dual import DUAL_ABLATIONS

from .heads import ExceedanceHead, ForecastOutput, SingleHead
from .spatial import SpatialEncoder
from .temporal import TemporalMLP, TemporalRNN, TemporalTransformer


@dataclass
class ModelConfig:
    in_channels: int
    out_channels: int
    model: str = "pact"
    encoder_type: str = "GraphSAGE"
    temporal_block: str = "Transformer"
    head_type: str = "dual"
    hidden_channels: int = 128
    num_layers: int = 2
    dropout: float = 0.05
    cnn_intermediate_channel: int = 29
    history_steps: int = 2
    max_time_steps: int = 32
    node_read_heads: int = 8
    time_read_heads: int = 8
    temporal_layers: int = 2
    temporal_ff_mult: float = 4.0
    temporal_dropout: float = 0.0
    head_dropout: float = 0.05
    station_feat_dim: int = 0
    peak_threshold_norm: list[float] | None = None
    peak_prior: float = 0.05
    dual_ablation: str = "none"


class PACT(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        c = config
        hidden = c.hidden_channels
        if c.history_steps < 0 or c.temporal_layers < 1:
            raise ValueError("History must be nonnegative and temporal depth positive.")
        self.spatial = SpatialEncoder(c.in_channels, hidden, c.num_layers, c.dropout,
                                      c.encoder_type, c.cnn_intermediate_channel)
        self.station_token = nn.Parameter(torch.zeros(1, hidden))
        self.station_meta = nn.Sequential(
            nn.Linear(c.station_feat_dim, hidden), nn.LeakyReLU(0.1),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden),
        ) if c.station_feat_dim else None
        self.node_readout = nn.MultiheadAttention(hidden, c.node_read_heads, batch_first=True)
        self.history_steps = c.history_steps
        if c.max_time_steps < c.history_steps + 1:
            raise ValueError("max_time_steps must cover the requested history window.")
        self.lag_embed = nn.Embedding(c.max_time_steps, hidden)
        ff_dim = int(hidden * c.temporal_ff_mult)
        if c.temporal_block == "Transformer":
            self.temporal = nn.Sequential(*(TemporalTransformer(hidden, c.time_read_heads, ff_dim,
                c.temporal_dropout) for _ in range(c.temporal_layers)))
        elif c.temporal_block == "MLP":
            self.temporal = nn.Sequential(*(TemporalMLP(hidden, ff_dim, c.temporal_dropout)
                for _ in range(c.temporal_layers)))
        elif c.temporal_block in ("LSTM", "GRU"):
            self.temporal = TemporalRNN(hidden, c.temporal_layers, c.temporal_dropout, c.temporal_block)
        else:
            raise ValueError(f"Unknown temporal block: {c.temporal_block}")
        self.horizon_embed = nn.Embedding(c.out_channels, hidden)
        self.forecast_readout = nn.MultiheadAttention(hidden, c.time_read_heads, batch_first=True)
        if c.head_type == "single":
            self.head = SingleHead(hidden, c.head_dropout)
        elif c.head_type == "dual":
            if c.peak_threshold_norm is None or len(c.peak_threshold_norm) != c.out_channels:
                raise ValueError("Dual head requires one TRAIN-derived threshold per horizon.")
            self.head = ExceedanceHead(hidden, c.head_dropout, c.peak_threshold_norm, c.peak_prior,
                                       fixed_gate=c.dual_ablation == "fixed_gate")
        else:
            raise ValueError(f"Unknown head: {c.head_type}")

    def forward(self, batch, station_feat=None):
        history = batch.x_hist
        steps = history.size(1)
        if steps != self.history_steps + 1:
            raise ValueError("Input history length does not match the model configuration.")
        grid_shape = self.spatial.grid_shape(batch)
        query = self.station_token
        if self.station_meta is not None:
            if station_feat is None:
                raise ValueError("This model requires station features.")
            query = query + self.station_meta(station_feat.reshape(1, -1))
        query = query.unsqueeze(0).expand(batch.num_graphs, -1, -1)
        station_history = []
        for x in history.unbind(dim=1):
            nodes = self.spatial(x, batch.edge_index, grid_shape)
            nodes, mask = to_dense_batch(nodes, batch.batch)
            station, _ = self.node_readout(query, nodes, nodes, key_padding_mask=~mask, need_weights=False)
            station_history.append(F.layer_norm(station.float(), (station.size(-1),)).squeeze(1))
        memory = torch.stack(station_history, dim=1)
        # Lag zero always means current forcing, independently of H.
        lag_ids = torch.arange(steps - 1, -1, -1, device=memory.device)
        memory = self.temporal(memory + self.lag_embed(lag_ids))
        horizons = self.horizon_embed.weight.unsqueeze(0).expand(batch.num_graphs, -1, -1)
        context, _ = self.forecast_readout(horizons, memory, memory, need_weights=False)
        if steps == 1:
            context = context + horizons  # Retain horizon identity in the H=0 control.
        context = F.layer_norm(context.float(), (context.size(-1),))
        return self.head(context)


class Baseline(nn.Module):
    """Spatial mean pooling; positive-history controls add a one-layer LSTM."""
    def __init__(self, config: ModelConfig):
        super().__init__()
        c = config
        if c.head_type != "single":
            raise ValueError("Baseline uses a single linear forecast head.")
        self.spatial = SpatialEncoder(c.in_channels, c.hidden_channels, c.num_layers,
                                      c.dropout, c.encoder_type, c.cnn_intermediate_channel)
        self.rnn = nn.LSTM(c.hidden_channels, c.hidden_channels, batch_first=True) if c.history_steps else None
        self.dropout = nn.Dropout(c.dropout)
        self.head = nn.Linear(c.hidden_channels, c.out_channels)

    def forward(self, batch, station_feat=None):
        grid_shape = self.spatial.grid_shape(batch)
        if self.rnn is None:
            context = global_mean_pool(self.spatial(batch.x, batch.edge_index, grid_shape), batch.batch)
        else:
            sequence = [global_mean_pool(self.spatial(x, batch.edge_index, grid_shape), batch.batch)
                        for x in batch.x_hist.unbind(dim=1)]
            context, _ = self.rnn(torch.stack(sequence, dim=1))
            context = context[:, -1]
        context = self.dropout(context)
        return ForecastOutput(self.head(context))


def build_model(config: ModelConfig):
    if config.dual_ablation not in DUAL_ABLATIONS:
        raise ValueError(f"Unknown dual ablation: {config.dual_ablation}")
    if config.dual_ablation != "none" and (config.model != "pact" or config.head_type != "dual"):
        raise ValueError("Dual ablations require a PACT dual head.")
    return {"pact": PACT, "baseline": Baseline}[config.model](config)
