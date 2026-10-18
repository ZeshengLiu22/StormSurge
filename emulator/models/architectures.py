"""Compose spatial encoding, station readout, temporal memory and forecast head."""

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool
from torch_geometric.utils import to_dense_batch

from .heads import ForecastOutput, SingleHead
from .spatial import SpatialEncoder
from .temporal import TemporalMLP, TemporalRNN, TemporalTransformer


@dataclass
class ModelConfig:
    """Python construction defaults; train.py supplies the CLI/config values.

    head_hidden overrides the PACT head MLP width (default: 2 * hidden_channels).
    temporal_hidden overrides the positive-history baseline LSTM width
    (default: hidden_channels); H=0 baselines keep the spatial output width.
    """

    in_channels: int
    out_channels: int
    hidden_channels: int
    model: str = "pact"
    encoder_type: str = "GraphSAGE"
    temporal_block: str = "Transformer"
    num_layers: int = 2
    dropout: float = 0.0
    cnn_intermediate_channel: int = 29
    history_steps: int = 2
    max_time_steps: int = 32
    node_read_heads: int = 8
    time_read_heads: int = 8
    temporal_layers: int = 2
    temporal_ff_mult: float = 4.0
    temporal_dropout: float = 0.05
    head_dropout: float = 0.0
    station_feat_dim: int = 0
    head_hidden: int | None = None
    temporal_hidden: int | None = None


class PACT(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        hidden = config.hidden_channels
        if config.history_steps < 0 or config.temporal_layers < 1:
            raise ValueError("History must be nonnegative and temporal depth positive.")
        self.spatial = SpatialEncoder(
            config.in_channels,
            hidden,
            config.num_layers,
            config.dropout,
            config.encoder_type,
            config.cnn_intermediate_channel,
        )
        self.station_token = nn.Parameter(torch.zeros(1, hidden))
        meta_hidden = max(16, hidden)
        self.station_meta = (
            nn.Sequential(
                nn.Linear(config.station_feat_dim, meta_hidden),
                nn.LeakyReLU(0.1),
                nn.Linear(meta_hidden, hidden),
                nn.LayerNorm(hidden),
            )
            if config.station_feat_dim
            else None
        )
        self.node_readout = nn.MultiheadAttention(hidden, config.node_read_heads, batch_first=True)

        self.history_steps = config.history_steps
        if config.max_time_steps < config.history_steps + 1:
            raise ValueError("max_time_steps must cover the requested history window.")
        self.lag_embed = nn.Embedding(config.max_time_steps, hidden)
        ff_dim = int(hidden * config.temporal_ff_mult)
        if config.temporal_block == "Transformer":
            self.temporal = nn.Sequential(
                *(
                    TemporalTransformer(
                        hidden, config.time_read_heads, ff_dim, config.temporal_dropout
                    )
                    for _ in range(config.temporal_layers)
                )
            )
        elif config.temporal_block == "MLP":
            self.temporal = nn.Sequential(
                *(
                    TemporalMLP(hidden, ff_dim, config.temporal_dropout)
                    for _ in range(config.temporal_layers)
                )
            )
        elif config.temporal_block in ("LSTM", "GRU"):
            self.temporal = TemporalRNN(
                hidden,
                config.temporal_layers,
                config.temporal_dropout,
                config.temporal_block,
            )
        else:
            raise ValueError(f"Unknown temporal block: {config.temporal_block}")

        self.horizon_embed = nn.Embedding(config.out_channels, hidden)
        self.forecast_readout = nn.MultiheadAttention(
            hidden, config.time_read_heads, batch_first=True
        )
        self.head = SingleHead(hidden, config.head_dropout, head_hidden=config.head_hidden)

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
            station, _ = self.node_readout(
                query, nodes, nodes, key_padding_mask=~mask, need_weights=False
            )
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
        self.spatial = SpatialEncoder(
            config.in_channels,
            config.hidden_channels,
            config.num_layers,
            config.dropout,
            config.encoder_type,
            config.cnn_intermediate_channel,
        )
        temporal_hidden = (
            config.hidden_channels if config.temporal_hidden is None else config.temporal_hidden
        )
        self.rnn = (
            nn.LSTM(config.hidden_channels, temporal_hidden, batch_first=True)
            if config.history_steps
            else None
        )
        self.dropout = nn.Dropout(config.dropout)
        self.head = nn.Linear(
            temporal_hidden if self.rnn is not None else config.hidden_channels,
            config.out_channels,
        )

    def forward(self, batch, station_feat=None):
        grid_shape = self.spatial.grid_shape(batch)
        if self.rnn is None:
            context = global_mean_pool(
                self.spatial(batch.x, batch.edge_index, grid_shape), batch.batch
            )
        else:
            sequence = [
                global_mean_pool(self.spatial(x, batch.edge_index, grid_shape), batch.batch)
                for x in batch.x_hist.unbind(dim=1)
            ]
            context, _ = self.rnn(torch.stack(sequence, dim=1))
            context = context[:, -1]
        context = self.dropout(context)
        return ForecastOutput(self.head(context))


def build_model(config: ModelConfig):
    return {"pact": PACT, "baseline": Baseline}[config.model](config)


def model_config_from_checkpoint(checkpoint):
    """Read current or historical Single metadata without relaxing state loading.

    Only explicitly retired construction keys are ignored. Unknown fields still
    raise an error, and callers must load ``model_state`` with ``strict=True``.
    """
    saved_config = dict(checkpoint["model_config"])
    head_type = saved_config.pop("head_type", "single")
    state = checkpoint.get("model_state", {})
    obsolete_branches = (
        "head.body.",
        "head.excess.",
        "head.gate.",
        "head.severity.",
        "head.shape.",
    )
    has_obsolete_branch = any(
        key.removeprefix("module.").startswith(obsolete_branches) for key in state
    )
    if head_type == "dual" or has_obsolete_branch:
        raise ValueError(
            "Legacy Dual-head checkpoints are no longer supported by this cleaned codebase."
        )
    if head_type != "single":
        raise ValueError(f"Unsupported checkpoint head type: {head_type!r}.")

    # Historical Single checkpoints saved these inactive construction settings.
    obsolete_keys = (
        "peak_threshold_norm",
        "peak_prior",
        "dual_ablation",
        "excess_formulation",
        "target_y_std",
        "severity_shape_eps",
        "exceedance_head_experiment",
        "exceedance_gate_pooling",
    )
    for key in obsolete_keys:
        saved_config.pop(key, None)
    return ModelConfig(**saved_config)
