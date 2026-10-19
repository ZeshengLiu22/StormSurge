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
    """Construction parameters for the PACT and spatial baseline models.

    Attributes:
        in_channels: Number of input node features.
        out_channels: Number of forecast target horizons.
        hidden_channels: Spatial output width and PACT attention/temporal width.
        model: Architecture key, pact or baseline.
        encoder_type: Spatial operator, GraphSAGE or CNN.
        temporal_block: PACT temporal block: Transformer, MLP, LSTM, or GRU.
        num_layers: Number of spatial layers.
        dropout: Spatial-layer and baseline readout dropout probability.
        cnn_intermediate_channel: Hidden width of nonfinal CNN spatial layers.
        history_steps: Number of forcing steps preceding the current step.
        max_time_steps: PACT lag-embedding capacity, including the current step.
        node_read_heads: Number of heads in PACT station-to-node attention.
        time_read_heads: Number of heads in PACT temporal and forecast attention.
        temporal_layers: Number of PACT temporal blocks or recurrent layers.
        temporal_ff_mult: PACT feedforward width as a multiple of hidden_channels.
        temporal_dropout: Dropout probability within PACT temporal blocks.
        head_dropout: Dropout probability within the PACT forecast head.
        station_feat_dim: PACT station-feature width; zero disables metadata.
        head_hidden: PACT head MLP width, or None for 2 * hidden_channels.
        temporal_hidden: Positive-history baseline LSTM width, or None for
            hidden_channels; ignored when history_steps is zero.

    These are Python construction defaults. The training entry point supplies
    its parsed command-line settings explicitly.
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
    """Station-conditioned spatial attention with temporal memory and horizon queries.
    """
    def __init__(self, config: ModelConfig):
        """Build the spatial encoder, attention readouts, temporal blocks, and head.

        Args:
            config: ModelConfig specifying feature widths, history length, spatial and
                temporal operators, optional station metadata, and forecast horizons.
                Attention widths must be divisible by their head counts.

        Returns:
            None. Registers the model layers, station token, and lag/horizon embeddings.

        Raises:
            ValueError: History, temporal depth, lag capacity, or the selected operator is
                invalid.
        """
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
        """Predict normalized targets from a batch of forcing histories.

        History is ordered from oldest forcing to current forcing. Lag zero always
        refers to the current step. Station attention produces one token per graph and
        time step; horizon queries read from the resulting temporal memory.

        Args:
            batch: PyG Batch with normalized x_hist [total_nodes, history_steps + 1,
                in_channels], edge_index [2, edges], node-to-graph batch indices, and
                num_graphs. CNN encoding also requires uniform grid_H/grid_W metadata and
                graph pointers.
            station_feat: Optional station-feature tensor with station_feat_dim elements,
                shared by all graphs. Required when station metadata is enabled; ignored
                otherwise.

        Returns:
            ForecastOutput: prediction tensor [num_graphs, out_channels] in training-
            standardized target units. The caller restores physical units with y_std and
            y_mean.

        Raises:
            ValueError: The history length, required station features, or CNN grid metadata
                is incompatible with the model.
        """
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
    """Spatial mean pooling with an optional one-layer temporal LSTM.
    """

    def __init__(self, config: ModelConfig):
        """Build a mean-pooling baseline with an LSTM for positive history lengths.

        Args:
            config: ModelConfig defining input/spatial widths, encoder settings, forecast
                width, dropout, and history_steps. A nonzero history_steps enables the LSTM;
                temporal_hidden overrides its output width.

        Returns:
            None. Registers spatial layers, the optional recurrent layer, dropout, and a
            linear multi-horizon head.
        """
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
        """Predict normalized targets from current forcing or pooled forcing history.

        Args:
            batch: PyG Batch containing normalized x [total_nodes, in_channels], edge_index,
                node-to-graph batch indices, and num_graphs. History-enabled models
                additionally use x_hist [total_nodes, steps, in_channels]; CNNs require
                uniform grid metadata.
            station_feat: Accepted for the common model interface; unused by this baseline.

        Returns:
            ForecastOutput: prediction [num_graphs, out_channels] in training-standardized
            target units. With an LSTM, the final recurrent state supplies the forecast
            context.
        """
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
    """Instantiate the architecture selected by a model configuration.

    Args:
        config: ModelConfig whose model field is pact or baseline.

    Returns:
        torch.nn.Module: A newly initialized PACT or Baseline instance.

    Raises:
        KeyError: The model field is not a supported architecture key.
    """
    return {"pact": PACT, "baseline": Baseline}[config.model](config)


def model_config_from_checkpoint(checkpoint):
    """Restore a model configuration from current or historical single-head metadata.

    This function validates construction metadata only. Callers must load the
    model_state with strict=True to validate the parameter schema.

    Args:
        checkpoint: Checkpoint mapping containing model_config and optionally
            model_state, whose parameter names are checked for unsupported retired head
            branches.

    Returns:
        ModelConfig: Configuration with explicitly retired, inactive construction keys
        removed. The input checkpoint is not modified.

    Raises:
        ValueError: The checkpoint declares an unsupported head type or contains retired
            dual-head parameters.
        TypeError: Unknown configuration fields remain after known retired keys are
            removed.
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
