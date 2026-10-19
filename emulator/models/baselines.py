"""Independent shallow baselines with resolution-independent spatial pooling.

Only tensor validation and layout operations are shared. Each architecture owns
its spatial layers, temporal layers (when present), and direct regression head.
"""

import torch
from torch import nn

from .heads import ForecastOutput


BASELINE_ARCHITECTURES = {
    "baseline_mlp": ("MLP", "None"),
    "baseline_cnn": ("CNN", "None"),
    "baseline_mlp_lstm": ("MLP", "LSTM"),
    "baseline_mlp_gru": ("MLP", "GRU"),
    "baseline_cnn_lstm": ("CNN", "LSTM"),
    "baseline_cnn_gru": ("CNN", "GRU"),
    "baseline_convlstm": ("CNN", "ConvLSTM"),
}
NEW_BASELINE_MODELS = tuple(BASELINE_ARCHITECTURES)
STATIC_BASELINE_MODELS = ("baseline_mlp", "baseline_cnn")


def _validate_batch(batch, in_channels, history_steps, *, require_history=False):
    """Validate PyG grid layout and feature/history shapes without changing data.

    Scalar dimensions or one value per graph are accepted. Nodes must remain
    contiguous, in row-major order within each graph. Coordinate ordering itself
    is a data contract; pointers and graph membership are checked here.
    """
    x = getattr(batch, "x", None)
    if not isinstance(x, torch.Tensor) or x.ndim != 2 or x.shape[1] != in_channels:
        raise ValueError("Input feature dimension must match config.in_channels.")
    ptr = getattr(batch, "ptr", None)
    if (
        not isinstance(ptr, torch.Tensor)
        or ptr.ndim != 1
        or ptr.numel() < 2
        or ptr.dtype not in (torch.int32, torch.int64)
    ):
        raise ValueError("Grid batches require integer graph pointers in batch.ptr.")
    size = ptr.numel() - 1
    if getattr(batch, "num_graphs", size) != size or ptr[0] != 0 or ptr[-1] != x.shape[0]:
        raise ValueError("Graph pointers must cover exactly the input nodes.")

    dimensions = []
    for name in ("grid_H", "grid_W"):
        try:
            values = torch.as_tensor(getattr(batch, name, []), device=x.device).reshape(-1)
        except (TypeError, ValueError, RuntimeError) as error:
            raise ValueError(f"Invalid {name} grid metadata.") from error
        if (
            values.numel() not in (1, size)
            or values.is_complex()
            or values.dtype == torch.bool
            or not torch.isfinite(values).all()
            or not torch.all(values == values[0])
        ):
            raise ValueError(f"Baselines require uniform {name} metadata for every graph.")
        value = values[0].item()
        if value <= 0 or int(value) != value:
            raise ValueError(f"{name} must be a positive integer.")
        dimensions.append(int(value))
    height, width = dimensions
    if not torch.all(ptr.diff() == height * width):
        raise ValueError("Node counts must match grid_H * grid_W for every graph.")

    membership = getattr(batch, "batch", None)
    if (
        not isinstance(membership, torch.Tensor)
        or membership.ndim != 1
        or membership.numel() != x.shape[0]
        or membership.dtype not in (torch.int32, torch.int64)
    ):
        raise ValueError("batch.batch must identify every node's graph.")
    expected = torch.arange(size, device=membership.device).repeat_interleave(height * width)
    if not torch.equal(membership, expected):
        raise ValueError("Graph membership must agree with contiguous node order and pointers.")

    history = getattr(batch, "x_hist", None)
    if require_history and history is None:
        raise ValueError("Temporal baselines require x_hist forcing history.")
    if history is not None and (
        not isinstance(history, torch.Tensor)
        or history.ndim != 3
        or history.shape != (x.shape[0], history_steps + 1, in_channels)
    ):
        raise ValueError("Input history length or feature dimension does not match the model configuration.")
    return size, height, width


def _nodes_to_grid(nodes, grid_shape):
    """Preserve row-major node ordering while producing [B, C, H, W]."""
    size, height, width = grid_shape
    return nodes.reshape(size, height, width, -1).permute(0, 3, 1, 2).contiguous()


def _history_to_grid(history, grid_shape):
    """Preserve chronological history and row-major nodes as [B, T, C, H, W]."""
    size, height, width = grid_shape
    steps, channels = history.shape[1:]
    return history.reshape(size, height, width, steps, channels).permute(0, 3, 4, 1, 2).contiguous()


class MLPBaseline(nn.Module):
    """Current-snapshot pointwise MLP with spatial pooling and a direct head."""

    def __init__(self, config):
        """Define the pointwise MLP, configurable pooling, projection, and head."""
        super().__init__()
        if config.history_steps != 0:
            raise ValueError("Static MLP baseline requires history_steps=0 (history_hours=0).")
        self.in_channels = config.in_channels
        self.spatial = nn.Sequential(
            nn.Linear(config.in_channels, config.baseline_stem_channels),
            nn.LeakyReLU(0.1),
            nn.Linear(config.baseline_stem_channels, config.hidden_channels),
            nn.LeakyReLU(0.1),
        )
        pool_grid = config.baseline_pool_grid if config.baseline_pooling == "adaptive" else 1
        self.pool = nn.AdaptiveAvgPool2d((pool_grid, pool_grid))
        self.projection = nn.Sequential(
            nn.Linear(config.hidden_channels * pool_grid ** 2, config.hidden_channels),
            nn.LeakyReLU(0.1),
        )
        self.dropout = nn.Dropout(config.dropout)
        self.head = nn.Linear(config.hidden_channels, config.out_channels)

    def forward(self, batch, station_feat=None):
        """Predict all horizons from current forcing after grid/history validation.

        Returns ForecastOutput with prediction [B, out_channels]. Invalid grid,
        feature, or history shapes raise ValueError. station_feat is unused.
        """
        grid_shape = _validate_batch(batch, self.in_channels, 0)
        features = _nodes_to_grid(self.spatial(batch.x), grid_shape)
        context = self.projection(self.pool(features).flatten(1))
        return ForecastOutput(self.head(self.dropout(context)))


class CNNBaseline(nn.Module):
    """Current-snapshot two-layer CNN with spatial pooling and a direct head."""

    def __init__(self, config):
        """Define two padded convolutions, configurable pooling, projection, and head."""
        super().__init__()
        if config.history_steps != 0:
            raise ValueError("Static CNN baseline requires history_steps=0 (history_hours=0).")
        self.in_channels = config.in_channels
        self.spatial = nn.Sequential(
            nn.Conv2d(config.in_channels, config.baseline_stem_channels, 3, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(config.baseline_stem_channels, config.hidden_channels, 3, padding=1),
            nn.LeakyReLU(0.1),
        )
        pool_grid = config.baseline_pool_grid if config.baseline_pooling == "adaptive" else 1
        self.pool = nn.AdaptiveAvgPool2d((pool_grid, pool_grid))
        self.projection = nn.Sequential(
            nn.Linear(config.hidden_channels * pool_grid ** 2, config.hidden_channels),
            nn.LeakyReLU(0.1),
        )
        self.dropout = nn.Dropout(config.dropout)
        self.head = nn.Linear(config.hidden_channels, config.out_channels)

    def forward(self, batch, station_feat=None):
        """Predict all horizons from the current grid after input validation.

        Returns ForecastOutput with prediction [B, out_channels]. Invalid grid,
        feature, or history shapes raise ValueError. station_feat is unused.
        """
        grid_shape = _validate_batch(batch, self.in_channels, 0)
        features = self.spatial(_nodes_to_grid(batch.x, grid_shape))
        context = self.projection(self.pool(features).flatten(1))
        return ForecastOutput(self.head(self.dropout(context)))


class MLPLSTMBaseline(nn.Module):
    """Shared pointwise MLP and pooled embeddings followed by one LSTM layer."""

    def __init__(self, config):
        """Define a shared pointwise MLP, pooling, projection, and single LSTM."""
        super().__init__()
        self.in_channels = config.in_channels
        self.history_steps = config.history_steps
        self.spatial = nn.Sequential(
            nn.Linear(config.in_channels, config.baseline_stem_channels),
            nn.LeakyReLU(0.1),
            nn.Linear(config.baseline_stem_channels, config.hidden_channels),
            nn.LeakyReLU(0.1),
        )
        pool_grid = config.baseline_pool_grid if config.baseline_pooling == "adaptive" else 1
        self.pool = nn.AdaptiveAvgPool2d((pool_grid, pool_grid))
        self.projection = nn.Sequential(
            nn.Linear(config.hidden_channels * pool_grid ** 2, config.hidden_channels),
            nn.LeakyReLU(0.1),
        )
        self.rnn = nn.LSTM(
            config.hidden_channels, config.hidden_channels, num_layers=1, batch_first=True
        )
        self.dropout = nn.Dropout(config.dropout)
        self.head = nn.Linear(config.hidden_channels, config.out_channels)

    def forward(self, batch, station_feat=None):
        """Encode each timestamp, pool its grid, and predict from the final LSTM output.

        Returns ForecastOutput with prediction [B, out_channels]. Invalid grid,
        feature, or history shapes raise ValueError. station_feat is unused.
        """
        grid_shape = _validate_batch(
            batch, self.in_channels, self.history_steps, require_history=True
        )
        features = _history_to_grid(self.spatial(batch.x_hist), grid_shape).flatten(0, 1)
        embeddings = self.projection(self.pool(features).flatten(1))
        sequence = embeddings.reshape(grid_shape[0], self.history_steps + 1, -1)
        output, _ = self.rnn(sequence)
        return ForecastOutput(self.head(self.dropout(output[:, -1, :])))


class MLPGRUBaseline(nn.Module):
    """Shared pointwise MLP and pooled embeddings followed by one GRU layer."""

    def __init__(self, config):
        """Define a shared pointwise MLP, pooling, projection, and single GRU."""
        super().__init__()
        self.in_channels = config.in_channels
        self.history_steps = config.history_steps
        self.spatial = nn.Sequential(
            nn.Linear(config.in_channels, config.baseline_stem_channels),
            nn.LeakyReLU(0.1),
            nn.Linear(config.baseline_stem_channels, config.hidden_channels),
            nn.LeakyReLU(0.1),
        )
        pool_grid = config.baseline_pool_grid if config.baseline_pooling == "adaptive" else 1
        self.pool = nn.AdaptiveAvgPool2d((pool_grid, pool_grid))
        self.projection = nn.Sequential(
            nn.Linear(config.hidden_channels * pool_grid ** 2, config.hidden_channels),
            nn.LeakyReLU(0.1),
        )
        self.rnn = nn.GRU(
            config.hidden_channels, config.hidden_channels, num_layers=1, batch_first=True
        )
        self.dropout = nn.Dropout(config.dropout)
        self.head = nn.Linear(config.hidden_channels, config.out_channels)

    def forward(self, batch, station_feat=None):
        """Encode each timestamp, pool its grid, and predict from the final GRU output.

        Returns ForecastOutput with prediction [B, out_channels]. Invalid grid,
        feature, or history shapes raise ValueError. station_feat is unused.
        """
        grid_shape = _validate_batch(
            batch, self.in_channels, self.history_steps, require_history=True
        )
        features = _history_to_grid(self.spatial(batch.x_hist), grid_shape).flatten(0, 1)
        embeddings = self.projection(self.pool(features).flatten(1))
        sequence = embeddings.reshape(grid_shape[0], self.history_steps + 1, -1)
        output, _ = self.rnn(sequence)
        return ForecastOutput(self.head(self.dropout(output[:, -1, :])))


class CNNLSTMBaseline(nn.Module):
    """Shared two-layer CNN and pooled embeddings followed by one LSTM layer."""

    def __init__(self, config):
        """Define a shared two-layer CNN, pooling, projection, and single LSTM."""
        super().__init__()
        self.in_channels = config.in_channels
        self.history_steps = config.history_steps
        self.spatial = nn.Sequential(
            nn.Conv2d(config.in_channels, config.baseline_stem_channels, 3, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(config.baseline_stem_channels, config.hidden_channels, 3, padding=1),
            nn.LeakyReLU(0.1),
        )
        pool_grid = config.baseline_pool_grid if config.baseline_pooling == "adaptive" else 1
        self.pool = nn.AdaptiveAvgPool2d((pool_grid, pool_grid))
        self.projection = nn.Sequential(
            nn.Linear(config.hidden_channels * pool_grid ** 2, config.hidden_channels),
            nn.LeakyReLU(0.1),
        )
        self.rnn = nn.LSTM(
            config.hidden_channels, config.hidden_channels, num_layers=1, batch_first=True
        )
        self.dropout = nn.Dropout(config.dropout)
        self.head = nn.Linear(config.hidden_channels, config.out_channels)

    def forward(self, batch, station_feat=None):
        """Encode chronological grids and predict from the final pooled-sequence output.

        Returns ForecastOutput with prediction [B, out_channels]. Invalid grid,
        feature, or history shapes raise ValueError. station_feat is unused.
        """
        grid_shape = _validate_batch(
            batch, self.in_channels, self.history_steps, require_history=True
        )
        snapshots = _history_to_grid(batch.x_hist, grid_shape).flatten(0, 1)
        embeddings = self.projection(self.pool(self.spatial(snapshots)).flatten(1))
        sequence = embeddings.reshape(grid_shape[0], self.history_steps + 1, -1)
        output, _ = self.rnn(sequence)
        return ForecastOutput(self.head(self.dropout(output[:, -1, :])))


class CNNGRUBaseline(nn.Module):
    """Traditional two-layer CNN and one GRU layer, independent of PACT."""

    def __init__(self, config):
        """Define a shared two-layer CNN, pooling, projection, and single GRU."""
        super().__init__()
        self.in_channels = config.in_channels
        self.history_steps = config.history_steps
        self.spatial = nn.Sequential(
            nn.Conv2d(config.in_channels, config.baseline_stem_channels, 3, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(config.baseline_stem_channels, config.hidden_channels, 3, padding=1),
            nn.LeakyReLU(0.1),
        )
        pool_grid = config.baseline_pool_grid if config.baseline_pooling == "adaptive" else 1
        self.pool = nn.AdaptiveAvgPool2d((pool_grid, pool_grid))
        self.projection = nn.Sequential(
            nn.Linear(config.hidden_channels * pool_grid ** 2, config.hidden_channels),
            nn.LeakyReLU(0.1),
        )
        self.rnn = nn.GRU(
            config.hidden_channels, config.hidden_channels, num_layers=1, batch_first=True
        )
        self.dropout = nn.Dropout(config.dropout)
        self.head = nn.Linear(config.hidden_channels, config.out_channels)

    def forward(self, batch, station_feat=None):
        """Encode chronological grids and predict from the final pooled-sequence output.

        Returns ForecastOutput with prediction [B, out_channels]. Invalid grid,
        feature, or history shapes raise ValueError. station_feat is unused.
        """
        grid_shape = _validate_batch(
            batch, self.in_channels, self.history_steps, require_history=True
        )
        snapshots = _history_to_grid(batch.x_hist, grid_shape).flatten(0, 1)
        embeddings = self.projection(self.pool(self.spatial(snapshots)).flatten(1))
        sequence = embeddings.reshape(grid_shape[0], self.history_steps + 1, -1)
        output, _ = self.rnn(sequence)
        return ForecastOutput(self.head(self.dropout(output[:, -1, :])))


class ConvLSTMBaseline(nn.Module):
    """One spatial input convolution and a single spatial ConvLSTM recurrence.

    Hidden and cell states retain the full grid resolution throughout recurrence.
    Only the final hidden map is pooled before direct multi-horizon prediction.
    """

    def __init__(self, config):
        """Define the spatial projection, convolutional gates, pooling, and direct head."""
        super().__init__()
        self.in_channels = config.in_channels
        self.history_steps = config.history_steps
        self.stem_channels = config.baseline_stem_channels
        self.input_projection = nn.Sequential(
            nn.Conv2d(config.in_channels, config.baseline_stem_channels, 3, padding=1),
            nn.LeakyReLU(0.1),
        )
        self.gates = nn.Conv2d(
            2 * config.baseline_stem_channels, 4 * config.baseline_stem_channels, 3, padding=1
        )
        pool_grid = config.baseline_pool_grid if config.baseline_pooling == "adaptive" else 1
        self.pool = nn.AdaptiveAvgPool2d((pool_grid, pool_grid))
        self.projection = nn.Sequential(
            nn.Linear(config.baseline_stem_channels * pool_grid ** 2, config.hidden_channels),
            nn.LeakyReLU(0.1),
        )
        self.dropout = nn.Dropout(config.dropout)
        self.head = nn.Linear(config.hidden_channels, config.out_channels)

    def forward(self, batch, station_feat=None):
        """Retain full-grid hidden/cell states until pooling the final hidden state.

        Returns ForecastOutput with prediction [B, out_channels]. Invalid grid,
        feature, or history shapes raise ValueError. station_feat is unused.
        """
        grid_shape = _validate_batch(
            batch, self.in_channels, self.history_steps, require_history=True
        )
        size, height, width = grid_shape
        snapshots = _history_to_grid(batch.x_hist, grid_shape).flatten(0, 1)
        projected = self.input_projection(snapshots).reshape(
            size, self.history_steps + 1, self.stem_channels, height, width
        )
        # Projected features supply the device and autocast-compatible state dtype.
        hidden = projected.new_zeros(size, self.stem_channels, height, width)
        cell = torch.zeros_like(hidden)
        for snapshot in projected.unbind(dim=1):
            i, f, o, g = self.gates(torch.cat((snapshot, hidden), dim=1)).chunk(4, dim=1)
            cell = torch.sigmoid(f) * cell + torch.sigmoid(i) * torch.tanh(g)
            hidden = torch.sigmoid(o) * torch.tanh(cell)
        context = self.projection(self.pool(hidden).flatten(1))
        return ForecastOutput(self.head(self.dropout(context)))
