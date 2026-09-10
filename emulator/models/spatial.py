"""Spatial encoders with the same node-in, node-out interface."""

import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv


class SpatialEncoder(nn.Module):
    def __init__(self, in_dim, hidden, depth, dropout, kind, cnn_width=29):
        super().__init__()
        if depth < 1 or (kind == "GraphSAGE" and depth < 2):
            raise ValueError("CNN needs at least one layer; GraphSAGE needs at least two.")
        self.kind = kind
        self.dropout = dropout
        if kind == "GraphSAGE":
            dims = [in_dim] + [hidden] * depth
            self.layers = nn.ModuleList(SAGEConv(a, b) for a, b in zip(dims, dims[1:]))
        elif kind == "CNN":
            dims = [in_dim] + [cnn_width] * (depth - 1) + [hidden]
            self.layers = nn.ModuleList(nn.Conv2d(a, b, 3, padding=1) for a, b in zip(dims, dims[1:]))
        else:
            raise ValueError(f"Unknown spatial encoder: {kind}")

    def grid_shape(self, batch):
        """Check the CNN's row-major, rectangular grid contract once per batch."""
        if self.kind != "CNN":
            return None
        dimensions = []
        for name in ("grid_H", "grid_W"):
            values = torch.as_tensor(getattr(batch, name, []), device=batch.x.device).reshape(-1)
            if values.numel() not in (1, batch.num_graphs) or not torch.all(values == values[0]):
                raise ValueError(f"CNN requires uniform {name} metadata for every graph.")
            dimensions.append(int(values[0]))
        height, width = dimensions
        if min(height, width) < 1 or not torch.all(batch.ptr.diff() == height * width):
            raise ValueError("CNN node counts must match grid_H * grid_W.")
        return batch.num_graphs, height, width

    def forward(self, x, edge_index, grid_shape):
        if self.kind == "CNN":
            size, height, width = grid_shape
            x = x.reshape(size, height, width, -1).permute(0, 3, 1, 2).contiguous()
        for layer in self.layers:
            x = layer(x, edge_index) if self.kind == "GraphSAGE" else layer(x)
            x = F.dropout(F.leaky_relu(x, 0.1), self.dropout, self.training)
        if self.kind == "CNN":
            x = x.permute(0, 2, 3, 1).contiguous().reshape(size * height * width, -1)
        return x
