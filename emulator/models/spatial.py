"""Spatial encoders with the same node-in, node-out interface."""

import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv


class SpatialEncoder(nn.Module):
    """GraphSAGE or rectangular-grid CNN preserving one output row per node.
    """
    def __init__(self, in_dim, hidden, depth, dropout, kind, cnn_width=29):
        """Construct spatial layers and their feature widths.

        Args:
            in_dim: Number of input features per node.
            hidden: Number of output features per node; also the intermediate GraphSAGE
                width.
            depth: Number of spatial layers; at least two for GraphSAGE or one for CNN.
            dropout: Dropout probability applied after the activation at every spatial
                layer.
            kind: Spatial operator name: GraphSAGE or CNN.
            cnn_width: Feature-channel width of nonfinal CNN layers; unused by GraphSAGE.

        Returns:
            None. Registers graph convolutions or 3-by-3 padded image convolutions.

        Raises:
            ValueError: The layer count or spatial operator is unsupported.
        """
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
        """Validate uniform CNN grid dimensions for a batched graph collection.

        CNN inputs must store nodes in row-major grid order. This method verifies
        dimensions and node counts; coordinate ordering is an input contract.

        Args:
            batch: PyG Batch exposing x, num_graphs, ptr, and, for CNNs, grid_H and grid_W
                scalars or per-graph values.

        Returns:
            tuple or None: (num_graphs, height, width) for CNNs, or None for GraphSAGE.

        Raises:
            ValueError: CNN dimensions are absent, nonuniform, nonpositive, or inconsistent
                with the number of nodes in each graph.
        """
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
        """Encode node features with the configured spatial operator.

        Args:
            x: Floating-point node features [total_nodes, in_dim]. CNN nodes must follow
                row-major order within each graph.
            edge_index: Integer graph connectivity [2, edges], used only by GraphSAGE.
            grid_shape: (num_graphs, height, width) validated by grid_shape for CNNs;
                ignored by GraphSAGE.

        Returns:
            torch.Tensor: Encoded node features [total_nodes, hidden] in the original
            graph/node order.
        """
        if self.kind == "CNN":
            size, height, width = grid_shape
            x = x.reshape(size, height, width, -1).permute(0, 3, 1, 2).contiguous()
        for layer in self.layers:
            x = layer(x, edge_index) if self.kind == "GraphSAGE" else layer(x)
            x = F.dropout(F.leaky_relu(x, 0.1, inplace=self.kind == "CNN"), self.dropout, self.training)
        if self.kind == "CNN":
            x = x.permute(0, 2, 3, 1).contiguous().reshape(size * height * width, -1)
        return x
