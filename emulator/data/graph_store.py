"""CPU graph storage and deterministic splits by complete year groups."""

from collections import defaultdict
from pathlib import Path
import random

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data


class ForcingGraphStore:
    def __init__(self, root_dir, station_filter=None):
        self.graphs = []
        self.graph_tags = []
        self.year_to_indices = defaultdict(list)
        # Filter filenames before loading: other stations never occupy RAM.
        files = sorted(Path(root_dir).glob("*_graphs.pt"))
        for path in files:
            parts = path.name.removesuffix("_graphs.pt").split("_")
            if len(parts) < 3:
                raise ValueError(f"Invalid graph filename: {path.name}")
            if station_filter is not None and parts[2] != station_filter:
                continue
            year = "_".join(parts[:2])
            for index, graph in enumerate(torch.load(path, map_location="cpu", weights_only=False)):
                self.year_to_indices[year].append(len(self.graphs))
                self.graph_tags.append(f'{"_".join(parts)}_{index}')
                self.graphs.append(graph)
        if not self.graphs:
            raise ValueError(f"No graphs for station {station_filter!r} in {root_dir}.")

    def split(self, train_ratio=0.6, val_ratio=0.2, shuffle_years=False, seed=42,
              future_only=False, future_year_threshold=2030):
        if not 0 < train_ratio < 1 or not 0 < val_ratio < 1 or train_ratio + val_ratio >= 1:
            raise ValueError("Train/val ratios must be positive and sum to less than one.")
        years = sorted(year for year in self.year_to_indices if not future_only
                       or any(int(y) > future_year_threshold for y in year.split("_")))
        if not years:
            raise ValueError("No year groups remain after filtering.")
        if shuffle_years:
            random.Random(seed).shuffle(years)
        count = len(years)
        n_train, n_val = max(1, round(train_ratio * count)), max(1, round(val_ratio * count))
        if count <= 2:
            n_train, n_val = 1, count - 1
        elif n_train + n_val >= count:
            n_train, n_val = count - 2, 1
        groups = {"train": years[:n_train], "val": years[n_train:n_train + n_val],
                  "test": years[n_train + n_val:]}
        return {part: sorted(i for year in selected for i in self.year_to_indices[year])
                for part, selected in groups.items()}


class ForcingGraphView(Dataset):
    def __init__(self, store, indices, history_steps, use_pmean=False):
        self.store = store
        self.indices = list(indices)
        self.window = history_steps + 1
        self.use_pmean = use_pmean
        if self.window < 1:
            raise ValueError("History must be nonnegative.")
        for index in self.indices:
            graph = store.graphs[index]
            history = getattr(graph, "x_hist", None)
            available = history.size(0) if history is not None else 1
            if available < self.window:
                raise ValueError(f"{store.graph_tags[index]} has {available} history steps; need {self.window}.")
            if use_pmean:
                pressure = getattr(graph, "p_mean_hist", None)
                if pressure is None:
                    pressure = getattr(graph, "p_mean_curr", None)
                if pressure is None or torch.as_tensor(pressure).numel() < self.window:
                    raise ValueError(f"{store.graph_tags[index]} lacks the {self.window}-step p_mean history requested by --use_pmean.")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        index = self.indices[item]
        graph = self.store.graphs[index]
        history = getattr(graph, "x_hist", None)
        history = history[-self.window:].permute(1, 0, 2) if history is not None else graph.x.unsqueeze(1)
        data = Data(x=graph.x, x_hist=history, edge_index=graph.edge_index, y=graph.y.view(1, -1),
                    tag=self.store.graph_tags[index], sample_id=torch.tensor([index]))
        for name in ("grid_H", "grid_W"):
            if name in graph:
                data[name] = int(graph[name])
        if self.use_pmean:
            pressure = graph.p_mean_hist if "p_mean_hist" in graph else graph.p_mean_curr
            data.p_mean_hist = torch.as_tensor(pressure).float().reshape(-1)[-self.window:].view(1, -1)
        return data
