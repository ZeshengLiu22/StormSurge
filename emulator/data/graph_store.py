"""CPU graph storage and deterministic splits by complete year groups."""

from collections import defaultdict
import math
from pathlib import Path
import random

import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data

from .targets import target_timestamps_from_graph, validate_target_timestamps


class ForcingGraphStore:
    def __init__(self, root_dir, station_filter=None, *, pattern="*graphs.pt", targets_only=False):
        self.graphs = []
        self.graph_tags = []
        self.year_to_indices = defaultdict(list)
        # Filter filenames before loading: other stations never occupy RAM.
        files = sorted(Path(root_dir).glob(pattern))
        for path in files:
            stem = path.name.removesuffix("_graphs.pt") if path.name.endswith("_graphs.pt") else path.stem
            parts = stem.split("_")
            if len(parts) < 3:
                raise ValueError(f"Invalid graph filename: {path.name}")
            if station_filter is not None and parts[2] != station_filter:
                continue
            year = "_".join(parts[:2])
            for index, graph in enumerate(torch.load(path, map_location="cpu", weights_only=False)):
                self.year_to_indices[year].append(len(self.graphs))
                self.graph_tags.append(f'{"_".join(parts)}_{index}')
                if targets_only:
                    # Data-only audits stream a source file at a time and retain
                    # labels/temporal evidence without retaining forcing tensors.
                    target = Data(y=graph.y.detach().clone(),
                                  source_history_steps=int(graph.x_hist.size(0)) if getattr(graph, "x_hist", None) is not None else 1)
                    for name in ("center_time", "target_timestamps", "time_index", "hour_start", "max_history_hours", "nc", "nc_tide"):
                        value = getattr(graph, name, None)
                        if value is not None:
                            target[name] = value.detach().clone() if isinstance(value, torch.Tensor) else value
                    graph = target
                self.graphs.append(graph)
        if not self.graphs:
            raise ValueError(f"No graphs for station {station_filter!r} in {root_dir}.")

    def target_timestamps(self, indices):
        """Return [N,K] UTC Unix seconds and reject overlapping target blocks."""
        rows = [target_timestamps_from_graph(self.graphs[index], self.graph_tags[index]) for index in indices]
        return validate_target_timestamps(np.stack(rows)) if rows else np.empty((0, 0), dtype=np.int64)

    def split(self, train_ratio=0.6, val_ratio=0.2, shuffle_years=False, seed=42,
              future_only=False, future_year_threshold=2030):
        ratio_sum = train_ratio + val_ratio
        no_heldout_test = math.isclose(ratio_sum, 1.0)
        if not train_ratio > 0 or not val_ratio > 0 or (ratio_sum > 1 and not no_heldout_test):
            raise ValueError("Train/val ratios must be positive and sum to at most one.")
        years = sorted(year for year in self.year_to_indices if not future_only
                       or any(int(y) > future_year_threshold for y in year.split("_")))
        if not years:
            raise ValueError("No year groups remain after filtering.")
        if shuffle_years:
            random.Random(seed).shuffle(years)
        count = len(years)
        if no_heldout_test:
            if count < 2:
                raise ValueError("TRAIN/VAL-only splitting requires at least two year groups.")
            n_train = min(count - 1, max(1, round(train_ratio * count)))
            groups = {"train": years[:n_train], "val": years[n_train:], "test": []}
        else:
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
    def __init__(self, store, indices, history_steps):
        self.store = store
        self.indices = list(indices)
        self.window = history_steps + 1
        if self.window < 1:
            raise ValueError("History must be nonnegative.")
        for index in self.indices:
            graph = store.graphs[index]
            history = getattr(graph, "x_hist", None)
            available = history.size(0) if history is not None else 1
            if available < self.window:
                raise ValueError(f"{store.graph_tags[index]} has {available} history steps; need {self.window}.")
        self.target_timestamps = store.target_timestamps(self.indices)
        self.episode_peak_targets = None

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        index = self.indices[item]
        graph = self.store.graphs[index]
        history = getattr(graph, "x_hist", None)
        history = history[-self.window:].permute(1, 0, 2) if history is not None else graph.x.unsqueeze(1)
        data = Data(x=graph.x, x_hist=history, edge_index=graph.edge_index, y=graph.y.view(1, -1),
                    tag=self.store.graph_tags[index], sample_id=torch.tensor([index]),
                    target_timestamps=torch.from_numpy(self.target_timestamps[item].copy()).view(1, -1))
        if self.episode_peak_targets is not None:
            targets = self.episode_peak_targets
            data.episode_id = torch.from_numpy(targets.episode_id[item].copy()).view(1, -1)
            data.is_episode_gt_peak = torch.from_numpy(targets.is_episode_gt_peak[item].copy()).view(1, -1)
        for name in ("grid_H", "grid_W"):
            if name in graph:
                data[name] = int(graph[name])
        return data
