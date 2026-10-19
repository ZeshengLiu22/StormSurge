"""CPU graph storage and deterministic splits by complete year groups."""

from collections import defaultdict
from copy import copy
import math
from pathlib import Path
import random

import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data

from .targets import target_timestamps_from_graph, validate_target_timestamps


PRESSURE_MODES = ("anomaly", "raw", "anomaly_plus_mean")


def transform_pressure_features(graph, pressure_mode="anomaly", *, graph_tag=None):
    """Select the unnormalized pressure representation for a stored graph.

    The complete stored history is transformed before history slicing or feature
    normalization. The default anomaly mode accepts legacy graphs without pressure
    metadata and performs no shape or metadata validation.

    Args:
        graph: PyG graph with current features x [N, 5] and chronological history x_hist
            [T, N, 5]. Feature index 2 holds pressure anomalies. Nondefault modes
            require finite absolute pressure means p_mean_hist [T] over the selected
            preprocessing grid, in the same units as the pressure feature; optional
            p_mean_curr must agree with the last mean.
        pressure_mode: anomaly returns the original graph; raw adds the spatial mean to
            pressure channel 2; anomaly_plus_mean appends that mean as a sixth feature.
        graph_tag: Optional source identifier included in validation errors.

    Returns:
        Data: The input object for anomaly mode, or a shallow graph copy with new
        float32 x_hist and x tensors for the selected representation. The current x is
        reconstructed as x_hist[-1]; other graph fields remain shared.

    Raises:
        ValueError: The mode is unsupported or a nondefault mode receives incompatible
            feature tensors or pressure metadata.
    """
    if pressure_mode not in PRESSURE_MODES:
        raise ValueError(f"Unknown pressure_mode {pressure_mode!r}; expected one of {PRESSURE_MODES}.")
    if pressure_mode == "anomaly":
        return graph

    context = f"pressure_mode={pressure_mode!r}, graph {graph_tag or getattr(graph, 'tag', None) or '<unknown>'}"

    def invalid(message):
        """Raise a pressure-validation error with the enclosing graph context.

        Args:
            message: Description of the invalid feature or metadata field.

        Returns:
            Never returns.

        Raises:
            ValueError: Always, including the pressure mode and graph identifier.
        """
        raise ValueError(f"{context}: {message}")

    x, history = getattr(graph, "x", None), getattr(graph, "x_hist", None)
    if not isinstance(x, torch.Tensor) or x.ndim != 2 or x.size(-1) != 5:
        invalid("x must have the original shape [N, 5].")
    if not isinstance(history, torch.Tensor) or history.ndim != 3 or history.size(-1) != 5:
        invalid("x_hist must have the original shape [T, N, 5].")
    if history.size(0) == 0 or history.size(1) != x.size(0):
        invalid("x_hist must be nonempty and its node count must match x.")
    if not x.is_floating_point() or not history.is_floating_point():
        invalid("x and x_hist must have floating-point dtypes.")
    if x.device != history.device:
        invalid("x and x_hist must be on the same device.")

    mean = getattr(graph, "p_mean_hist", None)
    if mean is None:
        invalid("missing required field p_mean_hist (absolute pressure means).")
    if not isinstance(mean, torch.Tensor) or mean.ndim != 1 or mean.size(0) != history.size(0):
        invalid("p_mean_hist must have shape [T], matching the complete x_hist length.")
    if mean.is_complex() or mean.dtype == torch.bool:
        invalid("p_mean_hist must contain real numeric pressure values.")
    mean = mean.to(device=history.device, dtype=torch.float32)
    if not torch.isfinite(mean).all():
        invalid("p_mean_hist must contain finite float32 values.")
    current = getattr(graph, "p_mean_curr", None)
    if current is not None:
        current = torch.as_tensor(current, device=history.device)
        if current.numel() != 1 or current.is_complex() or current.dtype == torch.bool:
            invalid("p_mean_curr must be a finite real scalar.")
        current = current.to(dtype=torch.float32).reshape(())
        if not torch.isfinite(current) or not torch.isclose(current, mean[-1], rtol=1e-6, atol=1e-5):
            invalid("p_mean_curr must be finite and agree with p_mean_hist[-1].")

    if pressure_mode == "raw":
        transformed = history.to(dtype=torch.float32, copy=True)
        transformed[..., 2] += mean[:, None]
    else:
        mean_feature = mean[:, None, None].expand(-1, history.size(1), 1)
        transformed = torch.cat((history.to(dtype=torch.float32), mean_feature), dim=-1)
    result = copy(graph)
    result.x_hist = transformed
    result.x = transformed[-1]
    return result


class ForcingGraphStore:
    """In-memory CPU graphs indexed by station-filtered source year groups.

    Graph filenames begin with year0_year1_station. ``graphs`` and
    ``graph_tags`` share indices; ``year_to_indices`` maps each year-pair string
    to the corresponding graph indices.
    """
    def __init__(self, root_dir, station_filter=None, *, pattern="*graphs.pt", targets_only=False, year_filter=None,
                 pressure_mode="anomaly"):
        """Load selected graph files and build the year-group index.

        Files are deserialized with ``torch.load(weights_only=False)`` and must come
        from a trusted source. File-level station and year filters are applied before
        deserialization; target-only mode still loads one complete source list at a time.

        Args:
            root_dir: Directory containing serialized PyG graph lists.
            station_filter: Exact station field to retain from each filename, or None to
                retain all stations.
            pattern: Path.glob pattern relative to root_dir; files are processed in sorted
                order.
            targets_only: Whether to retain cloned labels and timing/history-length metadata
                while discarding forcing tensors, for target-population audits.
            year_filter: Optional iterable of exact year-pair strings, such as 1979_1980, to
                retain before files are loaded.
            pressure_mode: Pressure representation passed to transform_pressure_features
                when targets_only is false.

        Returns:
            None. Populates graphs, graph_tags, and year_to_indices on the instance.

        Raises:
            ValueError: The pressure mode or filename is invalid, or no selected graphs are
                found.
        """
        if pressure_mode not in PRESSURE_MODES:
            raise ValueError(f"Unknown pressure_mode {pressure_mode!r}; expected one of {PRESSURE_MODES}.")
        self.graphs = []
        self.graph_tags = []
        self.year_to_indices = defaultdict(list)
        # Filter filenames before loading: other stations/years never occupy RAM.
        selected_years = None if year_filter is None else set(year_filter)
        files = sorted(Path(root_dir).glob(pattern))
        for path in files:
            stem = path.name.removesuffix("_graphs.pt") if path.name.endswith("_graphs.pt") else path.stem
            parts = stem.split("_")
            if len(parts) < 3:
                raise ValueError(f"Invalid graph filename: {path.name}")
            if station_filter is not None and parts[2] != station_filter:
                continue
            year = "_".join(parts[:2])
            if selected_years is not None and year not in selected_years:
                continue
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
                else:
                    graph = transform_pressure_features(graph, pressure_mode, graph_tag=f"{path} [{index}]")
                self.graphs.append(graph)
        if not self.graphs:
            raise ValueError(f"No graphs for station {station_filter!r} in {root_dir}.")

    def target_timestamps(self, indices):
        """Collect unique supervised UTC timestamps for selected graph windows.

        Args:
            indices: Iterable of indices into graphs, in the desired output row order.

        Returns:
            numpy.ndarray: int64 Unix seconds with shape [windows, horizons], or [0, 0] for
            an empty selection.

        Raises:
            ValueError: Timing metadata is invalid or selected target blocks overlap.
        """
        rows = [target_timestamps_from_graph(self.graphs[index], self.graph_tags[index]) for index in indices]
        return validate_target_timestamps(np.stack(rows)) if rows else np.empty((0, 0), dtype=np.int64)

    def split(self, train_ratio=0.6, val_ratio=0.2, shuffle_years=False, seed=42,
              future_only=False, future_year_threshold=2030):
        """Partition complete year groups into training, validation, and test sets.

        Group counts are rounded and adjusted to preserve training and, when enough
        groups exist, validation and test populations. A single group with a held-out
        test allocation yields training only; individual graphs are never split across
        partitions within their year group.

        Args:
            train_ratio: Positive fraction of selected year groups assigned to training
                before rounding.
            val_ratio: Positive validation fraction; the sum with train_ratio must be at
                most one, allowing math.isclose tolerance at one.
            shuffle_years: Whether to shuffle the sorted year-group keys before
                partitioning.
            seed: Seed for the local Python random generator used when shuffling.
            future_only: Whether to keep only groups containing a year strictly above
                future_year_threshold.
            future_year_threshold: Exclusive lower year bound used by future_only.

        Returns:
            dict: train, val, and test mapped to sorted graph-index lists. Ratios summing to
            one produce no test set and require at least two groups.

        Raises:
            ValueError: Ratios are invalid, filtering removes every group, or a
                train/validation-only split has fewer than two groups.
        """
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
    """Dataset view exposing a fixed forcing history and timestamped targets.
    """
    def __init__(self, store, indices, history_steps):
        """Validate a graph subset and its requested forcing-history window.

        Args:
            store: ForcingGraphStore with full forcing graphs and unique supervised
                timestamps.
            indices: Iterable of graph indices defining the dataset order.
            history_steps: Nonnegative number of stored forcing steps before the current
                step; each sample includes history_steps + 1 steps.

        Returns:
            None. Stores the selected indices and target timestamps, and initializes
            optional episode_peak_targets to None.

        Raises:
            ValueError: History is negative or unavailable, or selected target windows have
                invalid or overlapping timestamps.
        """
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
        """Return the number of selected forecast windows.

        Returns:
            int: Length of the stored index list.
        """
        return len(self.indices)

    def __getitem__(self, item):
        """Assemble one graph window with history in node-major order.

        Feature and edge tensors may share storage with the source graph.
        Timestamp and episode arrays are copied before conversion to tensors.

        Args:
            item: Integer position within this view, indexing its selected graph list.

        Returns:
            Data: Features x [N, F], x_hist [N, history_steps + 1, F], graph edges, physical
            targets y [1, K], target_timestamps [1, K] in Unix seconds, tag, and store-level
            sample_id [1]. Grid dimensions and canonical episode labels are included when
            available.
        """
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
