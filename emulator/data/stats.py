"""Training-only feature statistics, target moments, and physical thresholds."""

import numpy as np
import torch
import torch.distributed as dist

from .targets import supervised_targets, threshold_population


def fit_statistics(
    store, indices, x_norm="zscore", p_lo=1.0, p_hi=99.0, nodes_per_graph=256, seed=42, device=None
):
    """Fit feature normalization and per-horizon target moments on training graphs.

    Every distributed rank must call this function. Z-score feature statistics
    use current forcing only, with FP32 squares and FP64 reductions. Target moments
    use FP64 squares and reductions. Both moment paths cast to FP32 before computing
    variance and add 1e-6 before the square root. Percentiles are fitted on rank zero,
    scales are floored at 1e-6, and the result is broadcast. Stored history features
    do not contribute additional samples.

    Args:
        store: Graph store with unnormalized current features x [nodes, features] and
            physical target vectors y [horizons].
        indices: Nonempty sequence of source-training graph indices, identical across
            distributed ranks.
        x_norm: zscore uses all current node features; robust uses the midpoint and
            half-width between p_lo and p_hi; mag uses zero center and the p_hi
            percentile of absolute features.
        p_lo: Lower percentile in [0, p_hi), used only for robust normalization.
        p_hi: Upper percentile in (0, 100], used by robust and mag normalization.
        nodes_per_graph: Maximum nodes sampled without replacement per graph for
            percentile modes; nonpositive values select 256. Ignored by zscore.
        seed: NumPy seed used on rank zero for percentile-mode node sampling.
        device: Torch device for moment accumulation and distributed collectives;
            defaults to CPU and must support the active distributed backend.

    Returns:
        dict: CPU float32 tensors x_center and x_scale with shape [features], plus
        y_mean and y_std with shape [horizons]. Target statistics use finite labels
        independently at each horizon.

    Raises:
        ValueError: The training split is empty, percentile settings are invalid, or a
            target horizon has no finite training values.
    """
    if not indices:
        raise ValueError("Cannot fit statistics on an empty training split.")
    device = torch.device("cpu") if device is None else torch.device(device)
    distributed = dist.is_initialized()
    rank, world = (dist.get_rank(), dist.get_world_size()) if distributed else (0, 1)
    local_indices = indices[rank::world]
    features = store.graphs[indices[0]].x.size(-1)
    if x_norm == "zscore":
        total = torch.zeros(features, dtype=torch.float64, device=device)
        square = torch.zeros_like(total)
        count = torch.zeros((), dtype=torch.float64, device=device)
        for i in local_indices:
            x = store.graphs[i].x.to(device=device, dtype=torch.float32, non_blocking=True)
            total += x.sum(dim=0, dtype=torch.float64)
            square += (x * x).sum(dim=0, dtype=torch.float64)
            count += x.size(0)
        if distributed:
            for value in (total, square, count):
                dist.all_reduce(value)
        center = (total / count).float()
        variance = (square / count).float() - center**2
        scale = torch.sqrt(variance + 1e-6)
    else:
        if (
            x_norm not in ("robust", "mag")
            or not 0 < p_hi <= 100
            or (x_norm == "robust" and not 0 <= p_lo < p_hi)
        ):
            raise ValueError("Invalid feature percentile configuration.")
        # Nonpositive sample limits use the default of 256 nodes per graph.
        nodes_per_graph = int(nodes_per_graph) if nodes_per_graph > 0 else 256
        fitted = torch.zeros((2, features), dtype=torch.float32, device=device)
        if rank == 0:
            rng = np.random.default_rng(seed)
            sampled = []
            for i in indices:
                x = store.graphs[i].x.detach().cpu().numpy()
                if len(x) > nodes_per_graph:
                    x = x[rng.choice(len(x), size=nodes_per_graph, replace=False)]
                sampled.append(x.astype(np.float32, copy=False))
            values = np.concatenate(sampled)
            if x_norm == "robust":
                low, high = np.percentile(
                    values, [p_lo, p_hi], axis=0, overwrite_input=True
                ).astype(np.float32)
                center, scale = 0.5 * (low + high), 0.5 * (high - low)
            else:
                center = np.zeros(features, dtype=np.float32)
                scale = np.percentile(np.abs(values), p_hi, axis=0, overwrite_input=True)
            scale = np.maximum(scale, 1e-6).astype(np.float32)
            fitted.copy_(torch.as_tensor(np.stack((center, scale)), device=device))
        if distributed:
            dist.broadcast(fitted, src=0)
        center, scale = fitted[0], fitted[1]
    stats = {"x_center": center.cpu(), "x_scale": scale.cpu()}

    horizons = store.graphs[indices[0]].y.numel()
    total = torch.zeros(horizons, dtype=torch.float64, device=device)
    square = torch.zeros_like(total)
    count = torch.zeros_like(total)
    for i in local_indices:
        y = store.graphs[i].y.reshape(-1).to(device=device, dtype=torch.float64)
        finite = torch.isfinite(y)
        y = torch.where(finite, y, 0.0)
        total += y
        square += y**2
        count += finite
    if distributed:
        for value in (total, square, count):
            dist.all_reduce(value)
    if (count == 0).any():
        raise ValueError("Every target horizon requires at least one finite TRAIN target.")
    center = (total / count).float()
    variance = (square / count).float() - center**2
    stats.update(y_mean=center.cpu(), y_std=torch.sqrt(variance + 1e-6).cpu())

    return stats


def fit_loss_thresholds(store, indices, exceedance_percentile=95.0):
    """Fit one physical threshold from complete, unique training target windows.

    Any window containing a NaN target is excluded in full; infinite targets
    are rejected before exclusion. Timestamp uniqueness is also validated before
    exclusion, and every remaining target hour contributes once to the fitted quantile.

    Args:
        store: Source-training graph store containing physical target labels and
            explicit target timestamps or center times.
        indices: Nonempty selection from the source-training split only. Validation,
            test, and transfer splits must reuse the fitted result.
        exceedance_percentile: Percentile strictly between 0 and 100, using NumPy linear
            quantile interpolation; 95 selects the standard training Q95 threshold.

    Returns:
        dict: Threshold schema, quantile settings, tau_physical in target units,
        train_windows, and training target-hour, strict-exceedance, event-window, and
        episode population counts/rates.

    Raises:
        ValueError: The percentile or target metadata is invalid, windows overlap, or no
            fully finite training window remains.
    """
    if not 0 < exceedance_percentile < 100:
        raise ValueError("TRAIN exceedance_percentile must lie strictly between 0 and 100.")
    labels, timestamps = supervised_targets(store, indices)
    window_valid = np.isfinite(labels).all(axis=1)
    labels, timestamps = labels[window_valid], timestamps[window_valid]
    if not len(labels):
        raise ValueError("Cannot fit the TRAIN threshold without a fully finite target window.")
    chronological = np.argsort(timestamps, axis=None, kind="stable")
    hourly_values = labels.reshape(-1)[chronological]
    tau = float(np.quantile(hourly_values, exceedance_percentile / 100.0, method="linear"))
    population = threshold_population(labels, timestamps, tau)
    return dict(
        metric_schema="hourly_q95_v1",
        threshold_schema="train_hourly_q95_v1",
        exceedance_percentile=float(exceedance_percentile),
        quantile_method="linear",
        tau_physical=tau,
        fitted_on="train",
        train_windows=len(labels),
        **{f"train_{key}": value for key, value in population.items()},
    )
