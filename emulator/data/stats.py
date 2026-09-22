"""TRAIN statistics with the original sampling, device and arithmetic semantics."""

import numpy as np
import torch
import torch.distributed as dist

from .targets import supervised_targets, threshold_population


def fit_statistics(store, indices, x_norm="zscore", p_lo=1.0, p_hi=99.0,
                   nodes_per_graph=256, seed=42, device=None):
    """All ranks participate; return CPU statistics for checkpoint storage.

    X z-score uses FP32 squares, FP64 sums, then FP32 moments. Y uses FP64
    squares/sums and FP32 moments. These orders match the original trainer.
    Percentiles are fitted on rank zero and broadcast to the other ranks.
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
        variance = (square / count).float() - center ** 2
        scale = torch.sqrt(variance + 1e-6)
    else:
        if (x_norm not in ("robust", "mag") or not 0 < p_hi <= 100
                or (x_norm == "robust" and not 0 <= p_lo < p_hi)):
            raise ValueError("Invalid feature percentile configuration.")
        # Zero and negative values select the original automatic sample size.
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
                low, high = np.percentile(values, [p_lo, p_hi], axis=0, overwrite_input=True).astype(np.float32)
                center, scale = .5 * (low + high), .5 * (high - low)
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
    count = torch.zeros((), dtype=torch.float64, device=device)
    for i in local_indices:
        y = store.graphs[i].y.reshape(-1).to(device=device, dtype=torch.float64)
        total += y
        square += y ** 2
        count += 1.
    if distributed:
        for value in (total, square, count):
            dist.all_reduce(value)
    center = (total / count).float()
    variance = (square / count).float() - center ** 2
    stats.update(y_mean=center.cpu(), y_std=torch.sqrt(variance + 1e-6).cpu())

    return stats


def fit_loss_thresholds(store, indices, exceedance_percentile=95.0, stats=None):
    """Fit the one canonical threshold from unique TRAIN physical target hours.

    ``indices`` must contain only the source TRAIN split. All validation, test and
    transfer datasets reuse the returned threshold without calling this fitter.
    Normalization is per horizon, so ``tau_normalized`` is a length-K vector.
    """
    if not 0 < exceedance_percentile < 100:
        raise ValueError("TRAIN exceedance_percentile must lie strictly between 0 and 100.")
    labels, timestamps = supervised_targets(store, indices)
    chronological = np.argsort(timestamps, axis=None, kind="stable")
    hourly_values = labels.reshape(-1)[chronological]
    tau = float(np.quantile(hourly_values, exceedance_percentile / 100., method="linear"))
    population = threshold_population(labels, timestamps, tau)
    normalized = None
    if stats is not None:
        center = torch.as_tensor(stats["y_mean"], dtype=torch.float64).detach().cpu().numpy()
        scale = torch.as_tensor(stats["y_std"], dtype=torch.float64).detach().cpu().numpy()
        if center.size != labels.shape[1] or scale.size != labels.shape[1] or not np.isfinite(center).all() or not np.isfinite(scale).all() or np.any(scale <= 0):
            raise ValueError("Target normalization requires one finite mean and positive scale per horizon.")
        normalized = ((tau - center) / scale).reshape(-1).tolist()
    return dict(metric_schema="hourly_q95_v1", threshold_schema="train_hourly_q95_v1",
                exceedance_percentile=float(exceedance_percentile), quantile_method="linear",
                tau_physical=tau, tau_normalized=normalized, fitted_on="train",
                event_prior=population["event_window_rate"], train_windows=len(labels),
                **{f"train_{key}": value for key, value in population.items()})
