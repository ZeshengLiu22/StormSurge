"""TRAIN statistics with the original sampling, device and arithmetic semantics."""

import numpy as np
import torch
import torch.distributed as dist


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


def fit_loss_thresholds(store, indices, tail_frac=0.05, wmse_percentile=95.0, wmse_use_abs=True,
                        exceedance_percentile=95.0):
    """Fit independent tail/weighted-loss thresholds and a strict dual event on TRAIN."""
    if not indices:
        raise ValueError("Cannot fit loss thresholds on an empty training split.")
    if not 0 < tail_frac < 1 or not 0 <= wmse_percentile <= 100 or not 0 < exceedance_percentile < 100:
        raise ValueError("Invalid TRAIN loss percentile settings.")
    y = torch.stack([store.graphs[i].y.reshape(-1).float() for i in indices]).numpy()
    if not np.isfinite(y).all():
        raise ValueError("TRAIN labels must be finite to fit event thresholds.")
    peaks = y.max(axis=1)
    peak_threshold = float(np.percentile(peaks, 100 * (1 - tail_frac)))
    wmse_threshold = float(np.percentile(np.abs(y) if wmse_use_abs else y, wmse_percentile))
    tau_phys = float(np.percentile(peaks, exceedance_percentile))
    event_count = int(np.count_nonzero(peaks.astype(np.float64) > tau_phys))
    return dict(tail_threshold=peak_threshold, wmse_threshold=wmse_threshold, tau_phys=tau_phys,
                event_prior=event_count / len(indices), event_count=event_count,
                train_windows=len(indices), exceedance_percentile=exceedance_percentile)
