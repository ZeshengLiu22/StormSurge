"""TRAIN statistics with the original sampling, device and arithmetic semantics."""

import numpy as np
import torch
import torch.distributed as dist


def fit_statistics(store, indices, x_norm="zscore", p_lo=1.0, p_hi=99.0,
                   nodes_per_graph=256, seed=42, use_pmean=False, history_steps=0,
                   device=None):
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
        if x_norm not in ("robust", "mag") or not 0 <= p_lo < p_hi <= 100:
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

    if use_pmean:
        fitted = torch.zeros(2, dtype=torch.float32, device=device)
        if rank == 0:
            pressure = []
            for i in indices:
                graph = store.graphs[i]
                value = graph.p_mean_hist if "p_mean_hist" in graph else graph.p_mean_curr
                pressure.append(torch.as_tensor(value).detach().cpu().reshape(-1)[-(history_steps + 1):].float().numpy())
            values = np.concatenate(pressure)
            if x_norm == "zscore":
                center, scale = values.mean(dtype=np.float64), values.std(dtype=np.float64)
            elif x_norm == "robust":
                low, high = np.percentile(values, [p_lo, p_hi])
                center, scale = .5 * (float(low) + float(high)), .5 * (float(high) - float(low))
            else:
                center, scale = 0., np.percentile(np.abs(values), p_hi)
            fitted.copy_(torch.tensor([center, max(scale, 1e-6)], dtype=torch.float32, device=device))
        if distributed:
            dist.broadcast(fitted, src=0)
        stats.update(pmean_center=fitted[0].cpu(), pmean_scale=fitted[1].cpu())
    return stats


def fit_loss_thresholds(store, indices, tail_frac=0.05, wmse_percentile=95.0, wmse_use_abs=True):
    y = torch.stack([store.graphs[i].y.reshape(-1).float() for i in indices]).numpy()
    peak_threshold = float(np.percentile(y.max(axis=1), 100 * (1 - tail_frac)))
    wmse_threshold = float(np.percentile(np.abs(y) if wmse_use_abs else y, wmse_percentile))
    return peak_threshold, wmse_threshold
