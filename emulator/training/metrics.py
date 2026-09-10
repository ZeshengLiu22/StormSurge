"""Four physical-unit metrics from per-window errors; no prediction diagnostics."""

import math

import numpy as np


METRIC_NAMES = ("rmse_all", "mae_all", "rmse_peak5", "mae_peak5")


def summarize_windows(records, *, validation=False):
    """Rows: sample_id, GT window maximum, window MSE, window MAE.

    Peak metrics count each sample once. Validation keeps sampler padding in
    All and the original FP32/NumPy argsort calculation for Peak.
    """
    if not len(records):
        return dict.fromkeys(METRIC_NAMES)
    # Reconstruct dataset order before selecting peaks, regardless of rank order.
    _, first = np.unique(records[:, 0], return_index=True)
    rows = records[first]
    all_rows = records if validation else rows
    if not np.isfinite(all_rows[:, 1:]).all():
        raise ValueError("Cannot report metrics for nonfinite predictions or targets.")
    peak_rows = rows[:, 1:].astype(np.float32) if validation else rows[:, 1:]
    top = np.argsort(peak_rows[:, 0], kind="quicksort" if validation else "stable")[-max(1, math.ceil(0.05 * len(rows))):]
    return dict(zip(METRIC_NAMES, (float(np.sqrt(all_rows[:, 2].mean())), float(all_rows[:, 3].mean()),
                                  float(np.sqrt(peak_rows[top, 1].mean())), float(peak_rows[top, 2].mean()))))


def format_metrics(label, metrics):
    pairs = [f'{name}={value:.6f}' if value is not None else f'{name}=NA' for name, value in metrics.items()]
    return f'{label} ' + ' '.join(pairs)
