"""Physical trajectory and direct peak metrics from lightweight per-window records."""

import math

import numpy as np
import torch


# Keep the historical public keys and their numerical reduction paths intact.
METRIC_NAMES = ("rmse_all", "mae_all", "rmse_peak5", "mae_peak5")
PEAK_METRIC_STEMS = (
    "peak_magnitude_rmse", "peak_magnitude_mae", "peak_bias",
    "peak_underprediction_fraction", "peak_underprediction_mean",
    "peak_overprediction_fraction", "peak_overprediction_mean",
    "true_peak_point_rmse", "true_peak_point_mae", "true_peak_point_bias",
    "peak_timing_mae_steps",
)
PEAK_METRIC_NAMES = tuple(f"{name}_{population}" for population in ("all", "top5", "event")
                          for name in PEAK_METRIC_STEMS)


@torch.no_grad()
def physical_peak_columns(prediction_phys, target_phys):
    """Four FP64 scalars/window: truth peak, signed peak/point errors, timing.

    Hard maxima only. Torch max returns the first occurrence for tied peaks.
    Subtract physical values in FP64 for the new metrics, independent of AMP
    and of the existing FP32 trajectory-error reduction. Timing is in steps.
    """
    true_peak, true_index = target_phys.max(dim=1)
    pred_peak, pred_index = prediction_phys.max(dim=1)
    at_true_peak = prediction_phys.gather(1, true_index[:, None]).squeeze(1)
    return torch.stack((true_peak.double(), pred_peak.double() - true_peak.double(),
                        at_true_peak.double() - true_peak.double(),
                        (pred_index - true_index).abs().double()), dim=1)


def unique_rows_and_top5_indices(records, *, validation=False):
    """Restore sample-ID order and return the SAME legacy top5 membership.

    Evaluation uses FP32 truth peaks and NumPy quicksort. Training/inference
    summaries retain the existing input precision and stable sort convention.
    """
    _, first = np.unique(records[:, 0], return_index=True)
    rows = records[first]
    peaks = rows[:, 1].astype(np.float32) if validation else rows[:, 1]
    top = np.argsort(peaks, kind="quicksort" if validation else "stable")[-max(1, math.ceil(0.05 * len(rows))):]
    return rows, top


def _summarize_direct_peaks(rows):
    if not len(rows):
        return dict.fromkeys(PEAK_METRIC_STEMS)
    error, point, timing = rows[:, 4], rows[:, 5], rows[:, 6]
    under, over = error < 0, error > 0
    return dict(peak_magnitude_rmse=float(np.sqrt(np.mean(error ** 2))),
                peak_magnitude_mae=float(np.mean(np.abs(error))), peak_bias=float(error.mean()),
                peak_underprediction_fraction=float(under.mean()),
                peak_underprediction_mean=float(-error[under].mean()) if under.any() else None,
                peak_overprediction_fraction=float(over.mean()),
                peak_overprediction_mean=float(error[over].mean()) if over.any() else None,
                true_peak_point_rmse=float(np.sqrt(np.mean(point ** 2))),
                true_peak_point_mae=float(np.mean(np.abs(point))), true_peak_point_bias=float(point.mean()),
                peak_timing_mae_steps=float(timing.mean()))


def summarize_windows(records, *, validation=False, event_threshold=None):
    """Rows: ID, truth peak, window MSE/MAE, peak error, point error, timing.

    Four-column legacy callers still receive the original four metrics. New
    seven-column records add 33 peak metrics. Every new population counts each
    sample ID once; legacy evaluation All still includes DDP sampler padding.
    A missing TRAIN threshold makes event metrics unavailable, never TEST-fit.
    """
    names = METRIC_NAMES + PEAK_METRIC_NAMES if records.shape[1] >= 7 else METRIC_NAMES
    if event_threshold is not None and not math.isfinite(event_threshold):
        raise ValueError("Peak event metrics require a finite saved TRAIN threshold or None.")
    if not len(records):
        return dict.fromkeys(names)
    rows, top = unique_rows_and_top5_indices(records, validation=validation)
    all_rows = records if validation else rows
    if not np.isfinite(all_rows[:, 1:]).all():
        raise ValueError("Cannot report metrics for nonfinite predictions or targets.")
    peak_rows = rows[:, 1:4].astype(np.float32) if validation else rows[:, 1:4]
    metrics = dict(zip(METRIC_NAMES, (float(np.sqrt(all_rows[:, 2].mean())), float(all_rows[:, 3].mean()),
                                    float(np.sqrt(peak_rows[top, 1].mean())), float(peak_rows[top, 2].mean()))))
    if records.shape[1] >= 7:
        events = rows[rows[:, 1] > event_threshold] if event_threshold is not None else rows[:0]
        for population, selected in (("all", rows), ("top5", rows[top]), ("event", events)):
            metrics.update({f"{name}_{population}": value for name, value in _summarize_direct_peaks(selected).items()})
    return metrics


def format_metrics(label, metrics, *, extended=False):
    """Keep stdout concise; full dictionaries remain in JSONL/checkpoints/reports."""
    names = metrics if extended else (*METRIC_NAMES, "peak_magnitude_rmse_top5")
    pairs = [f'{name}={metrics[name]:.6f}' if metrics[name] is not None else f'{name}=NA'
             for name in names if name in metrics]
    return f'{label} ' + ' '.join(pairs)
