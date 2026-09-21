"""Physical trajectory and true-peak metrics from lightweight per-window records."""

import math

import numpy as np
import torch


# Keep the historical public keys and their numerical reduction paths intact.
METRIC_NAMES = ("rmse_all", "mae_all", "rmse_peak5", "mae_peak5")
PEAK_METRIC_STEMS = (
    "true_peak_rmse", "true_peak_mae", "true_peak_bias",
    "true_peak_underprediction_fraction", "peak_timing_mae_steps",
)
PEAK_METRIC_NAMES = tuple(f"{name}_{population}" for population in ("all", "top5", "event")
                          for name in PEAK_METRIC_STEMS)


@torch.no_grad()
def physical_peak_columns(prediction_phys, target_phys):
    """Three FP64 scalars/window: truth peak, aligned signed error, timing.

    The true target argmax anchors amplitude; the predicted argmax is used
    only for timing. Torch uses the first occurrence for tied peaks. Subtract
    physical values in FP64, independently of trajectory-error reduction.
    """
    true_peak, true_index = target_phys.max(dim=1)
    pred_index = prediction_phys.argmax(dim=1)
    at_true_peak = prediction_phys.gather(1, true_index[:, None]).squeeze(1)
    return torch.stack((true_peak.double(), at_true_peak.double() - true_peak.double(),
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


def _summarize_true_peaks(rows):
    if not len(rows):
        return dict.fromkeys(PEAK_METRIC_STEMS)
    error, timing = rows[:, 4], rows[:, 5]
    return dict(true_peak_rmse=float(np.sqrt(np.mean(error ** 2))),
                true_peak_mae=float(np.mean(np.abs(error))), true_peak_bias=float(error.mean()),
                true_peak_underprediction_fraction=float((error < 0).mean()),
                peak_timing_mae_steps=float(timing.mean()))


def summarize_windows(records, *, validation=False, event_threshold=None):
    """Rows: ID, truth peak, window MSE/MAE, true-peak signed error, timing.

    Six-column records provide 15 canonical peak metrics; four-column records
    provide trajectory metrics only. Peak populations count each sample ID
    once; evaluation All trajectory metrics still include DDP sampler padding.
    A missing TRAIN threshold makes event metrics unavailable, never TEST-fit.
    """
    names = METRIC_NAMES + PEAK_METRIC_NAMES if records.shape[1] == 6 else METRIC_NAMES
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
    if records.shape[1] == 6:
        events = rows[rows[:, 1] > event_threshold] if event_threshold is not None else rows[:0]
        for population, selected in (("all", rows), ("top5", rows[top]), ("event", events)):
            metrics.update({f"{name}_{population}": value for name, value in _summarize_true_peaks(selected).items()})
    return metrics


def format_metrics(label, metrics, *, extended=False):
    """Keep stdout concise; full dictionaries remain in JSONL/checkpoints/reports."""
    names = metrics if extended else (*METRIC_NAMES, "true_peak_rmse_top5")
    pairs = [f'{name}={metrics[name]:.6f}' if metrics[name] is not None else f'{name}=NA'
             for name in names if name in metrics]
    return f'{label} ' + ' '.join(pairs)
