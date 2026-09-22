"""Canonical physical-unit forecast metrics using one fixed TRAIN threshold.

Forecast arrays are [window, target horizon]. Empty regression populations and
undefined rates are represented by None, never NaN. Rates are fractions; keys
ending in ``_under_pct`` or ``_under_*mm_pct`` are percentages (0 to 100).
"""

import math

import numpy as np
import torch


METRIC_GROUPS = {
    "Overall": ("all_rmse", "all_mae"),
    "Extreme hours": (
        "extreme_hour_n", "extreme_hour_rate", "exceedance_rmse", "exceedance_mae",
        "exceedance_bias", "exceedance_under_pct", "exceedance_under_5mm_pct",
        "exceedance_under_10mm_pct", "exceedance_under_20mm_pct",
        "exceedance_precision", "exceedance_recall", "exceedance_f1",
    ),
    "Event windows": ("event_window_n", "event_window_rate", "event_window_rmse", "event_window_mae"),
    "Window peaks": ("window_peak_rmse", "window_peak_mae", "window_peak_bias"),
    "GT-aligned peaks": (
        "gt_aligned_peak_rmse", "gt_aligned_peak_mae", "gt_aligned_peak_bias",
        "gt_aligned_peak_under_pct", "gt_aligned_peak_under_5mm_pct",
        "gt_aligned_peak_under_10mm_pct", "gt_aligned_peak_under_20mm_pct",
    ),
    "Peak timing": ("peak_timing_mae_steps", "peak_timing_mae_hours"),
    "Event episodes": (
        "episode_n", "episode_peak_rmse", "episode_peak_mae", "episode_peak_bias",
        "episode_gt_aligned_peak_rmse", "episode_gt_aligned_peak_mae",
        "episode_gt_aligned_peak_bias", "episode_gt_aligned_peak_under_pct",
        "episode_peak_timing_mae_hours", "episode_detection_recall",
        "episode_excess_area_mae", "episode_excess_area_bias",
    ),
}
METRIC_KEYS = tuple(key for group in METRIC_GROUPS.values() for key in group)
EPOCH_METRIC_KEYS = tuple(key for name, group in METRIC_GROUPS.items() if name != "Event episodes" for key in group)
EPISODE_METRIC_KEYS = METRIC_GROUPS["Event episodes"]
METRIC_LABELS = dict(zip(METRIC_KEYS, (
    "AllRMSE", "AllMAE", "ExtremeHourN", "ExtremeHourRate", "ExceedanceRMSE",
    "ExceedanceMAE", "ExceedanceBias", "ExceedanceUnder%", "ExceedanceUnder5mm%",
    "ExceedanceUnder10mm%", "ExceedanceUnder20mm%", "ExceedancePrecision",
    "ExceedanceRecall", "ExceedanceF1", "EventWindowN", "EventWindowRate",
    "EventWindowRMSE", "EventWindowMAE", "WindowPeakRMSE", "WindowPeakMAE",
    "WindowPeakBias", "GTAlignedPeakRMSE", "GTAlignedPeakMAE", "GTAlignedPeakBias",
    "GTAlignedPeakUnder%", "GTAlignedPeakUnder5mm%", "GTAlignedPeakUnder10mm%",
    "GTAlignedPeakUnder20mm%", "PeakTimingMAESteps", "PeakTimingMAEHours",
    "EpisodeN", "EpisodePeakRMSE", "EpisodePeakMAE", "EpisodePeakBias",
    "EpisodeGTAlignedPeakRMSE", "EpisodeGTAlignedPeakMAE", "EpisodeGTAlignedPeakBias",
    "EpisodeGTAlignedPeakUnder%", "EpisodePeakTimingMAEHours", "EpisodeDetectionRecall",
    "EpisodeExcessAreaMAE", "EpisodeExcessAreaBias",
)))


def _numpy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _threshold(value):
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("Metrics require a finite tau_physical fitted on TRAIN target hours.")
    return value


def _values(value, name):
    array = _numpy(value).astype(np.float64)
    if array.ndim != 2 or array.shape[1] == 0:
        raise ValueError(f"{name} must have shape [windows, horizons] with at least one horizon.")
    if not np.isfinite(array).all():
        raise ValueError(f"Cannot evaluate nonfinite {name}.")
    return array


def _mean(value):
    return float(np.mean(value)) if np.size(value) else None


def _errors(error, prefix, *, bias=True):
    result = {f"{prefix}_rmse": float(np.sqrt(np.mean(error ** 2))) if error.size else None,
              f"{prefix}_mae": _mean(np.abs(error))}
    if bias:
        result[f"{prefix}_bias"] = _mean(error)
    return result


def _under(prediction, truth, prefix, *, tolerances=True):
    limits = (("", 0.), ("_5mm", .005), ("_10mm", .010), ("_20mm", .020)) if tolerances else (("", 0.),)
    return {f"{prefix}_under{label}_pct": float(100 * np.mean(prediction < truth - tolerance)) if truth.size else None
            for label, tolerance in limits}


def _timestamp_seconds(timestamps, shape):
    values = _numpy(timestamps)
    if values.shape != shape:
        raise ValueError(f"target_timestamps shape {values.shape} must equal target shape {shape}.")
    if np.issubdtype(values.dtype, np.number):
        if not np.isfinite(values).all() or not np.equal(values, np.floor(values)).all():
            raise ValueError("Numeric target timestamps must be finite integer UNIX seconds.")
        return values.astype(np.int64)
    try:
        dates = values.astype("datetime64[ns]")
    except (TypeError, ValueError) as error:
        raise ValueError("Target timestamps must be UNIX seconds, datetime64, or ISO datetime strings.") from error
    if np.isnat(dates).any():
        raise ValueError("Target timestamps must not contain NaT/missing values.")
    nanos = dates.astype(np.int64)
    if np.any(nanos % 1_000_000_000):
        raise ValueError("Target timestamps must have whole-second precision.")
    return nanos // 1_000_000_000


def _split_labels(split_ids, shape):
    if split_ids is None:
        return np.full(shape, "evaluation", dtype=str)
    labels = _numpy(split_ids)
    if labels.shape == (shape[0],):
        labels = np.repeat(labels[:, None], shape[1], axis=1)
    if labels.shape != shape:
        raise ValueError("split_ids must contain one split per window or per target hour.")
    return labels.astype(str)


def gt_event_episodes(y_true, target_timestamps, tau_physical, *, split_ids=None):
    """Return chronologically ordered flat target indices for each GT episode.

    Within each split, timestamps must be unique. Episodes join strict GT
    exceedances separated by exactly 3600 seconds, including across forecast
    blocks. A non-extreme hour, missing hour, or split boundary ends an episode.
    Predictions never participate in defining this population.
    """
    truth = _values(y_true, "y_true")
    tau = _threshold(tau_physical)
    times = _timestamp_seconds(target_timestamps, truth.shape).ravel()
    labels = _split_labels(split_ids, truth.shape).ravel()
    truth = truth.ravel()
    episodes = []
    for label in np.unique(labels):
        indices = np.flatnonzero(labels == label)
        indices = indices[np.argsort(times[indices], kind="stable")]
        sorted_times = times[indices]
        duplicate = np.flatnonzero(np.diff(sorted_times) == 0)
        if duplicate.size:
            timestamp = np.datetime64(int(sorted_times[duplicate[0]]), "s")
            raise ValueError(f"Duplicate target timestamp {timestamp} within split {label!r}; supervised target hours must occur exactly once.")
        extreme = indices[truth[indices] > tau]
        if extreme.size:
            breaks = np.flatnonzero(np.diff(times[extreme]) != 3600) + 1
            episodes.extend(np.split(extreme, breaks))
    return episodes


def _episode_metrics(prediction, truth, timestamps, tau, split_ids):
    episodes = gt_event_episodes(truth, timestamps, tau, split_ids=split_ids)
    times = _timestamp_seconds(timestamps, truth.shape).ravel()
    prediction, truth = prediction.ravel(), truth.ravel()
    peak_errors, aligned_predictions, aligned_truth, timings, detections, area_errors = [], [], [], [], [], []
    for indices in episodes:
        target, pred = truth[indices], prediction[indices]
        gt_index, pred_index = int(np.argmax(target)), int(np.argmax(pred))
        peak_errors.append(pred[pred_index] - target[gt_index])
        aligned_predictions.append(pred[gt_index])
        aligned_truth.append(target[gt_index])
        timings.append(abs(int(times[indices[pred_index]]) - int(times[indices[gt_index]])) / 3600.)
        detections.append(bool(np.any(pred > tau)))
        area_errors.append(float(np.maximum(pred - tau, 0).sum() - np.maximum(target - tau, 0).sum()))
    aligned_predictions, aligned_truth = np.asarray(aligned_predictions), np.asarray(aligned_truth)
    return dict(episode_n=len(episodes),
                **_errors(np.asarray(peak_errors), "episode_peak"),
                **_errors(aligned_predictions - aligned_truth, "episode_gt_aligned_peak"),
                **_under(aligned_predictions, aligned_truth, "episode_gt_aligned_peak", tolerances=False),
                episode_peak_timing_mae_hours=_mean(timings),
                episode_detection_recall=_mean(detections),
                episode_excess_area_mae=_mean(np.abs(area_errors)),
                episode_excess_area_bias=_mean(area_errors))


def evaluate_metrics(y_pred, y_true, tau_physical, *, target_timestamps=None,
                     split_ids=None, include_leadwise=True, target_interval_hours=1.):
    """Evaluate numpy/torch [N,K] physical predictions against fixed TRAIN tau.

    Epoch metrics need only arrays and tau. Supplying target timestamps enables
    final episode metrics and strict per-split timestamp uniqueness validation.
    Peak and timing metrics use GT Event Windows and first argmax on ties.
    Episode integrals use one hour per extreme target. Lead indices start at 0.
    """
    prediction, truth = _values(y_pred, "y_pred"), _values(y_true, "y_true")
    if prediction.shape != truth.shape:
        raise ValueError("y_pred and y_true must have identical [windows, horizons] shapes.")
    tau = _threshold(tau_physical)
    if not math.isfinite(target_interval_hours) or target_interval_hours <= 0:
        raise ValueError("target_interval_hours must be positive and finite.")
    if split_ids is not None and target_timestamps is None:
        raise ValueError("split_ids require target_timestamps for split-isolated episode evaluation.")
    error = prediction - truth
    extreme = truth > tau
    predicted_extreme = prediction > tau
    events = extreme.any(axis=1)
    event_truth, event_pred = truth[events], prediction[events]
    gt_index, pred_index = event_truth.argmax(axis=1), event_pred.argmax(axis=1)
    rows = np.arange(len(event_truth))
    aligned_truth, aligned_pred = event_truth[rows, gt_index], event_pred[rows, gt_index]
    peak_error = event_pred[rows, pred_index] - aligned_truth
    extreme_n, predicted_n, true_positive = int(extreme.sum()), int(predicted_extreme.sum()), int((extreme & predicted_extreme).sum())
    timing = _mean(np.abs(pred_index - gt_index))
    metrics = dict(
        **_errors(error, "all", bias=False),
        extreme_hour_n=extreme_n,
        extreme_hour_rate=extreme_n / truth.size if truth.size else None,
        **_errors(error[extreme], "exceedance"),
        **_under(prediction[extreme], truth[extreme], "exceedance"),
        exceedance_precision=true_positive / predicted_n if predicted_n else None,
        exceedance_recall=true_positive / extreme_n if extreme_n else None,
        exceedance_f1=2 * true_positive / (predicted_n + extreme_n) if predicted_n + extreme_n else None,
        event_window_n=int(events.sum()),
        event_window_rate=float(events.mean()) if events.size else None,
        **_errors(error[events], "event_window", bias=False),
        **_errors(peak_error, "window_peak"),
        **_errors(aligned_pred - aligned_truth, "gt_aligned_peak"),
        **_under(aligned_pred, aligned_truth, "gt_aligned_peak"),
        peak_timing_mae_steps=timing,
        peak_timing_mae_hours=timing * target_interval_hours if timing is not None else None,
    )
    if target_timestamps is not None:
        metrics.update(_episode_metrics(prediction, truth, target_timestamps, tau, split_ids))
    if include_leadwise:
        for lead in range(truth.shape[1]):
            selected = extreme[:, lead]
            metrics[f"extreme_hour_n_lead_{lead}"] = int(selected.sum())
            selected_error = error[selected, lead]
            metrics[f"exceedance_rmse_lead_{lead}"] = float(np.sqrt(np.mean(selected_error ** 2))) if selected_error.size else None
    return metrics


def format_metrics(label, metrics, *, extended=False):
    """Format a concise training line or the complete canonical metric dictionary."""
    names = metrics if extended else ("all_rmse", "all_mae", "exceedance_rmse", "event_window_rmse",
                                     "gt_aligned_peak_rmse", "extreme_hour_n", "event_window_n")
    pairs = []
    for key in names:
        if key not in metrics:
            continue
        name, value = METRIC_LABELS.get(key, key), metrics[key]
        text = "NA" if value is None else str(int(value)) if key.endswith("_n") else f"{value:.6f}"
        pairs.append(f"{name}={text}")
    return f"{label} " + " ".join(pairs)
