"""Canonical physical-unit forecast metrics using one fixed TRAIN threshold.

Forecast arrays are [window, target horizon]. Empty regression populations and
undefined metrics are represented by None, never NaN.
"""

import math

import numpy as np
import torch


METRIC_GROUPS = {
    "Overall": ("all_rmse", "all_mae"),
    "Extreme hours": ("exceedance_rmse", "exceedance_mae"),
    "Event episodes": (
        "episode_peak_rmse", "episode_peak_mae", "episode_peak_bias",
        "episode_gt_aligned_peak_rmse", "episode_gt_aligned_peak_mae",
        "episode_gt_aligned_peak_bias", "episode_peak_timing_mae_hours",
    ),
}
METRIC_KEYS = tuple(key for group in METRIC_GROUPS.values() for key in group)
EPOCH_METRIC_KEYS = METRIC_GROUPS["Overall"] + METRIC_GROUPS["Extreme hours"]
EPISODE_METRIC_KEYS = METRIC_GROUPS["Event episodes"]
METRIC_LABELS = dict(zip(METRIC_KEYS, (
    "AllRMSE", "AllMAE", "ExceedanceRMSE", "ExceedanceMAE",
    "EpisodePeakRMSE", "EpisodePeakMAE", "EpisodePeakBias",
    "EpisodeGTAlignedPeakRMSE", "EpisodeGTAlignedPeakMAE", "EpisodeGTAlignedPeakBias",
    "EpisodePeakTimingMAEHours",
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


def _values(value, name, *, allow_nonfinite=False):
    array = _numpy(value).astype(np.float64)
    if array.ndim != 2 or array.shape[1] == 0:
        raise ValueError(f"{name} must have shape [windows, horizons] with at least one horizon.")
    if not allow_nonfinite and not np.isfinite(array).all():
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


def _validate_unique_timestamps(times, labels):
    """Validate all target hours and return chronological indices per split."""
    groups = []
    for label in np.unique(labels):
        indices = np.flatnonzero(labels == label)
        indices = indices[np.argsort(times[indices], kind="stable")]
        sorted_times = times[indices]
        duplicate = np.flatnonzero(np.diff(sorted_times) == 0)
        if duplicate.size:
            timestamp = np.datetime64(int(sorted_times[duplicate[0]]), "s")
            raise ValueError(f"Duplicate target timestamp {timestamp} within split {label!r}; supervised target hours must occur exactly once.")
        groups.append(indices)
    return groups


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
    for indices in _validate_unique_timestamps(times, labels):
        extreme = indices[truth[indices] > tau]
        if extreme.size:
            breaks = np.flatnonzero(np.diff(times[extreme]) != 3600) + 1
            episodes.extend(np.split(extreme, breaks))
    return episodes


def gt_episode_peak_indices(y_true, episodes):
    """One earliest GT argmax per chronological episode, shared with training."""
    truth = _values(y_true, "y_true").ravel()
    return np.asarray([indices[int(np.argmax(truth[indices]))] for indices in episodes], dtype=np.int64)


def _episode_metrics(prediction, truth, timestamps, tau, split_ids):
    episodes = gt_event_episodes(truth, timestamps, tau, split_ids=split_ids)
    peaks = gt_episode_peak_indices(truth, episodes)
    times = _timestamp_seconds(timestamps, truth.shape).ravel()
    prediction, truth = prediction.ravel(), truth.ravel()
    peak_errors, timings = [], []
    for indices, gt_peak in zip(episodes, peaks):
        pred_peak = indices[int(np.argmax(prediction[indices]))]
        peak_errors.append(prediction[pred_peak] - truth[gt_peak])
        timings.append(abs(int(times[pred_peak]) - int(times[gt_peak])) / 3600.)
    return dict(**_errors(np.asarray(peak_errors), "episode_peak"),
                **_errors(prediction[peaks] - truth[peaks], "episode_gt_aligned_peak"),
                episode_peak_timing_mae_hours=_mean(timings))


def evaluate_metrics(y_pred, y_true, tau_physical, *, target_timestamps=None, split_ids=None):
    """Return four hourly metrics, or exactly eleven with target timestamps.

    Final episode populations use strict GT exceedance, exact hourly continuity
    and earliest maxima. Duplicate timestamps within a split are rejected.
    Hourly metrics omit individual nonfinite targets; episode metrics omit their
    entire windows. Predictions must be finite wherever targets are finite.
    """
    prediction = _values(y_pred, "y_pred", allow_nonfinite=True)
    truth = _values(y_true, "y_true", allow_nonfinite=True)
    if prediction.shape != truth.shape:
        raise ValueError("y_pred and y_true must have identical [windows, horizons] shapes.")
    point_valid = np.isfinite(truth)
    if not np.isfinite(prediction[point_valid]).all():
        raise ValueError("Cannot evaluate nonfinite y_pred where y_true is finite.")
    tau = _threshold(tau_physical)
    if split_ids is not None and target_timestamps is None:
        raise ValueError("split_ids require target_timestamps for split-isolated episode evaluation.")
    error = prediction[point_valid] - truth[point_valid]
    metrics = dict(**_errors(error, "all", bias=False),
                   **_errors(error[truth[point_valid] > tau], "exceedance", bias=False))
    if target_timestamps is not None:
        times = _timestamp_seconds(target_timestamps, truth.shape)
        labels = _split_labels(split_ids, truth.shape)
        window_valid = point_valid.all(axis=1)
        if not window_valid.all():
            _validate_unique_timestamps(times.ravel(), labels.ravel())
        metrics.update(_episode_metrics(prediction[window_valid], truth[window_valid],
                                        times[window_valid], tau, labels[window_valid]))
    return metrics


def format_metrics(label, metrics, *, extended=False):
    """Format a concise training line or the complete canonical metric dictionary."""
    names = METRIC_KEYS if extended else EPOCH_METRIC_KEYS
    pairs = []
    for key in names:
        if key not in metrics:
            continue
        name, value = METRIC_LABELS.get(key, key), metrics[key]
        text = "NA" if value is None else str(int(value)) if key.endswith("_n") else f"{value:.6f}"
        pairs.append(f"{name}={text}")
    return f"{label} " + " ".join(pairs)
