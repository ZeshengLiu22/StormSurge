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
    """Convert an array-like value to a NumPy array for metric evaluation.

    Args:
        value: NumPy-compatible object or torch tensor. Tensors are detached and moved
            to CPU before conversion.

    Returns:
        numpy.ndarray: Array representation, possibly sharing storage with the input.
    """
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _threshold(value):
    """Validate and convert the fixed training threshold.

    Args:
        value: Scalar convertible to float, in the same physical units as the targets.

    Returns:
        float: Finite physical threshold.

    Raises:
        ValueError: The threshold is nonfinite or cannot be converted to float.
    """
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("Metrics require a finite tau_physical fitted on TRAIN target hours.")
    return value


def _values(value, name, *, allow_nonfinite=False):
    """Validate the two-dimensional forecast-array shape and numeric values.

    Args:
        value: Array-like data or torch tensor with shape [windows, horizons]; zero
            windows are allowed, but at least one horizon is required.
        name: Array name to include in validation errors.
        allow_nonfinite: Whether to allow NaN and infinity for later masking/validation.

    Returns:
        numpy.ndarray: Float64 array with the input window/horizon shape.

    Raises:
        ValueError: The shape is invalid or nonfinite values are present when
            disallowed.
    """
    array = _numpy(value).astype(np.float64)
    if array.ndim != 2 or array.shape[1] == 0:
        raise ValueError(f"{name} must have shape [windows, horizons] with at least one horizon.")
    if not allow_nonfinite and not np.isfinite(array).all():
        raise ValueError(f"Cannot evaluate nonfinite {name}.")
    return array


def _mean(value):
    """Compute a scalar mean while representing empty populations explicitly.

    Args:
        value: Numeric array-like values; nonfinite values are not filtered.

    Returns:
        float or None: Arithmetic mean, or None when the input has zero elements.
    """
    return float(np.mean(value)) if np.size(value) else None


def _errors(error, prefix, *, bias=True):
    """Summarize prediction-minus-truth errors with RMSE, MAE, and optional bias.

    Args:
        error: NumPy array of signed errors in physical target units.
        prefix: Prefix for returned metric keys.
        bias: Whether to include the signed mean-error metric.

    Returns:
        dict: <prefix>_rmse, <prefix>_mae, and optionally <prefix>_bias, in the input
        units. Each value is None for an empty population.
    """
    result = {f"{prefix}_rmse": float(np.sqrt(np.mean(error ** 2))) if error.size else None,
              f"{prefix}_mae": _mean(np.abs(error))}
    if bias:
        result[f"{prefix}_bias"] = _mean(error)
    return result


def _under(prediction, truth, prefix, *, tolerances=True):
    """Measure strict underprediction percentages at selected amplitude tolerances.

    Args:
        prediction: NumPy prediction values in metres, aligned with truth.
        truth: NumPy ground-truth values in metres.
        prefix: Prefix for returned underprediction metric keys.
        tolerances: Whether to include 5, 10, and 20 mm tolerances in addition to zero.

    Returns:
        dict: Percentages in [0, 100] satisfying prediction < truth - tolerance, or None
        for an empty truth population.
    """
    limits = (("", 0.), ("_5mm", .005), ("_10mm", .010), ("_20mm", .020)) if tolerances else (("", 0.),)
    return {f"{prefix}_under{label}_pct": float(100 * np.mean(prediction < truth - tolerance)) if truth.size else None
            for label, tolerance in limits}


def _timestamp_seconds(timestamps, shape):
    """Validate target-time shape and convert to whole Unix seconds.

    Args:
        timestamps: Array-like times or torch tensor containing numeric integer Unix
            seconds, datetime64 values, or ISO datetime strings.
        shape: Required [windows, horizons] shape matching the forecast targets.

    Returns:
        numpy.ndarray: int64 Unix seconds with the requested shape. Uniqueness and
        hourly continuity are checked by downstream episode routines.

    Raises:
        ValueError: Shape differs, times are missing/nonfinite, numeric values are
            fractional, or datetime values have subsecond precision.
    """
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
    """Broadcast split identifiers to one string label per target hour.

    Args:
        split_ids: None, an array with one identifier per window, or an array with one
            identifier per target hour. Values are converted to strings.
        shape: Target-array shape [windows, horizons].

    Returns:
        numpy.ndarray: String labels with the target shape; None supplies the common
        label evaluation.

    Raises:
        ValueError: Identifiers cannot be interpreted as the required window or target
            shape.
    """
    if split_ids is None:
        return np.full(shape, "evaluation", dtype=str)
    labels = _numpy(split_ids)
    if labels.shape == (shape[0],):
        labels = np.repeat(labels[:, None], shape[1], axis=1)
    if labels.shape != shape:
        raise ValueError("split_ids must contain one split per window or per target hour.")
    return labels.astype(str)


def _validate_unique_timestamps(times, labels):
    """Order target indices chronologically within each split and reject duplicates.

    Args:
        times: Flat int64 array of UTC Unix seconds for all target hours.
        labels: Flat array of split labels aligned with times.

    Returns:
        list[numpy.ndarray]: Flat target indices in stable chronological order, one
        array per lexically sorted unique split label.

    Raises:
        ValueError: Two target hours have the same timestamp within a split. Repeated
            timestamps across distinct splits are allowed.
    """
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
    """Identify contiguous hourly episodes from strict ground-truth exceedances.

    Episode members satisfy y_true > tau_physical and are separated by exactly
    3600 seconds. A nonextreme hour, missing hour, or split boundary ends an episode.
    Forecast-window boundaries do not. Predictions never define this population.

    Args:
        y_true: Finite physical targets with shape [windows, horizons].
        target_timestamps: Whole-second target times matching y_true, as numeric Unix
            seconds, datetime64 values, or ISO strings.
        tau_physical: Finite source-training threshold in the same physical units as
            y_true.
        split_ids: Optional identifiers per window or per target hour; episodes and
            timestamp uniqueness are evaluated independently within each split.

    Returns:
        list[numpy.ndarray]: Arrays of flat row-major target indices, one per episode,
        each ordered chronologically. Splits are processed in sorted label order.

    Raises:
        ValueError: Targets, times, labels, or threshold are invalid, including
            duplicate timestamps within a split.
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
    """Select the earliest ground-truth maximum within each chronological episode.

    Args:
        y_true: Finite physical targets [windows, horizons], flattened in row-major
            order for indexing.
        episodes: Iterable of nonempty chronological flat-index arrays, as returned by
            gt_event_episodes.

    Returns:
        numpy.ndarray: int64 flat index of one ground-truth maximum per episode. Tied
        maxima select the first supplied index, which is the earliest time for canonical
        episodes.
    """
    truth = _values(y_true, "y_true").ravel()
    return np.asarray([indices[int(np.argmax(truth[indices]))] for indices in episodes], dtype=np.int64)


def _episode_metrics(prediction, truth, timestamps, tau, split_ids):
    """Evaluate amplitude and timing errors on ground-truth-defined episodes.

    The predicted peak is the earliest maximum within the ground-truth episode,
    not an independently detected forecast event. Aligned errors sample predictions
    at the canonical ground-truth peak time.

    Args:
        prediction: Finite physical predictions [windows, horizons].
        truth: Finite ground-truth targets with the same shape and units.
        timestamps: Whole-second target timestamps matching truth.
        tau: Finite training threshold in physical target units.
        split_ids: Optional split labels per window or target hour, preventing episodes
            from crossing split boundaries.

    Returns:
        dict: Episode-peak and ground-truth-aligned peak RMSE/MAE/bias in target units,
        plus mean absolute peak-timing error in hours. Empty episode populations yield
        None.
    """
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
    """Compute physical hourly metrics and optional ground-truth episode metrics.

    Hourly metrics omit individual missing targets. Episode metrics omit each
    entire window containing a missing target, while validating timestamps across
    all windows before exclusion. Ground-truth episodes use exact hourly continuity
    and earliest maxima. Predictions at missing target positions are ignored.

    Args:
        y_pred: Predictions [windows, horizons], in the same physical units as y_true;
            must be finite wherever y_true is finite.
        y_true: Physical targets of the identical shape. Nonfinite targets are treated
            as missing.
        tau_physical: Finite threshold fitted once on source-training target hours.
            Strict exceedance uses y_true > tau_physical.
        target_timestamps: Optional target-time array of the same shape. Supplying it
            enables episode metrics and requires unique timestamps within each split.
        split_ids: Optional split labels per window or target hour. Requires
            target_timestamps and isolates episodes across splits.

    Returns:
        dict: Four hourly metrics (all/exceedance RMSE and MAE), or eleven metrics when
        timestamps also enable seven episode amplitude/timing metrics. Amplitude metrics
        retain target units; timing MAE is in hours. Undefined values are None.

    Raises:
        ValueError: Shapes, required predictions, threshold, timestamps, or split labels
            are invalid.
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
    """Format canonical metrics as a concise log line.

    Args:
        label: Leading label, such as the split or epoch description.
        metrics: Mapping of canonical metric keys to numeric values or None.
        extended: Whether to include all canonical metrics rather than only the four
            hourly epoch metrics.

    Returns:
        str: Label followed by recognized present metrics, with six decimal places and
        NA for undefined values.
    """
    names = METRIC_KEYS if extended else EPOCH_METRIC_KEYS
    pairs = []
    for key in names:
        if key not in metrics:
            continue
        name, value = METRIC_LABELS.get(key, key), metrics[key]
        text = "NA" if value is None else str(int(value)) if key.endswith("_n") else f"{value:.6f}"
        pairs.append(f"{name}={text}")
    return f"{label} " + " ".join(pairs)
