"""Physical supervised targets and their explicit hourly timestamps.

Timestamp arrays use integer UTC seconds since the Unix epoch. The preprocessing
pipeline stores ``center_time`` and supervises lead zero at that exact time.
"""

from datetime import datetime, timezone

import numpy as np
import torch


HOUR_SECONDS = 3600


def _utc_seconds(value):
    """Convert a scalar time representation to whole UTC Unix seconds.

    Args:
        value: Integer Unix seconds, numpy.datetime64, datetime, or ISO datetime string.
            Naive datetimes are interpreted as UTC; aware datetimes are converted by
            their timezone offset.

    Returns:
        int: Seconds since 1970-01-01 00:00:00 UTC.

    Raises:
        ValueError: The value is invalid, missing, nonfinite, or has subsecond
            precision.
    """
    if isinstance(value, np.datetime64):
        if np.isnat(value):
            raise ValueError("Target timestamp must not be NaT.")
        seconds = value.astype("datetime64[s]")
        if value != seconds:
            raise ValueError("Target timestamps must have whole-second precision.")
        return int(seconds.astype(np.int64))
    if isinstance(value, (int, np.integer)):
        return int(value)
    try:
        parsed = value if isinstance(value, datetime) else datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid target timestamp {value!r}.") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    seconds = parsed.timestamp()
    if not np.isfinite(seconds) or seconds != int(seconds):
        raise ValueError("Target timestamps must be finite whole UTC seconds.")
    return int(seconds)


def validate_target_timestamps(timestamps):
    """Convert timestamp arrays to UTC seconds and reject repeated target hours.

    Uniqueness is required over the entire array, including across forecast
    windows. The function does not enforce chronological order or hourly spacing.

    Args:
        timestamps: Array-like timestamps or a torch tensor of any shape, containing
            integer Unix seconds, datetimes, or ISO datetime strings.

    Returns:
        numpy.ndarray: int64 Unix seconds with the input shape preserved.

    Raises:
        ValueError: Values are invalid, missing, subsecond, or duplicated.
    """
    if isinstance(timestamps, torch.Tensor):
        timestamps = timestamps.detach().cpu().numpy()
    values = np.asarray(timestamps)
    if np.issubdtype(values.dtype, np.integer):
        seconds = values.astype(np.int64)
    elif np.issubdtype(values.dtype, np.datetime64):
        if np.isnat(values).any() or np.any(values != values.astype("datetime64[s]")):
            raise ValueError("Target timestamps must be finite whole UTC seconds.")
        seconds = values.astype("datetime64[s]").astype(np.int64)
    else:
        seconds = np.asarray([_utc_seconds(value) for value in values.flat], dtype=np.int64).reshape(values.shape)
    unique, counts = np.unique(seconds, return_counts=True)
    duplicated = unique[counts > 1]
    if len(duplicated):
        example = str(np.datetime64(int(duplicated[0]), "s"))
        raise ValueError(
            f"Duplicate supervised target timestamps: {len(duplicated)} repeated timestamp(s), "
            f"including {example} UTC. Target blocks must not overlap; do not silently deduplicate."
        )
    return seconds


def target_timestamps_from_graph(graph, tag=None):
    """Read or reconstruct a graph window’s consecutive hourly target times.

    Args:
        graph: Graph exposing y and either target_timestamps or center_time. When both
            exist, the first timestamp must match center_time.
        tag: Optional graph identifier for validation errors.

    Returns:
        numpy.ndarray: int64 Unix seconds with shape [y.numel()], beginning at lead zero
        and increasing by 3600 seconds.

    Raises:
        ValueError: Timing metadata is absent, target lengths disagree, or timestamps
            are nonunique, nonhourly, or inconsistent with center_time.
    """
    horizons = graph.y.numel()
    explicit = getattr(graph, "target_timestamps", None)
    center = getattr(graph, "center_time", None)
    if explicit is not None:
        timestamps = validate_target_timestamps(explicit).reshape(-1)
        if len(timestamps) != horizons:
            raise ValueError(f"{tag or 'Graph'} target timestamp count does not match its {horizons} labels.")
        if center is not None and timestamps[0] != _utc_seconds(center):
            raise ValueError(f"{tag or 'Graph'} first target timestamp must equal center_time (lead zero).")
    elif center is not None:
        timestamps = _utc_seconds(center) + np.arange(horizons, dtype=np.int64) * HOUR_SECONDS
    else:
        raise ValueError(
            f"{tag or 'Graph'} has no center_time or target_timestamps metadata. "
            "Cannot establish supervised target times; regenerate timestamped data."
        )
    if np.any(np.diff(timestamps) != HOUR_SECONDS):
        raise ValueError(f"{tag or 'Graph'} target horizons must be consecutive hourly timestamps.")
    return timestamps


def supervised_targets(store, indices):
    """Collect physical target labels and unique timestamps from a graph subset.

    Args:
        store: Object exposing graphs and optionally graph_tags for diagnostic labels.
        indices: Nonempty iterable of graph indices defining the output window order.

    Returns:
        tuple: (labels, timestamps), each shaped [windows, horizons]. Labels are float64
        in their stored physical units and may contain NaN; timestamps are int64 UTC
        Unix seconds.

    Raises:
        ValueError: The selection is empty, labels contain infinity, or timing metadata
            is invalid or overlapping.
    """
    indices = list(indices)
    if not indices:
        raise ValueError("Cannot build a supervised target series from an empty split.")
    labels = np.stack([store.graphs[index].y.detach().cpu().numpy().reshape(-1) for index in indices]).astype(np.float64)
    if np.isinf(labels).any():
        raise ValueError("Supervised target labels must not contain infinity.")
    tags = getattr(store, "graph_tags", None)
    timestamps = np.stack([target_timestamps_from_graph(store.graphs[index], tags[index] if tags else str(index))
                           for index in indices])
    return labels, validate_target_timestamps(timestamps)


def threshold_population(labels, timestamps, tau_physical):
    """Count strict exceedances, affected forecast windows, and hourly episodes.

    An event window contains at least one extreme target. Episodes join
    strictly extreme targets exactly one hour apart, including across windows.

    Args:
        labels: Finite physical target values with shape [windows, horizons].
        timestamps: Unique whole-second UTC target times with the same shape as labels.
        tau_physical: Finite training-fitted threshold in the same physical units as
            labels; membership requires labels > tau_physical.

    Returns:
        dict: target_hour_count, extreme_hour_count/rate, event_window_count/rate, and
        episode_count. Rates are fractions, or None for an empty denominator.

    Raises:
        ValueError: Labels, timestamps, or the threshold violate the required shapes,
            finiteness, or uniqueness.
    """
    # Import lazily to keep data-module imports independent of training setup.
    from emulator.training.metrics import gt_event_episodes

    labels = np.asarray(labels, dtype=np.float64)
    timestamps = validate_target_timestamps(timestamps)
    if labels.ndim != 2 or labels.shape != timestamps.shape or not np.isfinite(labels).all():
        raise ValueError("Finite target labels and timestamps must have matching [windows, horizons] shapes.")
    if not np.isfinite(tau_physical):
        raise ValueError("The TRAIN physical threshold must be finite.")
    extreme = labels > tau_physical
    events = extreme.any(axis=1)
    return dict(target_hour_count=int(labels.size), extreme_hour_count=int(extreme.sum()),
                extreme_hour_rate=float(extreme.mean()) if labels.size else None,
                event_window_count=int(events.sum()),
                event_window_rate=float(events.mean()) if events.size else None,
                episode_count=len(gt_event_episodes(labels, timestamps, tau_physical)))
