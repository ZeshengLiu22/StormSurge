"""Physical supervised targets and their explicit hourly timestamps.

Timestamp arrays use integer UTC seconds since the Unix epoch. The preprocessing
pipeline stores ``center_time`` and supervises lead zero at that exact time.
"""

from datetime import datetime, timezone

import numpy as np
import torch


HOUR_SECONDS = 3600


def _utc_seconds(value):
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
    """Return UTC seconds, rejecting duplicates instead of deduplicating them."""
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
    """Read explicit timestamps, or expand preprocessing's recorded center time."""
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
    """Read raw physical graph.y only; inputs/history never enter this series."""
    indices = list(indices)
    if not indices:
        raise ValueError("Cannot build a supervised target series from an empty split.")
    labels = np.stack([store.graphs[index].y.detach().cpu().numpy().reshape(-1) for index in indices]).astype(np.float64)
    if not np.isfinite(labels).all():
        raise ValueError("Supervised target labels must be finite.")
    tags = getattr(store, "graph_tags", None)
    timestamps = np.stack([target_timestamps_from_graph(store.graphs[index], tags[index] if tags else str(index))
                           for index in indices])
    return labels, validate_target_timestamps(timestamps)


def threshold_population(labels, timestamps, tau_physical):
    """Count strict extreme hours, event windows and contiguous hourly episodes."""
    # Import lazily to keep data-module imports independent of training setup.
    from emulator.training.metrics import evaluate_metrics

    labels = np.asarray(labels, dtype=np.float64)
    timestamps = validate_target_timestamps(timestamps)
    if labels.ndim != 2 or labels.shape != timestamps.shape or not np.isfinite(labels).all():
        raise ValueError("Finite target labels and timestamps must have matching [windows, horizons] shapes.")
    if not np.isfinite(tau_physical):
        raise ValueError("The TRAIN physical threshold must be finite.")
    metrics = evaluate_metrics(labels, labels, tau_physical, target_timestamps=timestamps, include_leadwise=False)
    return dict(target_hour_count=int(labels.size), extreme_hour_count=metrics["extreme_hour_n"],
                extreme_hour_rate=metrics["extreme_hour_rate"], event_window_count=metrics["event_window_n"],
                event_window_rate=metrics["event_window_rate"], episode_count=metrics["episode_n"])
