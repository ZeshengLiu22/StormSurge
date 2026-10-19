"""Fixed TRAIN episode membership and one physical GT peak per episode."""

from dataclasses import dataclass

import numpy as np
from torch.utils.data import RandomSampler, SequentialSampler, DistributedSampler

from .metrics import (_timestamp_seconds, _validate_unique_timestamps, _values,
                      gt_event_episodes, gt_episode_peak_indices)


@dataclass
class EpisodePeakTargets:
    """Canonical training episode membership and ground-truth peak locations.

    Attributes:
        episode_id: int64 array [windows, horizons], with contiguous episode IDs
            at extreme targets and -1 elsewhere, including incomplete windows.
        is_episode_gt_peak: Boolean array of the same shape, true at exactly the
            earliest maximum of each eligible episode.
        metadata: Training threshold, population counts, tie/continuity rules,
            and p_episode_peak, the peak count divided by eligible target count.
    """
    episode_id: np.ndarray
    is_episode_gt_peak: np.ndarray
    metadata: dict


def build_episode_peak_targets(y_true, target_timestamps, tau_physical):
    """Build fixed training episode labels using the evaluation continuity rules.

    Episodes consist of strict ground-truth exceedances separated by exactly
    3600 seconds, including across forecast windows. Incomplete windows leave gaps;
    model predictions do not affect membership or peak placement.

    Args:
        y_true: Physical source-training targets [windows, horizons]. NaN is allowed,
            but every window containing it is excluded from episode supervision;
            infinity is rejected.
        target_timestamps: Target times with the same shape, represented as integer Unix
            seconds, datetime64, or ISO strings. Timestamps must be unique across the
            complete split.
        tau_physical: Finite threshold fitted on source-training target hours, in the
            same units as y_true.

    Returns:
        EpisodePeakTargets: Episode IDs and one earliest-maximum mask per episode,
        aligned to the original target shape, plus population metadata and fixed
        training peak prevalence.

    Raises:
        ValueError: Target shapes, values, timestamps, or threshold are invalid,
            including duplicate timestamps in excluded windows.
    """
    truth = _values(y_true, "TRAIN y_true", allow_nonfinite=True)
    if np.isinf(truth).any():
        raise ValueError("TRAIN y_true must not contain infinite targets.")
    times = _timestamp_seconds(target_timestamps, truth.shape)
    window_valid = np.isfinite(truth).all(axis=1)
    if not window_valid.all():
        _validate_unique_timestamps(times.ravel(), np.full(truth.size, "evaluation"))
    valid_truth = truth[window_valid]
    episodes = gt_event_episodes(valid_truth, times[window_valid], tau_physical)
    peaks = gt_episode_peak_indices(valid_truth, episodes)
    original_indices = np.flatnonzero(np.broadcast_to(window_valid[:, None], truth.shape))
    membership = np.full(truth.size, -1, dtype=np.int64)
    mask = np.zeros(truth.size, dtype=bool)
    for episode_id, indices in enumerate(episodes):
        membership[original_indices[indices]] = episode_id
    mask[original_indices[peaks]] = True
    count = len(peaks)
    eligible_count = valid_truth.size
    metadata = dict(fitted_on="train", episode_count=len(episodes),
                    canonical_peak_target_count=count, eligible_target_count=int(eligible_count),
                    duplicate_target_count=0,
                    p_episode_peak=count / eligible_count if eligible_count else 0.,
                    tau_physical=float(tau_physical),
                    canonical_occurrence_rule="unique target timestamp; overlapping blocks rejected",
                    episode_definition="GT > TRAIN tau; consecutive timestamps exactly 3600 seconds apart",
                    peak_tie_rule="earliest timestamp at GT maximum")
    return EpisodePeakTargets(membership.reshape(truth.shape), mask.reshape(truth.shape), metadata)


def validate_episode_peak_traversal(loader):
    """Require a loader traversal that visits each canonical training target once.

    For distributed training, the dataset size must be divisible by the world
    size so padding cannot duplicate episode peaks or alter their prevalence.

    Args:
        loader: DataLoader exposing dataset, sampler, and drop_last. Supported samplers
            are SequentialSampler, full-length RandomSampler without replacement, and
            unpadded/untruncated DistributedSampler.

    Returns:
        None. Performs validation without changing the loader or sampler.

    Raises:
        ValueError: The loader drops targets, its sampler repeats or omits samples, or
            the traversal cannot be verified.
    """
    sampler = loader.sampler
    if loader.drop_last:
        raise ValueError("Episode peak supervision requires every TRAIN target: drop_last must be false.")
    if isinstance(sampler, DistributedSampler):
        if sampler.drop_last or sampler.total_size != len(loader.dataset):
            raise ValueError("Episode peak supervision cannot use padded/truncated DistributedSampler: "
                             "TRAIN windows must be divisible by world size. Use one process or a "
                             "divisor of the TRAIN window count; sampling is unchanged.")
    elif isinstance(sampler, RandomSampler):
        if sampler.replacement or sampler.num_samples != len(loader.dataset):
            raise ValueError("Episode peak supervision requires one occurrence per TRAIN traversal.")
    elif not isinstance(sampler, SequentialSampler):
        raise ValueError("Episode peak supervision requires a verified traversal without replacement.")
