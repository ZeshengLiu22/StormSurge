"""Fixed TRAIN episode membership and one physical GT peak per episode."""

from dataclasses import dataclass

import numpy as np
from torch.utils.data import RandomSampler, SequentialSampler, DistributedSampler

from .metrics import _timestamp_seconds, _values, gt_event_episodes, gt_episode_peak_indices


@dataclass
class EpisodePeakTargets:
    episode_id: np.ndarray
    is_episode_gt_peak: np.ndarray
    metadata: dict


def build_episode_peak_targets(y_true, target_timestamps, tau_physical):
    """Use the evaluator's episode and tie rules over the entire TRAIN split.

    The current representation has unique target timestamps (six-hour blocks
    with six hourly targets). Repeated timestamps fail in the shared builder,
    before any supervision can be duplicated. No model output enters this code.
    """
    truth = _values(y_true, "TRAIN y_true")
    times = _timestamp_seconds(target_timestamps, truth.shape)
    episodes = gt_event_episodes(truth, times, tau_physical)
    peaks = gt_episode_peak_indices(truth, episodes)
    membership = np.full(truth.size, -1, dtype=np.int64)
    mask = np.zeros(truth.size, dtype=bool)
    for episode_id, indices in enumerate(episodes):
        membership[indices] = episode_id
    mask[peaks] = True
    count = len(peaks)
    metadata = dict(fitted_on="train", episode_count=len(episodes),
                    canonical_peak_target_count=count, eligible_target_count=int(truth.size),
                    duplicate_target_count=0,
                    p_episode_peak=count / truth.size if truth.size else 0.,
                    tau_physical=float(tau_physical),
                    canonical_occurrence_rule="unique target timestamp; overlapping blocks rejected",
                    episode_definition="GT > TRAIN tau; consecutive timestamps exactly 3600 seconds apart",
                    peak_tie_rule="earliest timestamp at GT maximum")
    return EpisodePeakTargets(membership.reshape(truth.shape), mask.reshape(truth.shape), metadata)


def validate_episode_peak_traversal(loader):
    """Reject traversal modes that repeat or omit canonical peak occurrences.

    Keep the existing sampler unchanged. In particular DistributedSampler's
    padding must not silently repeat peak supervision or change its prevalence.
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
