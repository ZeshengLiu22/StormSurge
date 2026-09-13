"""Optional event-only peak risk on the final PHYSICAL prediction, in square meters."""

import math
from typing import NamedTuple

import torch

from .excess_amplitude import peak_pool, validate_event_prior


def validate_peak_config(config):
    weight = config.peak_loss_weight
    pool = config.peak_pool
    if not math.isfinite(weight) or weight < 0:
        raise ValueError("--peak_loss_weight must be finite and nonnegative.")
    if pool not in ("max", "smoothmax"):
        raise ValueError("--peak_pool must be max or smoothmax.")
    if weight > 0 and pool == "smoothmax":
        beta = config.peak_pool_beta
        if not math.isfinite(beta) or beta <= 0:
            raise ValueError("--peak_pool_beta must be finite and positive for smoothmax (inverse meters).")
    return weight


def validate_event_threshold(event_threshold_phys):
    if event_threshold_phys is None or not math.isfinite(event_threshold_phys):
        raise ValueError("Final peak supervision requires a finite TRAIN event_threshold (tau_phys), "
                         "separate from the tail peak_threshold.")


class FinalPeakTerms(NamedTuple):
    event: torch.Tensor
    m_pred: torch.Tensor
    m_true: torch.Tensor
    loss: torch.Tensor

    def diagnostics(self):
        """Batch diagnostics using the training pool; means cover the whole batch.

        Pure tensor reductions, with no mutable criterion state or DDP collective.
        Call on detached inputs under no_grad for diagnostic-only use.
        Exact reported evaluation metrics instead use hard maxima.
        """
        return dict(peak_loss=self.loss, pred_peak_mean=self.m_pred.mean(),
                    true_peak_mean=self.m_true.mean(), peak_bias=(self.m_pred - self.m_true).mean(),
                    event_count=self.event.sum(), event_fraction=self.event.float().mean())


def final_peak_terms(prediction_phys, target_phys, event_threshold_phys, event_prior,
                     pool="max", beta=20.0):
    """mean(E * (pool(prediction_phys) - pool(target_phys))**2) / q_TRAIN.

    Reuse #2's bounded softmax-weighted mean for BOTH trajectories in smooth
    mode. Events always use strict hard truth peaks. The FP64 comparison keeps
    the fitted NumPy threshold exact, matching fit_loss_thresholds' event count.
    No normalized outputs, branch targets or batch-fitted denominators enter.
    """
    validate_event_prior(event_prior)
    validate_event_threshold(event_threshold_phys)
    m_pred = peak_pool(prediction_phys, pool, beta)
    m_true = peak_pool(target_phys, pool, beta)
    hard_true = m_true if pool == "max" else peak_pool(target_phys, "max")
    event = hard_true.double() > event_threshold_phys
    error = torch.where(event, m_pred - m_true, 0.0)
    loss = error.square().mean() / event_prior
    return FinalPeakTerms(event, m_pred, m_true, loss)
