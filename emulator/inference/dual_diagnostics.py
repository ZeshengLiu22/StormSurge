"""Offline, physical-unit diagnostics for a dual prediction export."""

import numpy as np


def summarize_dual(arrays, tau_phys, bins=10):
    """Use a fixed TRAIN threshold; tied scores enter the PR curve as one group.

    Average precision integrates the stepwise PR curve. pr_auc_trapezoid uses
    linear interpolation. Neither is defined when the evaluation set has no events.
    Empty reliability bins and undefined conditional metrics are JSON null.
    """
    truth = np.asarray(arrays["y_true"], dtype=np.float64)
    prediction = np.asarray(arrays["y_pred"], dtype=np.float64)
    probability = np.asarray(arrays["gate_probability"], dtype=np.float64).reshape(-1)
    body = np.asarray(arrays["body_phys"], dtype=np.float64)
    excess = np.asarray(arrays["excess_phys"], dtype=np.float64)
    if (truth.ndim != 2 or truth.shape != prediction.shape or truth.shape != body.shape
            or truth.shape != excess.shape or len(probability) != len(truth)):
        raise ValueError("Dual diagnostics require aligned (N,K) predictions and N probabilities.")
    if not all(np.isfinite(value).all() for value in (truth, prediction, probability, body, excess, tau_phys)):
        raise ValueError("Dual diagnostics require finite values.")
    if bins < 1 or np.any((probability < 0) | (probability > 1)):
        raise ValueError("Probabilities must be in [0,1] and bins positive.")
    event = np.any(truth > tau_phys, axis=1)
    count, positives = len(event), int(event.sum())
    edges = np.linspace(0, 1, bins + 1)
    membership = np.minimum(np.searchsorted(edges, probability, side="right") - 1, bins - 1)
    reliability = []
    for index in range(bins):
        selected = membership == index
        reliability.append(dict(lower=float(edges[index]), upper=float(edges[index + 1]),
                                count=int(selected.sum()),
                                mean_probability=float(probability[selected].mean()) if selected.any() else None,
                                event_rate=float(event[selected].mean()) if selected.any() else None))
    average_precision = pr_auc = None
    precision, recall = [], []
    if positives:
        order = np.argsort(-probability, kind="stable")
        ends = np.r_[np.flatnonzero(np.diff(probability[order])), count - 1]
        tp = np.cumsum(event[order])[ends]
        precision = np.r_[1., tp / (ends + 1)]
        recall = np.r_[0., tp / positives]
        average_precision = float(np.sum(np.diff(recall) * precision[1:]))
        pr_auc = float(np.sum(np.diff(recall) * (precision[1:] + precision[:-1]) * .5))
        precision, recall = precision.tolist(), recall.tolist()

    def rmse(error, selected):
        return float(np.sqrt(np.mean(error[selected] ** 2))) if selected.any() else None

    gate_groups = {}
    for name, selected in (("event", event), ("non_event", ~event)):
        values = probability[selected]
        gate_groups[name] = dict(count=len(values), mean=float(values.mean()) if len(values) else None,
                                 histogram=np.histogram(values, bins=edges)[0].tolist())
    error = prediction - truth
    return dict(samples=count, events=positives, event_prevalence=positives / count if count else None,
                brier=float(np.mean((probability - event) ** 2)) if count else None,
                average_precision=average_precision, pr_auc_trapezoid=pr_auc,
                precision=precision, recall=recall, reliability=reliability, gate_groups=gate_groups,
                event_window_rmse=rmse(error, event), non_event_window_rmse=rmse(error, ~event),
                body_rmse=rmse(body - np.minimum(truth, tau_phys), np.ones(count, dtype=bool)),
                excess_event_rmse=rmse(excess - np.maximum(truth - tau_phys, 0), event))
