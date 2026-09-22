"""Two checkpoint roles selected from one trajectory using VAL metrics only."""

import math
from pathlib import Path

from .checkpoints import atomic_save


TERMS = ("all_rmse", "exceedance_rmse", "gt_aligned_peak_rmse")
DEFAULT_WEIGHTS = (.65, .20, .15)
ROLES = ("overall", "eventaware")


def validate_score_weights(weights):
    """Return three finite nonnegative weights summing to one."""
    try:
        weights = tuple(float(value) for value in weights)
    except (TypeError, ValueError) as error:
        raise ValueError("Checkpoint score requires three numeric weights.") from error
    if (len(weights) != len(TERMS) or any(not math.isfinite(value) or value < 0 for value in weights)
            or not math.isclose(sum(weights), 1., rel_tol=0., abs_tol=1e-12)):
        raise ValueError("Checkpoint score weights must be finite, nonnegative, and sum to one.")
    return weights


def eventaware_score(val, weights=DEFAULT_WEIGHTS):
    """Raw physical RMSE sum; accepts only the three canonical VAL fields.

    Undefined components cause a clear failure, including empty validation
    extreme populations. There is no reference scale or alternate selector.
    """
    weights = validate_score_weights(weights)
    values = []
    for key in TERMS:
        value = val.get(key)
        if value is None or not math.isfinite(value) or value < 0:
            raise ValueError(f"Event-aware checkpoint selection requires finite nonnegative VAL {key}; verify the fixed TRAIN threshold and validation extreme population.")
        values.append(float(value))
    score = sum(weight * value for weight, value in zip(weights, values))
    if not math.isfinite(score):
        raise ValueError("Event-aware VAL score is nonfinite.")
    return score


class EventAwareTracker:
    """Retain earliest strict minima for both roles, saving the same epoch state.

    ``observe`` accepts one completed epoch's VAL dictionary. Training, loss,
    optimizer, scheduler, and TEST results are outside this selection interface.
    The primary role affects later loading/reporting only.
    """

    def __init__(self, output_dir, weights=DEFAULT_WEIGHTS):
        self.output = Path(output_dir)
        self.weights = validate_score_weights(weights)
        self.settings = dict(weights=dict(zip(TERMS, self.weights)),
                             selection_split="val", score_formula="raw_weighted_sum")
        self.best = dict.fromkeys(ROLES)
        self.paths = {role: self.output / f"best_{role}.pt" for role in ROLES}
        if any(path.exists() for path in self.paths.values()):
            raise ValueError("Checkpoint tracking requires a fresh run output directory.")
        self.last_epoch = 0

    def observe(self, epoch, val, checkpoint_factory):
        if not isinstance(epoch, int) or isinstance(epoch, bool) or epoch <= self.last_epoch:
            raise ValueError("Validation epochs must be positive integers increasing strictly.")
        score = eventaware_score(val, self.weights)
        candidate = dict(epoch=epoch, val=dict(val), score=score)
        improved = [role for role in ROLES if self.best[role] is None
                    or (val["all_rmse"] if role == "overall" else score)
                    < (self.best[role]["val"]["all_rmse"] if role == "overall" else self.best[role]["score"])]
        if improved:
            # One snapshot ensures two winning roles use identical training state.
            original = checkpoint_factory()
            if original.get("epoch") != epoch or original.get("val") != val:
                raise ValueError("Checkpoint factory must preserve its observed VAL epoch and metrics.")
            for role in improved:
                checkpoint = dict(original)
                metric = "all_rmse" if role == "overall" else "eventaware_score"
                value = val["all_rmse"] if role == "overall" else score
                checkpoint.update(checkpoint_role=role, checkpoint_roles=[role],
                                  selection_metric_key=metric, selection_metric_value=value,
                                  selection_split="val", eventaware_score=score,
                                  eventaware_settings=self.settings)
                atomic_save(checkpoint, self.paths[role])
                self.best[role] = dict(candidate, val=dict(candidate["val"]))
        self.last_epoch = epoch
        return score, improved

    def selection_summary(self, primary="overall"):
        if primary not in ROLES:
            raise ValueError(f"Unknown checkpoint role: {primary}.")
        if any(candidate is None for candidate in self.best.values()):
            raise ValueError("No completed validation epochs to select.")
        selected = self.best[primary]
        return dict(checkpoint_selection_mode=primary,
                    checkpoint_selection_metric_key="all_rmse" if primary == "overall" else "eventaware_score",
                    selected_epoch=selected["epoch"],
                    selected_val_all_rmse=selected["val"]["all_rmse"],
                    selected_val_exceedance_rmse=selected["val"]["exceedance_rmse"],
                    selected_val_gt_aligned_peak_rmse=selected["val"]["gt_aligned_peak_rmse"],
                    selected_eventaware_score=selected["score"],
                    eventaware_settings=self.settings,
                    checkpoints={role: dict(candidate, path=str(self.paths[role]))
                                 for role, candidate in self.best.items()})
