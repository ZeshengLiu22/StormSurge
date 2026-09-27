"""Two checkpoint roles selected from one trajectory using VAL metrics only."""

import math
from pathlib import Path

from .checkpoints import atomic_save


ROLES = ("overall", "exceedance")
SELECTION_METRICS = dict(overall="all_rmse", exceedance="exceedance_rmse")


def checkpoint_scores(val):
    """Independent physical VAL RMSE minima, with no composite selector."""
    scores = {}
    for role in ROLES:
        key = SELECTION_METRICS[role]
        value = val.get(key)
        if value is None or not math.isfinite(value) or value < 0:
            raise ValueError(f"Checkpoint selection requires finite nonnegative VAL {key}; verify the fixed TRAIN threshold and validation extreme population.")
        scores[role] = float(value)
    return scores


class CheckpointTracker:
    """Retain earliest strict minima for all roles, saving the same epoch state.

    ``observe`` accepts one completed epoch's VAL dictionary. Training, loss,
    optimizer, scheduler, and TEST results are outside this selection interface.
    The primary role affects later loading/reporting only.
    """

    def __init__(self, output_dir):
        self.output = Path(output_dir)
        self.best = dict.fromkeys(ROLES)
        self.paths = {role: self.output / f"best_{role}.pt" for role in ROLES}
        if any(path.exists() for path in self.paths.values()):
            raise ValueError("Checkpoint tracking requires a fresh run output directory.")
        self.last_epoch = 0

    def observe(self, epoch, val, checkpoint_factory):
        if not isinstance(epoch, int) or isinstance(epoch, bool) or epoch <= self.last_epoch:
            raise ValueError("Validation epochs must be positive integers increasing strictly.")
        scores = checkpoint_scores(val)
        candidate = dict(epoch=epoch, val=dict(val))
        improved = [role for role in ROLES if self.best[role] is None
                    or scores[role] < self.best[role]["selection_metric_value"]]
        if improved:
            # One snapshot ensures all winning roles use identical training state.
            original = checkpoint_factory()
            if original.get("epoch") != epoch or original.get("val") != val:
                raise ValueError("Checkpoint factory must preserve its observed VAL epoch and metrics.")
            for role in improved:
                checkpoint = dict(original)
                metric, value = SELECTION_METRICS[role], scores[role]
                checkpoint.update(checkpoint_role=role, checkpoint_roles=[role],
                                  selection_metric_key=metric, selection_metric_value=value,
                                  selection_split="val")
                atomic_save(checkpoint, self.paths[role])
                self.best[role] = dict(candidate, val=dict(candidate["val"]),
                                       selection_metric_key=metric, selection_metric_value=value)
        self.last_epoch = epoch
        return scores, improved

    def selection_summary(self, primary="exceedance"):
        if primary not in ROLES:
            raise ValueError(f"Unknown checkpoint role: {primary}.")
        if any(candidate is None for candidate in self.best.values()):
            raise ValueError("No completed validation epochs to select.")
        selected = self.best[primary]
        return dict(checkpoint_selection_mode=primary,
                    checkpoint_selection_metric_key=SELECTION_METRICS[primary],
                    checkpoint_selection_metric_value=selected["selection_metric_value"],
                    selected_epoch=selected["epoch"],
                    selected_val_all_rmse=selected["val"]["all_rmse"],
                    selected_val_exceedance_rmse=selected["val"]["exceedance_rmse"],
                    checkpoints={role: dict(candidate, path=str(self.paths[role]))
                                 for role, candidate in self.best.items()})
