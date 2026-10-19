"""Two checkpoint roles selected from one trajectory using VAL metrics only."""

import math
from pathlib import Path

from .checkpoints import atomic_save


ROLES = ("overall", "exceedance")
SELECTION_METRICS = dict(overall="all_rmse", exceedance="exceedance_rmse")


def checkpoint_scores(val):
    """Extract independent validation RMSE scores for the two checkpoint roles.

    Args:
        val: Validation metric mapping containing finite, nonnegative all_rmse and
            exceedance_rmse values in physical target units.

    Returns:
        dict: overall and exceedance mapped to their respective float RMSE scores.

    Raises:
        ValueError: Either required validation score is missing, nonfinite, or negative.
    """
    scores = {}
    for role in ROLES:
        key = SELECTION_METRICS[role]
        value = val.get(key)
        if value is None or not math.isfinite(value) or value < 0:
            raise ValueError(f"Checkpoint selection requires finite nonnegative VAL {key}; verify the fixed TRAIN threshold and validation extreme population.")
        scores[role] = float(value)
    return scores


class CheckpointTracker:
    """Retain the earliest strict validation RMSE minimum for each checkpoint role.

    The overall and exceedance roles are selected independently from one
    training trajectory. Only validation metrics participate in selection; the
    primary role controls later loading and reporting.
    """

    def __init__(self, output_dir):
        """Initialize role paths and selection state for a fresh training run.

        Args:
            output_dir: Directory for best_overall.pt and best_exceedance.pt; either
                existing role file causes an error.

        Returns:
            None. Initializes best candidates to None and last_epoch to zero. The directory
            is created when a checkpoint is first saved.

        Raises:
            ValueError: A role checkpoint already exists in the output directory.
        """
        self.output = Path(output_dir)
        self.best = dict.fromkeys(ROLES)
        self.paths = {role: self.output / f"best_{role}.pt" for role in ROLES}
        if any(path.exists() for path in self.paths.values()):
            raise ValueError("Checkpoint tracking requires a fresh run output directory.")
        self.last_epoch = 0

    def observe(self, epoch, val, checkpoint_factory):
        """Record one validation epoch and save every strictly improved role.

        Ties retain the earlier epoch. When both roles improve, their artifacts
        share the same factory snapshot and differ only in role-selection metadata.
        The method updates selection state and writes checkpoint files.

        Args:
            epoch: Positive integer epoch number, strictly greater than the previously
                observed epoch.
            val: Validation metrics containing both required RMSE scores.
            checkpoint_factory: Zero-argument callable returning the complete checkpoint
                dictionary. Called once only if a role improves; its epoch and val fields
                must match this observation.

        Returns:
            tuple: (scores, improved), where scores maps roles to their current RMSE and
            improved lists the roles saved for this epoch.

        Raises:
            ValueError: Epoch order, validation scores, or factory metadata is invalid.
        """
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
        """Describe the selected primary checkpoint and both retained role artifacts.

        Args:
            primary: Role used for primary loading/reporting: overall or exceedance.

        Returns:
            dict: Primary selection mode, metric and value, selected epoch and validation
            RMSEs, and per-role candidate metadata with checkpoint paths.

        Raises:
            ValueError: The role is unknown or no completed validation epoch has populated
                all roles.
        """
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
