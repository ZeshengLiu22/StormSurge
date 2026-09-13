"""Validation-only checkpoint selection, exact Pareto frontiers and safe artifact retention."""

from dataclasses import dataclass
import json
import math
import os
from pathlib import Path

import torch


SELECTION_METRICS = {
    "overall": "rmse_all",
    "peak5": "rmse_peak5",
    "peak_magnitude": "peak_magnitude_rmse_top5",
    "constrained_peak5": "rmse_peak5",
    "constrained_peak_magnitude": "peak_magnitude_rmse_top5",
}
SIMPLE_ROLES = ("overall", "peak5", "peak_magnitude")
PEAK_ROLES = ("peak5", "peak_magnitude")


def validate_checkpoint_settings(mode, overall_tol, save_aux):
    if mode not in SELECTION_METRICS:
        raise ValueError(f"Unknown checkpoint selection mode: {mode}")
    if not math.isfinite(overall_tol) or overall_tol < 0:
        raise ValueError("--checkpoint_overall_tol must be finite and nonnegative.")
    if save_aux not in (0, 1):
        raise ValueError("--save_aux_checkpoints must be a boolean.")


@dataclass(frozen=True)
class Candidate:
    epoch: int
    val: dict


def selection_key(candidate, role):
    """Overall preserves the earliest minimum; peaks break ties by overall then epoch."""
    if role == "overall":
        return candidate.val["rmse_all"], candidate.epoch
    return candidate.val[SELECTION_METRICS[role]], candidate.val["rmse_all"], candidate.epoch


def pareto_frontier(frontier, candidate, role):
    """Pure 2D update. Identical points retain the earliest epoch, in any insertion order."""
    metric = SELECTION_METRICS[role]

    def preferred(a, b):
        x, y = a.val["rmse_all"], a.val[metric]
        u, v = b.val["rmse_all"], b.val[metric]
        return x <= u and y <= v and (x < u or y < v or a.epoch <= b.epoch)

    if any(preferred(old, candidate) for old in frontier):
        return list(frontier)
    retained = [old for old in frontier if not preferred(candidate, old)]
    return sorted([*retained, candidate], key=lambda c: (c.val["rmse_all"], c.val[metric], c.epoch))


def constrained_winner(frontier, role, best_overall, overall_tol):
    """Resolve against the FINAL global best overall, with no slack or online cutoff."""
    limit = (1 + overall_tol) * best_overall
    eligible = [c for c in frontier if c.val["rmse_all"] <= limit]
    if not eligible:
        raise ValueError("No eligible validation checkpoint on the requested Pareto frontier.")
    return min(eligible, key=lambda c: selection_key(c, role)), dict(
        # A finite but enormous tolerance can overflow the product; every finite point is then eligible.
        overall_limit=limit if math.isfinite(limit) else "unbounded",
        eligible_candidate_count=len(eligible), frontier_candidate_count=len(frontier))


class CheckpointSelector:
    """Scalar VAL bookkeeping only. No model, loss, optimizer, scheduler or TEST input."""
    def __init__(self, mode="overall", overall_tol=.01, save_aux=False):
        validate_checkpoint_settings(mode, overall_tol, save_aux)
        self.mode, self.overall_tol, self.save_aux = mode, overall_tol, bool(save_aux)
        self.best = {role: None for role in SIMPLE_ROLES}
        self.frontiers = {role: [] for role in PEAK_ROLES if self.save_aux or mode == f"constrained_{role}"}
        self.last_epoch = 0

    @property
    def needs_candidates(self):
        return bool(self.frontiers) or self.save_aux

    def observe(self, epoch, val):
        if not isinstance(epoch, int) or isinstance(epoch, bool) or epoch <= self.last_epoch:
            raise ValueError("Validation epochs must be positive and strictly increasing.")
        for metric in (SELECTION_METRICS[role] for role in SIMPLE_ROLES):
            if metric not in val or val[metric] is None or not math.isfinite(val[metric]) or val[metric] < 0:
                raise ValueError(f"Checkpoint selection requires finite nonnegative VAL {metric}.")
        candidate = Candidate(epoch, dict(val))
        for role, current in self.best.items():
            if current is None or selection_key(candidate, role) < selection_key(current, role):
                self.best[role] = candidate
        for role, frontier in self.frontiers.items():
            self.frontiers[role] = pareto_frontier(frontier, candidate, role)
        self.last_epoch = epoch
        return candidate

    def retained_roles(self, epoch):
        roles = [role for role, c in self.best.items()
                 if c is not None and c.epoch == epoch and (self.save_aux or role == self.mode)]
        roles.extend(f"constrained_{role}" for role, frontier in self.frontiers.items()
                     if any(c.epoch == epoch for c in frontier))
        return roles

    def retained_epochs(self):
        epochs = {c.epoch for frontier in self.frontiers.values() for c in frontier}
        epochs.update(c.epoch for role, c in self.best.items()
                      if c is not None and (self.save_aux or role == self.mode))
        return epochs

    def resolve(self, role=None):
        role = self.mode if role is None else role
        if self.best["overall"] is None:
            raise ValueError("No completed validation epochs to select.")
        if role in SIMPLE_ROLES:
            return self.best[role], {}
        peak_role = role.removeprefix("constrained_")
        return constrained_winner(self.frontiers[peak_role], role,
                                  self.best["overall"].val["rmse_all"], self.overall_tol)

    def resolved_roles(self):
        roles = [self.mode, *[role for role in SELECTION_METRICS if role != self.mode]] if self.save_aux else [self.mode]
        return {role: self.resolve(role) for role in roles}

    def checkpoint_metadata(self, candidate, roles, *, provisional=False):
        role = roles[0]
        metric = SELECTION_METRICS[role]
        return dict(checkpoint_role="candidate" if provisional else role, checkpoint_roles=list(roles),
                    checkpoint_candidate=provisional, selection_metric_key=metric,
                    selection_metric_value=candidate.val[metric], checkpoint_selection_mode=self.mode,
                    checkpoint_overall_tol=self.overall_tol,
                    selection_by_role={r: dict(selection_metric_key=SELECTION_METRICS[r],
                                               selection_metric_value=candidate.val[SELECTION_METRICS[r]]) for r in roles})

    def summary(self, manifest_path=None):
        selected, constraint = self.resolve()
        result = dict(checkpoint_selection_mode=self.mode,
                      checkpoint_selection_metric_key=SELECTION_METRICS[self.mode],
                      checkpoint_overall_tol=self.overall_tol, save_aux_checkpoints=self.save_aux,
                      selected_epoch=selected.epoch, selected_val_rmse_all=selected.val["rmse_all"],
                      selected_val_rmse_peak5=selected.val["rmse_peak5"],
                      selected_val_peak_magnitude_rmse_top5=selected.val["peak_magnitude_rmse_top5"])
        for role, c in self.best.items():
            result[f"best_{role}_epoch"] = c.epoch
            result[f"best_{role}_val_{SELECTION_METRICS[role]}"] = c.val[SELECTION_METRICS[role]]
        result.update({f"selected_{name}": value for name, value in constraint.items()})
        if manifest_path is not None:
            result["manifest_path"] = str(manifest_path)
        return result


def _atomic_write(path, write):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        write(temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_save(checkpoint, path):
    _atomic_write(path, lambda temporary: torch.save(checkpoint, temporary))


def write_selection_manifest(selector, canonical, stem, role_paths):
    """Unavailable weight roles are null; shared paths explicitly list every retained role."""
    resolved = selector.resolved_roles()
    roles = dict.fromkeys(SELECTION_METRICS)
    for role, (candidate, constraint) in resolved.items():
        path = role_paths[role]
        shared = [other for other, other_path in role_paths.items() if other_path == path]
        roles[role] = dict(epoch=candidate.epoch, path=str(path), val=candidate.val,
                           **{name: candidate.val[name] for name in ("rmse_all", "rmse_peak5", "peak_magnitude_rmse_top5")},
                           selection_metric_key=SELECTION_METRICS[role], shared_roles=shared, **constraint)
    document = dict(checkpoint_selection_mode=selector.mode, checkpoint_overall_tol=selector.overall_tol,
                    save_aux_checkpoints=selector.save_aux,
                    primary=dict(role=selector.mode, **roles[selector.mode]), roles=roles)
    path = Path(canonical).parent / f"checkpoint_selection_{stem}.json"
    _atomic_write(path, lambda temporary: temporary.write_text(json.dumps(document, indent=2, allow_nan=False) + "\n"))
    return path


class CandidateCheckpointStore:
    """Rank-0 storage for the union of required frontiers/roles; never stores every epoch by default."""
    def __init__(self, canonical, stem):
        self.canonical, self.stem = Path(canonical), stem
        self.directory = self.canonical.parent / "checkpoint_candidates" / stem
        self.aux_directory = self.canonical.parent / "aux_checkpoints" / stem
        self.paths = {}

    def retain(self, selector, candidate, checkpoint_factory):
        needed = selector.retained_epochs()
        if candidate.epoch in needed:
            checkpoint = checkpoint_factory()
            checkpoint.update(selector.checkpoint_metadata(candidate, selector.retained_roles(candidate.epoch), provisional=True))
            path = self.directory / f"epoch_{candidate.epoch:04d}.pth"
            # Persist the replacement BEFORE deleting any dominated/unreferenced file.
            atomic_save(checkpoint, path)
            self.paths[candidate.epoch] = path
        for epoch in list(self.paths):
            if epoch not in needed:
                self.paths[epoch].unlink()
                del self.paths[epoch]

    def finalize(self, selector):
        resolved = selector.resolved_roles()
        by_epoch = {}
        for role, (candidate, _) in resolved.items():
            by_epoch.setdefault(candidate.epoch, []).append(role)
        selected, _ = selector.resolve()
        role_paths = {}
        # Keep every source candidate until ALL artifacts AND the manifest have been published.
        for epoch, roles in by_epoch.items():
            checkpoint = torch.load(self.paths[epoch], map_location="cpu", weights_only=False)
            candidate = resolved[roles[0]][0]
            if checkpoint["epoch"] != epoch or checkpoint["val"] != candidate.val:
                raise ValueError("Candidate checkpoint does not match its validation epoch/metrics.")
            checkpoint.update(selector.checkpoint_metadata(candidate, roles))
            destination = self.canonical if epoch == selected.epoch else self.aux_directory / f"epoch_{epoch:04d}.pth"
            atomic_save(checkpoint, destination)
            role_paths.update({role: destination for role in roles})
        manifest = write_selection_manifest(selector, self.canonical, self.stem, role_paths)
        for path in self.paths.values():
            path.unlink()
        self.paths.clear()
        for directory in (self.directory, self.directory.parent):
            try:
                directory.rmdir()
            except OSError:
                pass  # Other run stems or recovery files belong to their own writers.
        return manifest
