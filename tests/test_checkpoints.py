"""Exact VAL-only selection and recoverable, deduplicated checkpoint storage."""

import json
import math
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import torch

from emulator.training import checkpoints as cp


def metrics(overall, peak5, magnitude):
    return dict(rmse_all=overall, rmse_peak5=peak5, peak_magnitude_rmse_top5=magnitude,
                mae_all=overall / 2, mae_peak5=peak5 / 2)


CONFLICTING = [metrics(*row) for row in ((1.04, .5, .9), (1.02, .6, .4), (1., .8, .8),
                                        (1.005, .7, .6), (1.008, .9, .5), (1., .85, .75))]
WINNERS = dict(overall=3, peak5=1, peak_magnitude=2, constrained_peak5=4, constrained_peak_magnitude=5)


def observe_all(selector, values):
    for epoch, val in enumerate(values, 1):
        selector.observe(epoch, val)
    return selector


def snapshot(candidate):
    return dict(epoch=candidate.epoch, val=candidate.val,
                model_state={"weight": torch.tensor([candidate.epoch], dtype=torch.float64)})


class SelectorTests(unittest.TestCase):
    def test_all_five_distinct_winners_and_summary_source_keys(self):
        for mode, epoch in WINNERS.items():
            with self.subTest(mode=mode):
                selector = observe_all(cp.CheckpointSelector(mode), CONFLICTING)
                self.assertEqual(selector.resolve()[0].epoch, epoch)
                summary = selector.summary()
                self.assertEqual(summary["selected_epoch"], epoch)
                self.assertEqual(summary["checkpoint_selection_metric_key"], cp.SELECTION_METRICS[mode])
                for role in cp.SIMPLE_ROLES:
                    self.assertEqual(summary[f"best_{role}_epoch"], WINNERS[role])
                    self.assertEqual(summary[f"best_{role}_val_{cp.SELECTION_METRICS[role]}"],
                                     CONFLICTING[WINNERS[role] - 1][cp.SELECTION_METRICS[role]])
                if mode.startswith("constrained_"):
                    self.assertEqual(summary["selected_overall_limit"], 1.01)
                    self.assertGreater(summary["selected_eligible_candidate_count"], 0)

    def test_ties_follow_metric_overall_then_earliest_with_overall_exception(self):
        values = [metrics(2., .5, .5), metrics(1., .5, .5), metrics(1., .4, .4), metrics(1., .4, .4)]
        selector = observe_all(cp.CheckpointSelector(save_aux=True), values)
        self.assertEqual(selector.resolve("overall")[0].epoch, 2)
        for role in cp.SELECTION_METRICS:
            if role != "overall":
                self.assertEqual(selector.resolve(role)[0].epoch, 3)

    def test_zero_tolerance_has_no_slack_and_peak_ties_still_apply(self):
        values = [metrics(1., 2., 2.), metrics(math.nextafter(1., math.inf), 0., 0.), metrics(1., 1., 1.)]
        selector = observe_all(cp.CheckpointSelector(overall_tol=0., save_aux=True), values)
        self.assertEqual(selector.resolve("overall")[0].epoch, 1)
        for role in cp.PEAK_ROLES:
            candidate, info = selector.resolve(f"constrained_{role}")
            self.assertEqual(candidate.epoch, 3)
            self.assertEqual(info["overall_limit"], 1.)
        zero = observe_all(cp.CheckpointSelector(save_aux=True), [metrics(0., 3., 3.), metrics(1e-100, 0., 0.)])
        self.assertEqual(zero.resolve("constrained_peak5")[0].epoch, 1)

    def test_edge_sequences_match_full_history(self):
        sequences = [[metrics(1, 2, 3)], [metrics(1, 2, 3)] * 5,
                     [metrics(10 - i, 10 - i, i + 1) for i in range(10)],
                     [metrics(i + 1, 10 - i, 10 - i) for i in range(10)],
                     [metrics(10, 1, 1), metrics(9, 2, 2), metrics(1, 3, 3)],
                     [metrics(1, 9, 9), metrics(1.005, 8, 8), metrics(1.008, .1, .1)]]
        for values in sequences:
            for tolerance in (0., .01, 1., 1e100):
                selector = observe_all(cp.CheckpointSelector(overall_tol=tolerance, save_aux=True), values)
                limit = min(v["rmse_all"] for v in values) * (1 + tolerance)
                for role in cp.PEAK_ROLES:
                    metric = cp.SELECTION_METRICS[role]
                    expected = min((i + 1 for i, v in enumerate(values) if v["rmse_all"] <= limit),
                                   key=lambda e: (values[e - 1][metric], values[e - 1]["rmse_all"], e))
                    self.assertEqual(selector.resolve(f"constrained_{role}")[0].epoch, expected)

    def test_randomized_frontiers_equal_brute_force_for_all_insertion_orders(self):
        rng = random.Random(7036)
        # 250 histories x 2 targets x 6 tolerances x 3 insertion orders = 9,000 exact checks.
        for _ in range(250):
            points = [cp.Candidate(i + 1, metrics(*(rng.randrange(0, 100) / 20 for _ in range(3))))
                      for i in range(rng.randrange(1, 81))]
            minimum = min(p.val["rmse_all"] for p in points)
            shuffled = rng.sample(points, len(points))
            for role in cp.PEAK_ROLES:
                metric = cp.SELECTION_METRICS[role]
                expected_frontier = None
                for order in (points, list(reversed(points)), shuffled):
                    frontier = []
                    for point in order:
                        frontier = cp.pareto_frontier(frontier, point, role)
                    epochs = [p.epoch for p in frontier]
                    if expected_frontier is None:
                        expected_frontier = epochs
                    self.assertEqual(epochs, expected_frontier)
                    for tolerance in (0., .001, .01, .1, 1., 1e6):
                        feasible = [p for p in points if p.val["rmse_all"] <= (1 + tolerance) * minimum]
                        expected = min(feasible, key=lambda p: (p.val[metric], p.val["rmse_all"], p.epoch))
                        actual, info = cp.constrained_winner(frontier, role, minimum, tolerance)
                        self.assertEqual(actual.epoch, expected.epoch)
                        self.assertEqual(info["frontier_candidate_count"], len(frontier))
                        self.assertEqual(info["eligible_candidate_count"],
                                         sum(p.val["rmse_all"] <= info["overall_limit"] for p in frontier))

    def test_frontier_storage_is_only_enabled_for_requested_roles(self):
        for mode in cp.SELECTION_METRICS:
            for auxiliary in (False, True):
                selector = observe_all(cp.CheckpointSelector(mode, save_aux=auxiliary), CONFLICTING)
                expected = set(cp.PEAK_ROLES) if auxiliary else (
                    {mode.removeprefix("constrained_")} if mode.startswith("constrained_") else set())
                self.assertEqual(set(selector.frontiers), expected)
                self.assertEqual(selector.needs_candidates, bool(expected))
                self.assertEqual(set(selector.resolved_roles()), set(cp.SELECTION_METRICS) if auxiliary else {mode})

    def test_invalid_settings_missing_metrics_and_epoch_order_fail_explicitly(self):
        for tolerance in (-.01, float("nan"), float("inf"), -float("inf")):
            with self.assertRaisesRegex(ValueError, "finite and nonnegative"):
                cp.CheckpointSelector(overall_tol=tolerance)
        with self.assertRaises(ValueError):
            cp.CheckpointSelector("event")
        with self.assertRaises(ValueError):
            cp.CheckpointSelector(save_aux=2)
        selector = cp.CheckpointSelector()
        with self.assertRaisesRegex(ValueError, "No completed"):
            selector.resolve()
        for epoch in (0, -1, True, 1.5):
            with self.assertRaises(ValueError):
                selector.observe(epoch, CONFLICTING[0])
        for metric in (cp.SELECTION_METRICS[r] for r in cp.SIMPLE_ROLES):
            for bad in (None, float("nan"), float("inf"), -.1):
                with self.assertRaisesRegex(ValueError, metric):
                    selector.observe(1, dict(CONFLICTING[0], **{metric: bad}))
            missing = dict(CONFLICTING[0])
            missing.pop(metric)
            with self.assertRaisesRegex(ValueError, metric):
                selector.observe(1, missing)
        selector.observe(1, CONFLICTING[0])
        with self.assertRaises(ValueError):
            selector.observe(1, CONFLICTING[1])

    def test_finite_extreme_tolerance_overflow_is_unbounded_and_json_safe(self):
        selector = observe_all(cp.CheckpointSelector("constrained_peak5", 1e308),
                               [metrics(10., 4., 4.), metrics(20., 2., 2.)])
        candidate, info = selector.resolve()
        self.assertEqual(candidate.epoch, 2)
        self.assertEqual(info["overall_limit"], "unbounded")
        json.dumps(selector.summary(), allow_nan=False)


class CheckpointStoreTests(unittest.TestCase):
    def add(self, store, selector, epoch, val):
        candidate = selector.observe(epoch, val)
        store.retain(selector, candidate, lambda: snapshot(candidate))
        return candidate

    def test_every_role_loads_exact_epoch_state_and_unique_canonical(self):
        for mode in cp.SELECTION_METRICS:
            for auxiliary in (False, True):
                if not auxiliary and mode in cp.SIMPLE_ROLES:
                    continue  # These modes use the trainer's direct canonical path, tested in the pipeline.
                with self.subTest(mode=mode, auxiliary=auxiliary), tempfile.TemporaryDirectory() as tmp:
                    root = Path(tmp)
                    selector = cp.CheckpointSelector(mode, save_aux=auxiliary)
                    store = cp.CandidateCheckpointStore(root / "best_run.pth", "run")
                    for epoch, val in enumerate(CONFLICTING, 1):
                        self.add(store, selector, epoch, val)
                        self.assertEqual(set(store.paths), selector.retained_epochs())
                        self.assertFalse(list(root.rglob("best_*.pth")))
                    manifest = json.loads(store.finalize(selector).read_text())
                    self.assertEqual(len(list(root.rglob("best_*.pth"))), 1)
                    self.assertFalse(store.directory.exists())
                    available = set(cp.SELECTION_METRICS) if auxiliary else {mode}
                    for role, entry in manifest["roles"].items():
                        if role not in available:
                            self.assertIsNone(entry)
                            continue
                        checkpoint = torch.load(entry["path"], weights_only=False)
                        expected = WINNERS[role]
                        self.assertEqual((entry["epoch"], checkpoint["epoch"]), (expected, expected))
                        self.assertEqual(checkpoint["val"], CONFLICTING[expected - 1])
                        self.assertEqual(entry["val"], checkpoint["val"])
                        self.assertEqual(checkpoint["model_state"]["weight"].item(), expected)
                        self.assertIn(role, checkpoint["checkpoint_roles"])
                        self.assertEqual(checkpoint["checkpoint_role"], checkpoint["checkpoint_roles"][0])
                        self.assertFalse(checkpoint["checkpoint_candidate"])
                        self.assertEqual(entry["selection_metric_key"], cp.SELECTION_METRICS[role])
                        self.assertEqual(checkpoint["selection_by_role"][role]["selection_metric_value"],
                                         checkpoint["val"][cp.SELECTION_METRICS[role]])
                    self.assertEqual(manifest["primary"]["epoch"], WINNERS[mode])

    def test_all_roles_sharing_one_epoch_only_write_one_final_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            selector = cp.CheckpointSelector("constrained_peak_magnitude", save_aux=True)
            store = cp.CandidateCheckpointStore(root / "best_run.pth", "run")
            self.add(store, selector, 1, metrics(2., 2., 2.))
            self.add(store, selector, 2, metrics(1., 1., 1.))
            self.assertEqual(set(store.paths), {2})
            manifest = json.loads(store.finalize(selector).read_text())
            self.assertEqual(len(list(root.rglob("*.pth"))), 1)
            self.assertFalse(store.aux_directory.exists())
            for entry in manifest["roles"].values():
                self.assertEqual(entry["path"], str(store.canonical))
                self.assertEqual(set(entry["shared_roles"]), set(cp.SELECTION_METRICS))
            checkpoint = torch.load(store.canonical, weights_only=False)
            self.assertEqual(checkpoint["checkpoint_role"], selector.mode)
            self.assertEqual(set(checkpoint["checkpoint_roles"]), set(cp.SELECTION_METRICS))

    def test_one_frontier_cannot_delete_another_frontiers_copy(self):
        with tempfile.TemporaryDirectory() as tmp:
            selector = cp.CheckpointSelector(save_aux=True)
            store = cp.CandidateCheckpointStore(Path(tmp) / "best_run.pth", "run")
            self.add(store, selector, 1, metrics(1., 1., .1))
            first = store.paths[1]
            self.add(store, selector, 2, metrics(.9, .9, .2))
            self.assertNotIn(1, [p.epoch for p in selector.frontiers["peak5"]])
            self.assertTrue(first.exists())
            self.assertEqual(set(store.paths), {1, 2})
            self.add(store, selector, 3, metrics(.8, .8, .05))
            self.assertFalse(first.exists())
            self.assertEqual(set(store.paths), {3})

    def test_scalar_overall_role_retains_earliest_even_when_both_frontiers_drop_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            selector = cp.CheckpointSelector(save_aux=True)
            store = cp.CandidateCheckpointStore(Path(tmp) / "best_run.pth", "run")
            self.add(store, selector, 1, metrics(1., 2., 2.))
            self.add(store, selector, 2, metrics(1., 1., 1.))
            self.assertEqual(set(store.paths), {1, 2})
            self.assertEqual(selector.retained_roles(1), ["overall"])
            manifest = json.loads(store.finalize(selector).read_text())
            self.assertEqual(manifest["roles"]["overall"]["epoch"], 1)
            self.assertEqual(manifest["roles"]["constrained_peak5"]["epoch"], 2)

    def test_failed_new_save_keeps_old_dominated_candidate_and_removes_partial_temp(self):
        with tempfile.TemporaryDirectory() as tmp:
            selector = cp.CheckpointSelector("constrained_peak5")
            store = cp.CandidateCheckpointStore(Path(tmp) / "best_run.pth", "run")
            self.add(store, selector, 1, metrics(2., 2., 2.))
            previous = store.paths[1]

            def failing_save(checkpoint, path):
                Path(path).write_bytes(b"partial")
                self.assertTrue(previous.exists())
                raise OSError("disk full")

            with patch.object(cp.torch, "save", side_effect=failing_save), self.assertRaisesRegex(OSError, "disk full"):
                self.add(store, selector, 2, metrics(1., 1., 1.))
            self.assertEqual(set(store.paths), {1})
            self.assertTrue(previous.exists())
            self.assertEqual(list(store.directory.iterdir()), [previous])

    def test_new_candidate_is_persisted_before_obsolete_files_are_unlinked(self):
        with tempfile.TemporaryDirectory() as tmp:
            selector = cp.CheckpointSelector("constrained_peak5")
            store = cp.CandidateCheckpointStore(Path(tmp) / "best_run.pth", "run")
            self.add(store, selector, 1, metrics(2., 2., 2.))
            old = store.paths[1]
            save = cp.atomic_save

            def checked_save(checkpoint, path):
                self.assertTrue(old.exists())
                save(checkpoint, path)
                self.assertTrue(old.exists())
                self.assertEqual(torch.load(path, weights_only=False)["epoch"], 2)

            with patch.object(cp, "atomic_save", side_effect=checked_save):
                self.add(store, selector, 2, metrics(1., 1., 1.))
            self.assertFalse(old.exists())

    def test_failed_final_materialization_or_manifest_preserves_all_sources_and_retry(self):
        for failure in ("aux_save", "manifest"):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as tmp:
                selector = cp.CheckpointSelector(save_aux=True)
                store = cp.CandidateCheckpointStore(Path(tmp) / "best_run.pth", "run")
                for epoch, val in enumerate(CONFLICTING, 1):
                    self.add(store, selector, epoch, val)
                sources = dict(store.paths)
                save = cp.atomic_save

                def fail_aux(checkpoint, path):
                    if Path(path) != store.canonical:
                        raise OSError("finalization failed")
                    save(checkpoint, path)

                target = "atomic_save" if failure == "aux_save" else "write_selection_manifest"
                effect = fail_aux if failure == "aux_save" else OSError("finalization failed")
                with patch.object(cp, target, side_effect=effect), self.assertRaisesRegex(OSError, "finalization failed"):
                    store.finalize(selector)
                self.assertEqual(store.paths, sources)
                self.assertTrue(all(path.exists() for path in sources.values()))
                manifest = store.finalize(selector)
                self.assertTrue(manifest.exists())
                self.assertFalse(any(path.exists() for path in sources.values()))

    def test_cleanup_leaves_other_runs_and_recovery_files_untouched(self):
        with tempfile.TemporaryDirectory() as tmp:
            selector = cp.CheckpointSelector("constrained_peak5")
            store = cp.CandidateCheckpointStore(Path(tmp) / "best_run.pth", "run")
            self.add(store, selector, 1, metrics(1., 1., 1.))
            recovery = store.directory / "recovery.pth"
            recovery.write_bytes(b"recoverable")
            other = store.directory.parent / "another_run"
            other.mkdir()
            store.finalize(selector)
            self.assertEqual(recovery.read_bytes(), b"recoverable")
            self.assertTrue(other.exists())

    def test_corrupt_candidate_aborts_without_deleting_sources(self):
        with tempfile.TemporaryDirectory() as tmp:
            selector = cp.CheckpointSelector("constrained_peak5")
            store = cp.CandidateCheckpointStore(Path(tmp) / "best_run.pth", "run")
            self.add(store, selector, 1, metrics(1., 1., 1.))
            path = store.paths[1]
            checkpoint = torch.load(path, weights_only=False)
            checkpoint["epoch"] = 99
            torch.save(checkpoint, path)
            with self.assertRaisesRegex(ValueError, "does not match"):
                store.finalize(selector)
            self.assertTrue(path.exists())
            self.assertFalse(store.canonical.exists())


if __name__ == "__main__":
    unittest.main()
