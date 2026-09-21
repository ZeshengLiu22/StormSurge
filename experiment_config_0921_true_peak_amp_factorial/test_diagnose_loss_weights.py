"""Focused correctness checks for the experimental diagnostic, using synthetic data."""
from copy import deepcopy
from pathlib import Path
import sys
import tempfile
import types
import unittest

import torch
from torch_geometric.data import Batch, Data

import diagnose_loss_weights as diagnostic

REPO = Path(sys.argv[1]) if len(sys.argv) > 1 and Path(sys.argv[1]).is_dir() else Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from emulator.models import ModelConfig, build_model
from emulator.training import ForecastLoss, LossConfig


class DiagnosticTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(42)
        self.model = build_model(ModelConfig(in_channels=2, out_channels=3, hidden_channels=8,
                                            history_steps=1, node_read_heads=2, time_read_heads=2,
                                            peak_threshold_norm=[.5, .5, .5], peak_prior=.5,
                                            dropout=.1, head_dropout=.1, temporal_dropout=.1))
        self.stats = dict(x_center=torch.zeros(2), x_scale=torch.ones(2),
                          y_mean=torch.zeros(3), y_std=torch.ones(3))
        self.fitted = dict(tail_threshold=.5, wmse_threshold=.5, tau_phys=.5, event_prior=.5)
        self.batches = [Batch.from_data_list([
            Data(x=torch.rand(2, 2), x_hist=torch.rand(2, 2, 2),
                 edge_index=torch.tensor([[0, 1], [1, 0]]), y=torch.tensor([[.2, 1.1, .6]])),
            Data(x=torch.rand(2, 2), x_hist=torch.rand(2, 2, 2),
                 edge_index=torch.tensor([[0, 1], [1, 0]]), y=torch.tensor([[.1, .4, .2]]))]) for _ in range(4)]

    def test_components_equal_production_objective_and_true_peak_gradient(self):
        output = self.model(self.batches[0])
        terms = diagnostic.component_losses(output, self.batches[0].y, self.stats, self.fitted)
        criterion = ForecastLoss(LossConfig(loss_mode="mse_tail", tail_lambda=.025,
                                            excess_loss_weight=2, gate_loss_weight=.5,
                                            excess_amp_loss_weight=.0007), self.stats, .5, .5,
                                 event_prior=.5, event_threshold=.5)
        expected = terms["dual_base"] + .025 * terms["tail"] + .0007 * terms["amp_true"]
        torch.testing.assert_close(expected, criterion(output, output.prediction, self.batches[0].y))
        derivative, = torch.autograd.grad(terms["amp_true"], output.excess)
        self.assertNotEqual(derivative[0, 1].item(), 0)
        mask = torch.ones_like(derivative, dtype=torch.bool)
        mask[0, 1] = False
        self.assertTrue((derivative[mask] == 0).all())
        target = torch.full_like(self.batches[0].y, -.2)
        output = self.model(self.batches[0])
        terms = diagnostic.component_losses(output, target, self.stats, self.fitted)
        derivative, = torch.autograd.grad(terms["amp_true"], output.excess)
        self.assertEqual(terms["amp_true"].item(), 0)
        self.assertTrue((derivative == 0).all())

    def test_preservation_full_gradients_accumulation_and_weight_scaling(self):
        # Nonzero branch output weights exercise shared gradients beyond epoch zero.
        with torch.no_grad():
            self.model.head.excess[-1].weight.normal_(0, .02)
        self.model.eval()
        for p in self.model.parameters():
            p.grad = torch.ones_like(p) * .012
        before = diagnostic.state_digest(self.model)
        rng = torch.get_rng_state().clone()
        rows = diagnostic.diagnose(self.model, self.batches, self.stats, self.fitted, None,
                                   torch.device("cpu"), "fixture", 20, use_amp=False)
        self.assertFalse(self.model.training)
        self.assertEqual(diagnostic.state_digest(self.model), before)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertTrue(all(torch.equal(p.grad, torch.full_like(p, .012)) for p in self.model.parameters()))
        self.assertEqual(len(rows), 5 * len(diagnostic.COMPONENTS) * len(diagnostic.GROUPS))
        self.assertTrue(all(row["grad_norm"] > 0 for row in rows if row["component"] == "amp_true"))
        for row in rows:
            self.assertAlmostEqual(row["ratio_base"], row["grad_norm"] / row["base_grad_norm"])
        candidates = diagnostic.analytic_candidates(rows)
        reference = {(r["unit"], r["unit_id"], r["component"], r["group"]): r for r in rows}
        for candidate in candidates:
            row = reference[tuple(candidate[k] for k in ("unit", "unit_id", "component", "group"))]
            self.assertAlmostEqual(candidate["weighted_loss"], candidate["weight"] * row["loss"])
            self.assertAlmostEqual(candidate["weighted_ratio_base"], candidate["weight"] * row["ratio_base"])
        # Independent combined-loss backward verifies accumulation of actual vectors.
        params = list(self.model.parameters())
        self.model.train()
        losses = []
        for i, batch in enumerate(self.batches):
            torch.manual_seed(420000 + i)
            terms = diagnostic.component_losses(self.model(batch), batch.y, self.stats, self.fitted)
            losses.append(terms["amp_true"])
        grads = torch.autograd.grad(torch.stack(losses).mean(), params, allow_unused=True)
        norm = sum(g.double().square().sum() for g in grads if g is not None).sqrt().item()
        recorded = next(r["grad_norm"] for r in rows if r["unit"] == "accum4"
                        and r["component"] == "amp_true" and r["group"] == "full")
        self.assertAlmostEqual(recorded, norm, places=7)

    def test_split_excludes_unreadable_holdout_files(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            # Ten chronological years -> six TRAIN; holdouts are invalid torch files.
            for year in range(2000, 2010):
                path = root / f"{year}_{year+1}_CBBT_peryear_hist48_graphs.pt"
                if year < 2006:
                    torch.save([Data(x=torch.zeros(2, 2), y=torch.zeros(3))], path)
                else:
                    path.write_text("This VAL/TEST file must never be opened")
            args = types.SimpleNamespace(root_dir=root, station="CBBT", train_ratio=.6, val_ratio=.2,
                                         shuffle_years=0, seed=42, future_only=0, future_year_threshold=2030)
            store, splits, opened = diagnostic.train_only_store(args)
            self.assertEqual(len(store.graphs), 6)
            self.assertEqual(len(opened), 6)
            self.assertEqual(len(splits["test"]), 2)
            self.assertEqual(len(splits["val"]), 2)

    def test_zero_denominator_is_undefined(self):
        self.assertIsNone(diagnostic.ratio(1., 0.))
        self.assertIsNone(diagnostic.ratio(0., 0.))


if __name__ == "__main__":
    unittest.main(argv=[sys.argv[0]], verbosity=2)
