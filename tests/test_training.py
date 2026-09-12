"""Global peak metrics, one-pass evaluation, accumulation and storage isolation."""

import math
import unittest

import numpy as np
import torch
from torch_geometric.data import Data

from emulator.data import build_loader, normalize_inputs
from emulator.models import ForecastOutput
from emulator.training import ForecastLoss, LossConfig, run_epoch
from emulator.training.metrics import METRIC_NAMES, summarize_windows


class CountingModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(.2))
        self.calls = 0

    def forward(self, batch, station_feat=None):
        self.calls += 1
        return ForecastOutput(self.weight * batch.x.reshape(-1, 1).expand(-1, 2))


class TrainingTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.stats = dict(x_center=torch.tensor([2.]), x_scale=torch.tensor([3.]),
                          y_mean=torch.tensor([10., -2.]), y_std=torch.tensor([2., 4.]))
        self.data = [Data(x=torch.tensor([[float(i)]]), x_hist=torch.tensor([[[float(i)]]]),
                          edge_index=torch.empty(2, 0, dtype=torch.long),
                          y=torch.tensor([[float(i), -float(i)]]), sample_id=torch.tensor([i]), tag=str(i))
                     for i in range(41)]

    def loader(self, data=None, batch_size=7):
        return build_loader(self.data if data is None else data, None, batch_size, 0, False, False, 0, "fork")

    def test_one_forward_physical_units_and_global_top_five_percent(self):
        model = CountingModel()
        result = run_epoch(model, self.loader(), torch.device("cpu"), self.stats, save_predictions=True)
        self.assertEqual(model.calls, math.ceil(41 / 7))
        expected = .2 * (np.arange(41) - 2)[:, None] / 3 * np.array([2, 4]) + np.array([10, -2])
        np.testing.assert_allclose(result.predictions["y_pred"], expected, atol=2e-6)
        truth = result.predictions["y_true"]
        error = expected - truth
        self.assertAlmostEqual(result.metrics["rmse_all"], float(np.sqrt(np.mean(error ** 2))), places=5)
        self.assertAlmostEqual(result.metrics["mae_all"], float(np.mean(np.abs(error))), places=5)
        top = np.argsort(truth.max(axis=1))[-3:]
        self.assertAlmostEqual(result.metrics["rmse_peak5"], float(np.sqrt(np.mean(error[top] ** 2))), places=5)
        self.assertAlmostEqual(result.metrics["mae_peak5"], float(np.mean(np.abs(error[top]))), places=5)
        for batch_size in (1, 11, 41):
            other = run_epoch(CountingModel(), self.loader(batch_size=batch_size), torch.device("cpu"), self.stats)
            # Original All uses FP32 batch means, so reduction rounding can vary.
            self.assertEqual(other.metrics.keys(), result.metrics.keys())
            for name, value in result.metrics.items():
                if value is None:
                    self.assertIsNone(other.metrics[name])
                else:
                    np.testing.assert_allclose(other.metrics[name], value, rtol=2e-7)

    def test_padding_dedup_and_ties_are_independent_of_batching(self):
        rows = np.array([[i, 1., i * i, i] for i in range(41)], dtype=float)
        expected = summarize_windows(rows)
        shuffled = np.concatenate((rows[::2], rows[1::2], rows[[0, 5, 12]]))
        self.assertEqual(summarize_windows(shuffled), expected)
        self.assertEqual(expected["mae_peak5"], 39.)
        self.assertEqual(summarize_windows(np.empty((0, 4))), dict.fromkeys(METRIC_NAMES))

    def test_normalization_does_not_mutate_aliased_source_features(self):
        source = torch.tensor([[5.]])
        sample = Data(x=source, x_hist=source.unsqueeze(1))
        normalize_inputs(sample, self.stats)
        torch.testing.assert_close(source, torch.tensor([[5.]]))
        torch.testing.assert_close(sample.x, torch.tensor([[1.]]))
        torch.testing.assert_close(sample.x_hist, torch.tensor([[[1.]]]))

    def test_accumulation_preserves_equal_microbatch_weight_with_short_groups(self):
        for length in (5, 9):
            accumulated = CountingModel()
            criterion = ForecastLoss(LossConfig(), self.stats, 1, 1)
            run_epoch(accumulated, self.loader(self.data[:length], 2), torch.device("cpu"), self.stats,
                      optimizer=torch.optim.SGD(accumulated.parameters(), lr=.01),
                      criterion=criterion, grad_accum_steps=3)
            reference = CountingModel()
            optimizer = torch.optim.SGD(reference.parameters(), lr=.01)
            batches = list(self.loader(self.data[:length], 2))
            for start in range(0, len(batches), 3):
                optimizer.zero_grad()
                group = batches[start:start + 3]
                losses = []
                for batch in group:
                    normalize_inputs(batch, self.stats)
                    output = reference(batch)
                    prediction = output.prediction * self.stats["y_std"] + self.stats["y_mean"]
                    losses.append(criterion(output, prediction, batch.y))
                torch.stack(losses).mean().backward()
                optimizer.step()
            torch.testing.assert_close(reference.weight, accumulated.weight, rtol=1e-6, atol=1e-6)
            self.assertEqual(accumulated.calls, math.ceil(length / 2))
