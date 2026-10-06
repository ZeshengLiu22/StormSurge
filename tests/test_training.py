"""Global peak metrics, one-pass evaluation, accumulation and storage isolation."""

import math
import unittest

import numpy as np
import torch
from torch_geometric.data import Data

from emulator.data import build_loader, normalize_inputs
from emulator.models import ForecastOutput
from emulator.training import ForecastLoss, LossConfig, run_epoch
from emulator.training.metrics import evaluate_metrics


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
                          y=torch.tensor([[float(i), -float(i)]]), sample_id=torch.tensor([i]), tag=str(i),
                          target_timestamps=torch.tensor([[i*7200, i*7200+3600]]))
                     for i in range(41)]

    def loader(self, data=None, batch_size=7):
        return build_loader(self.data if data is None else data, None, batch_size, 0, False, False, 0, "fork")

    def test_one_forward_physical_units_and_fixed_hourly_threshold(self):
        model = CountingModel()
        result = run_epoch(model, self.loader(), torch.device("cpu"), self.stats, tau_physical=20., save_predictions=True)
        self.assertEqual(model.calls, math.ceil(41 / 7))
        expected = .2 * (np.arange(41) - 2)[:, None] / 3 * np.array([2, 4]) + np.array([10, -2])
        np.testing.assert_allclose(result.predictions["y_pred"], expected, atol=2e-6)
        truth = result.predictions["y_true"]
        error = expected - truth
        self.assertAlmostEqual(result.metrics["all_rmse"], float(np.sqrt(np.mean(error ** 2))), places=5)
        self.assertAlmostEqual(result.metrics["all_mae"], float(np.mean(np.abs(error))), places=5)
        extreme = truth > 20.
        self.assertAlmostEqual(result.metrics["exceedance_rmse"], float(np.sqrt(np.mean(error[extreme] ** 2))), places=5)
        self.assertAlmostEqual(result.metrics["exceedance_mae"], float(np.mean(np.abs(error[extreme]))), places=5)
        for batch_size in (1, 11, 41):
            other = run_epoch(CountingModel(), self.loader(batch_size=batch_size), torch.device("cpu"), self.stats, tau_physical=20.)
            # Original All uses FP32 batch means, so reduction rounding can vary.
            self.assertTrue(set(other.metrics).issubset(result.metrics))
            for name, value in other.metrics.items():
                if value is None:
                    self.assertIsNone(other.metrics[name])
                else:
                    np.testing.assert_allclose(result.metrics[name], value, rtol=2e-7)

    def test_hourly_metrics_ignore_only_nonfinite_target_elements(self):
        truth = np.array([[1., np.nan, 4.], [3., 5., 2.]])
        prediction = np.array([[3., np.nan, 1.], [5., 9., 2.]])
        metrics = evaluate_metrics(prediction, truth, 2.)
        self.assertAlmostEqual(metrics["all_rmse"], math.sqrt(33 / 5))
        self.assertAlmostEqual(metrics["all_mae"], 11 / 5)
        self.assertAlmostEqual(metrics["exceedance_rmse"], math.sqrt(29 / 3))
        self.assertAlmostEqual(metrics["exceedance_mae"], 3.)
        # The two finite targets in the first window still contribute.
        complete_only = evaluate_metrics(prediction[1:], truth[1:], 2.)
        self.assertNotEqual(metrics["all_rmse"], complete_only["all_rmse"])
        self.assertNotEqual(metrics["exceedance_rmse"], complete_only["exceedance_rmse"])
        # With no missing targets the original finite-data reductions are exact.
        truth[0, 1], prediction[0, 1] = 6., 7.
        finite = evaluate_metrics(prediction, truth, 2.)
        error = prediction - truth
        self.assertEqual(finite["all_rmse"], float(np.sqrt(np.mean(error ** 2))))
        self.assertEqual(finite["all_mae"], float(np.mean(np.abs(error))))
        self.assertEqual(finite["exceedance_rmse"], float(np.sqrt(np.mean(error[truth > 2.] ** 2))))
        self.assertEqual(finite["exceedance_mae"], float(np.mean(np.abs(error[truth > 2.]))))

    def test_nonfinite_predictions_fail_where_targets_are_finite(self):
        truth = np.array([[np.nan, 4.], [3., 5.]])
        prediction = np.array([[np.nan, 3.], [2., 4.]])
        times = np.arange(truth.size).reshape(truth.shape) * 3600
        for bad in (np.nan, np.inf, -np.inf):
            for index in ((0, 1), (1, 0)):
                with self.subTest(bad=bad, index=index):
                    invalid = prediction.copy()
                    invalid[index] = bad
                    with self.assertRaisesRegex(ValueError, "nonfinite y_pred.*y_true is finite"):
                        evaluate_metrics(invalid, truth, 2., target_timestamps=times)
        missing = evaluate_metrics(np.full((1, 2), np.nan), np.full((1, 2), np.nan), 2.,
                                   target_timestamps=times[:1])
        self.assertTrue(all(value is None for value in missing.values()))

    def test_nan_target_survives_training_and_prediction_export(self):
        data = [sample.clone() for sample in self.data[:5]]
        data[2].y[0, 0] = float("nan")
        model = CountingModel()
        criterion = ForecastLoss(LossConfig(exceedance_loss_weight=1.), self.stats, 1.,
                                 extreme_hour_prior=.25)
        result = run_epoch(model, self.loader(data, batch_size=2), torch.device("cpu"), self.stats,
                           tau_physical=1., optimizer=torch.optim.SGD(model.parameters(), lr=.001),
                           criterion=criterion, save_predictions=True)
        self.assertTrue(torch.isfinite(model.weight))
        self.assertTrue(np.isnan(result.predictions["y_true"][2, 0]))
        self.assertTrue(all(value is None or np.isfinite(value) for value in result.metrics.values()))
        self.assertTrue(all(value is None or np.isfinite(value) for value in result.losses.values()))
        truth, prediction = result.predictions["y_true"], result.predictions["y_pred"]
        valid = np.isfinite(truth)
        error = prediction.astype(np.float64)[valid] - truth[valid]
        self.assertEqual(result.metrics["all_rmse"], float(np.sqrt(np.mean(error ** 2))))
        self.assertEqual(result.metrics["all_mae"], float(np.mean(np.abs(error))))

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
            criterion = ForecastLoss(LossConfig(), self.stats, 1)
            run_epoch(accumulated, self.loader(self.data[:length], 2), torch.device("cpu"), self.stats, tau_physical=20.,
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
