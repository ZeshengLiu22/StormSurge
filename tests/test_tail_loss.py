"""Sample-additive tail objectives, CLI wiring, accumulation, and CPU DDP."""

import contextlib
from dataclasses import replace
from datetime import timedelta
import io
import itertools
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel
from torch_geometric.data import Batch, Data

import train
from emulator.data import ForcingGraphStore, build_loader
from emulator.models import ForecastOutput
from emulator.training import ForecastLoss, LossConfig, run_epoch
from test_pipeline import make_fixture


CORES = ("mse_tail", "wmse_tail", "mse_wtail")
TAIL_MODES = CORES + tuple(mode + "_slope" for mode in CORES)


def stats(dtype=torch.float32):
    return dict(x_center=torch.zeros(1, dtype=dtype), x_scale=torch.ones(1, dtype=dtype),
                y_mean=torch.zeros(2, dtype=dtype), y_std=torch.ones(2, dtype=dtype))


def tail_component(prediction, target, config):
    """Remove the unchanged base/slope terms to observe only the tail contribution."""
    normal = stats(prediction.dtype)
    output = ForecastOutput(prediction)
    enabled = ForecastLoss(config, normal, 1., 0.)(output, prediction, target)
    disabled = ForecastLoss(replace(config, tail_lambda=0.), normal, 1., 0.)(output, prediction, target)
    return enabled - disabled


class TailLinearModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([[.2], [-.15]]))
        self.bias = torch.nn.Parameter(torch.tensor([.1, -.2]))

    def forward(self, batch, station_feat=None):
        return ForecastOutput(torch.nn.functional.linear(batch.x, self.weight, self.bias))


def samples(dtype=torch.float32):
    # The first four windows contain no tail events; the next four all do.
    targets = [[.1, .2], [.3, .2], [-2., -1.], [.8, .9], [1., .4],
               [1.5, .2], [2., -.5], [1.2, 1.8], [-.5, .5], [1.1, .7]]
    return [Data(x=torch.tensor([[(index + 1) / 5]], dtype=dtype),
                 x_hist=torch.tensor([[[(index + 1) / 5]]], dtype=dtype),
                 edge_index=torch.empty(2, 0, dtype=torch.long),
                 y=torch.tensor([target], dtype=dtype), sample_id=torch.tensor([index]), tag=str(index))
            for index, target in enumerate(targets)]


def loader(data, batch_size):
    return build_loader(data, None, batch_size, 0, False, False, 0, "fork")


def ddp_tail_worker(rank, rendezvous, result_dir):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=Path(rendezvous).as_uri(), rank=rank, world_size=2,
                            timeout=timedelta(seconds=45))
    completed = []
    try:
        for mode in TAIL_MODES:
            config = LossConfig(loss_mode=mode, tail_frac=.25, tail_lambda=.3, slope_lambda=.2)
            data = samples(torch.float64)[:8]
            local = Batch.from_data_list(data[rank * 4:(rank + 1) * 4])
            full = Batch.from_data_list(data)
            assert int((local.y.amax(dim=1) >= 1.).sum()) == (0 if rank == 0 else 4)
            network = DistributedDataParallel(TailLinearModel().double())
            reference = TailLinearModel().double()
            tail_component(network(local).prediction, local.y, config).backward()
            tail_component(reference(full).prediction, full.y, config).backward()
            for actual, expected in zip(network.module.parameters(), reference.parameters()):
                torch.testing.assert_close(actual.grad, expected.grad, rtol=1e-12, atol=1e-12)

            # Exercise the unchanged engine's no_sync and microbatch normalization.
            data = samples()[:8]
            network = DistributedDataParallel(TailLinearModel())
            reference = TailLinearModel()
            normal = stats()
            criterion = ForecastLoss(config, normal, 1., 0.)
            run_epoch(network, loader(data[rank * 4:(rank + 1) * 4], 2), torch.device("cpu"), normal,
                      optimizer=torch.optim.SGD(network.parameters(), lr=.005), criterion=criterion,
                      grad_accum_steps=2, distributed=True)
            run_epoch(reference, loader(data, 8), torch.device("cpu"), normal,
                      optimizer=torch.optim.SGD(reference.parameters(), lr=.005), criterion=criterion)
            for actual, expected in zip(network.module.parameters(), reference.parameters()):
                torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)
            completed.append(mode)
        (Path(result_dir) / f"rank{rank}.json").write_text(json.dumps(completed))
    finally:
        dist.destroy_process_group()


class TailLossTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_equal_partitions_preserve_tail_loss_and_prediction_gradients(self):
        target = torch.tensor([[1., 0.], [2., -.5], [.2, .1], [-2., -1.],
                               [1.5, 1.2], [.8, .9], [-.1, .4], [.3, .7]], dtype=torch.float64)
        initial = torch.arange(16, dtype=torch.float64).reshape(8, 2) / 7
        for mode, fraction, size in itertools.product(TAIL_MODES, (.05, .25), (2, 4)):
            with self.subTest(mode=mode, fraction=fraction, microbatch_size=size):
                config = LossConfig(loss_mode=mode, tail_frac=fraction, tail_lambda=.3)
                whole_prediction = initial.clone().requires_grad_()
                divided_prediction = initial.clone().requires_grad_()
                whole = tail_component(whole_prediction, target, config)
                divided = torch.stack([tail_component(divided_prediction[start:start + size], target[start:start + size], config)
                                       for start in range(0, 8, size)]).mean()
                torch.testing.assert_close(divided, whole, rtol=1e-12, atol=1e-12)
                whole.backward()
                divided.backward()
                torch.testing.assert_close(divided_prediction.grad, whole_prediction.grad, rtol=1e-12, atol=1e-12)

    def test_tail_sample_gradient_is_independent_of_other_tail_samples(self):
        fixed_target = torch.tensor([1.5, -.5], dtype=torch.float64)
        fixed_prediction = torch.tensor([.2, .6], dtype=torch.float64)
        for mode in TAIL_MODES:
            config = LossConfig(loss_mode=mode, tail_frac=.2, tail_lambda=.3, wmse_s=.4)
            weight = 1 + config.wmse_alpha * torch.sigmoid(fixed_target.abs() / config.wmse_s)
            expected = 2 * (fixed_prediction - fixed_target) * config.tail_lambda / (config.tail_frac * 4 * 2)
            if mode.removesuffix("_slope") in ("wmse_tail", "mse_wtail"):
                expected = expected * weight
            for other_tails in (0, 1, 3):
                with self.subTest(mode=mode, other_tails=other_tails):
                    target = torch.zeros(4, 2, dtype=torch.float64)
                    target[0] = fixed_target
                    target[1:1 + other_tails] = torch.tensor([2., .5], dtype=torch.float64)
                    prediction = fixed_prediction.repeat(4, 1).requires_grad_()
                    gradient, = torch.autograd.grad(tail_component(prediction, target, config), prediction)
                    torch.testing.assert_close(gradient[0], expected, rtol=1e-12, atol=1e-12)

    def test_no_tail_events_produce_differentiable_zero_and_zero_gradient(self):
        target = torch.tensor([[-3., -1.], [.2, .7], [0., .9]], dtype=torch.float64)
        for mode in TAIL_MODES:
            with self.subTest(mode=mode):
                prediction = torch.tensor([[.2, .1], [-.5, .3], [1.2, .4]], dtype=torch.float64, requires_grad=True)
                config = LossConfig(loss_mode=mode, tail_frac=.05)
                tail = tail_component(prediction, target, config)
                self.assertTrue(tail.requires_grad)
                self.assertEqual(tail.item(), 0.)
                tail.backward()
                torch.testing.assert_close(prediction.grad, torch.zeros_like(prediction), rtol=0, atol=1e-14)

    def test_weighted_semantics_scale_and_inclusive_threshold(self):
        # Only [1, 0] is an event, exactly on the threshold; [-1, 0] is not.
        target = torch.tensor([[1., 0.], [-1., 0.]], dtype=torch.float64)
        positive_weight = 1 + 2 / (1 + math.exp(-1))
        for core, use_abs in itertools.product(CORES, (False, True)):
            with self.subTest(core=core, use_abs=use_abs):
                config = LossConfig(loss_mode=core, wmse_alpha=2., wmse_s=1., wmse_use_abs=use_abs,
                                    tail_frac=.25, tail_lambda=.3)
                prediction = torch.zeros_like(target, requires_grad=True)
                loss = ForecastLoss(config, stats(torch.float64), 1., 0.)(ForecastOutput(prediction), prediction, target)
                negative_weight = positive_weight if use_abs else 1 + 2 / (1 + math.exp(1))
                base = (positive_weight + negative_weight) / 4 if core == "wmse_tail" else .5
                tail = positive_weight if core in ("wmse_tail", "mse_wtail") else 1.
                self.assertAlmostEqual(loss.item(), base + .3 * tail, places=12)
                expected = torch.tensor([[-.5, 0.], [.5, 0.]], dtype=torch.float64)
                if core == "wmse_tail":
                    expected[:, 0] *= torch.tensor([positive_weight, negative_weight], dtype=torch.float64)
                expected[0, 0] -= .6 * tail
                loss.backward()
                torch.testing.assert_close(prediction.grad, expected, rtol=1e-12, atol=1e-12)

    def test_equal_microbatches_and_short_groups_match_concatenated_updates(self):
        for mode in TAIL_MODES:
            with self.subTest(mode=mode):
                data, normal = samples(), stats()
                config = LossConfig(loss_mode=mode, tail_frac=.25, tail_lambda=.3, slope_lambda=.2)
                criterion = ForecastLoss(config, normal, 1., 0.)
                accumulated, reference = TailLinearModel(), TailLinearModel()
                run_epoch(accumulated, loader(data, 2), torch.device("cpu"), normal,
                          optimizer=torch.optim.SGD(accumulated.parameters(), lr=.005), criterion=criterion,
                          grad_accum_steps=3)
                # Six samples in the first group, four in the final group; every microbatch has two.
                run_epoch(reference, loader(data, 6), torch.device("cpu"), normal,
                          optimizer=torch.optim.SGD(reference.parameters(), lr=.005), criterion=criterion)
                for actual, expected in zip(accumulated.parameters(), reference.parameters()):
                    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)

    def test_training_cli_fraction_reaches_loss_and_saved_threshold(self):
        self.assertEqual(LossConfig().tail_frac, .05)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            graphs, stations = make_fixture(root)
            store = ForcingGraphStore(graphs, "Battery")
            train_indices = store.split()["train"]
            peaks = np.asarray([store.graphs[index].y.max().item() for index in train_indices], dtype=np.float32)
            expected_threshold = float(np.percentile(peaks, 75))
            output_dir = root / "run"
            with patch.object(train, "ForecastLoss", wraps=ForecastLoss) as observed, contextlib.redirect_stdout(io.StringIO()):
                train.main(["--root_dir", str(graphs), "--station", "Battery", "--station_json_dir", str(stations),
                            "--output_dir", str(output_dir), "--model", "baseline", "--head_type", "single", "--history_hours", "0",
                            "--loss_mode", "mse_tail", "--tail_frac", ".25", "--tail_lambda", ".3", "--hidden_channels", "16",
                            "--epochs", "1", "--warmup_epochs", "0", "--device", "cpu", "--num_workers", "0", "--x_aug", "0"])
            config = observed.call_args.args[0]
            self.assertEqual((config.tail_frac, config.tail_lambda), (.25, .3))
            self.assertEqual(observed.call_args.args[2], expected_threshold)
            checkpoint = torch.load(next(output_dir.glob("best_*.pth")), weights_only=False)
            self.assertEqual(checkpoint["training_config"]["tail_frac"], .25)
            self.assertEqual(checkpoint["training_config"]["tail_lambda"], .3)
            self.assertEqual(checkpoint["loss_thresholds"]["tail_threshold"], expected_threshold)


@unittest.skipUnless(dist.is_available() and dist.is_gloo_available(), "CPU Gloo is required")
class TailDDPTests(unittest.TestCase):
    def test_equal_rank_batches_and_accumulation_match_concatenation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            mp.spawn(ddp_tail_worker, args=(str(root / "rendezvous"), str(root)), nprocs=2, join=True)
            for rank in range(2):
                self.assertEqual(json.loads((root / f"rank{rank}.json").read_text()), list(TAIL_MODES))


if __name__ == "__main__":
    unittest.main()
