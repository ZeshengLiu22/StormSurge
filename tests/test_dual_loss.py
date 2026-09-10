"""Mandatory dual supervision, resolved settings and real launcher logging."""

import contextlib
import copy
import io
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

from emulator.models import ForecastOutput
from emulator.training import ForecastLoss, LossConfig
from emulator.training.arguments import parse_args
from test_pipeline import make_fixture


class DualLossTests(unittest.TestCase):
    def test_dual_loss_cannot_be_disabled_in_any_prediction_mode(self):
        cores = ("mse", "wmse", "mse_tail", "wmse_tail", "mse_wtail")
        for mode in (*cores, *(core + "_slope" for core in cores)):
            with self.subTest(mode=mode), patch.dict(os.environ, RANK="0"):
                console = io.StringIO()
                with contextlib.redirect_stdout(console):
                    args = parse_args(["--model", "perceiver3", "--head_type", "dual",
                                       "--loss_mode", mode, "--dual_loss", "0",
                                       "--body_loss_weight", "0", "--excess_loss_weight", "0",
                                       "--gate_loss_weight", "0"])
                self.assertEqual(args.loss_mode, mode)
                self.assertEqual((args.dual_loss, args.body_loss_weight,
                                  args.excess_loss_weight, args.gate_loss_weight), (1, 1, 1, 1))
                self.assertEqual(console.getvalue().count("WARNING:"), 1)
                self.assertRegex(console.getvalue(), r"^\[\d{4}-\d{2}-\d{2}\|\d{2}:\d{2}:\d{2}\] WARNING:")

    def test_zero_terms_are_restored_but_positive_weights_are_preserved(self):
        for weight in ("body_loss_weight", "excess_loss_weight", "gate_loss_weight"):
            with self.subTest(weight=weight), patch.dict(os.environ, RANK="0"):
                console = io.StringIO()
                with contextlib.redirect_stdout(console):
                    args = parse_args(["--model", "perceiver3", "--body_loss_weight", ".3",
                                       "--excess_loss_weight", "2", "--gate_loss_weight", ".5",
                                       "--" + weight, "0"])
                expected = dict(body_loss_weight=.3, excess_loss_weight=2., gate_loss_weight=.5)
                expected[weight] = 1.
                for name, value in expected.items():
                    self.assertEqual(getattr(args, name), value)
                self.assertEqual(console.getvalue().count("WARNING:"), 1)
                self.assertIn(weight + "=1 (was 0)", console.getvalue())
        console = io.StringIO()
        with contextlib.redirect_stdout(console):
            args = parse_args(["--model", "perceiver3", "--body_loss_weight", ".3",
                               "--excess_loss_weight", "2", "--gate_loss_weight", ".5"])
        self.assertEqual((args.dual_loss, args.body_loss_weight, args.excess_loss_weight,
                          args.gate_loss_weight), (1, .3, 2, .5))
        self.assertEqual(console.getvalue(), "")

    def test_single_and_baseline_have_no_dual_loss_or_warning(self):
        for model in ("baseline", "perceiver3"):
            for requested in ("0", "1"):
                console = io.StringIO()
                with contextlib.redirect_stdout(console):
                    args = parse_args(["--model", model, "--head_type", "single", "--dual_loss", requested,
                                       "--body_loss_weight", "0", "--excess_loss_weight", "0", "--gate_loss_weight", "0"])
                    config = LossConfig(dual_loss=bool(args.dual_loss), body_loss_weight=0,
                                        excess_loss_weight=0, gate_loss_weight=0)
                    criterion = ForecastLoss(config, dict(y_mean=torch.zeros(2), y_std=torch.ones(2)), 1, 1)
                    pred = torch.tensor([[.2, .6]], requires_grad=True)
                    target = torch.tensor([[.4, .1]])
                    loss = criterion(ForecastOutput(pred), pred, target)
                self.assertEqual(args.dual_loss, 0)
                self.assertEqual(console.getvalue(), "")
                torch.testing.assert_close(loss, (pred - target).square().mean(), rtol=0, atol=0)

    def test_direct_loss_call_cannot_bypass_supervision_or_backpropagation(self):
        output = ForecastOutput(torch.zeros(2, 2, requires_grad=True), torch.zeros(2, 2, requires_grad=True),
                                torch.ones(2, 2, requires_grad=True), torch.zeros(2, 1, requires_grad=True),
                                torch.ones(1, 2))
        reference = copy.deepcopy(output)
        stats = dict(y_mean=torch.tensor([.1, -.2]), y_std=torch.tensor([2., 3.]))
        target = torch.tensor([[2., 0.], [0., 0.]]) * stats["y_std"] + stats["y_mean"]
        config = LossConfig(dual_loss=False, body_loss_weight=0, excess_loss_weight=0, gate_loss_weight=0)
        criterion = ForecastLoss(config, stats, 1, 1)
        prediction = output.prediction * stats["y_std"] + stats["y_mean"]
        console = io.StringIO()
        with patch.dict(os.environ, RANK="0"), contextlib.redirect_stdout(console):
            actual = criterion(output, prediction, target)
            criterion(output, prediction, target)  # A resolved config does not warn every batch.
        expected = ForecastLoss(LossConfig(), stats, 1, 1)(reference,
            reference.prediction * stats["y_std"] + stats["y_mean"], target)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        actual.backward()
        expected.backward()
        for name in ("prediction", "body", "excess", "gate_logits"):
            gradient = getattr(output, name).grad
            self.assertGreater(float(gradient.abs().sum()), 0, name)
            torch.testing.assert_close(gradient, getattr(reference, name).grad, rtol=0, atol=0)
        self.assertEqual(console.getvalue().count("WARNING:"), 1)
        self.assertTrue(config.dual_loss)

    def test_nonzero_rank_resolves_settings_without_duplicate_warning(self):
        console = io.StringIO()
        with patch.dict(os.environ, RANK="1"), contextlib.redirect_stdout(console):
            args = parse_args(["--model", "perceiver3", "--dual_loss", "0", "--gate_loss_weight", "0"])
        self.assertEqual((args.dual_loss, args.gate_loss_weight), (1, 1))
        self.assertEqual(console.getvalue(), "")

    def test_launcher_records_warning_and_effective_config(self):
        repo = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            graphs, stations = make_fixture(root)
            config = root / "dual.sh"
            config.write_text(f'''ROOT_DIR={shlex.quote(str(graphs))}
STATION_JSON_DIR={shlex.quote(str(stations))}
ALL_RESULTS_ROOT={shlex.quote(str(root / "results"))}
PYTHON_BIN={shlex.quote(sys.executable)}
STATION=Battery
MODEL=perceiver3
HEAD_TYPE=dual
DUAL_LOSS=0
BODY_LOSS_WEIGHT=0
EXCESS_LOSS_WEIGHT=0
GATE_LOSS_WEIGHT=0
ENCODER_TYPE=CNN
TEMPORAL_BLOCK=MLP
HISTORY_HOURS_LIST=(0)
LOSS_MODE_LIST=(mse)
LR_LIST=(.005)
HIDDEN_CHANNELS=16
EPOCHS=1
BATCH_SIZE=4
NUM_WORKERS=0
DISABLE_OOD=1
USE_TMUX=0
DO_CONDA=0
num_gpus=1
''')
            result = subprocess.run(["bash", "train.sh", str(config)], cwd=repo,
                                    env=dict(os.environ, CUDA_VISIBLE_DEVICES="", RANK="0", WORLD_SIZE="1", LOCAL_RANK="0", DRY_RUN="0"),
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            run_log = next((root / "results").rglob("train_*.log")).read_text()
            self.assertEqual(run_log.count("WARNING:"), 1)
            self.assertIn("head_type=dual requires dual loss", run_log)
            self.assertIn("Wall time:", run_log)
            saved = json.loads(next((root / "results").rglob("config_*.json")).read_text())
            checkpoint = torch.load(next((root / "results").rglob("best_*.pth")), weights_only=False)
            for settings in (saved, checkpoint["training_config"]):
                self.assertEqual(settings["dual_loss"], 1)
                for name in ("body_loss_weight", "excess_loss_weight", "gate_loss_weight"):
                    self.assertEqual(settings[name], 1)
                self.assertEqual(settings["loss_mode"], "mse")
