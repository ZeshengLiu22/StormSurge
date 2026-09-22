"""Launcher defaults, WQE placement configs, and shared parameter plumbing."""

import contextlib
import io
import itertools
import os
from pathlib import Path
import shlex
import subprocess
import tempfile
import unittest

from emulator.training.arguments import parse_args
from tools.generate_configs import generate


REPO = Path(__file__).resolve().parents[1]
PARAMETERS = dict(wqe_quantile_tau=.25, wqe_expectile_tau=.82,
                  wqe_quantile_weight=1. / 6., wqe_expectile_weight=5. / 6.)


def dry_run(config):
    result = subprocess.run(["bash", "train.sh", str(config)], cwd=REPO,
        env=dict(os.environ, DRY_RUN="1"), capture_output=True, text=True, check=True)
    commands = [shlex.split(line.removeprefix("CMD: "))[1:]
                for line in result.stdout.splitlines() if line.startswith("CMD: ")]
    return [parse_args(command) for command in commands], result.stdout


class WQEConfigTests(unittest.TestCase):
    def test_old_d0_configs_resolve_mse_mse_with_unchanged_branch_weights(self):
        configs = sorted((REPO / "configs/current").glob("*_D0_DualBase.sh"))
        self.assertEqual(len(configs), 4)
        for config in configs:
            with self.subTest(config=config.name):
                self.assertNotIn("EXCESS_LOSS_MODE=", config.read_text())
                commands, log = dry_run(config)
                self.assertEqual(len(commands), 1)
                args = commands[0]
                self.assertEqual((args.loss_mode, args.excess_loss_mode), ("mse", "mse"))
                self.assertEqual((args.body_loss_weight, args.excess_loss_weight, args.gate_loss_weight), (1., 2., .5))
                self.assertEqual((args.exceedance_loss_weight, args.excess_amp_loss_weight, args.shape_loss_weight), (0., 0., 0.))
                for name, expected in PARAMETERS.items():
                    self.assertEqual(getattr(args, name), expected)
                self.assertIn("Global prediction loss: mse", log)
                self.assertIn("Dual excess loss: mse", log)

    def test_generator_produces_exactly_twelve_matched_placement_configs(self):
        expected_modes = {"W1": ("wqe", "mse"), "W2": ("mse", "wqe"), "W3": ("wqe", "wqe")}
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            rows = generate(output, family="wqe")
            self.assertEqual(len(rows), 12)
            self.assertEqual(len(list(output.glob("*.sh"))), 12)
            self.assertEqual({(row["station"], row["variant"]) for row in rows},
                             set(itertools.product(("CBBT", "Lewes", "Battery", "Boston"), expected_modes)))
            for row in rows:
                with self.subTest(config=row["config"]):
                    path = output / row["config"]
                    self.assertEqual(path.read_text(), (REPO / "configs/wqe" / row["config"]).read_text())
                    commands, log = dry_run(path)
                    self.assertEqual(len(commands), 1)
                    args = commands[0]
                    modes = expected_modes[row["variant"]]
                    self.assertEqual((args.loss_mode, args.excess_loss_mode), modes)
                    self.assertEqual((row["loss_mode"], row["excess_loss_mode"]), modes)
                    self.assertEqual((args.head_type, args.excess_formulation, args.dual_ablation),
                                     ("dual", "direct", "none"))
                    self.assertEqual((args.body_loss_weight, args.excess_loss_weight, args.gate_loss_weight), (1., 2., .5))
                    self.assertEqual((args.exceedance_loss_weight, args.excess_amp_loss_weight, args.shape_loss_weight), (0., 0., 0.))
                    self.assertEqual((args.hidden_channels, args.history_hours, args.lr), (128, 24, .005))
                    for name, expected in PARAMETERS.items():
                        self.assertEqual(getattr(args, name), expected)
                        self.assertEqual(row[name], expected)
                    self.assertIn("Global prediction loss: " + modes[0], log)
                    self.assertIn("Dual excess loss: " + modes[1], log)
                    wqe_line = next(line for line in log.splitlines() if line.startswith("WQE: "))
                    logged = dict(item.split("=") for item in wqe_line.removeprefix("WQE: ").split())
                    for key, expected in (("q_tau", .25), ("e_tau", .82),
                                          ("q_weight", 1. / 6.), ("e_weight", 5. / 6.)):
                        self.assertAlmostEqual(float(logged[key]), expected, delta=5e-7)
            self.assertEqual((output / "manifest.csv").read_text(), (REPO / "configs/wqe/manifest.csv").read_text())

    def test_launcher_passes_custom_parameters_to_both_modes_and_logs_single_inactive(self):
        with tempfile.TemporaryDirectory() as temporary:
            config = Path(temporary) / "custom.sh"
            for head in ("single", "dual"):
                with self.subTest(head=head):
                    config.write_text(f"MODEL=perceiver3\nHEAD_TYPE={head}\n"
                        "LOSS_MODE_LIST=(mse wqe)\nEXCESS_LOSS_MODE=wqe\n"
                        "WQE_QUANTILE_TAU=.4\nWQE_EXPECTILE_TAU=.7\n"
                        "WQE_QUANTILE_WEIGHT=.3\nWQE_EXPECTILE_WEIGHT=.7\n"
                        "LR_LIST=(.005)\nHISTORY_HOURS_LIST=(24)\n")
                    commands, log = dry_run(config)
                    self.assertEqual({args.loss_mode for args in commands}, {"mse", "wqe"})
                    for args in commands:
                        self.assertEqual(args.excess_loss_mode, "wqe")
                        self.assertEqual((args.wqe_quantile_tau, args.wqe_expectile_tau,
                                          args.wqe_quantile_weight, args.wqe_expectile_weight), (.4, .7, .3, .7))
                    message = "Dual excess loss: inactive (head_type=single)" if head == "single" else "Dual excess loss: wqe"
                    self.assertIn(message, log)

    def test_cli_validates_modes_and_wqe_parameters(self):
        for mode in ("mse", "wqe", "wmse", "mse_slope", "wqe_slope", "wmse_slope"):
            args = parse_args(["--loss_mode", mode])
            self.assertEqual((args.loss_mode, args.excess_loss_mode), (mode, "mse"))
        invalid_options = (["--excess_loss_mode", "wmse"], ["--wqe_quantile_tau", "0"],
            ["--wqe_expectile_tau", "1"], ["--wqe_quantile_weight", "-1"],
            ["--wqe_expectile_weight", "nan"], ["--wqe_expectile_weight", "inf"],
            ["--wqe_quantile_weight", ".2", "--wqe_expectile_weight", ".2"])
        for options in invalid_options:
            with self.subTest(options=options), self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
                parse_args(options)


if __name__ == "__main__":
    unittest.main()
