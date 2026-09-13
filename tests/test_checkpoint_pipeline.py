"""Synthetic end-to-end selection: artifacts, frozen behavior, scheduler, and DDP."""

import contextlib
import hashlib
import io
import itertools
import json
import os
from pathlib import Path
import random
import tempfile
import time
import unittest
from unittest.mock import patch

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import infer
import train
from emulator.training import checkpoints as cp
from emulator.training.engine import EpochResult
from test_checkpoints import CONFLICTING, WINNERS
from test_config_interfaces import dry_commands, MODES, REPO
from test_pipeline import make_fixture


FIXTURES = Path(__file__).with_name("fixtures")
VARIANTS = ("single", "direct", "severity_shape")


def arguments(graphs, stations, output, variant="single"):
    args = ["--root_dir", str(graphs), "--station", "Battery", "--station_json_dir", str(stations),
            "--output_dir", str(output), "--model", "perceiver3", "--head_type", "single" if variant == "single" else "dual",
            "--hidden_channels", "16", "--node_read_heads", "2", "--time_read_heads", "2", "--history_hours", "12",
            "--epochs", "6", "--device", "cpu", "--num_workers", "0", "--run_tag", "selection_fixture"]
    if variant != "single":
        args += ["--excess_formulation", variant, "--excess_amp_loss_weight", ".3"]
    if variant == "severity_shape":
        args += ["--shape_loss_weight", ".2"]
    return args


def recorded_run(root, mode="overall", auxiliary=0, variant="single", extra=()):
    torch.manual_seed(824)
    graphs, stations = make_fixture(root)
    output = root / "run"
    args = arguments(graphs, stations, output, variant)
    args += ["--checkpoint_selection", mode, "--save_aux_checkpoints", str(auxiliary), *extra]
    state = dict(train_calls=0, val_calls=0, test_calls=0, loaders=0, states={})
    real_epoch, real_loader = train.run_epoch, train.build_loader

    def loader(*args, **kwargs):
        state["loaders"] += 1
        if state["loaders"] == 3:
            # Test construction itself follows final resolution/materialization, not just inference.
            assert state["val_calls"] == len(CONFLICTING)
            paths = list(output.glob("best_*.pth"))
            assert len(paths) == 1
            checkpoint = torch.load(paths[0], weights_only=False)
            assert checkpoint["epoch"] == WINNERS[mode]
            if auxiliary or mode != "overall":
                assert len(list(output.glob("checkpoint_selection_*.json"))) == 1
                assert not (output / "checkpoint_candidates").exists()
            state["test_loader_selected_epoch"] = checkpoint["epoch"]
        return real_loader(*args, **kwargs)

    def epoch(model, *args, **kwargs):
        if kwargs.get("save_predictions"):
            state["test_calls"] += 1
            assert state["test_loader_selected_epoch"] == WINNERS[mode]
            state["test_marker"] = next(model.parameters()).detach().flatten()[0].item()
            for key, tensor in state["states"][WINNERS[mode]].items():
                torch.testing.assert_close(model.state_dict()[key], tensor, rtol=0, atol=0)
            return real_epoch(model, *args, **kwargs)
        if kwargs.get("optimizer") is not None:
            state["train_calls"] += 1
            with torch.no_grad():
                for parameter in model.parameters():
                    parameter.fill_(state["train_calls"] / 100)
            state["states"][state["train_calls"]] = {key: value.detach().clone() for key, value in model.state_dict().items()}
        else:
            state["val_calls"] += 1
        return EpochResult(dict(CONFLICTING[state["train_calls"] - 1]))

    with contextlib.redirect_stdout(io.StringIO()), patch.object(train, "run_epoch", side_effect=epoch), \
            patch.object(train, "build_loader", side_effect=loader), patch.object(torch.optim.lr_scheduler.LambdaLR, "step"):
        train.main(args)
    return output, state


def tensor_digest(tensors):
    digest = hashlib.sha256()
    for name, tensor in sorted(tensors.items()):
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def trajectory_run(root, variant, mode="overall", auxiliary=0, scheduler="cosine", rop_metric="val_rmse_phys", module=train):
    """Real CPU optimizer/backprop on temporary random graphs; controlled VAL rankings only."""
    torch.manual_seed(824)
    graphs, stations = make_fixture(root)
    args = arguments(graphs, stations, root / "run", variant)
    args += ["--checkpoint_selection", mode, "--save_aux_checkpoints", str(auxiliary), "--peak_loss_weight", ".4",
             "--batch_size", "2", "--grad_accum_steps", "2", "--dropout", ".13", "--head_dropout", ".17",
             "--scheduler", scheduler, "--rop_metric", rop_metric, "--rop_patience", "0", "--warmup_epochs", "2"]
    history, scheduler_metrics = [], []
    real_epoch = module.run_epoch
    rop_step = torch.optim.lr_scheduler.ReduceLROnPlateau.step

    def step(scheduler, metric, *args, **kwargs):
        scheduler_metrics.append(metric)
        return rop_step(scheduler, metric, *args, **kwargs)

    def epoch(model, *args, **kwargs):
        optimizer = kwargs.get("optimizer")
        if optimizer is not None:
            losses, forwards = [], []
            loss_hook = kwargs["criterion"].register_forward_hook(lambda _, inputs, value: losses.append(value.item()))
            forward_hook = model.register_forward_hook(lambda _, inputs, value: forwards.append(
                tensor_digest({name: tensor for name, tensor in value._asdict().items() if isinstance(tensor, torch.Tensor)})))
            learning_rates = [group["lr"] for group in optimizer.param_groups]
            try:
                result = real_epoch(model, *args, **kwargs)
            finally:
                loss_hook.remove()
                forward_hook.remove()
            history.append(dict(lr=learning_rates, losses=losses, forwards=forwards, train_metrics=result.metrics,
                                weights=tensor_digest(model.state_dict()), rng=tensor_digest({"rng": torch.get_rng_state()}),
                                python_rng=hashlib.sha256(repr(random.getstate()).encode()).hexdigest(),
                                numpy_rng=hashlib.sha256(repr(np.random.get_state()).encode()).hexdigest()))
            return result
        result = real_epoch(model, *args, **kwargs)
        if not kwargs.get("save_predictions"):
            # Evaluate the real validation forward, then supply conflicting selection criteria.
            history[-1]["actual_val_metrics"] = dict(result.metrics)
            result.metrics.update(CONFLICTING[len(history) - 1])
        return result

    with contextlib.redirect_stdout(io.StringIO()), patch.object(module, "run_epoch", side_effect=epoch), \
            patch.object(torch.optim.lr_scheduler.ReduceLROnPlateau, "step", new=step):
        module.main(args)
    return dict(history=history, scheduler_metrics=scheduler_metrics)


def checkpoint_ddp_worker(rank, rendezvous, root_string):
    root = Path(root_string)
    dist.init_process_group("gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=2)
    real_epoch, real_save = train.run_epoch, torch.save
    real_manifest, real_select = cp.write_selection_manifest, train.CheckpointSelector
    observed = dict(saves=[], manifests=[], selections=[], validation=[], test_calls=0)

    def save(checkpoint, path, *args, **kwargs):
        observed["saves"].append(str(path))
        assert rank == 0
        return real_save(checkpoint, path, *args, **kwargs)

    def manifest(*args, **kwargs):
        observed["manifests"].append(rank)
        assert rank == 0
        return real_manifest(*args, **kwargs)

    class ObservedSelector(real_select):
        def observe(self, epoch, val):
            assert rank == 0
            observed["selections"].append(dict(val))
            return super().observe(epoch, val)

    def epoch(model, *args, **kwargs):
        if kwargs.get("save_predictions"):
            observed["test_calls"] += 1
            assert rank == 0
            assert len(observed["validation"]) == 3
            assert observed["manifests"] == [0]
        result = real_epoch(model, *args, **kwargs)
        if kwargs.get("optimizer") is None and not kwargs.get("save_predictions"):
            observed["validation"].append(dict(result.metrics))
        return result

    try:
        args = train.parse_args(["--root_dir", str(root / "graphs"), "--station", "Battery",
            "--output_dir", str(root / "run"), "--model", "baseline", "--head_type", "single",
            "--encoder_type", "CNN", "--temporal_block", "MLP", "--history_hours", "0", "--hidden_channels", "16",
            "--epochs", "3", "--batch_size", "2", "--num_workers", "0", "--device", "cpu", "--run_tag", "ddp",
            "--checkpoint_selection", "constrained_peak_magnitude", "--save_aux_checkpoints", "1"])
        train.configure_runtime(args.seed + rank, 1, True, False)
        with contextlib.redirect_stdout(io.StringIO()), patch.object(train, "run_epoch", side_effect=epoch), \
                patch.object(torch, "save", side_effect=save), patch.object(cp, "write_selection_manifest", side_effect=manifest), \
                patch.object(train, "CheckpointSelector", ObservedSelector):
            train.train(args, torch.device("cpu"), True, rank, time.perf_counter())
        (root / f"rank_{rank}.json").write_text(json.dumps(observed))
    finally:
        dist.destroy_process_group()


class CheckpointPipelineTests(unittest.TestCase):
    def test_frozen_post6_default_selects_same_epoch_weights_outputs_and_test(self):
        reference = json.loads((FIXTURES / "post6_checkpoint_selection.json").read_text())
        with tempfile.TemporaryDirectory() as tmp, patch.object(train, "CandidateCheckpointStore", side_effect=AssertionError("default must stay direct")):
            output, state = recorded_run(Path(tmp))
            checkpoint_path, = output.rglob("best_*.pth")
            self.assertRegex(checkpoint_path.name, r"^best_perceiver3_mse_[0-9a-f]{10}_selection_fixture\.pth$")
            summary = json.loads(next(output.glob("summary_*.json")).read_text())
            for key in ("best_epoch", "best_val_rmse", "val", "test"):
                self.assertEqual(summary[key], reference[key])
            self.assertEqual(state["test_marker"], reference["test_marker"])
            self.assertEqual(state["test_calls"], reference["test_calls"])
            with np.load(next(output.glob("test_preds_*.npz"))) as predictions:
                for key, values in reference["predictions"].items():
                    np.testing.assert_array_equal(predictions[key], np.asarray(values))
            self.assertFalse((output / "aux_checkpoints").exists())
            self.assertFalse((output / "checkpoint_candidates").exists())
            self.assertFalse(list(output.glob("checkpoint_selection_*.json")))

    def test_all_modes_aux_settings_and_head_variants_use_final_primary_once(self):
        for variant, mode, auxiliary in itertools.product(VARIANTS, cp.SELECTION_METRICS, (0, 1)):
            with self.subTest(variant=variant, mode=mode, auxiliary=auxiliary), tempfile.TemporaryDirectory() as tmp:
                output, state = recorded_run(Path(tmp), mode, auxiliary, variant)
                self.assertEqual((state["train_calls"], state["val_calls"], state["test_calls"], state["loaders"]), (6, 6, 1, 3))
                canonical, = output.rglob("best_*.pth")
                checkpoint = torch.load(canonical, weights_only=False)
                expected = WINNERS[mode]
                self.assertEqual(checkpoint["epoch"], expected)
                self.assertEqual(checkpoint["checkpoint_role"], mode)
                summary = json.loads(next(output.glob("summary_*.json")).read_text())
                self.assertEqual(summary["best_epoch"], expected)
                self.assertEqual(summary["best_val_rmse"], CONFLICTING[expected - 1]["rmse_all"])
                self.assertEqual(summary["val"], CONFLICTING[expected - 1])
                for role in cp.SIMPLE_ROLES:
                    self.assertEqual(summary["checkpoint_selection"][f"best_{role}_epoch"], WINNERS[role])
                config = json.loads(next(output.glob("config_*.json")).read_text())
                for key, value in dict(checkpoint_selection=mode, checkpoint_overall_tol=.01, save_aux_checkpoints=auxiliary).items():
                    self.assertEqual(config[key], value)
                    self.assertEqual(checkpoint["training_config"][key], value)
                    self.assertIn(f"{key.upper()}={value}", (output / "config_used.sh").read_text())
                digest = hashlib.sha256(json.dumps(config, sort_keys=True, default=str).encode()).hexdigest()[:10]
                self.assertIn(digest, canonical.name)
                if auxiliary or mode != "overall":
                    manifest = json.loads(next(output.glob("checkpoint_selection_*.json")).read_text())
                    for role, entry in manifest["roles"].items():
                        if entry is None:
                            continue
                        artifact = torch.load(entry["path"], weights_only=False)
                        self.assertEqual(artifact["epoch"], WINNERS[role])
                        self.assertEqual(artifact["val"], CONFLICTING[WINNERS[role] - 1])
                        self.assertIn(role, artifact["checkpoint_roles"])
                        for key, tensor in state["states"][WINNERS[role]].items():
                            torch.testing.assert_close(artifact["model_state"][key], tensor, rtol=0, atol=0)
                    self.assertEqual(len(list(output.rglob("*.pth"))), 5 if auxiliary else 1)

    def test_frozen_losses_forwards_updates_rng_and_cosine_unchanged_for_all_modes(self):
        reference = json.loads((FIXTURES / "post6_checkpoint_trajectories.json").read_text())
        for variant, mode in itertools.product(VARIANTS, cp.SELECTION_METRICS):
            with self.subTest(variant=variant, mode=mode), tempfile.TemporaryDirectory() as tmp:
                observed = trajectory_run(Path(tmp), variant, mode, auxiliary=1)
                self.assertEqual(observed, reference["variants"][variant])

    def test_rop_keeps_its_independent_two_metric_choices_across_selection_modes(self):
        for rop_metric, key in (("val_rmse_phys", "rmse_all"), ("val_rmse_peak", "rmse_peak5")):
            expected = None
            for mode in cp.SELECTION_METRICS:
                with self.subTest(rop_metric=rop_metric, mode=mode), tempfile.TemporaryDirectory() as tmp:
                    observed = trajectory_run(Path(tmp), "single", mode, auxiliary=1, scheduler="rop", rop_metric=rop_metric)
                    self.assertEqual(observed["scheduler_metrics"], [v[key] for v in CONFLICTING])
                    if expected is None:
                        expected = observed
                    self.assertEqual(observed, expected)

    def test_auxiliary_inference_uses_existing_explicit_checkpoint_interface(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output, state = recorded_run(root, "constrained_peak5", 1, "severity_shape")
            manifest = json.loads(next(output.glob("checkpoint_selection_*.json")).read_text())
            auxiliary = manifest["roles"]["peak_magnitude"]["path"]
            with contextlib.redirect_stdout(io.StringIO()), patch.object(infer, "run_epoch", wraps=infer.run_epoch) as inference:
                infer.main(["--ckpt", auxiliary, "--root_dir", str(root / "graphs"), "--out_dir", str(root / "infer"),
                            "--device", "cpu", "--num_workers", "0", "--save_npz"])
            self.assertEqual(state["test_calls"], 1)
            self.assertEqual(inference.call_count, 1)
            self.assertTrue((root / "infer" / "predictions.npz").exists())

    def test_real_two_rank_global_val_selection_and_only_rank_zero_writes_or_tests(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            torch.manual_seed(824)
            make_fixture(root)
            mp.spawn(checkpoint_ddp_worker, args=(str(root / "rendezvous"), str(root)), nprocs=2, join=True)
            zero, one = [json.loads((root / f"rank_{rank}.json").read_text()) for rank in (0, 1)]
            self.assertEqual(zero["validation"], one["validation"])
            self.assertEqual(zero["selections"], zero["validation"])
            self.assertTrue(zero["saves"])
            self.assertEqual(zero["manifests"], [0])
            for key in ("saves", "manifests", "selections"):
                self.assertEqual(one[key], [])
            self.assertEqual((zero["test_calls"], one["test_calls"]), (1, 0))
            manifest = json.loads(next((root / "run").glob("checkpoint_selection_*.json")).read_text())
            self.assertEqual(len(list((root / "run").rglob("best_*.pth"))), 1)
            for role, entry in manifest["roles"].items():
                checkpoint = torch.load(entry["path"], weights_only=False)
                self.assertEqual(entry["val"], zero["validation"][entry["epoch"] - 1])
                self.assertEqual(checkpoint["val"], entry["val"])


class CheckpointConfigTests(unittest.TestCase):
    def test_defaults_boolean_convention_and_invalid_values(self):
        args = train.parse_args([])
        self.assertEqual((args.checkpoint_selection, args.checkpoint_overall_tol, args.save_aux_checkpoints), ("overall", .01, 0))
        for value, expected in (("true", 1), ("YES", 1), ("1", 1), ("false", 0), ("No", 0), ("0", 0)):
            self.assertEqual(train.parse_args(["--save_aux_checkpoints", value]).save_aux_checkpoints, expected)
        self.assertEqual(train.parse_args(["--checkpoint_overall_tol", "0"]).checkpoint_overall_tol, 0)
        for flag, values in (("checkpoint_selection", ("event", "peak", "test")),
                             ("checkpoint_overall_tol", ("-.01", "nan", "inf", "-inf")),
                             ("save_aux_checkpoints", ("2", "perhaps"))):
            for value in values:
                with self.subTest(flag=flag, value=value), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    train.parse_args([f"--{flag}={value}"])

    def test_selection_and_peak_loss_are_independent_for_every_loss_mode(self):
        for loss, selection, weight in itertools.product(MODES, cp.SELECTION_METRICS, (0., .4)):
            args = train.parse_args(["--model", "perceiver3", "--head_type", "dual", "--loss_mode", loss,
                "--excess_formulation", "severity_shape", "--excess_amp_loss_weight", ".3", "--shape_loss_weight", ".2",
                "--peak_loss_weight", str(weight), "--checkpoint_selection", selection])
            self.assertEqual(args.checkpoint_selection, selection)
            self.assertEqual(args.peak_loss_weight, weight)
            self.assertEqual(args.loss_mode, loss)
        self.assertEqual(train.parse_args(["--peak_loss_weight", ".4"]).checkpoint_selection, "overall")

    def test_existing_shell_config_forwards_explicit_checkpoint_settings(self):
        config = REPO / "experiment_config/P0_QuickRun/train_config_NCEP_Battery_24h_single_mse.sh"
        original = config.read_bytes()
        with patch.dict(os.environ, CHECKPOINT_SELECTION="constrained_peak_magnitude", CHECKPOINT_OVERALL_TOL="0.025", SAVE_AUX_CHECKPOINTS="true"):
            commands = dry_commands(config)
        self.assertTrue(commands)
        for command in commands:
            args = train.parse_args(command)
            self.assertEqual((args.checkpoint_selection, args.checkpoint_overall_tol, args.save_aux_checkpoints),
                             ("constrained_peak_magnitude", .025, 1))
        self.assertEqual(config.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
