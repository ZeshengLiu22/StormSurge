"""Checkpoint persistence and inference only; training epochs are never executed."""

import contextlib
from dataclasses import asdict
import io
import itertools
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
from unittest.mock import patch

import numpy as np
import pytest
import torch

import train
from emulator.models import ModelConfig, build_model
from emulator.models.heads import AdditiveExceedanceHead, ExceedanceHead, ExceedanceHead_Experiment, SeverityShapeHead
from emulator.training.engine import EpochResult
from test_hard_gate_diagnostics import evaluate_fixture
from test_models import graph_batch
from test_pipeline import make_fixture


@pytest.fixture(autouse=True)
def single_thread():
    torch.set_num_threads(1)


@pytest.mark.parametrize("scope", ["event", "all"], ids=["F2", "F3"])
def test_actual_checkpoint_snapshot_and_additive_inference_without_training(tmp_path, scope):
    torch.manual_seed(42)
    graphs, _ = make_fixture(tmp_path)
    destination = tmp_path / "snapshot"
    metrics = dict(rmse_all=1., mae_all=1., rmse_peak5=1., mae_peak5=1., peak_magnitude_rmse_top5=1.)
    passes = [EpochResult(metrics), EpochResult(metrics), EpochResult(metrics), EpochResult(metrics,
              dict(y_true=np.zeros((0, 4)), y_pred=np.zeros((0, 4)), tags=np.array([], dtype=str)))]
    constructed = []

    def construct(config):
        model = build_model(config)
        constructed.append(model)
        return model

    def forbidden_step(*args, **kwargs):
        raise AssertionError("No optimizer step is allowed in this persistence test.")

    # Exercise train.py's real dataclass collection and checkpoint_snapshot,
    # while replacing ALL epochs and disallowing optimizer updates.
    with patch.object(train, "run_epoch", side_effect=passes) as epochs, \
            patch.object(train, "build_model", side_effect=construct), \
            patch.object(torch.optim.Adam, "step", new=forbidden_step), \
            patch.object(torch.optim.lr_scheduler.LambdaLR, "step"), contextlib.redirect_stdout(io.StringIO()):
        train.main(["--root_dir", str(graphs), "--station", "Battery", "--use_station_meta", "0",
                    "--output_dir", str(destination), "--model", "perceiver3", "--head_type", "dual",
                    "--hidden_channels", "8", "--node_read_heads", "2", "--time_read_heads", "2",
                    "--temporal_block", "MLP", "--history_hours", "0", "--epochs", "1",
                    "--batch_size", "2", "--num_workers", "0", "--device", "cpu",
                    "--direct_dual_reconstruction", "additive", "--excess_supervision_scope", scope,
                    "--body_loss_weight", "1", "--excess_loss_weight", "2", "--gate_loss_weight", ".5"])
    assert epochs.call_count == 4
    checkpoint = next(destination.glob("best_*.pth"))
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert saved["model_config"]["direct_dual_reconstruction"] == "additive"
    assert saved["training_config"]["excess_supervision_scope"] == scope
    assert "excess_supervision_scope" not in saved["model_config"]
    assert saved["training_config"]["checkpoint_selection"] == "overall"
    restored = build_model(ModelConfig(**saved["model_config"])).eval()
    restored.load_state_dict(saved["model_state"], strict=True)
    assert type(restored.head) is AdditiveExceedanceHead
    batch = graph_batch(steps=1)
    for expected, actual in zip(constructed[0].eval()(batch), restored(batch)):
        if expected is None:
            assert actual is None
        else:
            torch.testing.assert_close(expected, actual, rtol=0, atol=0)
    snapshot = json.loads(next(destination.glob("config_*.json")).read_text())
    shell = (destination / "config_used.sh").read_text()
    for key, value in (("direct_dual_reconstruction", "additive"), ("excess_supervision_scope", scope)):
        assert snapshot[key] == value
        assert f"{key.upper()}={value}" in shell
    original_checkpoint = checkpoint.read_bytes()
    normal, diagnostic = tmp_path / "normal", tmp_path / "diagnostic"
    plain = evaluate_fixture(graphs, checkpoint, normal, diagnostics=False)
    extra = evaluate_fixture(graphs, checkpoint, diagnostic)
    assert plain["metrics"] == extra["metrics"]
    report = json.loads((diagnostic / "dual_diagnostics.json").read_text())
    assert "hard_gate_0p5" not in report
    assert report["direct_dual_reconstruction"] == "additive"
    assert report["excess_supervision_scope"] == scope
    with np.load(normal / "predictions.npz") as before, np.load(diagnostic / "predictions.npz") as after, \
            np.load(diagnostic / "dual_diagnostics.npz") as arrays:
        for name in before.files:
            np.testing.assert_array_equal(before[name], after[name])
        np.testing.assert_allclose(arrays["y_pred"], arrays["body_phys"] + arrays["excess_phys"], atol=1e-6)
        np.testing.assert_array_equal(arrays["contribution_phys"], arrays["excess_phys"])
        assert not np.allclose(arrays["y_pred"], arrays["body_phys"]
                               + arrays["gate_probability"][:, None] * arrays["excess_phys"])
    assert checkpoint.read_bytes() == original_checkpoint
    assert len(list(tmp_path.rglob("*.pth"))) == 1


HISTORICAL = [({}, ExceedanceHead)] + [
    (dict(exceedance_head_experiment=variant, exceedance_gate_pooling=pool), ExceedanceHead_Experiment)
    for variant, pool in itertools.product(("legacy", "c1", "c2", "c2r", "c3"), ("mean", "learned"))
] + [(dict(excess_formulation="severity_shape", target_y_std=[.25, .5, 2., 5.], dual_ablation=ablation), SeverityShapeHead)
     for ablation in ("none", "fixed_gate")]


@pytest.mark.parametrize("options,head_type", HISTORICAL)
def test_historical_checkpoint_schema_strict_roundtrip(tmp_path, options, head_type):
    config = ModelConfig(3, 4, 8, peak_threshold_norm=[1.] * 4, **options)
    original = build_model(config).eval()
    values = asdict(config)
    values.pop("direct_dual_reconstruction")
    if head_type is ExceedanceHead:
        # Also cover production files from before both optional head studies.
        for name in ("exceedance_head_experiment", "exceedance_gate_pooling", "excess_formulation",
                     "target_y_std", "severity_shape_eps"):
            values.pop(name)
    path = tmp_path / "historical.pth"
    torch.save(dict(model_config=values, model_state=original.state_dict()), path)
    saved = torch.load(path, map_location="cpu", weights_only=False)
    restored_config = ModelConfig(**saved["model_config"])
    assert restored_config.direct_dual_reconstruction == "soft_gate"
    restored = build_model(restored_config).eval()
    restored.load_state_dict(saved["model_state"], strict=True)
    assert type(restored.head) is head_type
    batch = graph_batch()
    expected, actual = original(batch), restored(batch)
    for before, after in zip(expected, actual):
        if before is None:
            assert after is None
        else:
            torch.testing.assert_close(before, after, rtol=0, atol=0)
    torch.testing.assert_close(actual.prediction, actual.body + actual.gate_probability * actual.excess, rtol=0, atol=0)


@pytest.mark.parametrize("mode,scope", [("soft_gate", "event"), ("additive", "event"), ("additive", "all")])
def test_launcher_resolved_snapshot_with_no_training_process(tmp_path, mode, scope):
    # The launcher runs this empty stub, never train.py or a model fit.
    stub = tmp_path / "no_training.py"
    stub.write_text("raise SystemExit(0)\n")
    config = tmp_path / "config.sh"
    config.write_text(f"MODEL=perceiver3\nDIRECT_DUAL_RECONSTRUCTION={mode}\nEXCESS_SUPERVISION_SCOPE={scope}\n"
                      f"TRAIN_PY={shlex.quote(str(stub))}\nPYTHON_BIN={shlex.quote(sys.executable)}\n"
                      f"ALL_RESULTS_ROOT={shlex.quote(str(tmp_path / 'results'))}\n"
                      "DO_CONDA=0\nUSE_TMUX=0\nnum_gpus=1\nDRY_RUN=0\n")
    environment = dict(os.environ)
    environment.pop("SLURM_NTASKS_PER_NODE", None)
    subprocess.run(["bash", "train.sh", str(config)], cwd=Path(__file__).resolve().parents[1],
                   env=environment, check=True, capture_output=True, text=True)
    snapshot = next((tmp_path / "results").glob("*/config_used.sh"))
    result = subprocess.run(["bash", "-c", 'source "$1"; printf "%s %s" "$DIRECT_DUAL_RECONSTRUCTION" "$EXCESS_SUPERVISION_SCOPE"',
                             "snapshot", str(snapshot)], check=True, capture_output=True, text=True)
    assert result.stdout == f"{mode} {scope}"
    assert not list(tmp_path.rglob("*.pth"))
