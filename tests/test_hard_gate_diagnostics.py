"""Offline F1 metrics and inference exports, using unfitted temporary checkpoints."""

import contextlib
from dataclasses import asdict
import io
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch

import infer
from emulator.data import ForcingGraphStore
from emulator.inference.dual_diagnostics import hard_gate_prediction, summarize_hard_gate
from emulator.models import ModelConfig, build_model
from emulator.training.arguments import parse_args
from emulator.training.metrics import physical_peak_columns, prediction_window_records, summarize_windows
from test_peak_metrics import make_records
from test_pipeline import make_fixture


@pytest.fixture(autouse=True)
def single_thread():
    torch.set_num_threads(1)


def inference_fixture(root, *, config_overrides=None, training_overrides=None, historical=False):
    """Create the existing checkpoint schema directly, without running a trainer."""
    torch.manual_seed(42)
    graphs, _ = make_fixture(root, years=2)
    stats = dict(x_center=torch.zeros(3), x_scale=torch.ones(3),
                 y_mean=torch.tensor([.2, -.5, 1., 0.]), y_std=torch.tensor([.25, .5, 2., 5.]))
    tau = 1.
    values = dict(in_channels=3, out_channels=4, hidden_channels=8, history_steps=0,
                  temporal_block="MLP", temporal_layers=1, temporal_dropout=0.,
                  node_read_heads=2, time_read_heads=2, peak_prior=.17,
                  peak_threshold_norm=((tau - stats["y_mean"]) / stats["y_std"]).tolist())
    values.update(config_overrides or {})
    config = ModelConfig(**values)
    model = build_model(config).eval()
    training = vars(parse_args(["--model", "perceiver3", "--station", "Battery", "--history_hours", "0"]))
    training.update(training_overrides or {})
    saved_config = asdict(config)
    if historical:
        saved_config.pop("direct_dual_reconstruction", None)
        training.pop("excess_supervision_scope", None)
    checkpoint = root / "selected.pth"
    torch.save(dict(model_config=saved_config, training_config=training, model_state=model.state_dict(),
                    normalization=stats, station="Battery", station_feat=None,
                    split_tags=dict(test=ForcingGraphStore(graphs, "Battery").graph_tags),
                    dual_metadata=dict(tau_phys=tau, event_prior=.17, fitted_on="train")), checkpoint)
    return graphs, checkpoint


def evaluate_fixture(graphs, checkpoint, out, *, diagnostics=True):
    with contextlib.redirect_stdout(io.StringIO()):
        infer.main(["--ckpt", str(checkpoint), "--root_dir", str(graphs), "--out_dir", str(out),
                    "--device", "cpu", "--num_workers", "0", "--batch_size", "2", "--save_npz",
                    *(["--dual_diagnostics"] if diagnostics else [])])
    return json.loads((out / "metrics.json").read_text())


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("column_probability", [False, True])
def test_fixed_half_threshold_is_inclusive_and_inputs_are_unchanged(dtype, column_probability):
    body = np.array([[.1, .2], [.3, .4], [.5, .6]], dtype=dtype)
    excess = np.array([[1., 2.], [3., 4.], [5., 6.]], dtype=dtype)
    probability = np.array([.2, .5, .8], dtype=dtype)
    soft = body + probability[:, None] * excess
    before = [value.copy() for value in (body, excess, probability, soft)]
    actual = hard_gate_prediction(body, excess, probability[:, None] if column_probability else probability)
    expected = body + np.array([False, True, True])[:, None] * excess
    np.testing.assert_array_equal(actual, expected)
    assert actual.dtype == dtype
    for value, saved in zip((body, excess, probability, soft), before):
        np.testing.assert_array_equal(value, saved)
        assert not np.shares_memory(actual, value)


@pytest.mark.parametrize("body,excess,probability", [
    (np.ones(3), np.ones(3), np.ones(3)),
    (np.ones((3, 2)), np.ones((3, 1)), np.ones(3)),
    (np.ones((3, 2)), np.ones((3, 2)), np.ones(2)),
    (np.ones((3, 2)), np.ones((3, 2)), np.ones((1, 3))),
    (np.ones((3, 2)), np.ones((3, 2)), np.ones((3, 2))),
    (np.ones((3, 2)), np.ones((3, 2)), np.ones((3, 1, 1))),
    (np.ones((3, 0)), np.ones((3, 0)), np.ones(3)),
    (np.full((3, 2), np.nan), np.ones((3, 2)), np.ones(3)),
    (np.ones((3, 2)), np.full((3, 2), np.inf), np.ones(3)),
    (np.ones((3, 2)), np.ones((3, 2)), np.array([.2, np.nan, .8])),
    (np.ones((3, 2)), np.ones((3, 2)), np.array([.2, -1., .8])),
    (np.ones((3, 2)), np.ones((3, 2)), np.array([.2, 2., .8])),
])
def test_hard_gate_rejects_bad_dimensions_and_nonfinite_values(body, excess, probability):
    with pytest.raises(ValueError):
        hard_gate_prediction(body, excess, probability)


@pytest.mark.parametrize("threshold", [np.nan, np.inf, -.1, 1.1, [.5, .5]])
def test_hard_gate_rejects_invalid_threshold(threshold):
    with pytest.raises(ValueError):
        hard_gate_prediction(np.ones((3, 2)), np.ones((3, 2)), np.ones(3), threshold)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_f1_metric_path_matches_existing_record_and_metric_implementation(dtype):
    rng = np.random.default_rng(17)
    truth = rng.normal(size=(81, 4)).astype(dtype)
    truth[-5:] = [1., 3., 3., 0.]  # Exercise tied top-5 and first-occurrence peak timing.
    arrays = dict(y_true=truth, body_phys=rng.normal(size=truth.shape).astype(dtype),
                  excess_phys=rng.uniform(0, 2, size=truth.shape).astype(dtype),
                  gate_probability=np.resize(np.array([.2, .5, .8], dtype=dtype), len(truth)))
    arrays["y_pred"] = arrays["body_phys"] + arrays["gate_probability"][:, None] * arrays["excess_phys"]
    soft = arrays["y_pred"].copy()
    manual = arrays["body_phys"] + (arrays["gate_probability"] >= .5)[:, None] * arrays["excess_phys"]
    expected = summarize_windows(make_records(manual, truth), event_threshold=1.)
    with patch("emulator.training.metrics.physical_peak_columns", wraps=physical_peak_columns) as reused:
        actual = summarize_hard_gate(arrays, 1.)
    assert reused.call_count == 1
    assert actual == expected  # Includes all 10 requested metrics, plus existing event metrics.
    np.testing.assert_array_equal(arrays["y_pred"], soft)
    json.dumps(actual, allow_nan=False)


def test_empty_diagnostics_and_original_inference_record_conversion():
    arrays = dict(y_true=np.empty((0, 4)), body_phys=np.empty((0, 4)), excess_phys=np.empty((0, 4)),
                  gate_probability=np.empty(0))
    assert all(value is None for value in summarize_hard_gate(arrays, 1.).values())
    fixture = json.loads((Path(__file__).parent / "fixtures/post3_peak_metrics.json").read_text())
    prediction, truth = np.array(fixture["y_pred"], dtype=np.float32), np.array(fixture["y_true"], dtype=np.float32)
    records = prediction_window_records(prediction, truth)
    np.testing.assert_array_equal(records, make_records(prediction, truth))
    metrics = summarize_windows(records)
    for key, value in fixture["inference_metrics"].items():
        assert metrics[key] == value


@pytest.mark.parametrize("prediction,truth", [
    (np.ones(3), np.ones(3)), (np.ones((2, 3)), np.ones((2, 1))),
    (np.ones((2, 0)), np.ones((2, 0))),
    (np.full((2, 3), np.inf), np.ones((2, 3))),
    (np.ones((2, 3)), np.full((2, 3), np.nan)),
])
def test_prediction_records_validate_alignment_and_finiteness(prediction, truth):
    with pytest.raises(ValueError):
        prediction_window_records(prediction, truth)


def test_historical_f0_infer_adds_only_offline_hard_gate_metrics(tmp_path):
    graphs, checkpoint = inference_fixture(tmp_path, historical=True)
    original_checkpoint = checkpoint.read_bytes()
    normal, diagnostic = tmp_path / "normal", tmp_path / "diagnostic"
    normal_metrics = evaluate_fixture(graphs, checkpoint, normal, diagnostics=False)
    with patch.object(infer, "run_epoch", wraps=infer.run_epoch) as passes:
        diagnostic_metrics = evaluate_fixture(graphs, checkpoint, diagnostic)
    assert passes.call_count == 2  # One pass per year; F1 adds no forward pass.
    assert normal_metrics["metrics"] == diagnostic_metrics["metrics"]
    assert not (normal / "dual_diagnostics.json").exists()
    assert "hard_gate_0p5" not in diagnostic_metrics
    report = json.loads((diagnostic / "dual_diagnostics.json").read_text())
    assert report["hard_gate_0p5"]["threshold"] == .5
    with np.load(normal / "predictions.npz") as before, np.load(diagnostic / "predictions.npz") as after, \
            np.load(diagnostic / "dual_diagnostics.npz") as arrays:
        assert set(after.files) == {"y_pred", "y_true", "tags"}
        for name in before.files:
            np.testing.assert_array_equal(before[name], after[name])
            np.testing.assert_array_equal(before[name], arrays[name])
        np.testing.assert_allclose(arrays["y_pred"], arrays["body_phys"]
                                   + arrays["gate_probability"][:, None] * arrays["excess_phys"], atol=1e-6)
        manual = arrays["body_phys"] + (arrays["gate_probability"] >= .5)[:, None] * arrays["excess_phys"]
        assert not np.array_equal(manual, arrays["y_pred"])
        assert report["hard_gate_0p5"]["overall"] == summarize_windows(make_records(manual, arrays["y_true"]), event_threshold=1.)
        for year in report["years"]:
            selected = np.char.startswith(arrays["tags"], year)
            assert report["hard_gate_0p5"]["by_year"][year] == summarize_windows(
                make_records(manual[selected], arrays["y_true"][selected]), event_threshold=1.)
    assert checkpoint.read_bytes() == original_checkpoint
    assert list(tmp_path.rglob("*.pth")) == [checkpoint]


@pytest.mark.parametrize("options", [
    {"exceedance_head_experiment": "c2r", "exceedance_gate_pooling": "learned"},
    {"excess_formulation": "severity_shape", "target_y_std": [.25, .5, 2., 5.]},
    {"dual_ablation": "fixed_gate"},
])
def test_non_f0_historical_heads_keep_existing_diagnostics_without_f1_label(tmp_path, options):
    graphs, checkpoint = inference_fixture(tmp_path, config_overrides=options, historical=True)
    output = tmp_path / "diagnostics"
    evaluate_fixture(graphs, checkpoint, output)
    report = json.loads((output / "dual_diagnostics.json").read_text())
    assert "hard_gate_0p5" not in report
    assert report["overall"]["samples"] == 3
