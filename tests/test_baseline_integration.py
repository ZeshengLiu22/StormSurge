"""Shallow baselines using the shared pressure, loss and training pipelines."""

import contextlib
import io
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch_geometric.data import Batch

import infer
import train
from emulator.data import ForcingGraphStore, ForcingGraphView, fit_statistics, normalize_inputs
from emulator.models import NEW_BASELINE_MODELS, STATIC_BASELINE_MODELS, ModelConfig, build_model
from emulator.models import model_config_from_checkpoint
from emulator.training import ForecastLoss, LossConfig
from emulator.training.episode_peaks import build_episode_peak_targets
from test_pressure_modes import MODES, save_pressure_graphs


@pytest.fixture(autouse=True)
def single_cpu_thread():
    """Keep the synthetic models small on shared CPU nodes."""
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize("name", NEW_BASELINE_MODELS)
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("weights", [(0.0, 0.0), (0.4, 0.7)])
def test_pressure_pipeline_and_physical_loss(tmp_path, name, mode, weights):
    """Use real graph transformations and canonical peaks, including missing labels."""
    save_pressure_graphs(tmp_path, count=5)
    store = ForcingGraphStore(tmp_path, "Battery", pressure_mode=mode)
    steps = 0 if name in STATIC_BASELINE_MODELS else 4
    view = ForcingGraphView(store, [0, 1, 2], steps)
    stats = fit_statistics(store, [0, 1, 2], x_norm="robust")
    batch = Batch.from_data_list([view[index] for index in range(len(view))])
    original = store.graphs[0].x.clone()
    normalize_inputs(batch, stats)
    channels = 6 if mode == "anomaly_plus_mean" else 5
    model = build_model(ModelConfig(
        model=name, in_channels=channels, out_channels=4, hidden_channels=8,
        baseline_stem_channels=4, history_steps=steps, dropout=0.0,
    ))
    output = model(batch)
    physical = output.prediction * stats["y_std"] + stats["y_mean"]
    target = batch.y.clone()
    target[1, 0] = float("nan")
    episodes = build_episode_peak_targets(target.numpy(), view.target_timestamps, 2.0)
    peak_mask = torch.from_numpy(episodes.is_episode_gt_peak)
    complete = torch.isfinite(target).all(dim=1)
    tail_mask = complete[:, None] & (target > 2.0)
    eligible = complete.sum() * target.shape[1]
    prior = float(tail_mask.sum() / eligible)
    peak_prior = episodes.metadata["p_episode_peak"]
    criterion = ForecastLoss(
        LossConfig(*weights), stats, 2.0, extreme_hour_prior=prior,
        p_episode_peak=peak_prior,
    )
    loss = criterion(output, physical, target, episode_gt_peak_mask=peak_mask)
    finite = torch.isfinite(target)
    errors = (physical[finite] - target[finite]).square()
    expected = errors.mean()
    expected = expected + weights[0] * (physical[tail_mask] - target[tail_mask]).square().sum() / eligible / prior
    expected = expected + weights[1] * (physical[peak_mask] - target[peak_mask]).square().sum() / eligible / peak_prior
    torch.testing.assert_close(loss, expected)
    loss.backward()
    for parameter in model.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
    torch.testing.assert_close(store.graphs[0].x, original)


@pytest.mark.parametrize("name", NEW_BASELINE_MODELS)
def test_one_epoch_training_checkpoint_and_inference(tmp_path, name):
    """Train one tiny CPU epoch through train.main and replay its saved checkpoint."""
    graphs = tmp_path / "source" / "graphs"
    graphs.mkdir(parents=True)
    save_pressure_graphs(graphs)
    output = tmp_path / "training"
    history = 0 if name in STATIC_BASELINE_MODELS else 12
    options = [
        "--root_dir", str(graphs), "--station", "Battery", "--output_dir", str(output),
        "--model", name, "--history_hours", str(history), "--hidden_channels", "8",
        "--baseline_stem_channels", "4", "--baseline_pooling", "adaptive",
        "--baseline_pool_grid", "2", "--pressure_mode", "anomaly_plus_mean",
        "--dropout", "0", "--epochs", "1", "--warmup_epochs", "0",
        "--batch_size", "2", "--num_workers", "0", "--device", "cpu",
        "--use_station_meta", "0", "--x_aug", "0", "--x_norm", "robust",
        "--exceedance_loss_weight", "0.4", "--episode_gt_aligned_peak_weight", "0.7",
    ]
    with contextlib.redirect_stdout(io.StringIO()):
        train.main(options)
    checkpoint_path = output / "best.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = model_config_from_checkpoint(checkpoint)
    assert config.model == name
    assert config.in_channels == 6
    assert config.history_steps * 6 == history
    assert config.baseline_pooling == "adaptive"
    assert config.baseline_pool_grid == 2
    assert config.baseline_stem_channels == 4
    model = build_model(config)
    result = model.load_state_dict(checkpoint["model_state"], strict=True)
    assert not result.missing_keys and not result.unexpected_keys
    training = checkpoint["training_config"]
    assert training["exceedance_loss_weight"] == 0.4
    assert training["episode_gt_aligned_peak_weight"] == 0.7
    assert checkpoint["model_parameters"]["total"] == sum(p.numel() for p in model.parameters())
    assert checkpoint["episode_peak_metadata"]["p_episode_peak"] > 0
    restored_output = tmp_path / "inference"
    with contextlib.redirect_stdout(io.StringIO()):
        infer.main([
            "--ckpt", str(checkpoint_path), "--root_dir", str(graphs),
            "--out_dir", str(restored_output), "--model", name,
            "--device", "cpu", "--num_workers", "0", "--batch_size", "2", "--save_npz",
        ])
    report = json.loads((restored_output / "metrics.json").read_text())
    assert report["pressure_mode"] == "anomaly_plus_mean"
    assert report["normalization_origin"] == "source_checkpoint_train"
    with np.load(next(output.glob("test_preds_*.npz"))) as expected:
        with np.load(restored_output / "predictions.npz") as actual:
            for field in ("y_pred", "y_true", "tags", "target_timestamps"):
                np.testing.assert_array_equal(actual[field], expected[field])
