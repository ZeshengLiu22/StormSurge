"""Independent shallow architectures: grid/history contracts, gradients and checkpoints."""

from dataclasses import asdict
import io
import itertools
import math

import pytest
import torch
from torch import nn
from torch_geometric.data import Batch, Data

from emulator.models import (
    BASELINE_ARCHITECTURES,
    NEW_BASELINE_MODELS,
    STATIC_BASELINE_MODELS,
    Baseline,
    CNNBaseline,
    CNNGRUBaseline,
    CNNLSTMBaseline,
    ConvLSTMBaseline,
    MLPBaseline,
    MLPGRUBaseline,
    MLPLSTMBaseline,
    ModelConfig,
    PACT,
    build_model,
    model_config_from_checkpoint,
)
from emulator.models.baselines import _history_to_grid, _nodes_to_grid
from emulator.models.spatial import SpatialEncoder
from emulator.models.temporal import TemporalRNN


CLASSES = {
    "baseline_mlp": MLPBaseline,
    "baseline_cnn": CNNBaseline,
    "baseline_mlp_lstm": MLPLSTMBaseline,
    "baseline_mlp_gru": MLPGRUBaseline,
    "baseline_cnn_lstm": CNNLSTMBaseline,
    "baseline_cnn_gru": CNNGRUBaseline,
    "baseline_convlstm": ConvLSTMBaseline,
}
TEMPORAL_MODELS = tuple(name for name in NEW_BASELINE_MODELS if name not in STATIC_BASELINE_MODELS)
POOLING = (("adaptive", 4), ("global_mean", 4), ("adaptive", 2))


@pytest.fixture(autouse=True, scope="module")
def single_threaded_models():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def baseline_config(name, **overrides):
    values = dict(
        in_channels=5,
        out_channels=4,
        hidden_channels=8,
        baseline_stem_channels=3,
        model=name,
        history_steps=0 if name in STATIC_BASELINE_MODELS else 2,
        dropout=0.05,
    )
    values.update(overrides)
    return ModelConfig(**values)


def graph_batch(config, height=3, width=5, count=2):
    graphs = []
    for _ in range(count):
        history = torch.randn(height * width, config.history_steps + 1, config.in_channels)
        graphs.append(Data(
            x=history[:, -1].clone(),
            x_hist=history,
            edge_index=torch.empty(2, 0, dtype=torch.long),
            grid_H=height,
            grid_W=width,
            y=torch.randn(1, config.out_channels),
        ))
    return Batch.from_data_list(graphs)


@pytest.mark.parametrize("name", NEW_BASELINE_MODELS)
def test_registered_independent_architectures(name):
    # Legacy operator/depth settings cannot select or alter a new architecture.
    config = baseline_config(name, encoder_type="unused", temporal_block="unused", temporal_layers=5)
    model = build_model(config)
    assert type(model) is CLASSES[name]
    assert type(model).__bases__ == (nn.Module,)
    assert not any(isinstance(layer, (SpatialEncoder, TemporalRNN, nn.MultiheadAttention))
                   for layer in model.modules())
    assert model.pool.output_size == (4, 4)
    assert model.head.out_features == config.out_channels
    assert set(BASELINE_ARCHITECTURES) == set(CLASSES)
    if name.endswith(("_lstm", "_gru")):
        recurrent_type = nn.LSTM if name.endswith("_lstm") else nn.GRU
        assert type(model.rnn) is recurrent_type
        assert model.rnn.num_layers == 1
        assert model.rnn.input_size == model.rnn.hidden_size == config.hidden_channels
        assert model.rnn.batch_first
    if name.startswith("baseline_mlp"):
        assert sum(isinstance(layer, nn.Linear) for layer in model.spatial) == 2
        assert not any(isinstance(layer, nn.Conv2d) for layer in model.modules())
    elif name.startswith("baseline_cnn"):
        assert sum(isinstance(layer, nn.Conv2d) for layer in model.spatial) == 2
    else:
        assert sum(isinstance(layer, nn.Conv2d) for layer in model.modules()) == 2
        assert model.gates.out_channels == 4 * config.baseline_stem_channels


@pytest.mark.parametrize("name", NEW_BASELINE_MODELS)
@pytest.mark.parametrize("pooling,grid", POOLING)
@pytest.mark.parametrize("channels,count", tuple(itertools.product((5, 6), (1, 3))))
def test_forward_pooling_resolution_and_finite_gradients(name, pooling, grid, channels, count):
    config = baseline_config(name, in_channels=channels, baseline_pooling=pooling,
                             baseline_pool_grid=grid)
    model = build_model(config)
    assert model.pool.output_size == ((grid, grid) if pooling == "adaptive" else (1, 1))
    original_parameters = sum(parameter.numel() for parameter in model.parameters())
    for height, width in ((3, 5), (6, 4)):
        batch = graph_batch(config, height, width, count)
        output = model(batch)
        assert output.prediction.shape == (count, 4)
        output.prediction.square().mean().backward()
        for parameter in model.parameters():
            assert parameter.grad is not None
            assert torch.isfinite(parameter.grad).all()
        model.zero_grad(set_to_none=True)
    assert sum(parameter.numel() for parameter in model.parameters()) == original_parameters


@pytest.mark.parametrize("name", TEMPORAL_MODELS)
@pytest.mark.parametrize("hours", (0, 12, 24, 48))
def test_configurable_history_uses_every_timestamp(name, hours):
    torch.manual_seed(31)
    config = baseline_config(name, history_steps=hours // 6, dropout=0)
    model = build_model(config)
    batch = graph_batch(config)
    batch.x_hist.requires_grad_()
    model(batch).prediction.square().sum().backward()
    gradients = batch.x_hist.grad
    assert torch.isfinite(gradients).all()
    assert torch.all(gradients.abs().sum(dim=(0, 2)) > 0)


@pytest.mark.parametrize("name", STATIC_BASELINE_MODELS)
def test_static_history_rejection_and_current_input_only(name):
    with pytest.raises(ValueError, match="history_steps=0"):
        baseline_config(name, history_steps=2)
    # Public classes also reject positive history if constructed directly.
    with pytest.raises(ValueError, match="history_steps=0"):
        CLASSES[name](ModelConfig(5, 4, 8, history_steps=2))
    config = baseline_config(name)
    model = build_model(config).eval()
    batch = graph_batch(config)
    expected = model(batch).prediction
    batch.x_hist.fill_(10000)
    torch.testing.assert_close(model(batch).prediction, expected, rtol=0, atol=0)
    del batch.x_hist
    torch.testing.assert_close(model(batch).prediction, expected, rtol=0, atol=0)
    batch.x_hist = torch.randn(batch.x.shape[0], 2, config.in_channels)
    with pytest.raises(ValueError, match="history"):
        model(batch)


@pytest.mark.parametrize("name", NEW_BASELINE_MODELS)
@pytest.mark.parametrize("invalid", (
    "missing_height", "empty", "nonuniform", "nonpositive", "fractional", "nan",
    "wrong_product", "unequal_counts", "bad_pointer_start", "bad_pointer_end",
    "bad_membership", "missing_membership", "wrong_features", "wrong_history",
    "history_features", "history_nodes",
))
def test_malformed_input_rejected(name, invalid):
    config = baseline_config(name)
    batch = graph_batch(config)
    if invalid == "missing_height":
        del batch.grid_H
    elif invalid == "empty":
        batch.grid_H = torch.empty(0)
    elif invalid == "nonuniform":
        batch.grid_H[1] += 1
    elif invalid == "nonpositive":
        batch.grid_H.zero_()
    elif invalid == "fractional":
        batch.grid_H = torch.tensor([3.5, 3.5])
    elif invalid == "nan":
        batch.grid_H = torch.tensor([float("nan")])
    elif invalid == "wrong_product":
        batch.grid_W += 1
    elif invalid == "unequal_counts":
        batch.ptr[1] -= 1
    elif invalid == "bad_pointer_start":
        batch.ptr[0] = 1
    elif invalid == "bad_pointer_end":
        batch.ptr[-1] -= 1
    elif invalid == "bad_membership":
        batch.batch[0] = 1
    elif invalid == "missing_membership":
        del batch.batch
    elif invalid == "wrong_features":
        batch.x = batch.x[:, :-1]
    elif invalid == "wrong_history":
        batch.x_hist = torch.cat((batch.x_hist, batch.x_hist[:, :1]), dim=1)
    elif invalid == "history_features":
        batch.x_hist = batch.x_hist[:, :, :-1]
    else:
        batch.x_hist = batch.x_hist[:-1]
    with pytest.raises(ValueError):
        build_model(config)(batch)


@pytest.mark.parametrize("name", TEMPORAL_MODELS)
def test_missing_history_rejected(name):
    config = baseline_config(name)
    batch = graph_batch(config)
    del batch.x_hist
    with pytest.raises(ValueError, match="require x_hist"):
        build_model(config)(batch)


def test_row_major_and_chronological_layout():
    size, height, width, steps, channels = 2, 3, 5, 4, 6
    history = torch.arange(size * height * width * steps * channels).reshape(-1, steps, channels)
    grids = _history_to_grid(history, (size, height, width))
    current = _nodes_to_grid(history[:, -1], (size, height, width))
    for graph, row, column, step in itertools.product(range(size), range(height), range(width), range(steps)):
        index = graph * height * width + row * width + column
        torch.testing.assert_close(grids[graph, step, :, row, column], history[index, step])
        torch.testing.assert_close(current[graph, :, row, column], history[index, -1])


@pytest.mark.parametrize("name", TEMPORAL_MODELS[:-1])
def test_shared_spatial_weights_match_individual_timestamp_processing(name):
    config = baseline_config(name, history_steps=4, dropout=0)
    model = build_model(config).eval()
    batch = graph_batch(config)
    embeddings = []
    for timestamp in batch.x_hist.unbind(1):
        if name.startswith("baseline_mlp"):
            features = _nodes_to_grid(model.spatial(timestamp), (2, 3, 5))
        else:
            features = model.spatial(_nodes_to_grid(timestamp, (2, 3, 5)))
        embeddings.append(model.projection(model.pool(features).flatten(1)))
    output, _ = model.rnn(torch.stack(embeddings, dim=1))
    expected = model.head(output[:, -1])
    torch.testing.assert_close(model(batch).prediction, expected)


@pytest.mark.parametrize("name", NEW_BASELINE_MODELS)
def test_no_target_or_station_feature_dependence(name):
    config = baseline_config(name)
    batch = graph_batch(config)
    model = build_model(config).eval()
    expected = model(batch).prediction
    batch.y.fill_(1e8)
    torch.testing.assert_close(model(batch, torch.randn(19)).prediction, expected, rtol=0, atol=0)
    del batch.y
    torch.testing.assert_close(model(batch).prediction, expected, rtol=0, atol=0)


def test_convlstm_spatial_states_gating_and_reset():
    config = baseline_config("baseline_convlstm", history_steps=4, dropout=0)
    model = build_model(config).double().eval()
    batch = graph_batch(config)
    batch.x, batch.x_hist = batch.x.double(), batch.x_hist.double()
    with torch.no_grad():
        model.gates.weight.zero_()
        model.gates.bias.zero_()
        model.gates.bias[3 * config.baseline_stem_channels:] = math.atanh(0.2)
    gate_inputs, pooled_inputs = [], []
    gate_hook = model.gates.register_forward_pre_hook(lambda module, args: gate_inputs.append(args[0].clone()))
    pool_hook = model.pool.register_forward_pre_hook(lambda module, args: pooled_inputs.append(args[0].clone()))
    expected = model(batch).prediction
    assert len(gate_inputs) == config.history_steps + 1
    for features in gate_inputs:
        assert features.shape == (2, 2 * config.baseline_stem_channels, 3, 5)
        assert features.dtype == torch.float64
        assert features.device == batch.x_hist.device
    assert torch.count_nonzero(gate_inputs[0][:, config.baseline_stem_channels:]) == 0
    cell = 0.2 * (1 - 0.5 ** (config.history_steps + 1))
    final_hidden = 0.5 * math.tanh(cell)
    torch.testing.assert_close(pooled_inputs[0], torch.full_like(pooled_inputs[0], final_hidden))
    gate_hook.remove()
    pool_hook.remove()
    torch.testing.assert_close(model(batch).prediction, expected, rtol=0, atol=0)


@pytest.mark.parametrize("name", NEW_BASELINE_MODELS)
def test_cpu_bfloat16_autocast_gradients(name):
    config = baseline_config(name)
    model = build_model(config)
    batch = graph_batch(config)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        prediction = model(batch).prediction
    assert prediction.dtype == torch.bfloat16
    prediction.float().square().mean().backward()
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
               for parameter in model.parameters())


@pytest.mark.parametrize("name", NEW_BASELINE_MODELS)
@pytest.mark.parametrize("pooling,grid", POOLING)
def test_checkpoint_strict_round_trip(name, pooling, grid):
    config = baseline_config(name, baseline_pooling=pooling, baseline_pool_grid=grid, in_channels=6)
    model = build_model(config).eval()
    batch = graph_batch(config)
    expected = model(batch).prediction
    stream = io.BytesIO()
    torch.save({"model_config": asdict(config), "model_state": model.state_dict()}, stream)
    stream.seek(0)
    checkpoint = torch.load(stream, weights_only=False)
    restored_config = model_config_from_checkpoint(checkpoint)
    assert restored_config == config
    restored = build_model(restored_config).eval()
    restored.load_state_dict(checkpoint["model_state"], strict=True)
    torch.testing.assert_close(restored(batch).prediction, expected, rtol=0, atol=0)


@pytest.mark.parametrize("name", ("pact", "baseline"))
@pytest.mark.parametrize("encoder", ("GraphSAGE", "CNN"))
def test_historical_checkpoint_without_new_fields(name, encoder):
    config = ModelConfig(5, 4, 8, model=name, encoder_type=encoder, history_steps=2)
    model = build_model(config).eval()
    assert type(model) is (PACT if name == "pact" else Baseline)
    saved = asdict(config)
    for field in ("baseline_pooling", "baseline_pool_grid", "baseline_stem_channels"):
        del saved[field]
    saved.update(head_type="single", peak_threshold_norm=[1.0] * 4)
    checkpoint = {"model_config": saved, "model_state": model.state_dict()}
    restored = build_model(model_config_from_checkpoint(checkpoint)).eval()
    restored.load_state_dict(checkpoint["model_state"], strict=True)
    batch = graph_batch(config)
    torch.testing.assert_close(restored(batch).prediction, model(batch).prediction, rtol=0, atol=0)
    # New pooling settings must not alter any legacy layer or state key.
    ignored = asdict(config)
    ignored.update(baseline_pooling="unused", baseline_pool_grid=-3, baseline_stem_channels=0)
    unchanged = build_model(ModelConfig(**ignored)).eval()
    unchanged.load_state_dict(checkpoint["model_state"], strict=True)
    torch.testing.assert_close(unchanged(batch).prediction, model(batch).prediction, rtol=0, atol=0)


@pytest.mark.parametrize("field,value", (
    ("model", "unknown"), ("baseline_pooling", "attention"),
    ("baseline_pool_grid", 0), ("baseline_pool_grid", 1.5),
    ("baseline_stem_channels", -1), ("baseline_stem_channels", True),
    ("hidden_channels", 0), ("in_channels", 0), ("out_channels", -1),
    ("history_steps", -1), ("history_steps", 1.5), ("dropout", 1.5),
))
def test_invalid_configuration(field, value):
    with pytest.raises(ValueError):
        baseline_config("baseline_cnn_gru", **{field: value})
