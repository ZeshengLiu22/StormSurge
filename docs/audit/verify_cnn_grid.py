"""Compare CNN computations and grid checks with untouched original source.

Run from the repository root with its training environment:
    python docs/audit/verify_cnn_grid.py --original /path/to/Emulator
"""

import argparse
import ast
import copy
from datetime import datetime, timezone
import hashlib
import itertools
import json
from pathlib import Path
import platform
import sys
import types

import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.data import Batch, Data
import torch_geometric

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from emulator.models import ModelConfig, build_model
from emulator.models.spatial import SpatialEncoder


def original_cnn(path):
    """Execute the original AST definitions without importing its obsolete API."""
    names = {"_uniform_grid_dim", "_infer_grid_batch_shape", "GridCNNEncoder"}
    tree = ast.parse(path.read_text(), filename=str(path))
    definitions = [node for node in tree.body if getattr(node, "name", None) in names]
    assert {node.name for node in definitions} == names
    namespace = dict(torch=torch, nn=nn, F=F)
    exec(compile(ast.Module(body=definitions, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


def make_batch(height, width, count, steps=9):
    graphs = []
    for _ in range(count):
        history = torch.randn(height * width, steps, 5)
        graphs.append(Data(x=history[:, -1].clone(), x_hist=history,
                           edge_index=torch.empty(2, 0, dtype=torch.long), grid_H=height, grid_W=width))
    return Batch.from_data_list(graphs)


def exact(left, right):
    if left is None or right is None:
        assert left is right
    else:
        torch.testing.assert_close(left, right, rtol=0, atol=0)


def compare_pair(left, right, left_call, right_call, left_inputs, right_inputs):
    """Check outputs, input/parameter gradients, RNG, and one Adam update."""
    assert left.state_dict().keys() == right.state_dict().keys()
    for key in left.state_dict():
        exact(left.state_dict()[key], right.state_dict()[key])
    results, rng_states = [], []
    rng = torch.get_rng_state()
    for call in (left_call, right_call):
        torch.set_rng_state(rng)
        output = call()
        tensors = (output,) if torch.is_tensor(output) else output
        sum(value.float().square().mean() for value in tensors if value is not None).backward()
        results.append(tensors)
        rng_states.append(torch.get_rng_state())
    assert len(results[0]) == len(results[1])
    for before, after in zip(*results):
        exact(before, after)
    exact(*rng_states)
    for before, after in zip(left_inputs, right_inputs):
        exact(before.grad, after.grad)
    old_parameters, new_parameters = dict(left.named_parameters()), dict(right.named_parameters())
    assert old_parameters.keys() == new_parameters.keys()
    for key in old_parameters:
        exact(old_parameters[key].grad, new_parameters[key].grad)
    for model in (left, right):
        torch.optim.Adam(model.parameters(), lr=.003, weight_decay=1e-5).step()
    for key in left.state_dict():
        exact(left.state_dict()[key], right.state_dict()[key])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "docs/audit/evidence/cnn_grid_equivalence.json")
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(42)
    original_path = args.original / "emulator/models/architectures.py"
    original = original_cnn(original_path)
    infer_grid = original["_infer_grid_batch_shape"]
    encoder = SpatialEncoder(5, 128, 2, .05, "CNN")

    formats = {
        "pyg_batch_tensor": lambda value: torch.tensor([value] * 3),
        "scalar_int": int,
        "scalar_tensor": torch.tensor,
        "single_item_list": lambda value: [value],
        "batch_tuple": lambda value: (value,) * 3,
        "column_tensor": lambda value: torch.full((3, 1), value),
        "integer_valued_float_tensor": lambda value: torch.full((3,), float(value)),
    }
    for format_name, convert in formats.items():
        batch = make_batch(2, 3, 3)
        batch.grid_H, batch.grid_W = convert(2), convert(3)
        before = batch.clone()
        rng = torch.get_rng_state()
        assert infer_grid(batch, batch.x.size(0)) == encoder.grid_shape(batch) == (3, 2, 3), format_name
        exact(rng, torch.get_rng_state())
        for name in ("x", "x_hist", "ptr", "batch"):
            exact(before[name], batch[name])

    invalid = {}
    for name in ("missing_dimension", "empty_dimension", "nonuniform_dimension", "nonpositive_dimension", "wrong_grid_product", "unequal_graph_sizes"):
        batch = make_batch(2, 3, 2)
        if name == "missing_dimension":
            del batch.grid_H
        elif name == "empty_dimension":
            batch.grid_H = torch.tensor([], dtype=torch.long)
        elif name == "nonuniform_dimension":
            batch.grid_H[1] = 3
        elif name == "nonpositive_dimension":
            batch.grid_H.zero_()
        elif name == "wrong_grid_product":
            batch.grid_W.fill_(4)
        else:
            graphs = batch.to_data_list()
            graphs[0].x, graphs[0].x_hist = graphs[0].x[:-1], graphs[0].x_hist[:-1]
            graphs[1].x = torch.cat((graphs[1].x, graphs[1].x[:1]))
            graphs[1].x_hist = torch.cat((graphs[1].x_hist, graphs[1].x_hist[:1]))
            batch = Batch.from_data_list(graphs)
        for check in (lambda: infer_grid(batch, batch.x.size(0)), lambda: encoder.grid_shape(batch)):
            try:
                check()
            except ValueError:
                pass
            else:
                raise AssertionError(f"Expected rejection: {name}")
        invalid[name] = "both reject"
    print(f"Grid checks: {len(formats)} valid metadata representations agree; {len(invalid)} malformed cases both fail.", flush=True)

    shapes = [(1, 1), (1, 7), (7, 1), (2, 3), (3, 2), (11, 17)]
    spatial_cases = 0
    for (height, width), count, depth, training in itertools.product(shapes, (1, 3), (1, 2, 3), (False, True)):
        batch = make_batch(height, width, count)
        old_shape, new_shape = infer_grid(batch, batch.x.size(0)), encoder.grid_shape(batch)
        assert old_shape == new_shape
        cnn_width = 29 if depth == 2 else 11
        torch.manual_seed(2026)
        old = original["GridCNNEncoder"](5, 128, depth, .05, cnn_width).train(training)
        torch.manual_seed(2026)
        new = SpatialEncoder(5, 128, depth, .05, "CNN", cnn_width).train(training)
        history = torch.randn(batch.x.size(0), 9, 5)
        x_old, x_new = history.clone()[:, -1], history.clone()[:, -1]
        if not training:
            x_old, x_new = x_old.contiguous(), x_new.contiguous()
        x_old.requires_grad_()
        x_new.requires_grad_()
        compare_pair(old, new, lambda: old(x_old, old_shape),
                     lambda: new(x_new, batch.edge_index, new_shape), [x_old], [x_new])
        spatial_cases += 1
    print(f"Original CNN versus current CNN: {spatial_cases} exact output/gradient/Adam comparisons passed.", flush=True)

    model_cases = 0
    variants = [("baseline", "single", "LSTM")]
    variants += [("pact", head, temporal) for head, temporal in itertools.product(("single", "dual"), ("MLP", "LSTM", "GRU", "Transformer"))]
    for (kind, head, temporal), history, shape, training in itertools.product(variants, (0, 8), ((2, 3), (11, 17)), (False, True)):
        config = ModelConfig(5, 4, hidden_channels=128, model=kind, encoder_type="CNN", head_type=head,
                             temporal_block=temporal, history_steps=history, dropout=.05, head_dropout=.05,
                             station_feat_dim=6 if kind == "pact" else 0, peak_threshold_norm=[1.] * 4)
        new = build_model(config).train(training)
        old = copy.deepcopy(new)
        # Isolate this table row: keep the entire current model, replacing only its grid check.
        old.spatial.grid_shape = types.MethodType(lambda self, batch: infer_grid(batch, batch.x.size(0)), old.spatial)
        left = make_batch(*shape, count=2, steps=history + 1)
        right = left.clone()
        for batch in (left, right):
            batch.x.requires_grad_()
            batch.x_hist.requires_grad_()
        features_left = torch.randn(6, requires_grad=True)
        features_right = features_left.detach().clone().requires_grad_()
        compare_pair(old, new, lambda: old(left, features_left), lambda: new(right, features_right),
                     [left.x, left.x_hist, features_left], [right.x, right.x_hist, features_right])
        model_cases += 1
    print(f"Current models with original versus current grid checks: {model_cases} exact comparisons passed.", flush=True)

    result = dict(
        generated_utc=datetime.now(timezone.utc).isoformat(), python=platform.python_version(),
        torch=torch.__version__, torch_geometric=torch_geometric.__version__, device="cpu", dtype="float32",
        gpu_rerun=False, gpu_note="Current host NVIDIA driver unavailable; historical GPU evidence is separate.",
        reference="Unmodified AST definitions read from original emulator/models/architectures.py",
        original_source_sha256=hashlib.sha256(original_path.read_bytes()).hexdigest(),
        current_source_sha256={name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in
                              ("emulator/models/spatial.py", "emulator/models/architectures.py", "emulator/data/graph_store.py")},
        valid_metadata_formats=list(formats), malformed_cases=invalid,
        original_cnn_comparisons=dict(cases=spatial_cases, grid_shapes=shapes, batch_sizes=[1, 3], depths=[1, 2, 3],
                                      in_channels=5, hidden_channels=128, cnn_widths=[11, 29], dropout=.05,
                                      modes=["train", "eval"], layouts=["contiguous", "noncontiguous history slice"],
                                      same_seed_initialization="exactly equal"),
        isolated_grid_check_comparisons=dict(cases=model_cases, models=variants, history_hours=[0, 48],
                                             grid_shapes=[[2, 3], [11, 17]], batch_size=2, hidden_channels=128,
                                             scope="Current full model with only grid validation replaced by original; not whole original PACT equivalence."),
        max_output_difference=0.0, max_input_gradient_difference=0.0, max_parameter_gradient_difference=0.0,
        max_parameter_difference_after_one_adam_step=0.0, rng_states="exactly equal",
        initial_finding=dict(grid_shape=[1, 1], batch_size=1, depth=2, mode="eval", input_channels=5,
                             hidden_channels=128, cnn_width=29, max_output_difference=0.0,
                             max_input_gradient_difference=1.4551915228366852e-11,
                             max_parameter_gradient_difference=2.0372681319713593e-10,
                             cause="Out-of-place versus original in-place LeakyReLU; repeating the case with in-place activation removes the differences."),
        decision="Retain current grid checks; restore original in-place LeakyReLU for CNN only. Configs and GraphSAGE computation unchanged.",
        limits="Valid positive-integer rectangular grids in normally collated PyG batches. Does not establish equality for manually inconsistent batch/ptr metadata or long-run forecast skill of the changed PACT architecture.",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"PASS: evidence saved to {args.output}", flush=True)


if __name__ == "__main__":
    main()
