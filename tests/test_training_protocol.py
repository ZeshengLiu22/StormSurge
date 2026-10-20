"""Focused regression tests for the validated attention/Adam production policies."""

from contextlib import redirect_stderr
from copy import deepcopy
from dataclasses import asdict, replace
from datetime import timedelta
import hashlib
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
from torch import nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch_geometric.data import Batch, Data

from emulator.models import ModelConfig, NEW_BASELINE_MODELS, build_model, model_config_from_checkpoint
from emulator.models.node_attention import node_attention
from emulator.training.arguments import parse_args
from emulator.training.weight_decay import build_adam, format_optimizer_groups


ROOT = Path(__file__).resolve().parents[1]


def config(backend="math", encoder="GraphSAGE", **kwargs):
    """Small real model with the same attention and Adam code as production."""
    return ModelConfig(3, 4, 8, node_attention_backend=backend, encoder_type=encoder,
                       history_steps=1, node_read_heads=2, time_read_heads=2,
                       temporal_block="GRU", temporal_dropout=0, **kwargs)


def batch(encoder="GraphSAGE"):
    """GNN batches have actual padding; CNNs retain their required uniform grids."""
    return Batch.from_data_list([
        Data(x=torch.randn(n, 3), x_hist=torch.randn(n, 2, 3), y=torch.randn(1, 4),
             edge_index=torch.stack((torch.arange(n), torch.arange(n))), grid_H=2, grid_W=3)
        for n in ((6, 4) if encoder == "GraphSAGE" else (6, 6))
    ])


def flags():
    """Global settings must survive the local backend context unchanged."""
    return tuple(getattr(torch.backends.cuda, name + "_sdp_enabled")()
                 for name in ("flash", "mem_efficient", "math", "cudnn"))


def group_signature(model, optimizer):
    """Name every parameter in every group for cross-rank comparisons."""
    names = {id(p): name for name, p in model.named_parameters()}
    return [(g["weight_decay"], [names[id(p)] for p in g["params"]]) for g in optimizer.param_groups]


def ddp_worker(rank, directory):
    """Four CPU ranks exercise real wrapped model grouping and one Adam update."""
    torch.set_num_threads(1)
    os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo")
    dist.init_process_group("gloo", init_method="file://" + str(Path(directory) / "rendezvous"),
                            rank=rank, world_size=4, timeout=timedelta(seconds=90))
    results = []
    try:
        for backend, model_key in (("math", "pact"), ("auto", "pact"), ("math", "baseline")):
            torch.manual_seed(42)
            model = build_model(replace(config(backend), model=model_key))
            wrapped = nn.parallel.DistributedDataParallel(model)
            optimizer = build_adam(wrapped, .005)
            assert group_signature(model, optimizer) == group_signature(model, build_adam(model, .005))
            signatures = [None] * 4
            dist.all_gather_object(signatures, group_signature(model, optimizer))
            assert all(signature == signatures[0] for signature in signatures)
            torch.manual_seed(1000 + rank)
            data = batch()
            output = wrapped(data).prediction
            loss = (output - data.y).square().mean()
            assert torch.isfinite(output).all() and torch.isfinite(loss)
            loss.backward()
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
            optimizer.step()
            digest = hashlib.sha256(b"".join(p.detach().numpy().tobytes() for p in model.parameters())).hexdigest()
            digests = [None] * 4
            dist.all_gather_object(digests, digest)
            assert len(set(digests)) == 1
            results.append(dict(backend=backend, model=model_key, loss=float(loss.detach()),
                                groups=signatures[0], parameters_equal_after_update=True))
            del optimizer, wrapped, model
        (Path(directory) / f"rank{rank}.json").write_text(json.dumps(results))
    finally:
        dist.destroy_process_group()


class BackendTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_auto_preserves_original_call_and_gradients_with_real_padding(self):
        for training in (False, True):
            for padded in (False, True):
                with self.subTest(training=training, padded=padded):
                    torch.manual_seed(123)
                    readout = nn.MultiheadAttention(8, 2, dropout=.2, batch_first=True).train(training)
                    query = torch.randn(2, 1, 8, requires_grad=True)
                    nodes = torch.randn(2, 5, 8, requires_grad=True)
                    mask = torch.zeros(2, 5, dtype=torch.bool)
                    mask[1, -2:] = padded
                    rng = torch.get_rng_state()
                    expected = readout(query, nodes, nodes, key_padding_mask=mask, need_weights=False)[0]
                    gradients = torch.autograd.grad(expected.square().sum(), (query, nodes, *readout.parameters()))
                    torch.set_rng_state(rng)
                    actual = node_attention(readout, query, nodes, mask, "auto")[0]
                    actual_gradients = torch.autograd.grad(actual.square().sum(), (query, nodes, *readout.parameters()))
                    self.assertTrue(torch.equal(expected, actual))
                    self.assertTrue(all(torch.equal(a, b) for a, b in zip(gradients, actual_gradients)))

    def test_math_preserves_nonempty_padding_and_masks_its_gradients(self):
        readout = nn.MultiheadAttention(8, 2, batch_first=True).double()
        query = torch.randn(2, 1, 8, dtype=torch.float64, requires_grad=True)
        nodes = torch.randn(2, 5, 8, dtype=torch.float64, requires_grad=True)
        mask = torch.tensor([[False] * 5, [False, False, False, True, True]])
        expected = node_attention(readout, query, nodes, mask, "math")[0]
        gradients = torch.autograd.grad(expected.sum(), nodes)[0]
        self.assertEqual(float(gradients[1, 3:].abs().max()), 0.)
        changed = nodes.detach().clone()
        changed[1, 3:] = 1e8
        self.assertTrue(torch.equal(node_attention(readout, query, changed, mask, "math")[0], expected))
        unpadded = node_attention(readout, query[1:2], nodes[1:2, :3], None, "math")[0]
        torch.testing.assert_close(unpadded, expected[1:2], rtol=1e-13, atol=1e-13)

    def test_actual_dispatch_and_forecast_flags_are_local_for_cnn_and_gnn(self):
        for encoder in ("CNN", "GraphSAGE"):
            for backend in ("auto", "math"):
                with self.subTest(encoder=encoder, backend=backend):
                    model = build_model(config(backend, encoder)).train()
                    original_flags = flags()
                    node_forward, forecast_forward = model.node_readout.forward, model.forecast_readout.forward
                    observed_masks = []
                    def marked_node(*args, **kwargs):
                        observed_masks.append(kwargs["key_padding_mask"].clone())
                        with torch.profiler.record_function("test:node"):
                            return node_forward(*args, **kwargs)
                    def marked_forecast(*args, **kwargs):
                        self.assertEqual(flags(), original_flags)
                        with torch.profiler.record_function("test:forecast"):
                            return forecast_forward(*args, **kwargs)
                    with patch.object(model.node_readout, "forward", marked_node), patch.object(model.forecast_readout, "forward", marked_forecast):
                        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profiler:
                            output = model(batch(encoder)).prediction
                            loss = output.square().mean()
                            loss.backward()
                    self.assertTrue(torch.isfinite(output).all() and torch.isfinite(loss))
                    self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
                    self.assertEqual(flags(), original_flags)
                    self.assertEqual(any(bool(mask.any()) for mask in observed_masks), encoder == "GraphSAGE")
                    scopes = {"node": set(), "forecast": set()}
                    for event in profiler.events():
                        if "scaled_dot_product" not in event.name:
                            continue
                        parent = event.cpu_parent
                        while parent is not None and not parent.name.startswith("test:"):
                            parent = parent.cpu_parent
                        if parent is not None:
                            scopes[parent.name.split(":")[1]].add(event.name)
                    self.assertTrue(scopes["forecast"])
                    if backend == "math":
                        self.assertIn("aten::_scaled_dot_product_attention_math", scopes["node"])
                    else:
                        self.assertIn("aten::scaled_dot_product_attention", scopes["node"])

    def test_invalid_or_unavailable_explicit_backend_never_falls_back(self):
        readout = nn.MultiheadAttention(8, 2, batch_first=True)
        q, nodes = torch.randn(2, 1, 8), torch.randn(2, 4, 8)
        with self.assertRaises(ValueError):
            node_attention(readout, q, nodes, None, "invalid")
        with patch("emulator.models.node_attention.sdpa_kernel", side_effect=RuntimeError("unavailable")):
            with self.assertRaisesRegex(RuntimeError, "unavailable"):
                node_attention(readout, q, nodes, None, "math")

    def test_legacy_and_new_checkpoints_preserve_backend_and_parameter_schema(self):
        data = batch()
        for backend in ("auto", "math"):
            cfg = config(backend)
            torch.manual_seed(42)
            model = build_model(cfg).eval()
            checkpoint = dict(model_config=asdict(cfg), model_state=model.state_dict())
            for legacy in (False, True) if backend == "auto" else (False,):
                saved = deepcopy(checkpoint)
                if legacy:
                    del saved["model_config"]["node_attention_backend"]
                original = deepcopy(saved["model_config"])
                with patch.dict(os.environ, NODE_ATTENTION_BACKEND="math"):
                    restored_config = model_config_from_checkpoint(saved)
                self.assertEqual(restored_config.node_attention_backend, backend)
                self.assertEqual(saved["model_config"], original)
                restored = build_model(restored_config).eval()
                restored.load_state_dict(saved["model_state"], strict=True)
                self.assertEqual(list(model.state_dict()), list(restored.state_dict()))
                with torch.no_grad():
                    torch.testing.assert_close(restored(data).prediction, model(data).prediction, rtol=0, atol=0)
        self.assertEqual(ModelConfig(3, 4, 8).node_attention_backend, "auto")


class WeightDecayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_membership_covers_spatial_temporal_forecast_and_all_baselines(self):
        configs = [replace(config(encoder=encoder), temporal_block=temporal)
                   for encoder in ("CNN", "GraphSAGE") for temporal in ("GRU", "LSTM", "MLP", "Transformer")]
        configs += [replace(config(), model=key, history_steps=0 if key in ("baseline_mlp", "baseline_cnn") else 1)
                    for key in ("baseline", *NEW_BASELINE_MODELS)]
        for cfg in configs:
            with self.subTest(model=cfg.model, encoder=cfg.encoder_type, temporal=cfg.temporal_block):
                model = build_model(cfg)
                optimizer = build_adam(model, .005)
                flat = [p for group in optimizer.param_groups for p in group["params"]]
                self.assertEqual(len(flat), len({id(p) for p in flat}))
                self.assertEqual({id(p) for p in flat}, {id(p) for p in model.parameters() if p.requires_grad})
                actual = {id(p): group["weight_decay"] for group in optimizer.param_groups for p in group["params"]}
                for name, parameter in model.named_parameters():
                    expected = 0. if name.startswith(("horizon_embed.", "forecast_readout.")) else 1e-5
                    self.assertEqual(actual[id(parameter)], expected, name)
                self.assertEqual(len(optimizer.param_groups), 2 if cfg.model == "pact" else 1)
                self.assertIs(type(optimizer), torch.optim.Adam)
                self.assertEqual(optimizer.defaults["weight_decay"], 0.)
                self.assertFalse(optimizer._optimizer_step_pre_hooks)
                self.assertFalse(any(p._backward_hooks for p in model.parameters()))
                message = format_optimizer_groups(optimizer, "module")
                self.assertIn("weight_decay_policy=module", message)
                self.assertIn("tensors=", message)
                self.assertIn("parameters=", message)

    def test_missing_partial_shared_and_frozen_modules(self):
        model = nn.Module()
        model.horizon_embed = nn.Embedding(4, 8)
        model.alias = model.horizon_embed
        model.head = nn.Linear(8, 4)
        model.head.bias.requires_grad_(False)
        for wrapped in (model, nn.DataParallel(model)):
            optimizer = build_adam(wrapped, .005)
            actual = {id(p): group["weight_decay"] for group in optimizer.param_groups for p in group["params"]}
            self.assertEqual(actual, {id(model.horizon_embed.weight): 0., id(model.head.weight): 1e-5})
        self.assertEqual(build_adam(nn.Linear(3, 2), .005).param_groups[0]["weight_decay"], 1e-5)

    def test_updates_and_moments_match_independent_native_adam_reference(self):
        for dtype in (torch.float32, torch.float64):
            for foreach in (False, True):
                with self.subTest(dtype=dtype, foreach=foreach):
                    model = build_model(config()).to(dtype)
                    optimizer = build_adam(model, .005, foreach=foreach)
                    reference = {name: nn.Parameter(p.detach().clone()) for name, p in model.named_parameters()}
                    native = torch.optim.Adam([
                        dict(params=[p], weight_decay=0. if name.startswith(("horizon_embed.", "forecast_readout.")) else 1e-5)
                        for name, p in reference.items()
                    ], lr=.005, foreach=foreach)
                    for step in range(3):
                        for index, (name, p) in enumerate(model.named_parameters()):
                            p.grad = torch.full_like(p, 0. if step == 0 else (index % 3 - 1) * .002 * step)
                            reference[name].grad = p.grad.clone()
                        optimizer.step()
                        native.step()
                        for name, p in model.named_parameters():
                            self.assertTrue(torch.equal(p, reference[name]), name)
                            for key in ("exp_avg", "exp_avg_sq"):
                                self.assertTrue(torch.equal(optimizer.state[p][key], native.state[reference[name]][key]), name)

    def test_legacy_global_is_explicit_opt_in(self):
        model = build_model(config())
        optimizer = build_adam(model, .005, policy="legacy_global")
        self.assertEqual(len(optimizer.param_groups), 1)
        self.assertEqual(optimizer.param_groups[0]["weight_decay"], 1e-5)
        self.assertEqual(len(optimizer.param_groups[0]["params"]), len(list(model.parameters())))
        with self.assertRaises(ValueError):
            build_adam(model, .005, policy="global")

    def test_four_rank_ddp_group_consistency_and_finite_updates(self):
        with tempfile.TemporaryDirectory(prefix="stormsurge-protocol-ddp-") as directory:
            mp.spawn(ddp_worker, args=(directory,), nprocs=4, join=True)
            rows = [json.loads((Path(directory) / f"rank{rank}.json").read_text()) for rank in range(4)]
            self.assertTrue(all(len(row) == 3 for row in rows))
            self.assertTrue(all(case["parameters_equal_after_update"] for row in rows for case in row))


class InterfaceTests(unittest.TestCase):
    def test_training_optimizer_uses_selected_policy_and_existing_schedule(self):
        import train

        for backend in ("math", "auto"):
            args = parse_args(["--node_attention_backend", backend, "--lr", ".005"])
            model = build_model(config(backend))
            with patch.object(train, "log_message") as log:
                optimizer, scheduler = train._build_optimizer_and_scheduler(args, model)
            log.assert_called_once_with(format_optimizer_groups(optimizer, "module"))
            self.assertEqual([g["weight_decay"] for g in optimizer.param_groups], [0., 1e-5])
            self.assertEqual([g["initial_lr"] for g in optimizer.param_groups], [.005, .005])
            self.assertEqual(args.warmup_epochs, 5)
            self.assertEqual(args.max_grad_norm, 0.)
            self.assertIsInstance(scheduler, torch.optim.lr_scheduler.LambdaLR)

    def test_cli_environment_overrides_and_validation(self):
        with patch.dict(os.environ, {}, clear=True):
            args = parse_args([])
            self.assertEqual((args.node_attention_backend, args.weight_decay_policy), ("math", "module"))
        with patch.dict(os.environ, NODE_ATTENTION_BACKEND="auto", WEIGHT_DECAY_POLICY="legacy_global"):
            args = parse_args([])
            self.assertEqual((args.node_attention_backend, args.weight_decay_policy), ("auto", "legacy_global"))
            args = parse_args(["--node_attention_backend", "math", "--weight_decay_policy", "module"])
            self.assertEqual((args.node_attention_backend, args.weight_decay_policy), ("math", "module"))
        for key in ("NODE_ATTENTION_BACKEND", "WEIGHT_DECAY_POLICY"):
            with patch.dict(os.environ, {key: "invalid"}), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parse_args([])

    def test_active_configs_and_pbs_explicit_defaults_and_environment_override(self):
        configs = list((ROOT / "configs").rglob("train_config_*.sh"))
        self.assertTrue(configs)
        for path in [*configs, *(ROOT / "PBS-launcher").rglob("*.pbs")]:
            text = path.read_text()
            self.assertIn('NODE_ATTENTION_BACKEND="${NODE_ATTENTION_BACKEND:-math}"', text, str(path))
            self.assertIn('WEIGHT_DECAY_POLICY="${WEIGHT_DECAY_POLICY:-module}"', text, str(path))
        env = dict(os.environ, DRY_RUN="1", USE_TMUX="0", PYTHON_BIN=sys.executable, S0_PYTHON_BIN=sys.executable,
                   NODE_ATTENTION_BACKEND="auto", WEIGHT_DECAY_POLICY="legacy_global")
        result = subprocess.run(["bash", str(ROOT / "train.sh"), str(configs[0])], cwd=ROOT,
                                env=env, capture_output=True, text=True, check=True)
        command = next(shlex.split(line[5:])[1:] for line in result.stdout.splitlines() if line.startswith("CMD: "))
        args = parse_args(command)
        self.assertEqual((args.node_attention_backend, args.weight_decay_policy), ("auto", "legacy_global"))


class LauncherTests(unittest.TestCase):
    """Reuse the investigation's fake child/tmux handoff test, without training."""

    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="stormsurge-protocol-launcher-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.environment = dict(os.environ)
        for key in ("NODE_ATTENTION_BACKEND", "WEIGHT_DECAY_POLICY", "PACT_RUN_NAME",
                    "SESSION_NAME", "PYTHON_RUN_TAG_BASE", "SLURM_NTASKS_PER_NODE",
                    "SLURM_JOB_ID", "PACT_RUNSTAMP", "QSUB_LOCAL_STATUS_FILE"):
            self.environment.pop(key, None)
        self.environment.update(DRY_RUN="0", USE_TMUX="0", PYTHON_BIN=sys.executable,
                                S0_PYTHON_BIN=sys.executable, CUDA_VISIBLE_DEVICES="")
        self.capture = self.root / "argv.json"
        self.training = self.root / "capture_training.py"
        self.training.write_text(
            "import json, sys\n"
            f"with open({str(self.capture)!r}, 'w') as handle: json.dump(sys.argv[1:], handle)\n"
        )

    def config(self, *, tmux=False, **overrides):
        values = dict(MODEL="perceiver3", TRAIN_PY=str(self.training), num_gpus="1",
                      USE_TMUX="1" if tmux else "0", ALL_RESULTS_ROOT=str(self.root / "results"),
                      RUN_DIR_NAME_STYLE="runname_timestamp", PACT_RUN_NAME="launcher_test",
                      PACT_RUNSTAMP="FIXED", SESSION_NAME="fake_launcher_session", NUM_WORKERS="0")
        values.update(overrides)
        path = self.root / "test_config.sh"
        path.write_text("".join(f"{key}={shlex.quote(value)}\n" for key, value in values.items()))
        return path

    def launch(self, config, **overrides):
        return subprocess.run(["bash", str(ROOT / "train.sh"), str(config)], cwd=ROOT,
                              env=self.environment | overrides, capture_output=True, text=True)

    def check_child_and_snapshot(self, result):
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        args = parse_args(json.loads(self.capture.read_text()))
        self.assertEqual((args.node_attention_backend, args.weight_decay_policy), ("auto", "legacy_global"))
        snapshot = self.root / "results/launcher_test__FIXED/config_used.sh"
        self.assertIn('NODE_ATTENTION_BACKEND="auto"', snapshot.read_text())
        self.assertIn('WEIGHT_DECAY_POLICY="legacy_global"', snapshot.read_text())
        replay = self.launch(snapshot, DRY_RUN="1")
        self.assertEqual(replay.returncode, 0, replay.stdout + replay.stderr)
        command = next(shlex.split(line[5:])[1:] for line in replay.stdout.splitlines() if line.startswith("CMD: "))
        args = parse_args(command)
        self.assertEqual((args.node_attention_backend, args.weight_decay_policy), ("auto", "legacy_global"))

    def test_unexported_config_reaches_child_snapshot_and_replay(self):
        self.check_child_and_snapshot(self.launch(self.config(
            NODE_ATTENTION_BACKEND="auto", WEIGHT_DECAY_POLICY="legacy_global")))

    def test_tmux_stale_environment_cannot_replace_selected_policies(self):
        binary = self.root / "bin/tmux"
        binary.parent.mkdir()
        binary.write_text(
            f"#!{sys.executable}\n"
            "import os, subprocess, sys\n"
            "assert sys.argv[1] == 'new-session'\n"
            "environment = {'PATH': os.environ['PATH'], 'NODE_ATTENTION_BACKEND': 'math', 'WEIGHT_DECAY_POLICY': 'module'}\n"
            "directory = sys.argv[sys.argv.index('-c') + 1]\n"
            "result = subprocess.run(['bash', '-c', sys.argv[-1]], cwd=directory, env=environment)\n"
            "raise SystemExit(result.returncode)\n"
        )
        binary.chmod(0o755)
        self.check_child_and_snapshot(self.launch(
            self.config(tmux=True), NODE_ATTENTION_BACKEND="auto", WEIGHT_DECAY_POLICY="legacy_global",
            PATH=str(binary.parent) + os.pathsep + self.environment["PATH"]))

    def test_invalid_values_fail_before_creating_outputs(self):
        for key in ("NODE_ATTENTION_BACKEND", "WEIGHT_DECAY_POLICY"):
            with self.subTest(key=key):
                result = self.launch(self.config(**{key: "invalid"}))
                self.assertEqual(result.returncode, 2)
                self.assertIn(key + " must be", result.stderr)
                self.assertFalse((self.root / "results").exists())


if __name__ == "__main__":
    unittest.main()
