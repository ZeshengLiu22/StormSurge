#!/usr/bin/env python3
"""Experimental TRAIN-only loss/gradient scale diagnosis; never launches a factorial.

Uses the production model, loss helpers, normalization, and run_epoch unchanged.
The only optimizer trajectory is Dual Base. Candidate weights are post-processing.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import contextmanager
import csv
from dataclasses import asdict, fields
import gc
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time

sys.dont_write_bytecode = True
import numpy as np
import torch

COMPONENTS = ("base", "tail", "body", "excess", "gate", "amp_true", "dual_base")
TAIL_WEIGHTS = (0.01, 0.025, 0.05, 0.1)
AMP_WEIGHTS = (1e-4, 3e-4, 7e-4, 1e-3, 3e-3)
GROUPS = ("shared", "temporal", "excess", "full")


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, default=str, allow_nan=False) + "\n")


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def state_digest(model):
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def production_args(repo, station):
    """Read the active shell config, then explicitly apply the requested protocol."""
    from emulator.training.arguments import parse_args
    defaults = vars(parse_args(["--model", "perceiver3"]))
    special = dict(lr="LR_LIST", loss_mode="LOSS_MODE_LIST", tail_lambda="TAIL_LAMBDA_LIST",
                   slope_lambda="SLOPE_LAMBDA_LIST", slope_mask_s="SLOPE_MASK_S_LIST",
                   wmse_q="WMSE_Q_LIST", amp="USE_AMP", tf32="USE_TF32")
    mapping = {key: special.get(key, key.upper()) for key in defaults}
    config = repo / "configs/configs_train/NCEP" / f"train_config_NCEP_{station}_P3_Base.sh"
    names = sorted(set(mapping.values()) | {"num_gpus", "HISTORY_HOURS_LIST"})
    script = 'source "$1"; shift; for name in "$@"; do printf "%s\\0%s\\0" "$name" "${!name}"; done'
    result = subprocess.run(["bash", "-c", script, "bash", str(config), *names],
                            cwd=repo, check=True, capture_output=True)
    values = result.stdout.decode().split("\0")[:-1]
    shell = dict(zip(values[::2], values[1::2]))
    argv = []
    flags = {"amp", "tf32", "pin_memory", "persistent_workers"}
    for name, key in mapping.items():
        if name in {"history_hours", "output_dir", "run_tag", "filter", "device"}:
            continue
        value = shell.get(key, "")
        if not value:
            continue
        if name in flags:
            if value == "1":
                argv.append("--" + name)
        else:
            argv.extend(["--" + name, value])
    overrides = dict(model="perceiver3", station=station, encoder_type="GraphSAGE",
                     temporal_block="Transformer", head_type="dual", excess_formulation="direct",
                     history_hours=24, hidden_channels=128, exceedance_percentile=95,
                     body_loss_weight=1, excess_loss_weight=2, gate_loss_weight=.5,
                     lr=.005, batch_size=256, grad_accum_steps=4, seed=42, deterministic=0,
                     x_norm="zscore", x_aug=0, x_clip=0, max_grad_norm=0,
                     loss_mode="mse", tail_lambda=0, slope_lambda=0,
                     excess_amp_loss_weight=0, shape_loss_weight=0, dual_ablation="none",
                     epochs=300, scheduler="cosine", warmup_epochs=5)
    for key, value in overrides.items():
        argv.extend(["--" + key, str(value)])
    args = parse_args(argv)
    args.root_dir = str((repo / args.root_dir).resolve())
    args.station_json_dir = (repo / args.station_json_dir).resolve()
    assert args.exceedance_head_experiment is None and args.dual_loss == 1
    assert args.amp and args.amp_dtype == "bf16" and args.tf32
    return args, dict(source=str(config), shell=shell, explicit_overrides=overrides)


def train_only_store(args):
    """Split filenames by the production year rule before opening any graph file."""
    from emulator.data import ForcingGraphStore
    manifest = ForcingGraphStore.__new__(ForcingGraphStore)
    manifest.year_to_indices = defaultdict(list)
    paths = []
    for path in sorted(Path(args.root_dir).glob("*graphs.pt")):
        parts = path.name.removesuffix("_graphs.pt").split("_")
        if len(parts) < 3:
            raise ValueError(f"Invalid graph filename: {path.name}")
        if parts[2] == args.station:
            manifest.year_to_indices["_".join(parts[:2])].append(len(paths))
            paths.append(path)
    split = manifest.split(**{key: getattr(args, key) for key in
                             ("train_ratio", "val_ratio", "shuffle_years", "seed",
                              "future_only", "future_year_threshold")})
    store = ForcingGraphStore.__new__(ForcingGraphStore)
    store.graphs, store.graph_tags = [], []
    store.year_to_indices = defaultdict(list)
    opened = []
    for index in split["train"]:
        path = paths[index]
        stem = path.name.removesuffix("_graphs.pt")
        year = "_".join(stem.split("_")[:2])
        opened.append(dict(path=str(path), bytes=path.stat().st_size, mtime_ns=path.stat().st_mtime_ns))
        for j, graph in enumerate(torch.load(path, map_location="cpu", weights_only=False)):
            store.year_to_indices[year].append(len(store.graphs))
            store.graph_tags.append(f"{stem}_{j}")
            store.graphs.append(graph)
    if not store.graphs:
        raise ValueError("Empty TRAIN split")
    return store, {part: [paths[i].name for i in ids] for part, ids in split.items()}, opened


def component_losses(output, target, stats, fitted, tail_frac=.05):
    from emulator.training.losses import dual_loss_terms
    from emulator.training.excess_amplitude import excess_amplitude_terms
    prediction = output.prediction.float() * stats["y_std"] + stats["y_mean"]
    error = (prediction - target).square()
    target_norm = (target - stats["y_mean"]) / stats["y_std"]
    body, excess, gate = dual_loss_terms(output, target_norm, stats["y_std"])
    tail_mask = (target.amax(dim=1) >= fitted["tail_threshold"]).to(error.dtype)
    tail = (error.mean(dim=1) * tail_mask).mean() / tail_frac
    amp = excess_amplitude_terms(output.excess, target_norm, output.threshold, stats["y_std"],
                                fitted["event_prior"], target_phys=target,
                                event_threshold_phys=fitted["tau_phys"])
    base = error.mean()
    return dict(base=base, tail=tail, body=body, excess=excess, gate=gate,
                amp_true=amp.loss, dual_base=base + (body + 2 * excess + .5 * gate))


def parameter_groups(model):
    named = [(name, param) for name, param in model.named_parameters() if param.requires_grad]
    indices = dict(shared=[i for i, (name, _) in enumerate(named) if not name.startswith("head.")],
                   temporal=[i for i, (name, _) in enumerate(named) if name.startswith("temporal.")],
                   excess=[i for i, (name, _) in enumerate(named) if name.startswith("head.excess.")],
                   full=list(range(len(named))))
    if any(not ids for ids in indices.values()):
        raise ValueError("Expected production shared, temporal and direct excess parameters")
    return named, indices


def vector_norm(vector):
    return float(torch.linalg.vector_norm(vector.double()).item())


def ratio(numerator, denominator):
    # Undefined ratios are empty in CSV, never silently stabilized with epsilon.
    return numerator / denominator if denominator > 0 else None


def cosine(left, right):
    norm = vector_norm(left) * vector_norm(right)
    return float(torch.dot(left.double(), right.double()).item()) / norm if norm > 0 else None


def rows_for_vectors(station, epoch, unit, unit_id, count, events, tails, losses, vectors):
    rows = []
    for group in GROUPS:
        base_norm, dual_norm = (vector_norm(vectors[name][group]) for name in ("base", "dual_base"))
        for component in COMPONENTS:
            vec = vectors[component][group]
            norm = vector_norm(vec)
            rows.append(dict(station=station, epoch=epoch, unit=unit, unit_id=unit_id,
                             windows=count, events=events, tails=tails, component=component,
                             group=group, loss=losses[component], grad_norm=norm,
                             base_grad_norm=base_norm, dual_base_grad_norm=dual_norm,
                             ratio_base=ratio(norm, base_norm), ratio_dual_base=ratio(norm, dual_norm),
                             cosine_base=cosine(vec, vectors["base"][group]),
                             cosine_dual_base=cosine(vec, vectors["dual_base"][group])))
    return rows


@contextmanager
def isolated_diagnostic(model, device):
    """Preserve model tensors, .grad buffers, module modes, and CPU/CUDA RNG."""
    before = state_digest(model)
    modes = {module: module.training for module in model.modules()}
    before_grads = [None if p.grad is None else p.grad.detach().clone() for p in model.parameters()]
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        try:
            model.train()
            yield
        finally:
            for module, mode in modes.items():
                module.training = mode
    assert state_digest(model) == before, "Diagnostic changed model parameters/buffers"
    for param, previous in zip(model.parameters(), before_grads):
        if previous is None:
            assert param.grad is None, "Diagnostic wrote a .grad buffer"
        else:
            torch.testing.assert_close(param.grad, previous, rtol=0, atol=0)


def diagnose(model, batches, stats, fitted, station_feat, device, station, epoch, *,
             use_amp=True, tail_frac=.05, accum_steps=4):
    """Full-parameter backward passes, no optimizer and no detached backbone."""
    from emulator.data.normalization import normalize_inputs
    from emulator.training import ForecastLoss, LossConfig
    named, indices = parameter_groups(model)
    params = [p for _, p in named]
    base_criterion = ForecastLoss(LossConfig(tail_lambda=0, slope_lambda=0,
                                           excess_loss_weight=2, gate_loss_weight=.5),
                                 stats, fitted["tail_threshold"], fitted["wmse_threshold"],
                                 event_prior=fitted["event_prior"], event_threshold=fitted["tau_phys"])
    rows = []
    with isolated_diagnostic(model, device):
        group_vectors, group_losses = None, defaultdict(float)
        counts = [0, 0, 0]
        for batch_id, source in enumerate(batches):
            # Fixed dropout realization per batch, identical at every sampled epoch.
            torch.manual_seed(420000 + batch_id)
            batch = source.clone().to(device)
            normalize_inputs(batch, stats, x_clip=0, augmentation=None)
            target = batch.y.float()
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
                output = model(batch, station_feat=station_feat)
            terms = component_losses(output, target, stats, fitted, tail_frac)
            prediction = output.prediction.float() * stats["y_std"] + stats["y_mean"]
            torch.testing.assert_close(terms["dual_base"], base_criterion(output, prediction, target), rtol=0, atol=0)
            vectors = {}
            for i, component in enumerate(COMPONENTS):
                grads = torch.autograd.grad(terms[component], params,
                                            retain_graph=i + 1 < len(COMPONENTS), allow_unused=True)
                grads = [torch.zeros_like(p) if g is None else g.detach() for p, g in zip(params, grads)]
                vectors[component] = {group: torch.cat([grads[j].reshape(-1) for j in ids]).float()
                                      for group, ids in indices.items()}
            losses = {key: float(value.detach().item()) for key, value in terms.items()}
            if not all(math.isfinite(value) for value in losses.values()):
                raise FloatingPointError("Nonfinite diagnostic loss")
            count = target.size(0)
            events = int((target.amax(dim=1).double() > fitted["tau_phys"]).sum())
            tails = int((target.amax(dim=1) >= fitted["tail_threshold"]).sum())
            new_rows = rows_for_vectors(station, epoch, "batch", batch_id, count, events, tails, losses, vectors)
            if not all(math.isfinite(row["grad_norm"]) for row in new_rows):
                raise FloatingPointError("Nonfinite diagnostic gradient")
            rows.extend(new_rows)
            if group_vectors is None:
                group_vectors = {key: {g: v.clone() for g, v in groups.items()} for key, groups in vectors.items()}
            else:
                for key, groups in vectors.items():
                    for group, vec in groups.items():
                        group_vectors[key][group].add_(vec)
            for key, value in losses.items():
                group_losses[key] += value
            counts = [x + y for x, y in zip(counts, (count, events, tails))]
            end = (batch_id + 1) % accum_steps == 0 or batch_id + 1 == len(batches)
            if end:
                n = batch_id % accum_steps + 1
                for groups in group_vectors.values():
                    for vec in groups.values():
                        vec.div_(n)
                rows.extend(rows_for_vectors(station, epoch, "accum4", batch_id // accum_steps,
                                             *counts, {k: v / n for k, v in group_losses.items()}, group_vectors))
                group_vectors, group_losses, counts = None, defaultdict(float), [0, 0, 0]
            del output, terms, grads, vectors, batch
    return rows


def statistics(values):
    numbers = np.asarray([v for v in values if v is not None and math.isfinite(v)], dtype=float)
    if not len(numbers):
        return dict(n=0, mean=None, median=None, min=None, max=None)
    return dict(n=len(numbers), mean=float(numbers.mean()), median=float(np.median(numbers)),
                min=float(numbers.min()), max=float(numbers.max()))


def summarize(rows, keys, metrics):
    grouped = defaultdict(list)
    for row in rows:
        grouped[tuple(row[k] for k in keys)].append(row)
    result = []
    for key, values in sorted(grouped.items()):
        record = dict(zip(keys, key))
        for metric in metrics:
            record.update({f"{metric}_{stat}": value for stat, value in statistics(v[metric] for v in values).items()})
        result.append(record)
    return result


def analytic_candidates(rows):
    candidates = []
    weights = dict(tail=TAIL_WEIGHTS, amp_true=AMP_WEIGHTS, body=(1.,), excess=(2.,), gate=(.5,))
    for row in rows:
        for weight in weights.get(row["component"], ()):
            record = {k: row[k] for k in ("station", "epoch", "unit", "unit_id", "component", "group")}
            record.update(weight=weight, weighted_loss=weight * row["loss"],
                          weighted_grad_norm=weight * row["grad_norm"],
                          weighted_ratio_base=None if row["ratio_base"] is None else weight * row["ratio_base"],
                          weighted_ratio_dual_base=None if row["ratio_dual_base"] is None else weight * row["ratio_dual_base"])
            candidates.append(record)
    return candidates


def fmt(value):
    return "undefined" if value is None else f"{value:.4g}"


def write_reports(output_dir):
    rows = []
    for path in sorted(output_dir.glob("*/batch_gradients.json")):
        rows.extend(json.loads(path.read_text()))
    if not rows:
        return
    keys = ["station", "epoch", "unit", "component", "group"]
    metrics = ["loss", "grad_norm", "ratio_base", "ratio_dual_base", "cosine_base", "cosine_dual_base"]
    summary = summarize(rows, keys, metrics)
    candidates = analytic_candidates(rows)
    candidate_summary = summarize(candidates, keys + ["weight"],
                                  ["weighted_loss", "weighted_grad_norm", "weighted_ratio_base", "weighted_ratio_dual_base"])
    write_csv(output_dir / "batch_gradients.csv", rows)
    write_csv(output_dir / "component_summary.csv", summary)
    write_csv(output_dir / "candidate_batches.csv", candidates)
    write_csv(output_dir / "candidate_summary.csv", candidate_summary)
    lines = ["# TRAIN loss-scale diagnosis", "",
             "Experimental analysis only. No factorial has been generated or launched.", "",
             "`base` is physical trajectory MSE. `dual_base = base + body + 2 excess + 0.5 gate` is the actual trained objective. "
             "The requested R uses base MSE; a second ratio uses the complete Dual Base gradient, including cancellation.", "",
             "Sixteen fixed, uniformly shuffled TRAIN batches of 256 by default; no event balancing or replacement. "
             "Each checkpoint uses the same windows and training-mode dropout seeds. Every component receives a separate "
             "full autograd backward through the same forward graph. No optimizer updates, parameter/buffer changes, "
             "`.grad` writes, or training RNG consumption occur during diagnosis.", "",
             "`shared` includes every trainable parameter outside the head, `temporal` is its temporal Transformer subset, "
             "`excess` is head.excess, and `full` is all trainable parameters. Norms are Euclidean. Unused derivatives are zero. "
             "Zero denominators produce undefined ratios. `accum4` measures the norm of the mean gradient of four batches "
             "at a frozen state (not the mean of four norms). This is a complementary local accumulation estimate.", "",
             "Production BF16 autocast/TF32; physical losses FP32; norm reduction FP64. Raw gradients precede Adam, "
             "weight decay and any optimizer preconditioning. A coefficient scales the recorded loss and gradient linearly; "
             "it does not predict the trajectory after retraining. Diagnostics do not use VAL/TEST or select checkpoints.", "",
             "The single-GPU trajectory uses effective batch 1024, production Adam (weight_decay=1e-5), "
             "five-epoch warmup and the first 50 epochs of the 300-epoch cosine schedule. Early stopping is not used.", "",
             "## Unweighted loss components", "",
             "| Station | Epoch | Component | Mean | Median | Min | Max |",
             "|---|---:|---|---:|---:|---:|---:|"]
    for row in summary:
        if row["unit"] == "batch" and row["group"] == "full":
            lines.append("| " + " | ".join([row["station"], str(row["epoch"]), row["component"]] +
                         [fmt(row[f"loss_{s}"]) for s in ("mean", "median", "min", "max")]) + " |")
    lines += ["", "## CBBT candidate and existing dual contributions", "",
              "Batch summaries; gradient ratios here refer to shared parameters. Full/temporal/excess and accumulation "
              "statistics, including median and range, are in candidate_summary.csv.", "",
              "| Epoch | Component | Weight | Weighted loss mean | R/base mean | R/base median | R/base min–max | R/Dual Base mean | R/Dual Base max |",
              "|---:|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in candidate_summary:
        if row["station"] == "CBBT" and row["unit"] == "batch" and row["group"] == "shared":
            lines.append("| " + " | ".join([str(row["epoch"]), row["component"], fmt(row["weight"]),
                         fmt(row["weighted_loss_mean"]), fmt(row["weighted_ratio_base_mean"]),
                         fmt(row["weighted_ratio_base_median"]),
                         fmt(row["weighted_ratio_base_min"]) + "–" + fmt(row["weighted_ratio_base_max"]),
                         fmt(row["weighted_ratio_dual_base_mean"]), fmt(row["weighted_ratio_dual_base_max"])]) + " |")
    lines += ["", "## Files", "",
              "- component_summary.csv: all component loss/norm/ratio/cosine mean, median, minimum and maximum.",
              "- candidate_summary.csv: analytical weights and existing weighted dual contributions for every station/state/group.",
              "- batch_gradients.csv and candidate_batches.csv: individual batches and frozen accumulation groups.",
              "- STATION/metadata.json, fixed_batches.json, states.csv, trajectory.csv: provenance, selected TRAIN windows, "
              "model integrity checks, learning rates and TRAIN-only trajectory progress.",
              "- STATION/checkpoints/: fixed-epoch states, including optimizer/scheduler and RNG for audit; no best-model selection."]
    (output_dir / "REPORT.md").write_text("\n".join(lines) + "\n")


def run_station(options, station):
    from emulator.common import configure_runtime
    from emulator.data import (ForcingGraphView, build_loader, fit_statistics, fit_loss_thresholds,
                               load_station_json, station_features_from_json)
    from emulator.models import ModelConfig, build_model
    from emulator.training import ForecastLoss, LossConfig, run_epoch
    started = time.perf_counter()
    dest = options.output / station
    dest.mkdir(exist_ok=False)
    (dest / "checkpoints").mkdir()
    args, source = production_args(options.repo, station)
    device = torch.device(options.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    configure_runtime(args.seed, args.torch_threads, bool(args.deterministic), bool(args.tf32))
    print(f"{station}: loading TRAIN only", flush=True)
    store, split_files, opened = train_only_store(args)
    indices = list(range(len(store.graphs)))
    stats_cpu = fit_statistics(store, indices, args.x_norm, args.x_p_lo, args.x_p_hi,
                               args.x_nodes_per_graph, args.seed, device=device)
    fitted = fit_loss_thresholds(store, indices, args.tail_frac, args.wmse_q,
                                bool(args.wmse_use_abs), args.exceedance_percentile)
    stats = {key: value.to(device) for key, value in stats_cpu.items()}
    station_feat = station_features_from_json(load_station_json(args.station_json_dir, station),
                                              use_site_elevation=args.use_site_elevation,
                                              use_bathymetry=args.use_bathymetry).to(device)
    model_values = {f.name: getattr(args, f.name) for f in fields(ModelConfig) if hasattr(args, f.name)}
    graph = store.graphs[0]
    model_values.update(in_channels=graph.x.size(-1), out_channels=graph.y.numel(), model="pact",
                        temporal_layers=args.transformer_layers, temporal_ff_mult=args.transformer_ff_mult,
                        temporal_dropout=args.transformer_dropout, history_steps=args.history_hours // 6,
                        station_feat_dim=station_feat.numel(),
                        peak_threshold_norm=((fitted["tau_phys"] - stats_cpu["y_mean"]) / stats_cpu["y_std"]).tolist(),
                        peak_prior=fitted["event_prior"])
    model_config = ModelConfig(**model_values)
    model = build_model(model_config).to(device)
    loss_config = LossConfig(**{f.name: getattr(args, f.name) for f in fields(LossConfig)})
    criterion = ForecastLoss(loss_config, stats, fitted["tail_threshold"], fitted["wmse_threshold"],
                             event_prior=fitted["event_prior"], event_threshold=fitted["tau_phys"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)

    def multiplier(epoch):
        if epoch < args.warmup_epochs:
            return args.warmup_start_factor + (1 - args.warmup_start_factor) * epoch / args.warmup_epochs
        progress = (epoch - args.warmup_epochs) / max(args.epochs - args.warmup_epochs, 1)
        minimum = args.min_lr / args.lr
        return minimum + (1 - minimum) * (1 + math.cos(math.pi * progress)) / 2

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)
    loader_args = dict(batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=args.pin_memory,
                       persistent_workers=args.persistent_workers, prefetch_factor=args.prefetch_factor,
                       mp_context=args.mp_context)
    train_generator = torch.Generator().manual_seed(args.seed)
    train_loader = build_loader(ForcingGraphView(store, indices, args.history_hours // 6), None,
                                shuffle=True, generator=train_generator, **loader_args)
    batch_generator = torch.Generator().manual_seed(options.batch_seed)
    selected = torch.randperm(len(indices), generator=batch_generator)[:options.batches * args.batch_size].tolist()
    if len(selected) != options.batches * args.batch_size:
        raise ValueError("Not enough TRAIN windows for distinct complete diagnostic batches")
    diagnostic_loader = build_loader(ForcingGraphView(store, selected, args.history_hours // 6), None,
                                     generator=torch.Generator().manual_seed(options.batch_seed), **loader_args)
    batches = list(diagnostic_loader)
    batch_manifest = [dict(batch=i, indices=selected[i*args.batch_size:(i+1)*args.batch_size], tags=list(b.tag),
                           events=int((b.y.amax(dim=1).double() > fitted["tau_phys"]).sum()),
                           tails=int((b.y.amax(dim=1) >= fitted["tail_threshold"]).sum())) for i, b in enumerate(batches)]
    named, groups = parameter_groups(model)
    metadata = dict(station=station, training_args=vars(args), source_config=source,
                    model_config=asdict(model_config), loss_config=asdict(loss_config), thresholds=fitted,
                    normalization={k: v.tolist() for k, v in stats_cpu.items()}, file_splits=split_files,
                    opened_graph_files=opened, train_windows=len(indices), train_batches=len(train_loader),
                    single_gpu=True, effective_batch_size=args.batch_size * args.grad_accum_steps,
                    batch_seed=options.batch_seed, diagnostic_batches=options.batches,
                    diagnostic_dropout="training mode, seed 420000 + fixed batch index at every state; RNG restored",
                    parameters={g: dict(count=sum(named[i][1].numel() for i in ids), names=[named[i][0] for i in ids])
                                for g, ids in groups.items()}, initial_model_sha256=state_digest(model),
                    torch=torch.__version__, device=str(device), gpu=torch.cuda.get_device_name(device) if device.type == "cuda" else None)
    write_json(dest / "metadata.json", metadata)
    write_json(dest / "fixed_batches.json", batch_manifest)
    sample_epochs = {e for e in options.sample_epochs if e <= options.stop_epoch} if station == "CBBT" else {options.stop_epoch}
    trajectory, diagnostics, states = [], [], []

    def snapshot(epoch):
        before = state_digest(model)
        cpu_rng = torch.get_rng_state().clone()
        cuda_rng = torch.cuda.get_rng_state(device).clone() if device.type == "cuda" else None
        sampler_rng = train_generator.get_state().clone()
        diagnostic_start = time.perf_counter()
        diagnostics.extend(diagnose(model, batches, stats, fitted, station_feat, device, station, epoch,
                                    use_amp=bool(args.amp) and device.type == "cuda", tail_frac=args.tail_frac))
        assert torch.equal(cpu_rng, torch.get_rng_state())
        assert torch.equal(sampler_rng, train_generator.get_state())
        if cuda_rng is not None:
            assert torch.equal(cuda_rng, torch.cuda.get_rng_state(device))
        after = state_digest(model)
        assert before == after
        states.append(dict(epoch=epoch, model_before_sha256=before, model_after_sha256=after,
                           cpu_cuda_sampler_rng_unchanged=True, diagnostic_seconds=time.perf_counter()-diagnostic_start))
        torch.save(dict(epoch=epoch, model_state=model.state_dict(), model_config=asdict(model_config),
                        optimizer_state=optimizer.state_dict(), scheduler_state=scheduler.state_dict(),
                        normalization=stats_cpu, thresholds=fitted, training_config=vars(args),
                        cpu_rng=cpu_rng, cuda_rng=cuda_rng, train_generator_rng=sampler_rng),
                   dest / "checkpoints" / f"epoch_{epoch:03d}.pt")
        write_json(dest / "batch_gradients.json", diagnostics)
        write_csv(dest / "states.csv", states)
        write_reports(options.output)
        print(f"{station}: epoch {epoch} diagnosis saved ({time.perf_counter()-diagnostic_start:.1f}s)", flush=True)

    print(f"{station}: {len(indices)} TRAIN windows, {len(train_loader)} batches/epoch; q_E={fitted['event_prior']:.8f}", flush=True)
    if 0 in sample_epochs:
        snapshot(0)
    for epoch in range(1, options.stop_epoch + 1):
        epoch_start = time.perf_counter()
        learning_rate = optimizer.param_groups[0]["lr"]
        result = run_epoch(model, train_loader, device, stats, station_feat=station_feat,
                           optimizer=optimizer, criterion=criterion, grad_accum_steps=args.grad_accum_steps,
                           max_grad_norm=0, use_amp=bool(args.amp) and device.type == "cuda",
                           amp_dtype=torch.bfloat16, x_clip=0, augmentation=None,
                           event_threshold=fitted["tau_phys"])
        scheduler.step()
        if not math.isfinite(result.metrics["rmse_all"]):
            raise FloatingPointError("Nonfinite TRAIN trajectory")
        trajectory.append(dict(epoch=epoch, lr_used=learning_rate, next_lr=optimizer.param_groups[0]["lr"],
                               train_rmse=result.metrics["rmse_all"], train_mae=result.metrics["mae_all"],
                               seconds=time.perf_counter()-epoch_start,
                               optimizer_steps=epoch * math.ceil(len(train_loader)/args.grad_accum_steps)))
        write_csv(dest / "trajectory.csv", trajectory)
        if epoch in sample_epochs:
            snapshot(epoch)
    write_json(dest / "complete.json", dict(station=station, stop_epoch=options.stop_epoch,
                                            sampled_epochs=sorted(sample_epochs), seconds=time.perf_counter()-started))
    print(f"{station}: complete in {time.perf_counter()-started:.1f}s", flush=True)
    del model, optimizer, train_loader, batches, store, stats
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stations", nargs="+", choices=("CBBT", "Lewes", "Battery", "Boston"),
                        default=["CBBT", "Lewes", "Battery", "Boston"])
    parser.add_argument("--stop-epoch", type=int, default=50)
    parser.add_argument("--sample-epochs", type=int, nargs="+", default=[0, 5, 20, 50])
    parser.add_argument("--batches", type=int, default=16)
    parser.add_argument("--batch-seed", type=int, default=1042)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--report-only", action="store_true")
    options = parser.parse_args()
    options.repo, options.output = options.repo.resolve(), options.output.resolve()
    if options.batches < 4 or options.batches % 4 or not 0 <= options.stop_epoch <= 50:
        parser.error("Use complete groups of four diagnostic batches, and 0–50 trajectory epochs.")
    sys.path.insert(0, str(options.repo))
    if options.report_only:
        write_reports(options.output)
        return
    options.output.mkdir(parents=True, exist_ok=True)
    provenance = dict(command=sys.argv, script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                      commit=subprocess.check_output(["git", "-C", str(options.repo), "rev-parse", "HEAD"], text=True).strip(),
                      tracked_changes=subprocess.check_output(["git", "-C", str(options.repo), "diff", "HEAD", "--stat"], text=True),
                      options=vars(options), intent="TRAIN scale diagnosis only; no factorial; no validation/test evaluation")
    if provenance["tracked_changes"]:
        raise RuntimeError("Diagnosis requires unchanged tracked main sources")
    branch = subprocess.check_output(["git", "-C", str(options.repo), "branch", "--show-current"], text=True).strip()
    if branch != "main":
        raise RuntimeError("Diagnosis must run on main")
    write_json(options.output / "provenance.json", provenance)
    for station in options.stations:
        run_station(options, station)
    write_reports(options.output)


if __name__ == "__main__":
    main()
