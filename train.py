#!/usr/bin/env python3
"""Train fresh v2 models. Epoch logs contain only eight physical-unit errors."""

from dataclasses import asdict, fields
import json
import math
import os
import hashlib
import re
import shlex
from datetime import datetime
from pathlib import Path
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler

from emulator.common import configure_runtime
from emulator.common.runtime import log_message
from emulator.common.dual import initial_gate_prior
from emulator.data import (ForcingGraphStore, ForcingGraphView, build_loader,
                           fit_loss_thresholds, fit_statistics, load_station_json, station_features_from_json)
from emulator.models import ModelConfig, build_model
from emulator.training import ForecastLoss, LossConfig, format_metrics, run_epoch
from emulator.training.arguments import parse_args


def main(argv=None):
    wall_start = time.perf_counter()
    args = parse_args(argv)
    distributed = int(os.environ.get("WORLD_SIZE", "1")) > 1
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    configure_runtime(args.seed + rank, args.torch_threads, bool(args.deterministic), bool(args.tf32))
    cuda = args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available())
    device = torch.device("cuda", local_rank) if cuda else torch.device("cpu")
    if cuda:
        torch.cuda.set_device(device)
    if distributed:
        dist.init_process_group("nccl" if cuda else "gloo")
    try:
        train(args, device, distributed, rank, wall_start)
    finally:
        if distributed:
            dist.destroy_process_group()


def train(args, device, distributed, rank, wall_start):
    world = dist.get_world_size() if distributed else 1
    launch = [datetime.now().strftime("%Y%m%d_%H%M%S_%f") if rank == 0 else None]
    if distributed:
        dist.broadcast_object_list(launch, src=0)
    args.run_tag = args.run_tag or launch[0]
    output_dir = Path(args.output_dir or Path("All_Results") / f"{launch[0]}_{args.station or 'ALL'}_{args.model}").resolve()
    args.output_dir = str(output_dir)
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", args.run_tag).strip("-")[:40] or "run"
    config_id = hashlib.sha256(json.dumps(vars(args), sort_keys=True, default=str).encode()).hexdigest()[:10]
    stem = f"{args.model}_{args.loss_mode}_{config_id}_{slug}"
    checkpoint_path = output_dir / f"best_{stem}.pth"
    metrics_path = output_dir / f"metrics_{stem}.jsonl"
    summary_path = output_dir / f"summary_{stem}.json"
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        with (output_dir / f"config_{stem}.json").open("x") as handle:
            json.dump(vars(args), handle, indent=2, default=str)
        if not (output_dir / "config_used.sh").exists():
            snapshot = ["#!/usr/bin/env bash", "# Resolved train.py arguments.",
                        f"TRAIN_PY={shlex.quote(str(Path(__file__).resolve()))}"]
            for key, value in vars(args).items():
                value = "" if value is None else int(value) if isinstance(value, bool) else value
                snapshot.append(f"{key.upper()}={shlex.quote(str(value))}")
            (output_dir / "config_used.sh").write_text("\n".join(snapshot) + "\n")
    if distributed:
        dist.barrier()
    store = ForcingGraphStore(args.root_dir, args.station)
    split_config = {name: getattr(args, name) for name in
                    ("train_ratio", "val_ratio", "shuffle_years", "seed", "future_only", "future_year_threshold")}
    splits = store.split(**split_config)
    if not splits["train"] or not splits["val"]:
        raise ValueError("Training needs nonempty train and validation year groups.")
    history_steps = args.history_hours // 6
    train_data = ForcingGraphView(store, splits["train"], history_steps)
    val_data = ForcingGraphView(store, splits["val"], history_steps)
    # Z-score moments are reduced across ranks on the training device, as before.
    stats_cpu = fit_statistics(store, splits["train"], args.x_norm, args.x_p_lo, args.x_p_hi,
                               args.x_nodes_per_graph, args.seed, device=device)
    thresholds = [None]
    if rank == 0:
        thresholds[0] = fit_loss_thresholds(store, splits["train"], args.tail_frac,
                                            args.wmse_q, bool(args.wmse_use_abs), args.exceedance_percentile)
    if distributed:
        dist.broadcast_object_list(thresholds, src=0)
    fitted = thresholds[0]
    dual_metadata = (dict(tau_phys=fitted["tau_phys"], event_prior=fitted["event_prior"],
                          gate_init_prior=initial_gate_prior(fitted["event_prior"])
                          if args.dual_ablation != "fixed_gate" else fitted["event_prior"],
                          event_count=fitted["event_count"], train_windows=fitted["train_windows"],
                          exceedance_percentile=args.exceedance_percentile, dual_ablation=args.dual_ablation,
                          event_definition="max_h(Y_h) > tau_phys", fitted_on="train")
                     if args.head_type == "dual" else None)
    stats = {key: value.to(device) for key, value in stats_cpu.items()}
    station_feat = None
    if args.model == "perceiver3" and args.use_station_meta and args.station:
        station_json = load_station_json(args.station_json_dir, args.station)
        station_feat = station_features_from_json(station_json, use_site_elevation=args.use_site_elevation,
                                                  use_bathymetry=args.use_bathymetry).to(device)
    model_values = {field.name: getattr(args, field.name) for field in fields(ModelConfig) if hasattr(args, field.name)}
    train_graph = store.graphs[splits["train"][0]]
    model_values.update(in_channels=train_graph.x.size(-1), out_channels=train_graph.y.numel(),
                        model="pact" if args.model == "perceiver3" else "baseline",
                        temporal_layers=args.transformer_layers, temporal_ff_mult=args.transformer_ff_mult,
                        temporal_dropout=args.transformer_dropout, history_steps=history_steps,
                        station_feat_dim=station_feat.numel() if station_feat is not None else 0,
                        peak_threshold_norm=((fitted["tau_phys"] - stats_cpu["y_mean"]) / stats_cpu["y_std"]).tolist()
                        if args.head_type == "dual" else None, peak_prior=fitted["event_prior"])
    model_config = ModelConfig(**model_values)
    model = build_model(model_config).to(device)
    if distributed:
        model = DistributedDataParallel(model, device_ids=[device.index] if device.type == "cuda" else None)
    loss_config = LossConfig(**{field.name: getattr(args, field.name) for field in fields(LossConfig)})
    criterion = ForecastLoss(loss_config, stats, fitted["tail_threshold"], fitted["wmse_threshold"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    def lr_multiplier(epoch):
        if epoch < args.warmup_epochs:
            return args.warmup_start_factor + (1 - args.warmup_start_factor) * epoch / args.warmup_epochs
        progress = (epoch - args.warmup_epochs) / max(args.epochs - args.warmup_epochs, 1)
        minimum = args.min_lr / args.lr
        return minimum + (1 - minimum) * (1 + math.cos(math.pi * progress)) / 2

    if args.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_multiplier)
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=args.rop_factor,
            patience=args.rop_patience, threshold=args.rop_threshold, cooldown=args.rop_cooldown, min_lr=args.rop_min_lr)
    use_amp = bool(args.amp) and device.type == "cuda"
    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}[args.amp_dtype]
    scaler = torch.amp.GradScaler("cuda") if use_amp and amp_dtype == torch.float16 else None
    loader_options = dict(batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=args.pin_memory,
                          persistent_workers=args.persistent_workers, prefetch_factor=args.prefetch_factor,
                          mp_context=args.mp_context)
    train_sampler = DistributedSampler(train_data, num_replicas=world, rank=rank, shuffle=True, drop_last=False) if distributed else None
    val_sampler = DistributedSampler(val_data, num_replicas=world, rank=rank, shuffle=False, drop_last=False) if distributed else None
    train_loader = build_loader(train_data, train_sampler, shuffle=True, **loader_options)
    val_loader = build_loader(val_data, val_sampler, **loader_options)
    epoch_options = dict(device=device, stats=stats, station_feat=station_feat, use_amp=use_amp,
                         amp_dtype=amp_dtype, x_clip=args.x_clip, distributed=distributed)
    best_rmse, best_epoch = float("inf"), 0
    start = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        training = run_epoch(model, train_loader, optimizer=optimizer, criterion=criterion, scaler=scaler,
                             grad_accum_steps=args.grad_accum_steps, max_grad_norm=args.max_grad_norm,
                             augmentation=(args.x_aug_prob, args.x_aug_scale, args.x_aug_bias) if args.x_aug else None,
                             **epoch_options)
        validation = run_epoch(model, val_loader, **epoch_options)
        if args.scheduler == "cosine":
            scheduler.step()
        else:
            metric = "rmse_all" if args.rop_metric == "val_rmse_phys" else "rmse_peak5"
            scheduler.step(validation.metrics[metric])
        if rank == 0:
            log_message(f'Epoch {epoch:03d}/{args.epochs} | {format_metrics("Train", training.metrics)} | '
                        f'{format_metrics("Val", validation.metrics)}')
            with metrics_path.open("a") as handle:
                handle.write(json.dumps({"epoch": epoch, "train": training.metrics, "val": validation.metrics}) + "\n")
            if validation.metrics["rmse_all"] < best_rmse:
                best_rmse, best_epoch = validation.metrics["rmse_all"], epoch
                network = model.module if distributed else model
                checkpoint = {"model_config": asdict(model_config), "model_state": network.state_dict(),
                              "normalization": stats_cpu, "station_feat": station_feat.cpu() if station_feat is not None else None,
                              "station": args.station, "split_config": split_config,
                              "split_tags": {key: [store.graph_tags[i] for i in indices] for key, indices in splits.items()},
                              "training_config": vars(args), "loss_thresholds": fitted, "dual_metadata": dual_metadata,
                              "epoch": epoch, "val": validation.metrics}
                torch.save(checkpoint, checkpoint_path)
    if distributed:
        dist.barrier()
    elapsed = time.perf_counter() - start
    if rank == 0:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        network = model.module if distributed else model
        network.load_state_dict(checkpoint["model_state"], strict=True)
        if args.test_root_dir:
            test_store = ForcingGraphStore(args.test_root_dir, args.station)
            test_indices = list(range(len(test_store.graphs)))
        else:
            test_store, test_indices = store, splits["test"]
        test_data = ForcingGraphView(test_store, test_indices, history_steps)
        result = run_epoch(network, build_loader(test_data, None, **loader_options), device, stats,
                           station_feat=station_feat, use_amp=use_amp, amp_dtype=amp_dtype,
                           x_clip=args.x_clip, save_predictions=True)
        np.savez_compressed(output_dir / f"test_preds_{stem}.npz", **result.predictions)
        summary = {"best_epoch": best_epoch, "best_val_rmse": best_rmse, "training_seconds": elapsed,
                   "loss_thresholds": fitted, "dual_metadata": dual_metadata,
                   "test": result.metrics, "test_scope": "external_all_years" if args.test_root_dir else "held_out_years"}
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        wall_seconds = time.perf_counter() - wall_start
        summary["wall_seconds"] = wall_seconds
        summary["run_tag"] = args.run_tag
        summary_path.write_text(json.dumps(summary, indent=2))
        hours, remainder = divmod(wall_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        log_message(f"Wall time: {int(hours):02d}:{int(minutes):02d}:{seconds:06.3f} ({wall_seconds:.3f} s)")
    if distributed:
        dist.barrier()


if __name__ == "__main__":
    main()
