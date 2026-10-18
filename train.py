#!/usr/bin/env python3
"""Train one trajectory with the fixed TRAIN hourly threshold and two VAL checkpoint roles."""

import hashlib
import json
import math
import os
import re
import shlex
import time
from dataclasses import asdict, dataclass, fields
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from emulator.common import configure_runtime
from emulator.common.runtime import log_message
from emulator.data import (
    ForcingGraphStore,
    ForcingGraphView,
    build_loader,
    fit_loss_thresholds,
    fit_statistics,
    load_station_json,
    station_features_from_json,
)
from emulator.models import (
    ModelConfig,
    build_model,
    count_model_parameters,
    format_parameter_counts,
)
from emulator.training import ForecastLoss, LossConfig, format_metrics, run_epoch
from emulator.training.arguments import parse_args
from emulator.training.checkpoint_selection import ROLES, CheckpointTracker
from emulator.training.checkpoints import atomic_save
from emulator.training.episode_peaks import build_episode_peak_targets
from emulator.training.reporting import (
    compact_comparison_report,
    comparison_report,
    threshold_label,
    format_loss_components,
)


def main(argv=None):
    wall_start = time.perf_counter()
    args = parse_args(argv)
    distributed = int(os.environ.get("WORLD_SIZE", "1")) > 1
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    configure_runtime(
        args.seed + rank, args.torch_threads, bool(args.deterministic), bool(args.tf32)
    )
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


@dataclass
class RunPaths:
    directory: Path
    stem: str
    checkpoint: Path
    metrics: Path
    summary: Path


@dataclass
class TrainingData:
    store: ForcingGraphStore
    splits: dict
    split_config: dict
    history_steps: int
    train: ForcingGraphView
    val: ForcingGraphView
    statistics: dict
    thresholds: dict
    episode_peaks: dict


@dataclass
class ModelComponents:
    model: torch.nn.Module
    config: ModelConfig
    parameters: dict
    criterion: ForecastLoss
    station_features: torch.Tensor | None
    statistics: dict


@dataclass
class TrainingLoaders:
    train: DataLoader
    val: DataLoader
    sampler: DistributedSampler | None
    options: dict
    epoch_options: dict
    scaler: torch.amp.GradScaler | None


def _prepare_output_paths(args, distributed, rank):
    """Resolve a shared run identity before any rank starts loading data."""
    launch = [datetime.now().strftime("%Y%m%d_%H%M%S_%f") if rank == 0 else None]
    if distributed:
        dist.broadcast_object_list(launch, src=0)
    args.run_tag = args.run_tag or launch[0]
    output_dir = Path(
        args.output_dir
        or Path("All_Results") / f"{launch[0]}_{args.station or 'ALL'}_{args.model}"
    ).resolve()
    args.output_dir = str(output_dir)
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", args.run_tag).strip("-")[:40] or "run"
    config_id = hashlib.sha256(
        json.dumps(vars(args), sort_keys=True, default=str).encode()
    ).hexdigest()[:10]
    stem = f"{args.model}_mse_{config_id}_{slug}"
    checkpoint_path = output_dir / f"best_{stem}.pth"
    metrics_path = output_dir / f"metrics_{stem}.jsonl"
    summary_path = output_dir / f"summary_{stem}.json"
    if rank == 0:
        log_message(
            f"Checkpoint roles (VAL-only): {', '.join(ROLES)}; primary={args.checkpoint_selection}"
        )
        log_message("Global prediction loss: physical MSE")
        log_message(f"Episode GT-aligned peak MSE weight: {args.episode_gt_aligned_peak_weight:g}")
        log_message(
            f"Strict TRAIN-Q{args.exceedance_percentile:g} Tail MSE weight: "
            f"{args.exceedance_loss_weight:g}"
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        with (output_dir / f"config_{stem}.json").open("x") as handle:
            json.dump(vars(args), handle, indent=2, default=str)
        if not (output_dir / "config_used.sh").exists():
            snapshot = [
                "#!/usr/bin/env bash",
                "# Resolved train.py arguments.",
                f"TRAIN_PY={shlex.quote(str(Path(__file__).resolve()))}",
            ]
            for key, value in vars(args).items():
                value = "" if value is None else int(value) if isinstance(value, bool) else value
                snapshot.append(f"{key.upper()}={shlex.quote(str(value))}")
            (output_dir / "config_used.sh").write_text("\n".join(snapshot) + "\n")
    if distributed:
        dist.barrier()
    return RunPaths(output_dir, stem, checkpoint_path, metrics_path, summary_path)


def _prepare_datasets(args, device, distributed, rank, output_dir):
    """Split by year and fit normalization, Q95 and episode targets on TRAIN only."""
    store = ForcingGraphStore(args.root_dir, args.station)
    split_config = {
        name: getattr(args, name)
        for name in (
            "train_ratio",
            "val_ratio",
            "shuffle_years",
            "seed",
            "future_only",
            "future_year_threshold",
        )
    }
    splits = store.split(**split_config)
    if not splits["train"] or not splits["val"]:
        raise ValueError("Training needs nonempty train and validation year groups.")
    history_steps = args.history_hours // 6
    train_data = ForcingGraphView(store, splits["train"], history_steps)
    val_data = ForcingGraphView(store, splits["val"], history_steps)
    # Z-score moments are reduced across ranks on the training device, as before.
    stats_cpu = fit_statistics(
        store,
        splits["train"],
        args.x_norm,
        args.x_p_lo,
        args.x_p_hi,
        args.x_nodes_per_graph,
        args.seed,
        device=device,
    )
    thresholds = [None]
    if rank == 0:
        thresholds[0] = fit_loss_thresholds(
            store, splits["train"], exceedance_percentile=args.exceedance_percentile
        )
    if distributed:
        dist.broadcast_object_list(thresholds, src=0)
    fitted = thresholds[0]
    train_truth = np.stack(
        [store.graphs[i].y.detach().cpu().numpy().reshape(-1) for i in splits["train"]]
    )
    train_data.episode_peak_targets = build_episode_peak_targets(
        train_truth, train_data.target_timestamps, fitted["tau_physical"]
    )
    episode_peak_metadata = train_data.episode_peak_targets.metadata
    if rank == 0:
        (output_dir / "episode_peak_metadata.json").write_text(
            json.dumps(episode_peak_metadata, indent=2) + "\n"
        )
        log_message(
            f'TRAIN episodes={episode_peak_metadata["episode_count"]} '
            f'canonical peaks={episode_peak_metadata["canonical_peak_target_count"]} '
            f'p_episode_peak={episode_peak_metadata["p_episode_peak"]:.9g}; unique target timestamps'
        )
        (output_dir / "threshold_metadata.json").write_text(json.dumps(fitted, indent=2) + "\n")
        log_message(threshold_label(fitted))
    return TrainingData(
        store,
        splits,
        split_config,
        history_steps,
        train_data,
        val_data,
        stats_cpu,
        fitted,
        episode_peak_metadata,
    )


def _build_model_and_objective(args, data, device, distributed, rank):
    store = data.store
    splits = data.splits
    history_steps = data.history_steps
    stats_cpu, fitted = data.statistics, data.thresholds
    episode_peak_metadata = data.episode_peaks
    stats = {key: value.to(device) for key, value in stats_cpu.items()}
    station_feat = None
    if args.model == "perceiver3" and args.use_station_meta and args.station:
        station_json = load_station_json(args.station_json_dir, args.station)
        station_feat = station_features_from_json(
            station_json,
            use_site_elevation=args.use_site_elevation,
            use_bathymetry=args.use_bathymetry,
        ).to(device)
    model_values = {
        field.name: getattr(args, field.name)
        for field in fields(ModelConfig)
        if hasattr(args, field.name)
    }
    train_graph = store.graphs[splits["train"][0]]
    model_values.update(
        in_channels=train_graph.x.size(-1),
        out_channels=train_graph.y.numel(),
        model="pact" if args.model == "perceiver3" else "baseline",
        temporal_layers=args.transformer_layers,
        temporal_ff_mult=args.transformer_ff_mult,
        temporal_dropout=args.transformer_dropout,
        history_steps=history_steps,
        station_feat_dim=station_feat.numel() if station_feat is not None else 0,
    )
    model_config = ModelConfig(**model_values)
    model = build_model(model_config).to(device)
    model_parameters = count_model_parameters(model)
    if rank == 0:
        log_message(format_parameter_counts(model_parameters))
    if distributed:
        model = DistributedDataParallel(
            model, device_ids=[device.index] if device.type == "cuda" else None
        )
    loss_config = LossConfig(
        **{field.name: getattr(args, field.name) for field in fields(LossConfig)}
    )
    criterion = ForecastLoss(
        loss_config,
        stats,
        fitted["tau_physical"],
        extreme_hour_prior=fitted["train_extreme_hour_rate"],
        p_episode_peak=episode_peak_metadata["p_episode_peak"],
    ).to(device)
    return ModelComponents(model, model_config, model_parameters, criterion, station_feat, stats)


def _build_optimizer_and_scheduler(args, model):
    """Preserve the existing Adam and warmup/cosine or plateau schedule."""
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)

    def lr_multiplier(epoch):
        if epoch < args.warmup_epochs:
            return (
                args.warmup_start_factor
                + (1 - args.warmup_start_factor) * epoch / args.warmup_epochs
            )
        progress = (epoch - args.warmup_epochs) / max(args.epochs - args.warmup_epochs, 1)
        minimum = args.min_lr / args.lr
        return minimum + (1 - minimum) * (1 + math.cos(math.pi * progress)) / 2

    if args.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_multiplier)
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            factor=args.rop_factor,
            patience=args.rop_patience,
            threshold=args.rop_threshold,
            cooldown=args.rop_cooldown,
            min_lr=args.rop_min_lr,
        )
    return optimizer, scheduler


def _prepare_loaders(args, data, components, device, distributed, rank):
    world = dist.get_world_size() if distributed else 1
    train_data, val_data = data.train, data.val
    fitted = data.thresholds
    station_feat = components.station_features
    stats = components.statistics
    use_amp = bool(args.amp) and device.type == "cuda"
    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}[args.amp_dtype]
    scaler = torch.amp.GradScaler("cuda") if use_amp and amp_dtype == torch.float16 else None
    loader_options = dict(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        persistent_workers=args.persistent_workers,
        prefetch_factor=args.prefetch_factor,
        mp_context=args.mp_context,
    )
    train_sampler = (
        DistributedSampler(
            train_data, num_replicas=world, rank=rank, shuffle=True, drop_last=False
        )
        if distributed
        else None
    )
    val_sampler = (
        DistributedSampler(val_data, num_replicas=world, rank=rank, shuffle=False, drop_last=False)
        if distributed
        else None
    )
    # Isolate single-process data order from initialization/dropout RNG draws.
    # Seed once; successive epochs advance this stream. DDP keeps its sampler RNG.
    train_generator = torch.Generator().manual_seed(args.seed) if not distributed else None
    train_loader = build_loader(
        train_data, train_sampler, shuffle=True, generator=train_generator, **loader_options
    )
    val_loader = build_loader(val_data, val_sampler, **loader_options)
    epoch_options = dict(
        device=device,
        stats=stats,
        station_feat=station_feat,
        use_amp=use_amp,
        amp_dtype=amp_dtype,
        x_clip=args.x_clip,
        distributed=distributed,
        tau_physical=fitted["tau_physical"],
    )
    return TrainingLoaders(
        train_loader, val_loader, train_sampler, loader_options, epoch_options, scaler
    )


def _train_epochs(
    args, data, components, loaders, paths, optimizer, scheduler, tracker, distributed, rank
):
    """Train once and select both checkpoint roles using validation metrics only."""
    store = data.store
    splits = data.splits
    split_config = data.split_config
    stats_cpu, fitted = data.statistics, data.thresholds
    episode_peak_metadata = data.episode_peaks
    model = components.model
    model_config = components.config
    model_parameters = components.parameters
    criterion = components.criterion
    station_feat = components.station_features
    train_loader, val_loader = loaders.train, loaders.val
    train_sampler = loaders.sampler
    epoch_options = loaders.epoch_options
    scaler = loaders.scaler
    output_dir = paths.directory
    checkpoint_path, metrics_path = paths.checkpoint, paths.metrics
    start = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        training = run_epoch(
            model,
            train_loader,
            optimizer=optimizer,
            criterion=criterion,
            scaler=scaler,
            grad_accum_steps=args.grad_accum_steps,
            max_grad_norm=args.max_grad_norm,
            augmentation=(
                (args.x_aug_prob, args.x_aug_scale, args.x_aug_bias) if args.x_aug else None
            ),
            **epoch_options,
        )
        validation = run_epoch(model, val_loader, **epoch_options)
        if args.scheduler == "cosine":
            scheduler.step()
        else:
            metric = "all_rmse" if args.rop_metric == "val_all_rmse" else "exceedance_rmse"
            scheduler.step(validation.metrics[metric])
        if rank == 0:
            log_message(
                f'Epoch {epoch:03d}/{args.epochs} | {format_metrics("Train", training.metrics)} | '
                f'{format_metrics("Val", validation.metrics)}'
            )
            epoch_record = {"epoch": epoch, "train": training.metrics, "val": validation.metrics}
            if training.losses is not None:
                epoch_record["loss"] = training.losses
                log_message(format_loss_components(training.losses))

            def checkpoint_snapshot():
                network = model.module if distributed else model
                return {
                    "model_config": asdict(model_config),
                    "model_state": network.state_dict(),
                    "normalization": stats_cpu,
                    "station_feat": station_feat.cpu() if station_feat is not None else None,
                    "station": args.station,
                    "split_config": split_config,
                    "split_tags": {
                        key: [store.graph_tags[i] for i in indices]
                        for key, indices in splits.items()
                    },
                    "training_config": vars(args),
                    "threshold_metadata": fitted,
                    "episode_peak_metadata": episode_peak_metadata,
                    **fitted,
                    "epoch": epoch,
                    "val": validation.metrics,
                    "model_parameters": model_parameters,
                }

            scores, improved = tracker.observe(epoch, validation.metrics, checkpoint_snapshot)
            epoch_record["checkpoint_scores"] = scores
            for role in improved:
                log_message(
                    f'[Best {role}] epoch={epoch} Val AllRMSE={validation.metrics["all_rmse"]:.9f} '
                    f'ExceedanceRMSE={validation.metrics["exceedance_rmse"]:.9f}'
                )
            if args.checkpoint_selection in improved:
                chosen = torch.load(
                    tracker.paths[args.checkpoint_selection],
                    map_location="cpu",
                    weights_only=False,
                )
                atomic_save(chosen, checkpoint_path)
                atomic_save(chosen, output_dir / "best.pt")
            with metrics_path.open("a") as handle:
                handle.write(json.dumps(epoch_record) + "\n")
    if distributed:
        dist.barrier()
    elapsed = time.perf_counter() - start
    return elapsed


def _evaluate_selected_checkpoints(
    args, data, components, loaders, paths, tracker, device, distributed
):
    """Reload each selected state strictly and export fresh physical predictions."""
    store = data.store
    splits = data.splits
    history_steps = data.history_steps
    val_data = data.val
    fitted = data.thresholds
    model = components.model
    loader_options, epoch_options = loaders.options, loaders.epoch_options
    output_dir = paths.directory
    stem = paths.stem
    network = model.module if distributed else model
    final_options = dict(epoch_options, distributed=False)
    if args.test_root_dir:
        test_store = ForcingGraphStore(args.test_root_dir, args.station)
        test_indices = list(range(len(test_store.graphs)))
        test_scope, test_is_val_mirror = "external_all_years", False
    elif splits["test"]:
        test_store, test_indices = store, splits["test"]
        test_scope, test_is_val_mirror = "held_out_years", False
    else:
        # Final reporting only: preserve the canonical empty TEST in checkpoints.
        test_store, test_indices = store, splits["val"]
        test_scope, test_is_val_mirror = "val_mirror_no_heldout_test", True
    test_data = ForcingGraphView(test_store, test_indices, history_steps)
    evaluations = {}
    if test_is_val_mirror:
        warning = "\n".join(
            (
                "=" * 72,
                "WARNING: THIS RUN HAS NO HELD-OUT TEST SPLIT.",
                'FINAL "TEST" EVALUATION IS A MIRROR OF VALIDATION DATA ONLY.',
                "THESE TEST METRICS MUST NOT BE REPORTED AS HELD-OUT PERFORMANCE.",
                "USE EXTERNAL INFERENCE FOR REAL EVALUATION.",
                "=" * 72,
            )
        )
        print(warning, flush=True)
        (output_dir / "TEST_IS_VAL_MIRROR_WARNING.txt").write_text(warning + "\n")
    log_message("FINAL MULTI-CHECKPOINT RE-EVALUATION")
    log_message("Water-level errors and biases are displayed in mm; timing is displayed in hours.")
    for role in ROLES:
        checkpoint = torch.load(tracker.paths[role], map_location=device, weights_only=False)
        network.load_state_dict(checkpoint["model_state"], strict=True)
        evaluations[role] = dict(
            epoch=checkpoint["epoch"],
            path=str(tracker.paths[role]),
            selection_split=checkpoint["selection_split"],
            selection_metric_key=checkpoint["selection_metric_key"],
            selection_metric_value=checkpoint["selection_metric_value"],
        )
        for split, dataset in (("val", val_data), ("test", test_data)):
            result = run_epoch(
                network,
                build_loader(dataset, None, **loader_options),
                save_predictions=True,
                **final_options,
            )
            evaluations[role][split] = result.metrics
            exported = dict(
                result.predictions,
                tau_physical=fitted["tau_physical"],
                exceedance_percentile=args.exceedance_percentile,
                metric_schema=fitted["metric_schema"],
                threshold_schema=fitted["threshold_schema"],
                station=args.station or "ALL",
                split=split,
            )
            if split == "test":
                exported.update(test_scope=test_scope, test_is_val_mirror=test_is_val_mirror)
            np.savez_compressed(output_dir / f"{split}_predictions_{role}.npz", **exported)
            if role == args.checkpoint_selection and split == "test":
                np.savez_compressed(output_dir / f"test_preds_{stem}.npz", **exported)
    return evaluations, test_scope, test_is_val_mirror


def _write_run_artifacts(
    args,
    data,
    components,
    paths,
    tracker,
    evaluations,
    test_scope,
    test_is_val_mirror,
    elapsed,
    device,
    wall_start,
):
    fitted = data.thresholds
    episode_peak_metadata = data.episode_peaks
    model_parameters = components.parameters
    output_dir = paths.directory
    primary = evaluations[args.checkpoint_selection]
    best_epoch, best_rmse = primary["epoch"], primary["val"]["all_rmse"]
    comparison = comparison_report(evaluations, fitted)
    (output_dir / "checkpoint_comparison.md").write_text(comparison)
    comparison_metadata = dict(threshold_metadata=fitted, method=threshold_label(fitted))
    (output_dir / "checkpoint_comparison.json").write_text(
        json.dumps({**evaluations, **comparison_metadata}, indent=2, allow_nan=False) + "\n"
    )
    summary = {
        "best_epoch": best_epoch,
        "best_val_rmse": best_rmse,
        "training_seconds": elapsed,
        **comparison_metadata,
        **fitted,
        "episode_peak_metadata": episode_peak_metadata,
        "model_parameters": model_parameters,
        "val": primary["val"],
        "test": primary["test"],
        "checkpoint_selection": tracker.selection_summary(args.checkpoint_selection),
        "checkpoint_comparison": evaluations,
        "test_scope": test_scope,
        "test_is_val_mirror": test_is_val_mirror,
    }
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    wall_seconds = time.perf_counter() - wall_start
    summary["wall_seconds"] = wall_seconds
    summary["run_tag"] = args.run_tag
    paths.summary.write_text(json.dumps(summary, indent=2))
    log_message(f"Best checkpoint: epoch {best_epoch:03d} | selection={args.checkpoint_selection}")
    for line in compact_comparison_report(evaluations).splitlines():
        log_message(line)
    hours, remainder = divmod(wall_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    log_message(
        f"Wall time: {int(hours):02d}:{int(minutes):02d}:{seconds:06.3f} ({wall_seconds:.3f} s)"
    )


def train(args, device, distributed, rank, wall_start):
    paths = _prepare_output_paths(args, distributed, rank)
    tracker = CheckpointTracker(paths.directory) if rank == 0 else None
    data = _prepare_datasets(args, device, distributed, rank, paths.directory)
    components = _build_model_and_objective(args, data, device, distributed, rank)
    optimizer, scheduler = _build_optimizer_and_scheduler(args, components.model)
    loaders = _prepare_loaders(args, data, components, device, distributed, rank)

    elapsed = _train_epochs(
        args,
        data,
        components,
        loaders,
        paths,
        optimizer,
        scheduler,
        tracker,
        distributed,
        rank,
    )
    if rank == 0:
        evaluations, test_scope, test_is_val_mirror = _evaluate_selected_checkpoints(
            args,
            data,
            components,
            loaders,
            paths,
            tracker,
            device,
            distributed,
        )
        _write_run_artifacts(
            args,
            data,
            components,
            paths,
            tracker,
            evaluations,
            test_scope,
            test_is_val_mirror,
            elapsed,
            device,
            wall_start,
        )
    if distributed:
        dist.barrier()


if __name__ == "__main__":
    main()
