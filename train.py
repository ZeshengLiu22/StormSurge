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
    BASELINE_ARCHITECTURES,
    NEW_BASELINE_MODELS,
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
    """Configure the execution device and run one training experiment.

    Args:
        argv (list[str] | None): Training command-line arguments without the
            program name. None reads the process command line.

    Returns:
        None: Writes checkpoints, metrics, and configuration snapshots to the
        resolved output directory and prints progress to standard output.

    Notes:
        WORLD_SIZE, RANK, and LOCAL_RANK control distributed execution. The
        process group is destroyed on completion or training failure.
    """
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
    """Resolved artifact locations for one training run.

    Attributes:
        directory (Path): Absolute run directory shared by all ranks.
        stem (str): Model, configuration hash, and run-tag filename component.
        checkpoint (Path): Primary selected checkpoint, also saved as best.pt.
        metrics (Path): JSON Lines file containing per-epoch metrics.
        summary (Path): JSON file containing final evaluation and timing.
    """
    directory: Path
    stem: str
    checkpoint: Path
    metrics: Path
    summary: Path


@dataclass
class TrainingData:
    """Datasets and TRAIN-fitted quantities shared by training stages.

    Attributes:
        store (ForcingGraphStore): Loaded graphs in the selected pressure mode.
        splits (dict): TRAIN/VAL/TEST names mapped to store graph indices.
        split_config (dict): Year-split settings saved for reproducibility.
        history_steps (int): Number of six-hour lags, excluding the current step.
        train (ForcingGraphView): TRAIN view with episode-peak supervision.
        val (ForcingGraphView): Validation view with the same history length.
        statistics (dict): CPU tensors for input and target normalization.
        thresholds (dict): TRAIN hourly exceedance threshold and prevalence.
        episode_peaks (dict): TRAIN episode counts and peak prevalence metadata.
    """
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
    """Model, objective, and device-resident inputs for one run.

    Attributes:
        model (torch.nn.Module): Forecast network, optionally wrapped in DDP.
        config (ModelConfig): Architecture specification saved in checkpoints.
        parameters (dict): Model parameter counts for reporting.
        criterion (ForecastLoss): Physical-space training objective.
        station_features (torch.Tensor | None): Station descriptor vector on
            the execution device, or None when metadata is disabled.
        statistics (dict): Normalization tensors on the execution device.
    """
    model: torch.nn.Module
    config: ModelConfig
    parameters: dict
    criterion: ForecastLoss
    station_features: torch.Tensor | None
    statistics: dict


@dataclass
class TrainingLoaders:
    """Batch iterators and execution options reused across epochs.

    Attributes:
        train (DataLoader): Shuffled TRAIN batches.
        val (DataLoader): Deterministically ordered validation batches.
        sampler (DistributedSampler | None): TRAIN sampler whose epoch must
            advance in distributed runs; None for single-process training.
        options (dict): Loader construction settings for final evaluation.
        epoch_options (dict): Shared run_epoch device, normalization, threshold,
            precision, and distributed settings.
        scaler (torch.amp.GradScaler | None): CUDA float16 gradient scaler;
            None for other precision modes.
    """
    train: DataLoader
    val: DataLoader
    sampler: DistributedSampler | None
    options: dict
    epoch_options: dict
    scaler: torch.amp.GradScaler | None


def _prepare_output_paths(args, distributed, rank):
    """Resolve a shared run identity and save the effective configuration.

    Args:
        args (argparse.Namespace): Validated training settings. The run_tag and
            output_dir attributes are resolved and updated in place.
        distributed (bool): Whether an initialized process group is active.
        rank (int): Global process rank; rank zero creates the run artifacts.

    Returns:
        RunPaths: Absolute artifact paths and the configuration-derived stem.

    Notes:
        All ranks must participate in the broadcast and barrier when distributed
        is true. Rank zero creates the directory and configuration snapshots.
    """
    launch = [datetime.now().strftime("%Y%m%d_%H%M%S_%f") if rank == 0 else None]
    if distributed:
        dist.broadcast_object_list(launch, src=0)
    if args.model in NEW_BASELINE_MODELS:
        source = Path(args.root_dir)
        source_tag = source.parent.name if source.name == "graphs" else source.name
        pool_tag = (
            f"adaptive{args.baseline_pool_grid}x{args.baseline_pool_grid}"
            if args.baseline_pooling == "adaptive" else "global_mean"
        )
        default_name = (
            f"{args.model}_{args.station or 'ALL'}_{source_tag}_hist{args.history_hours}h_"
            f"{pool_tag}_p{args.pressure_mode}_seed{args.seed}"
        )
        args.run_tag = args.run_tag or default_name
        directory_name = re.sub(r"[^A-Za-z0-9_.-]+", "-", args.run_tag).strip("-") or "run"
    else:
        args.run_tag = args.run_tag or launch[0]
        directory_name = f"{args.station or 'ALL'}_{args.model}"
    output_dir = Path(
        args.output_dir or Path("All_Results") / f"{launch[0]}_{directory_name}"
    ).resolve()
    if args.model in NEW_BASELINE_MODELS:
        existing_result = [None]
        if rank == 0:
            markers = (
                "config_*.json", "best*.pt", "best*.pth", "threshold_metadata.json",
                "episode_peak_metadata.json", "metrics*", "summary*",
            )
            existing_result[0] = next(
                (str(path) for pattern in markers for path in output_dir.glob(pattern)), None
            )
        if distributed:
            dist.broadcast_object_list(existing_result, src=0)
        if existing_result[0] is not None:
            raise FileExistsError(
                f"Refusing to overwrite existing baseline results: {existing_result[0]}. "
                "Choose a fresh --output_dir."
            )
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
    """Split by year and fit normalization and hourly thresholds on TRAIN.

    Args:
        args (argparse.Namespace): Validated graph, split, normalization, history,
            pressure-mode, and exceedance-percentile settings.
        device (torch.device): Device used for distributed moment reductions.
        distributed (bool): Whether ranks share a process group.
        rank (int): Global rank; rank zero fits and broadcasts the threshold.
        output_dir (Path): Existing directory for threshold and episode metadata.

    Returns:
        TrainingData: Store, split indices, TRAIN/VAL views, CPU normalization
        tensors, hourly threshold, and TRAIN episode metadata. Episode targets
        are attached to the TRAIN view; histories include the current step.

    Raises:
        ValueError: The TRAIN or validation year group is empty.

    Notes:
        Rank zero writes threshold_metadata.json and episode_peak_metadata.json.
        The configured percentile defaults to Q95; validation targets never
        contribute to normalization, threshold, or episode-supervision fitting.
    """
    store = ForcingGraphStore(args.root_dir, args.station, pressure_mode=args.pressure_mode)
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
    # Reduce TRAIN z-score moments across ranks on the training device.
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
    """Construct the forecast network and its physical-space loss.

    Args:
        args (argparse.Namespace): Validated architecture, station-metadata, and
            loss settings from the training parser.
        data (TrainingData): Graphs, history length, and TRAIN-fitted quantities.
        device (torch.device): Destination for model and normalization tensors.
        distributed (bool): Whether to wrap the model in DistributedDataParallel.
        rank (int): Global process rank; only rank zero logs parameter counts.

    Returns:
        ModelComponents: Model, serializable configuration, parameter counts,
        objective, optional station vector, and device-resident statistics.
    """
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
        and (args.model in NEW_BASELINE_MODELS or not field.name.startswith("baseline_"))
    }
    train_graph = store.graphs[splits["train"][0]]
    model_values.update(
        in_channels=train_graph.x.size(-1),
        out_channels=train_graph.y.numel(),
        model="pact" if args.model == "perceiver3" else args.model,
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


def _baseline_architecture_metadata(args, config, parameters):
    """Describe effective shallow architecture settings for saved provenance.

    Args:
        args: Training arguments including pressure mode and source graph path.
        config: Constructed ModelConfig with inferred input and output widths.
        parameters: Parameter counts returned by count_model_parameters.

    Returns:
        dict: Effective architecture metadata for new baselines; empty for legacy
        models, whose existing configuration reporting remains unchanged.
    """
    if config.model not in NEW_BASELINE_MODELS:
        return {}
    spatial, temporal = BASELINE_ARCHITECTURES[config.model]
    return {
        "model": config.model,
        "spatial_encoder": spatial,
        "temporal_module": temporal,
        "baseline_pooling": config.baseline_pooling,
        "baseline_pool_grid": config.baseline_pool_grid if config.baseline_pooling == "adaptive" else None,
        "baseline_stem_channels": config.baseline_stem_channels,
        "hidden_channels": config.hidden_channels,
        "dropout": config.dropout,
        "history_hours": config.history_steps * 6,
        "in_channels": config.in_channels,
        "out_channels": config.out_channels,
        "pressure_mode": args.pressure_mode,
        "parameter_count": parameters["total"],
        "source_root_dir": args.root_dir,
    }


def _build_optimizer_and_scheduler(args, model):
    """Build Adam and the selected learning-rate schedule.

    Args:
        args (argparse.Namespace): Learning rate, schedule choice, total epochs,
            linear-warmup settings, and ReduceLROnPlateau settings.
        model (torch.nn.Module): Network whose parameters will be optimized.

    Returns:
        tuple: Adam optimizer and either a LambdaLR warmup/cosine scheduler or
        a ReduceLROnPlateau scheduler. Adam uses weight decay of 1e-5.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)

    def lr_multiplier(epoch):
        """Compute the warmup/cosine factor relative to the initial rate.

        Args:
            epoch (int): Zero-based scheduler epoch, including the initial step.

        Returns:
            float: Multiplier applied to args.lr; the cosine segment approaches
            args.min_lr at the configured final epoch.
        """
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
    """Build TRAIN/VAL loaders and configure mixed precision.

    Args:
        args (argparse.Namespace): Batch, worker, random-seed, precision, and
            input-clipping settings.
        data (TrainingData): TRAIN/VAL views and fixed TRAIN threshold metadata.
        components (ModelComponents): Device-resident normalization and station
            features passed to run_epoch.
        device (torch.device): Execution device; AMP is enabled only on CUDA.
        distributed (bool): Whether to shard both views across a process group.
        rank (int): Global rank used to select each distributed shard.

    Returns:
        TrainingLoaders: Loaders, TRAIN sampler, reusable execution settings,
        and a CUDA float16 gradient scaler when required.
    """
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
    """Train once and select both checkpoint roles using validation metrics.

    Args:
        args (argparse.Namespace): Epoch, augmentation, accumulation, clipping,
            scheduling, and primary-checkpoint settings.
        data (TrainingData): Split definitions and TRAIN-fitted metadata saved
            with each selected checkpoint.
        components (ModelComponents): Model and loss to optimize in place.
        loaders (TrainingLoaders): Batch iterators and precision settings.
        paths (RunPaths): Destination paths for checkpoints and epoch metrics.
        optimizer (torch.optim.Optimizer): Optimizer updated at gradient-accumulation
            boundaries, including the final partial group.
        scheduler: Learning-rate scheduler stepped once per completed epoch.
        tracker (CheckpointTracker | None): Rank-zero checkpoint selector;
            None on other ranks.
        distributed (bool): Whether the model is wrapped in DDP.
        rank (int): Global rank; only rank zero writes artifacts and logs.

    Returns:
        float: Elapsed training-loop time in seconds, including validation and
        checkpoint I/O. Rank zero appends JSON Lines metrics and saves selected
        states, including aliases for the primary checkpoint role.
    """
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
                """Capture the current model state and its training provenance.

                Returns:
                    dict: Model state, normalization, station vector, split tags, threshold
                    and episode metadata, epoch, validation metrics, and parameter counts.
                    Values are assembled from the enclosing epoch and are not deep-copied;
                    the checkpoint tracker serializes the snapshot when a role improves.
                """
                network = model.module if distributed else model
                snapshot = {
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
                if model_config.model in NEW_BASELINE_MODELS:
                    snapshot["architecture_metadata"] = _baseline_architecture_metadata(
                        args, model_config, model_parameters
                    )
                return snapshot

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
    """Reload selected checkpoints and export VAL and TEST predictions.

    Args:
        args (argparse.Namespace): External-test root, station, pressure mode,
            exceedance percentile, and primary-checkpoint role.
        data (TrainingData): Source splits, validation view, and TRAIN threshold.
        components (ModelComponents): Model whose state is replaced in place.
        loaders (TrainingLoaders): Loader and normalization options for evaluation.
        paths (RunPaths): Existing output directory and primary artifact stem.
        tracker (CheckpointTracker): Saved paths for both checkpoint roles.
        device (torch.device): Device on which checkpoints are loaded.
        distributed (bool): Whether components.model must be unwrapped from DDP.

    Returns:
        tuple[dict, str, bool]: Per-role checkpoint provenance and VAL/TEST
        metrics, test-scope label, and whether TEST mirrors validation because
        no held-out or external test set exists. Predictions are exported as
        NPZ files with physical water levels in meters.

    Notes:
        Called on rank zero only. Each final evaluation uses the complete view
        without distributed sampling. A validation-mirror run writes an explicit
        warning file and retains an empty TEST split in checkpoint provenance.
    """
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
        test_store = ForcingGraphStore(args.test_root_dir, args.station, pressure_mode=args.pressure_mode)
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
    """Write final checkpoint comparisons and the primary-run summary.

    Args:
        args (argparse.Namespace): Primary checkpoint role and resolved run tag.
        data (TrainingData): Fixed TRAIN threshold and episode metadata.
        components (ModelComponents): Parameter counts included in the summary.
        paths (RunPaths): Existing run directory and summary destination.
        tracker (CheckpointTracker): Checkpoint selection history and provenance.
        evaluations (dict): Per-role VAL/TEST results from final re-evaluation.
        test_scope (str): External, held-out, or validation-mirror scope label.
        test_is_val_mirror (bool): Whether the reported TEST uses validation data.
        elapsed (float): Training-loop duration in seconds.
        device (torch.device): Device synchronized before recording wall time.
        wall_start (float): Initial time.perf_counter reading for the whole run.

    Returns:
        None: Writes Markdown/JSON comparisons and a JSON summary, and logs
        selection metrics and elapsed wall time. Called on rank zero only.
    """
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
    if components.config.model in NEW_BASELINE_MODELS:
        summary.update(
            model_config=asdict(components.config),
            architecture_metadata=_baseline_architecture_metadata(args, components.config, model_parameters),
        )
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
    """Execute data preparation, training, checkpoint selection, and reporting.

    Args:
        args (argparse.Namespace): Validated training options; run_tag and
            output_dir are resolved in place during setup.
        device (torch.device): Execution device configured by main.
        distributed (bool): Whether an initialized process group is active.
        rank (int): Global rank; rank zero writes artifacts and final reports.
        wall_start (float): Initial time.perf_counter reading for wall-time logs.

    Returns:
        None: Trains the model and writes configuration, selected checkpoints,
        epoch metrics, final predictions, and summary artifacts.

    Notes:
        All distributed ranks must call this function. Final re-evaluation runs
        only on rank zero, and all ranks synchronize before returning.
    """
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
