#!/usr/bin/env python3
"""Evaluate Single-head checkpoints with saved TRAIN statistics and thresholds."""

import argparse
import json
import re
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from emulator.common import configure_runtime
from emulator.common.cli import parse_bool_int, temporal_block_name
from emulator.common.runtime import log_message
from emulator.data import (
    ForcingGraphStore,
    ForcingGraphView,
    build_loader,
    load_station_json,
    station_features_from_json,
)
from emulator.inference import classify_past_future, infer_dataset_tag, parse_year_tag
from emulator.models import (
    build_model,
    count_model_parameters,
    format_parameter_counts,
    model_config_from_checkpoint,
)
from emulator.training import format_metrics, run_epoch
from emulator.training.metrics import evaluate_metrics
from emulator.training.reporting import metric_report, threshold_label


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", required=True, help="Path to .pth checkpoint from train.py")
    parser.add_argument(
        "--root_dir",
        required=True,
        help="Train/val root used for year-split test if --test_root_dir is empty",
    )
    parser.add_argument(
        "--test_root_dir",
        type=str,
        default="",
        help="If set: use ALL years from this root (e.g., CMIP6)",
    )
    parser.add_argument(
        "--source_name",
        default="",
        help="Source label for pair provenance; defaults to the source root name.",
    )
    parser.add_argument(
        "--target_name",
        default="",
        help="Target label for pair provenance; defaults to the evaluation root name.",
    )
    parser.add_argument(
        "--pressure_mode",
        choices=("anomaly", "raw", "anomaly_plus_mean"),
        default=None,
        help="Expected pressure representation. Omit to read the checkpoint setting.",
    )
    parser.add_argument("--station", type=str, default=None)
    parser.add_argument("--station_json_dir", type=str, default="./station_json")
    parser.add_argument(
        "--use_site_elevation",
        type=parse_bool_int,
        choices=[0, 1],
        default=None,
        help="Expected PACT site-elevation setting. Omit to read the checkpoint.",
    )
    parser.add_argument(
        "--use_bathymetry",
        type=parse_bool_int,
        choices=[0, 1],
        default=None,
        help="Expected PACT bathymetry setting. Omit to read the checkpoint.",
    )
    parser.add_argument("--model", type=str, default="", choices=["", "baseline", "perceiver3"])
    parser.add_argument(
        "--encoder_type",
        type=str,
        default="",
        choices=["", "GraphSAGE", "CNN"],
        help=("Expected checkpoint encoder. Omit to read the saved checkpoint setting."),
    )
    parser.add_argument(
        "--temporal_block",
        type=temporal_block_name,
        default=None,
        choices=["MLP", "LSTM", "GRU", "Transformer"],
        help=(
            "Expected PACT temporal block. Omit to read the saved checkpoint setting. "
            "'attn' is accepted as an alias."
        ),
    )
    parser.add_argument(
        "--cnn_intermediate_channel",
        type=int,
        default=None,
        help="Expected CNN intermediate width. Omit to read the saved checkpoint setting.",
    )
    parser.add_argument(
        "--history_hours",
        type=int,
        default=-1,
        help="Expected history_hours; -1 reads the checkpoint",
    )
    parser.add_argument("--tf32", action="store_true")
    parser.add_argument("--torch_threads", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--pin_memory", action="store_true")
    parser.add_argument("--persistent_workers", action="store_true")
    parser.add_argument("--prefetch_factor", type=int, default=2)
    parser.add_argument("--mp_context", type=str, default="fork", choices=["fork", "spawn"])
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--amp_dtype", type=str, default="bf16", choices=["bf16", "fp16"])
    parser.add_argument(
        "--gpu_sync_timing",
        action="store_true",
        help="Synchronize CUDA for accurate timing",
    )
    parser.add_argument("--save_npz", action="store_true")
    parser.add_argument(
        "--model_label",
        type=str,
        default="",
        help=(
            "Human-readable model/checkpoint label used in automatic run folder names "
            "(for example, P3_Best). Empty infers it from the checkpoint filename."
        ),
    )
    parser.add_argument(
        "--inference_results_root",
        type=str,
        default="./All_Inference_Results",
        help="Parent directory used when --out_dir is omitted.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="",
        help=(
            "Exact directory for metrics and predictions. When omitted, infer.py creates "
            "All_Inference_Results/<Station>_<ModelLabel>_<Source>_To_<Target>_<timestamp>/outputs/."
        ),
    )
    parser.add_argument(
        "--years",
        type=str,
        default="",
        help=(
            "Optional comma-separated list of year tags to evaluate "
            "(e.g., '1979_1980,1980_1981'). Default: all years."
        ),
    )
    parser.add_argument(
        "--scope",
        choices=("test", "all"),
        default=None,
        help="Explicit scope: test uses saved tags; all uses every sample in the supplied root.",
    )
    parser.add_argument(
        "--strict_years",
        action="store_true",
        help="Require every requested --years group and exactly that evaluated year set.",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args(argv)
    if args.test_root_dir and args.scope == "test":
        parser.error(
            "--test_root_dir selects external data; it cannot be combined with --scope test."
        )
    if args.strict_years and not args.years.strip():
        parser.error("--strict_years requires an explicit --years list.")
    if args.batch_size < 1 or args.torch_threads < 1:
        parser.error("Batch size and thread count must be positive.")
    return args


def resolve_pressure_mode(checkpoint, supplied=None):
    """Recover forcing preprocessing, treating an explicit CLI mode as an assertion."""
    saved = checkpoint["training_config"].get("pressure_mode", "anomaly")
    if saved not in ("anomaly", "raw", "anomaly_plus_mean"):
        raise ValueError(f"Unsupported checkpoint pressure_mode: {saved!r}.")
    if supplied is not None and supplied != saved:
        raise ValueError(
            f"--pressure_mode={supplied} does not match checkpoint pressure_mode={saved}."
        )
    return saved


def _validate_input_dimensions(store, indices, config, normalization, pressure_mode):
    """Fail before inference if saved TRAIN statistics or forcing channels disagree."""
    channels = config.in_channels
    for name in ("x_center", "x_scale"):
        value = normalization.get(name)
        if not isinstance(value, torch.Tensor) or tuple(value.shape) != (channels,):
            shape = tuple(value.shape) if isinstance(value, torch.Tensor) else None
            raise ValueError(
                f"Checkpoint normalization {name} has shape {shape}; expected ({channels},) "
                f"for model_config.in_channels={channels} and pressure_mode={pressure_mode}."
            )
    for index in indices:
        graph = store.graphs[index]
        for name, rank in (("x", 2), ("x_hist", 3)):
            value = getattr(graph, name, None)
            # Historical zero-history anomaly graphs need not store x_hist.
            if name == "x_hist" and value is None:
                continue
            if not isinstance(value, torch.Tensor) or value.ndim != rank or value.size(-1) != channels:
                shape = tuple(value.shape) if isinstance(value, torch.Tensor) else None
                raise ValueError(
                    f"Graph {store.graph_tags[index]} {name} has shape {shape} after "
                    f"pressure_mode={pressure_mode}; expected {rank} dimensions and "
                    f"{channels} channels to match checkpoint model_config.in_channels."
                )


def _validate_checkpoint_options(args, config, training):
    """Treat architecture flags as assertions about the saved checkpoint."""
    expected = {
        "model": "perceiver3" if config.model == "pact" else "baseline",
        "encoder_type": config.encoder_type,
    }
    if config.encoder_type == "CNN":
        expected["cnn_intermediate_channel"] = config.cnn_intermediate_channel
    if config.model == "pact":
        expected.update(
            temporal_block=config.temporal_block,
            use_site_elevation=training["use_site_elevation"],
            use_bathymetry=training["use_bathymetry"],
        )
    for name, saved in expected.items():
        supplied = getattr(args, name)
        if supplied is not None and supplied != "" and supplied != saved:
            raise ValueError(f"--{name}={supplied} does not match checkpoint setting {saved}.")
    if args.history_hours != -1 and args.history_hours != config.history_steps * 6:
        raise ValueError("--history_hours must match the checkpoint history window.")
    return expected["model"]


def _select_evaluation_samples(args, checkpoint, station, root, all_years):
    """Select exact target years without fitting statistics or crossing saved TEST boundaries."""
    if not all_years and not checkpoint["split_tags"]["test"]:
        raise ValueError(
            "Checkpoint has no held-out source TEST split. "
            "Provide --test_root_dir for external transfer evaluation, "
            "or explicitly use --scope all if source-all evaluation is intended."
        )

    requested_years = [year.strip() for year in args.years.split(",") if year.strip()]
    if args.strict_years:
        if not requested_years:
            raise ValueError("--strict_years requires at least one evaluation year group.")
        if len(requested_years) != len(set(requested_years)):
            raise ValueError(
                "Duplicate requested evaluation year groups are not allowed with --strict_years."
            )
        invalid = [
            year
            for year in requested_years
            if not re.fullmatch(r"\d{4}_\d{4}", year) or int(year[5:]) != int(year[:4]) + 1
        ]
        if invalid:
            raise ValueError(f"Invalid requested evaluation year groups: {', '.join(invalid)}")

    # External/source-all evaluations can load just the explicitly requested years.
    # Held-out evaluation still loads the full store to validate all saved TEST tags.
    if all_years and requested_years:
        available_years = set()
        for path in Path(root).glob("*graphs.pt"):
            parts = path.stem.split("_")
            if len(parts) >= 3 and (station is None or parts[2] == station):
                available_years.add("_".join(parts[:2]))
        missing_years = set(requested_years) - available_years
        if args.strict_years and missing_years:
            raise ValueError(
                "Requested evaluation year groups missing from target: "
                + ", ".join(sorted(missing_years))
            )
        if not set(requested_years) & available_years:
            raise ValueError("The requested evaluation set is empty.")

    # Missing station data must never silently select other stations.
    store = ForcingGraphStore(
        root,
        station,
        pressure_mode=resolve_pressure_mode(checkpoint, getattr(args, "pressure_mode", None)),
        year_filter=requested_years if all_years and requested_years else None,
    )
    if all_years:
        indices = list(range(len(store.graphs)))
    else:
        expected_tags = set(checkpoint["split_tags"]["test"])
        missing = expected_tags - set(store.graph_tags)
        if missing:
            raise ValueError(
                f"Saved test samples are missing: {len(missing)}. "
                "Select --test_root_dir for external data."
            )
        indices = [i for i, tag in enumerate(store.graph_tags) if tag in expected_tags]
    if args.strict_years:
        missing_years = set(requested_years) - set(store.year_to_indices)
        if missing_years:
            raise ValueError(
                "Requested evaluation year groups missing from target: "
                + ", ".join(sorted(missing_years))
            )

    requested_years = sorted(set(requested_years))
    if args.years:
        indices = [
            index
            for index in indices
            if "_".join(store.graph_tags[index].split("_")[:2]) in requested_years
        ]
    evaluated_years = sorted({"_".join(store.graph_tags[i].split("_")[:2]) for i in indices})
    if args.strict_years and evaluated_years != requested_years:
        raise ValueError(
            "Evaluated year groups do not match requested years within the selected scope: "
            f"requested={requested_years}, evaluated={evaluated_years}"
        )
    if not indices:
        raise ValueError("The requested evaluation set is empty.")
    return store, indices, requested_years, evaluated_years


def _prepare_output(args, source, target, station_tag, model_name):
    """Resolve the established artifact names and record the effective invocation."""
    label = args.model_label.strip()
    if not label:
        stem = Path(args.ckpt).stem
        prefix, marker = f"{source}_{station_tag}_", f"_{station_tag}_"
        if stem.startswith(prefix):
            label = stem[len(prefix) :]
        elif marker in stem:
            label = stem.split(marker, 1)[1]
        else:
            label = model_name
    label = label or model_name
    args.model_label = label
    if args.out_dir:
        out_dir = Path(args.out_dir).expanduser().resolve()
    else:
        name = re.sub(
            r"[^A-Za-z0-9_.-]+", "_", f"{station_tag}_{label}_{source}_To_{target}"
        ).strip("_")
        out_dir = (
            Path(args.inference_results_root).expanduser().resolve()
            / f"{name}_{datetime.now():%Y%m%d_%H%M%S}"
            / "outputs"
        )
    args.out_dir = str(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(vars(args), indent=2, default=str))


def _evaluate_years(
    args, config, model, device, stats, station_feat, store, indices, *, x_clip, tau_physical
):
    """Evaluate once per year; retain predictions for metrics independently of NPZ export."""
    cuda = device.type == "cuda"
    year_to_indices = defaultdict(list)
    for index in indices:
        year_to_indices["_".join(store.graph_tags[index].split("_")[:2])].append(index)

    results, yearly_predictions, timing_year_seconds = {}, [], []
    for year, year_indices in sorted(year_to_indices.items()):
        dataset = ForcingGraphView(store, year_indices, config.history_steps)
        loader = build_loader(
            dataset,
            None,
            args.batch_size,
            args.num_workers,
            args.pin_memory,
            args.persistent_workers,
            args.prefetch_factor,
            args.mp_context,
        )
        if cuda and args.gpu_sync_timing:
            torch.cuda.synchronize()
        start = time.perf_counter()
        result = run_epoch(
            model,
            loader,
            device,
            stats,
            station_feat=station_feat,
            x_clip=x_clip,
            use_amp=args.amp and cuda,
            amp_dtype={"bf16": torch.bfloat16, "fp16": torch.float16}[args.amp_dtype],
            save_predictions=True,
            tau_physical=tau_physical,
        )
        if cuda and args.gpu_sync_timing:
            torch.cuda.synchronize()
        seconds = time.perf_counter() - start
        results[year] = dict(
            samples=len(dataset),
            rmse=result.metrics["all_rmse"],
            mae=result.metrics["all_mae"],
            seconds=seconds,
            unit="meters",
            **result.metrics,
        )
        log_message(
            f"[Year {year} | physical] samples={len(dataset)} "
            f'rmse={result.metrics["all_rmse"]:.6e} '
            f'mae={result.metrics["all_mae"]:.6e} time={seconds:.2f}s'
        )
        if parse_year_tag(year) != (2014, 2015):
            timing_year_seconds.append(seconds)
        # Reports always use the evaluated predictions; save_npz controls disk I/O only.
        yearly_predictions.append(result.predictions)

    arrays = {
        name: np.concatenate([item[name] for item in yearly_predictions])
        for name in yearly_predictions[0]
    }
    elapsed = sum(results[year]["seconds"] for year in year_to_indices)
    return results, arrays, timing_year_seconds, elapsed


def _summarize_predictions(
    arrays, results, checkpoint, station, external, tau_physical, timing_year_seconds
):
    """Keep source split boundaries for episode metrics and the established past/future groups."""
    saved_split_by_tag = {
        tag: split for split, split_tags in checkpoint["split_tags"].items() for tag in split_tags
    }
    source_station = not external and station == checkpoint["station"]
    arrays["split_ids"] = np.asarray(
        [
            saved_split_by_tag.get(tag, "evaluation") if source_station else "external"
            for tag in arrays["tags"]
        ],
        dtype=str,
    )
    metrics = evaluate_metrics(
        arrays["y_pred"],
        arrays["y_true"],
        tau_physical,
        target_timestamps=arrays["target_timestamps"],
        split_ids=arrays["split_ids"],
    )

    groups = np.array([classify_past_future(parse_year_tag(tag)[0]) for tag in arrays["tags"]])
    for key, mask, years_label in (
        ("_overall", np.ones(len(arrays["tags"]), dtype=bool), None),
        ("_overall_past", groups == "past", "1979-2014"),
        ("_overall_future", groups == "future", "2070-2099"),
    ):
        group_metrics = evaluate_metrics(
            arrays["y_pred"][mask],
            arrays["y_true"][mask],
            tau_physical,
            target_timestamps=arrays["target_timestamps"][mask],
            split_ids=arrays["split_ids"][mask],
        )
        rmse, mae = group_metrics["all_rmse"], group_metrics["all_mae"]
        results[key] = dict(rmse=rmse, mae=mae, unit="meters", **group_metrics)
        if years_label is not None:
            results[key]["years"] = years_label
        log_message(
            f'[{key.removeprefix("_")} {years_label or "ALL"} | physical] '
            f"AllRMSE={rmse} AllMAE={mae} (n_graphs={int(mask.sum())})"
        )

    average = float(np.mean(timing_year_seconds)) if timing_year_seconds else None
    results["_avg_time_per_year_excl_2014_2015"] = dict(
        seconds=average, n_years=len(timing_year_seconds), excluded="2014-2015"
    )
    log_message(
        f"[Timing | excl 2014-2015] avg_time_per_year={average}s "
        f"over {len(timing_year_seconds)} years"
    )
    return metrics


def _write_reports(
    args,
    checkpoint,
    config,
    pair_metadata,
    results,
    metrics,
    arrays,
    *,
    model_parameters,
    evaluation_scope,
    external,
    all_years,
    elapsed,
    wall_start,
):
    """Write physical-unit predictions and source-checkpoint provenance."""
    training = checkpoint["training_config"]
    threshold_metadata = checkpoint["threshold_metadata"]
    tau_physical = threshold_metadata["tau_physical"]
    source = pair_metadata["source_name"]
    target = pair_metadata["target_name"]
    station_tag = pair_metadata["station"]
    model_name = pair_metadata["model"]
    evaluated_years = pair_metadata["evaluated_years"]
    out_dir = Path(args.out_dir)
    label = args.model_label
    test_tag = target
    model_file_tag = "" if config.encoder_type == "GraphSAGE" else f"_{config.encoder_type}"
    if config.model == "pact":
        if config.temporal_block != "Transformer":
            model_file_tag += f"_{config.temporal_block}"
        model_file_tag += "_single"
    report_stem = f"{test_tag}_{station_tag}_{model_name}{model_file_tag}"
    export_metadata = dict(
        tau_physical=tau_physical,
        exceedance_percentile=threshold_metadata["exceedance_percentile"],
        metric_schema=threshold_metadata["metric_schema"],
        threshold_schema=threshold_metadata["threshold_schema"],
        **pair_metadata,
        split="external" if external else ("mixed" if all_years else "test"),
    )
    if args.save_npz:
        np.savez_compressed(out_dir / "predictions.npz", **arrays, **export_metadata)
    wall_seconds = time.perf_counter() - wall_start
    metadata = {
        "metrics": metrics,
        "model_parameters": model_parameters,
        "runtime_seconds": elapsed,
        "wall_seconds": wall_seconds,
        "samples": len(arrays["tags"]),
        "scope": evaluation_scope,
        "method": threshold_label(threshold_metadata),
        "years": evaluated_years,
        "results": results,
        "threshold_metadata": threshold_metadata,
        **threshold_metadata,
        **pair_metadata,
    }
    (out_dir / "metrics.json").write_text(json.dumps(metadata, indent=2))

    split_parameters = None
    if not external:
        split_parameters = dict(
            train_frac=training["train_ratio"],
            val_frac=training["val_ratio"],
            shuffle_years=bool(training["shuffle_years"]),
            future_only=bool(training["future_only"]),
            future_year_threshold=training["future_year_threshold"],
            split_seed=training["seed"],
        )
    report = dict(
        timestamp=datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
        test_tag=test_tag,
        source_tag=source,
        target_tag=target,
        station=station_tag,
        model=model_name,
        model_label=label,
        encoder_type=config.encoder_type,
        model_parameters=model_parameters,
        temporal_block=config.temporal_block,
        head_type="single",
        cnn_intermediate_channel=(
            config.cnn_intermediate_channel if config.encoder_type == "CNN" else None
        ),
        time_encoding="relative_lag" if config.model == "pact" else None,
        zero_history_query_residual=config.model == "pact" and config.history_steps == 0,
        use_site_elevation=training["use_site_elevation"],
        use_bathymetry=training["use_bathymetry"],
        station_feat_dim=config.station_feat_dim,
        history_hours=config.history_steps * 6,
        ckpt=args.ckpt,
        root_dir=args.root_dir,
        test_root_dir=args.test_root_dir,
        evaluation_scope=metadata["scope"],
        split_parameters=split_parameters,
        years_evaluated=metadata["years"],
        inference_args=vars(args).copy(),
        checkpoint_args=training,
        results=results,
        metrics=metrics,
        threshold_metadata=threshold_metadata,
        method=threshold_label(threshold_metadata),
        metric_space="physical",
        metric_note=(
            "RMSE/MAE on denormalized predictions in original y units. "
            "Episode metrics use fixed GT hourly episodes within each evaluation split. "
            "Argmax ties use the earliest timestamp; timing is in hours. "
            "All extreme populations use the saved TRAIN hourly tau_physical."
        ),
        x_clip=training["x_clip"],
    )
    report.update(pair_metadata)
    report.update(
        {
            key: threshold_metadata[key]
            for key in (
                "tau_physical",
                "exceedance_percentile",
                "metric_schema",
                "threshold_schema",
            )
        }
    )
    (out_dir / f"metrics_per_year_{report_stem}.json").write_text(
        json.dumps(report, indent=2, default=str)
    )
    (out_dir / "metrics.md").write_text(metric_report(metrics, threshold_metadata))
    metric_label = "External" if external else ("SourceAll" if all_years else "Test")
    log_message(format_metrics(metric_label, metrics))
    log_message(f"Wall time: {wall_seconds:.3f} s | Outputs: {out_dir}")


def main(argv=None):
    wall_start = time.perf_counter()
    args = parse_args(argv)
    checkpoint = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    config = model_config_from_checkpoint(checkpoint)
    threshold_metadata = checkpoint["threshold_metadata"]
    if (
        threshold_metadata["metric_schema"] != "hourly_q95_v1"
        or threshold_metadata["threshold_schema"] != "train_hourly_q95_v1"
    ):
        raise ValueError("Inference requires the current hourly TRAIN threshold schema.")
    tau_physical = threshold_metadata["tau_physical"]
    log_message(threshold_label(threshold_metadata))
    training = checkpoint["training_config"]
    args.pressure_mode = resolve_pressure_mode(checkpoint, args.pressure_mode)
    model_name = _validate_checkpoint_options(args, config, training)
    log_message(f"Pressure representation: {args.pressure_mode} (source checkpoint)")

    configure_runtime(
        training["seed"], args.torch_threads, bool(training["deterministic"]), args.tf32
    )
    cuda = args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available())
    device = torch.device("cuda" if cuda else "cpu")
    model = build_model(config).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model_parameters = count_model_parameters(model)
    log_message(format_parameter_counts(model_parameters))

    # Cross-dataset evaluation always retains the source TRAIN normalization.
    stats = checkpoint["normalization"]
    station = args.station or checkpoint["station"]
    station_feat = checkpoint["station_feat"]
    if station != checkpoint["station"] and config.station_feat_dim:
        metadata = load_station_json(args.station_json_dir, station)
        station_feat = station_features_from_json(
            metadata,
            use_site_elevation=training["use_site_elevation"],
            use_bathymetry=training["use_bathymetry"],
        )
    if station_feat is not None:
        station_feat = station_feat.to(device)

    external = bool(args.test_root_dir)
    all_years = external or args.scope == "all"
    evaluation_scope = (
        "external_all_years" if external else "source_all_years" if all_years else "held_out_years"
    )
    root = args.test_root_dir or args.root_dir
    store, indices, requested_years, evaluated_years = _select_evaluation_samples(
        args, checkpoint, station, root, all_years
    )
    _validate_input_dimensions(store, indices, config, stats, args.pressure_mode)
    stats = {name: value.to(device) for name, value in stats.items()}
    source = args.source_name or infer_dataset_tag(args.root_dir)
    target = args.target_name or infer_dataset_tag(root)
    station_tag = station or "ALL"
    pair_metadata = dict(
        source_name=source,
        target_name=target,
        source_root=str(Path(args.root_dir).expanduser().resolve()),
        target_root=str(Path(root).expanduser().resolve()),
        checkpoint=str(Path(args.ckpt).expanduser().resolve()),
        requested_years=requested_years,
        evaluated_years=evaluated_years,
        requested_year_count=len(requested_years),
        evaluated_year_count=len(evaluated_years),
        strict_years=args.strict_years,
        station=station_tag,
        model=model_name,
        encoder_type=config.encoder_type,
        temporal_block=config.temporal_block,
        head_type="single",
        history_steps=config.history_steps,
        history_hours=config.history_steps * 6,
        normalization_origin="source_checkpoint_train",
        normalization_type=training.get("x_norm", ""),
        pressure_mode=args.pressure_mode,
        evaluation_threshold_origin="source_checkpoint_train",
        source_tau_physical=tau_physical,
        source_exceedance_percentile=threshold_metadata["exceedance_percentile"],
    )
    _prepare_output(args, source, target, station_tag, model_name)
    results, arrays, timing_year_seconds, elapsed = _evaluate_years(
        args,
        config,
        model,
        device,
        stats,
        station_feat,
        store,
        indices,
        x_clip=training["x_clip"],
        tau_physical=tau_physical,
    )
    metrics = _summarize_predictions(
        arrays, results, checkpoint, station, external, tau_physical, timing_year_seconds
    )

    _write_reports(
        args,
        checkpoint,
        config,
        pair_metadata,
        results,
        metrics,
        arrays,
        model_parameters=model_parameters,
        evaluation_scope=evaluation_scope,
        external=external,
        all_years=all_years,
        elapsed=elapsed,
        wall_start=wall_start,
    )


if __name__ == "__main__":
    main()
