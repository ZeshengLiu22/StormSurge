#!/usr/bin/env python3
"""Evaluate fresh v2 checkpoints through the established inference CLI."""

import argparse
from collections import defaultdict
from datetime import datetime
import json
from pathlib import Path
import re
import time

import numpy as np
import torch

from emulator.common import configure_runtime
from emulator.common.cli import head_type_name, parse_bool_int, temporal_block_name
from emulator.common.runtime import log_message
from emulator.data import ForcingGraphStore, ForcingGraphView, build_loader, load_station_json, station_features_from_json
from emulator.models import ModelConfig, build_model
from emulator.inference import classify_past_future, infer_dataset_tag, parse_year_tag
from emulator.inference.dual_diagnostics import summarize_dual
from emulator.training import format_metrics, run_epoch
from emulator.training.metrics import summarize_windows


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", required=True, help="Path to .pth checkpoint from train.py")
    parser.add_argument(
        "--root_dir",
        required=True,
        help="Train/val root used for year-split test if --test_root_dir is empty",
    )
    parser.add_argument("--test_root_dir", type=str, default="", help="If set: use ALL years from this root (e.g., CMIP6)")
    parser.add_argument("--station", type=str, default=None)
    parser.add_argument("--station_json_dir", type=str, default="./station_json")
    parser.add_argument(
        "--use_site_elevation", type=parse_bool_int, choices=[0, 1], default=None,
        help="Expected PACT site-elevation setting. Omit to read the checkpoint.",
    )
    parser.add_argument(
        "--use_bathymetry", type=parse_bool_int, choices=[0, 1], default=None,
        help="Expected PACT bathymetry setting. Omit to read the checkpoint.",
    )
    parser.add_argument(
        "--strict_station_test",
        action="store_true",
        help="Require the requested station (always enforced in v2).",
    )
    parser.add_argument("--model", type=str, default="", choices=["", "baseline", "perceiver3"])
    parser.add_argument(
        "--encoder_type",
        type=str,
        default="",
        choices=["", "GraphSAGE", "CNN"],
        help=(
            "Expected checkpoint encoder. Omit to read the saved checkpoint setting."
        ),
    )
    parser.add_argument(
        "--temporal_block",
        type=temporal_block_name,
        default=None,
        choices=["MLP", "LSTM", "GRU", "Transformer"],
        help=(
            "Expected PACT temporal block. Omit to read the saved checkpoint setting. 'attn' is accepted as an alias."
        ),
    )
    parser.add_argument(
        "--head_type",
        type=head_type_name,
        default=None,
        choices=["single", "dual"],
        help=(
            "Expected PACT prediction head. Omit to read the saved checkpoint setting."
        ),
    )
    parser.add_argument(
        "--cnn_intermediate_channel", type=int, default=None,
        help="Expected CNN intermediate width. Omit to read the saved checkpoint setting.",
    )
    parser.add_argument("--history_hours", type=int, default=-1, help="Expected history_hours; -1 reads the checkpoint")
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
    parser.add_argument("--gpu_sync_timing", action="store_true", help="Synchronize CUDA for accurate timing")
    parser.add_argument("--save_npz", action="store_true")
    parser.add_argument("--dual_diagnostics", action="store_true",
                        help="Export gate/body/excess/event arrays and calibration metrics for a dual checkpoint.")
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
        help="Optional comma-separated list of year tags to evaluate (e.g., '1979_1980,1980_1981'). Default: all years.",
    )
    parser.add_argument("--scope", choices=("test", "all"), default=None,
                        help="Optional explicit scope; test uses saved tags, all uses every sample in the supplied root.")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args(argv)
    if args.test_root_dir and args.scope == "test":
        parser.error("--test_root_dir selects external data; it cannot be combined with --scope test.")
    if args.batch_size < 1 or args.torch_threads < 1:
        parser.error("Batch size and thread count must be positive.")
    return args


def main(argv=None):
    wall_start = time.perf_counter()
    args = parse_args(argv)
    checkpoint = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    config = ModelConfig(**checkpoint["model_config"])
    dual_metadata = checkpoint.get("dual_metadata")
    if args.dual_diagnostics and (config.head_type != "dual" or dual_metadata is None):
        raise ValueError("--dual_diagnostics requires a dual checkpoint with TRAIN event metadata.")
    training = checkpoint["training_config"]
    expected = {"model": "perceiver3" if config.model == "pact" else "baseline",
                "encoder_type": config.encoder_type}
    if config.encoder_type == "CNN":
        expected["cnn_intermediate_channel"] = config.cnn_intermediate_channel
    if config.model == "pact":
        expected.update(temporal_block=config.temporal_block, head_type=config.head_type,
                        use_site_elevation=training["use_site_elevation"], use_bathymetry=training["use_bathymetry"])
    for name, saved in expected.items():
        supplied = getattr(args, name)
        if supplied is not None and supplied != "" and supplied != saved:
            raise ValueError(f"--{name}={supplied} does not match checkpoint setting {saved}.")
    if args.history_hours != -1 and args.history_hours != config.history_steps * 6:
        raise ValueError("--history_hours must match the checkpoint history window.")
    configure_runtime(training["seed"], args.torch_threads, bool(training["deterministic"]), args.tf32)
    cuda = args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available())
    device = torch.device("cuda" if cuda else "cpu")
    model = build_model(config).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    stats = {name: value.to(device) for name, value in checkpoint["normalization"].items()}
    station = args.station or checkpoint["station"]
    station_feat = checkpoint["station_feat"]
    if station != checkpoint["station"] and config.station_feat_dim:
        metadata = load_station_json(args.station_json_dir, station)
        station_feat = station_features_from_json(metadata, use_site_elevation=training["use_site_elevation"],
                                                  use_bathymetry=training["use_bathymetry"])
    if station_feat is not None:
        station_feat = station_feat.to(device)
    external = bool(args.test_root_dir) or args.scope == "all"
    root = args.test_root_dir or args.root_dir
    # Station filtering is always strict; missing station data never selects other stations.
    store = ForcingGraphStore(root, station)
    if external:
        indices = list(range(len(store.graphs)))
    else:
        expected_tags = set(checkpoint["split_tags"]["test"])
        missing = expected_tags - set(store.graph_tags)
        if missing:
            raise ValueError(f"Saved test samples are missing: {len(missing)}. Select --test_root_dir for external data.")
        indices = [i for i, tag in enumerate(store.graph_tags) if tag in expected_tags]
    if args.years:
        years = {year.strip() for year in args.years.split(",") if year.strip()}
        indices = [i for i in indices if "_".join(store.graph_tags[i].split("_")[:2]) in years]
    if not indices:
        raise ValueError("The requested evaluation set is empty.")
    source, target = infer_dataset_tag(args.root_dir), infer_dataset_tag(root)
    model_name, station_tag = expected["model"], station or "ALL"
    label = args.model_label.strip()
    if not label:
        stem = Path(args.ckpt).stem
        prefix, marker = f"{source}_{station_tag}_", f"_{station_tag}_"
        label = stem[len(prefix):] if stem.startswith(prefix) else stem.split(marker, 1)[1] if marker in stem else model_name
    label = label or model_name
    args.model_label = label
    if args.out_dir:
        out_dir = Path(args.out_dir).expanduser().resolve()
    else:
        name = re.sub(r"[^A-Za-z0-9_.-]+", "_", f"{station_tag}_{label}_{source}_To_{target}").strip("_")
        out_dir = Path(args.inference_results_root).expanduser().resolve() / f"{name}_{datetime.now():%Y%m%d_%H%M%S}" / "outputs"
    args.out_dir = str(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(vars(args), indent=2, default=str))
    year_to_indices = defaultdict(list)
    for index in indices:
        year_to_indices["_".join(store.graph_tags[index].split("_")[:2])].append(index)
    results, yearly_predictions, timing_year_seconds = {}, [], []
    for year, year_indices in sorted(year_to_indices.items()):
        dataset = ForcingGraphView(store, year_indices, config.history_steps)
        loader = build_loader(dataset, None, args.batch_size, args.num_workers, args.pin_memory,
                              args.persistent_workers, args.prefetch_factor, args.mp_context)
        if cuda and args.gpu_sync_timing:
            torch.cuda.synchronize()
        start = time.perf_counter()
        result = run_epoch(model, loader, device, stats, station_feat=station_feat, x_clip=training["x_clip"],
                           use_amp=args.amp and cuda,
                           amp_dtype={"bf16": torch.bfloat16, "fp16": torch.float16}[args.amp_dtype],
                           save_predictions=True, save_dual_diagnostics=args.dual_diagnostics)
        if cuda and args.gpu_sync_timing:
            torch.cuda.synchronize()
        seconds = time.perf_counter() - start
        results[year] = dict(samples=len(dataset), rmse=result.metrics["rmse_all"],
                             mae=result.metrics["mae_all"], seconds=seconds, unit="physical")
        log_message(f'[Year {year} | physical] samples={len(dataset)} '
                    f'rmse={result.metrics["rmse_all"]:.6e} mae={result.metrics["mae_all"]:.6e} time={seconds:.2f}s')
        if parse_year_tag(year) != (2014, 2015):
            timing_year_seconds.append(seconds)
        # Reports always use the evaluated predictions; save_npz controls disk I/O only.
        yearly_predictions.append(result.predictions)
    arrays = {name: np.concatenate([item[name] for item in yearly_predictions])
              for name in yearly_predictions[0]}
    elapsed = sum(results[year]["seconds"] for year in year_to_indices)
    error = arrays["y_pred"].astype(np.float64) - arrays["y_true"].astype(np.float64)
    records = np.column_stack((np.arange(len(indices)), arrays["y_true"].max(axis=1),
                               np.mean(error ** 2, axis=1), np.mean(np.abs(error), axis=1)))
    metrics = summarize_windows(records)
    groups = np.array([classify_past_future(parse_year_tag(tag)[0]) for tag in arrays["tags"]])
    for key, mask, years_label in (
        ("_overall", np.ones(len(indices), dtype=bool), None),
        ("_overall_past", groups == "past", "1979-2014"),
        ("_overall_future", groups == "future", "2070-2099"),
    ):
        selected = error[mask]
        rmse = float(np.sqrt(np.mean(selected ** 2))) if selected.size else float("nan")
        mae = float(np.mean(np.abs(selected))) if selected.size else float("nan")
        results[key] = dict(rmse=rmse, mae=mae, unit="physical")
        if years_label is not None:
            results[key]["years"] = years_label
        log_message(f'[{key.removeprefix("_")} {years_label or "ALL"} | physical] '
                    f'rmse={rmse:.6e} mae={mae:.6e} (n_graphs={int(mask.sum())})')
    average = float(np.mean(timing_year_seconds)) if timing_year_seconds else float("nan")
    results["_avg_time_per_year_excl_2014_2015"] = dict(seconds=average, n_years=len(timing_year_seconds), excluded="2014-2015")
    log_message(f"[Timing | excl 2014-2015] avg_time_per_year={average:.2f}s over {len(timing_year_seconds)} years")

    test_tag = target
    model_file_tag = "" if config.encoder_type == "GraphSAGE" else f"_{config.encoder_type}"
    if config.model == "pact":
        if config.temporal_block != "Transformer":
            model_file_tag += f"_{config.temporal_block}"
        if config.head_type != "dual":
            model_file_tag += f"_{config.head_type}"
    report_stem = f"{test_tag}_{station_tag}_{model_name}{model_file_tag}"
    if args.save_npz:
        np.savez_compressed(out_dir / "predictions.npz", **{name: arrays[name] for name in ("y_true", "y_pred", "tags")})
        np.savez(out_dir / f"preds_{report_stem}_ALLYEARS.npz", y_true=arrays["y_true"],
                 y_pred=arrays["y_pred"], tags=arrays["tags"].astype(object))
    if args.dual_diagnostics:
        tau_phys = dual_metadata["tau_phys"]
        event = np.any(arrays["y_true"].astype(np.float64) > tau_phys, axis=1)
        np.savez_compressed(out_dir / "dual_diagnostics.npz", **arrays, event=event,
                            contribution_phys=arrays["gate_probability"][:, None] * arrays["excess_phys"],
                            tau_phys=tau_phys, event_prior=dual_metadata["event_prior"],
                            dual_ablation=config.dual_ablation)
        diagnostic_report = dict(train=dual_metadata,
                                 scope="external_all_years" if external else "held_out_years",
                                 years=sorted(year_to_indices), overall=summarize_dual(arrays, tau_phys),
                                 by_year={year: summarize_dual(item, tau_phys)
                                          for year, item in zip(sorted(year_to_indices), yearly_predictions)},
                                 metric_note="PR average_precision is stepwise; pr_auc_trapezoid is linearly interpolated. "
                                             "Undefined metrics and empty bins are null. Threshold is fixed from TRAIN.")
        (out_dir / "dual_diagnostics.json").write_text(json.dumps(diagnostic_report, indent=2, allow_nan=False))
        log_message(f"Dual diagnostics: {out_dir / 'dual_diagnostics.json'}")
    wall_seconds = time.perf_counter() - wall_start
    metadata = {"metrics": metrics, "runtime_seconds": elapsed, "wall_seconds": wall_seconds,
                "samples": len(indices), "scope": "external_all_years" if external else "held_out_years",
                "years": sorted(year_to_indices), "results": results, "dual_metadata": dual_metadata}
    (out_dir / "metrics.json").write_text(json.dumps(metadata, indent=2))
    report = dict(timestamp=datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
                  test_tag=test_tag, source_tag=source, target_tag=target, station=station_tag,
                  model=model_name, model_label=label, encoder_type=config.encoder_type,
                  temporal_block=config.temporal_block, head_type=config.head_type,
                  cnn_intermediate_channel=config.cnn_intermediate_channel if config.encoder_type == "CNN" else None,
                  time_encoding="relative_lag" if config.model == "pact" else None,
                  zero_history_query_residual=config.model == "pact" and config.history_steps == 0,
                  use_site_elevation=training["use_site_elevation"], use_bathymetry=training["use_bathymetry"],
                  station_feat_dim=config.station_feat_dim, history_hours=config.history_steps * 6,
                  ckpt=args.ckpt, root_dir=args.root_dir, test_root_dir=args.test_root_dir,
                  evaluation_scope=metadata["scope"],
                  split_parameters=dict(train_frac=training["train_ratio"], val_frac=training["val_ratio"],
                                        shuffle_years=bool(training["shuffle_years"]), future_only=bool(training["future_only"]),
                                        future_year_threshold=training["future_year_threshold"], split_seed=training["seed"])
                  if not external else None,
                  years_evaluated=metadata["years"], inference_args=vars(args).copy(), checkpoint_args=training,
                  results=results, metric_space="physical", metric_note="RMSE/MAE on denormalized predictions in original y units.",
                  x_clip=training["x_clip"], dual_metadata=dual_metadata, dual_ablation=config.dual_ablation)
    (out_dir / f"metrics_per_year_{report_stem}.json").write_text(json.dumps(report, indent=2, default=str))
    log_message(format_metrics("External" if external else "Test", metrics))
    log_message(f"Wall time: {wall_seconds:.3f} s | Outputs: {out_dir}")


if __name__ == "__main__":
    main()
