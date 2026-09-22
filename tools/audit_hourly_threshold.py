#!/usr/bin/env python3
"""Data-only TRAIN hourly threshold and fixed-threshold split population audit."""

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from emulator.data import ForcingGraphStore, fit_loss_thresholds, supervised_targets, threshold_population
from emulator.training.metrics import gt_event_episodes
from emulator.training.reporting import threshold_label


def distribution(values, *, histogram=False):
    values = np.asarray(values, dtype=np.float64)
    if not values.size:
        return dict(n=0, min=None, mean=None, median=None, p90=None, p95=None, max=None)
    result = dict(n=int(values.size), min=float(values.min()), mean=float(values.mean()),
                  median=float(np.quantile(values, .5, method="linear")),
                  p90=float(np.quantile(values, .9, method="linear")),
                  p95=float(np.quantile(values, .95, method="linear")), max=float(values.max()))
    if histogram:
        unique, counts = np.unique(values, return_counts=True)
        result["histogram"] = {str(int(value)): int(count) for value, count in zip(unique, counts)}
    return result


def split_audit(labels, timestamps, tau):
    population = threshold_population(labels, timestamps, tau)
    extreme = labels > tau
    event = extreme.any(axis=1)
    flat_order = np.argsort(timestamps, axis=None, kind="stable")
    ordered_times = timestamps.reshape(-1)[flat_order]
    episodes = gt_event_episodes(labels, timestamps, tau)
    assert len(episodes) == population["episode_count"]
    return dict(**population, window_count=int(len(labels)),
                first_target_time=str(np.datetime64(int(ordered_times[0]), "s")),
                last_target_time=str(np.datetime64(int(ordered_times[-1]), "s")),
                target_timestamps_unique=True, target_horizons=int(labels.shape[1]),
                target_interval_hours=1, target_lead_hours=list(range(labels.shape[1])),
                episode_duration_hours=distribution([len(episode) for episode in episodes], histogram=True),
                true_excess_m=distribution(labels[extreme] - tau),
                extreme_hours_per_event_window=distribution(extreme.sum(axis=1)[event], histogram=True),
                event_window_gt_peak_m=distribution(labels[event].max(axis=1)),
                episode_gt_peak_m=distribution([labels.reshape(-1)[episode].max() for episode in episodes]))


def audit_station(root_dir, station, split_config, percentile):
    store = ForcingGraphStore(root_dir, station, targets_only=True)
    splits = store.split(**split_config)
    # Assert nonoverlap across the whole station as additional leakage evidence.
    store.target_timestamps(range(len(store.graphs)))
    train_labels, _ = supervised_targets(store, splits["train"])
    # Exactly the existing Y normalization arithmetic: FP64 sums, FP32 moments.
    labels_t = torch.from_numpy(train_labels)
    mean = labels_t.mean(dim=0).float()
    variance = labels_t.square().mean(dim=0).float() - mean.square()
    std = torch.sqrt(variance + 1e-6)
    fitted = fit_loss_thresholds(store, splits["train"], percentile, stats=dict(y_mean=mean, y_std=std))
    target_nc_max_error = 0.
    centers_checked = 0
    for graph, tag in zip(store.graphs, store.graph_tags):
        if getattr(graph, "nc", None) is None or getattr(graph, "nc_tide", None) is None:
            raise ValueError(f"{tag}: source graph lacks nc/nc_tide needed to verify physical labels.")
        target_nc_max_error = max(target_nc_max_error, float((graph.y - (graph.nc - graph.nc_tide)).abs().max()))
        season_year = tag.split("_", 1)[0]
        start = np.datetime64(f"{season_year}-11-01T00:00:00", "s").astype(np.int64)
        center = np.datetime64(graph.center_time, "s").astype(np.int64)
        if graph.y.numel() != 6 or center != start + int(graph.hour_start) * 3600 or int(graph.hour_start) != int(graph.time_index) * 6:
            raise ValueError(f"{tag}: saved metadata contradicts six-hour disjoint blocks from Nov 1 00:00.")
        centers_checked += 1
    if target_nc_max_error != 0:
        raise ValueError(f"Source graph physical y != nc - nc_tide; maximum discrepancy {target_nc_max_error}.")
    audits = {}
    for split, indices in splits.items():
        labels, times = supervised_targets(store, indices)
        selected = set(indices)
        audit = split_audit(labels, times, fitted["tau_physical"])
        audit["year_groups"] = [year for year, indices_for_year in store.year_to_indices.items() if selected.intersection(indices_for_year)]
        audits[split] = audit
    return dict(station=station, source_graph_root=str(Path(root_dir).resolve()), threshold=fitted,
                normalization=dict(y_mean=mean.tolist(), y_std=std.tolist()),
                timestamp_proof=dict(graphs_checked=centers_checked, all_station_targets_unique=True,
                                     graph_target_definition="float32(nc) - float32(nc_tide)",
                                     physical_target_recomposition_max_error=target_nc_max_error,
                                     center_definition="season Nov 1 00:00 UTC + 6 * time_index hours",
                                     target_definition="center_time + [0, 1, 2, 3, 4, 5] hours",
                                     disjoint_block_proof="sample i targets indices 6i through 6i+5; next block starts at 6i+6",
                                     stored_forcing_history_steps=sorted({int(g.source_history_steps) for g in store.graphs}),
                                     stored_history_hours=sorted({int(g.max_history_hours) for g in store.graphs}),
                                     input_history_definition="center_time + [-H, -H+6, ..., 0] hours",
                                     threshold_inputs="TRAIN graph.y only; no x, x_hist, VAL, TEST or OOD values"),
                splits=audits)


def population_rows(report):
    """Create standalone CSV rows, retaining each station's TRAIN definition."""
    return [dict(station=station, split=split, tau_physical=audit["threshold"]["tau_physical"],
                 **{key: values[key] for key in ("target_hour_count", "extreme_hour_count", "extreme_hour_rate", "event_window_count", "event_window_rate", "episode_count")},
                 exceedance_percentile=audit["threshold"]["exceedance_percentile"],
                 metric_schema=audit["threshold"]["metric_schema"],
                 threshold_schema=audit["threshold"]["threshold_schema"],
                 quantile_method=audit["threshold"]["quantile_method"],
                 threshold_definition=threshold_label(audit["threshold"]))
            for station, audit in report["stations"].items() for split, values in audit["splits"].items()]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root-dir", required=True)
    parser.add_argument("--stations", nargs="+", default=["CBBT", "Lewes", "Battery", "Boston"])
    parser.add_argument("--output-dir", default="results/hourly_q95_audit")
    parser.add_argument("--exceedance-percentile", type=float, default=95.)
    parser.add_argument("--train-ratio", type=float, default=.6)
    parser.add_argument("--val-ratio", type=float, default=.2)
    parser.add_argument("--shuffle-years", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    split_config = dict(train_ratio=args.train_ratio, val_ratio=args.val_ratio,
                        shuffle_years=args.shuffle_years, seed=args.seed)
    report = dict(metric_schema="hourly_q95_v1", threshold_schema="train_hourly_q95_v1",
                  threshold_formula="np.quantile(unique TRAIN physical hourly graph.y, exceedance_percentile / 100, method='linear')",
                  split_config=split_config, stations={})
    for station in args.stations:
        print(f"Auditing {station} from supervised graph targets ...", flush=True)
        audit = audit_station(args.root_dir, station, split_config, args.exceedance_percentile)
        report["stations"][station] = audit
        (output / f"{station}.json").write_text(json.dumps(audit, indent=2, allow_nan=False) + "\n")
        print(f"  {threshold_label(audit['threshold'])}", flush=True)
        for split, values in audit["splits"].items():
            print(f"  {split.upper():5s} hours={values['target_hour_count']} extreme={values['extreme_hour_rate']:.5%} event_windows={values['event_window_rate']:.5%} episodes={values['episode_count']}", flush=True)
    (output / "audit.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    rows = population_rows(report)
    with (output / "populations.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = [f"# TRAIN hourly target Q{args.exceedance_percentile:g} data audit", "", f"Source: `{Path(args.root_dir).resolve()}`", "",
             "Threshold: `np.quantile(TRAIN graph.y flattened, percentile / 100, method='linear')` in meters.",
             "TRAIN threshold is reused unchanged for VAL and TEST. Strict exceedance: `y > tau`.", "",
             f"Schemas: `{report['metric_schema']}` / `{report['threshold_schema']}`.", "",
             "| Station | Split | Tau (m) | Target hours | Extreme hours (%) | Event windows (%) | Episodes |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(f"| {row['station']} | {row['split']} | {row['tau_physical']:.9f} | {row['target_hour_count']} | {row['extreme_hour_rate'] * 100:.5f} | {row['event_window_rate'] * 100:.5f} | {row['episode_count']} |")
    lines += ["", "All stored graph centers and raw label decompositions were checked. Targets are `[t,t+1,...,t+5]` at one-hour intervals. Centers advance by six hours, with target indices `[6i,...,6i+5]`; no target timestamp repeats within or across splits. Missing seasonal hours break episodes.",
              "", "Stored forcing history has nine slices at `[t-48,t-42,...,t]`. For configured history H the loader keeps `[t-H,t-H+6,...,t]`. No input or history values enter threshold fitting.",
              "", "JSON artifacts include exact split year groups, normalized thresholds, episode duration histograms, extreme hours per event window, and true excess / event-window peak / episode peak distributions. No model training was launched."]
    (output / "REPORT.md").write_text("\n".join(lines) + "\n")
    print(f"Wrote {output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
