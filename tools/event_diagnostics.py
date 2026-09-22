#!/usr/bin/env python3
"""Compare new prediction exports on identical GT hours, windows and episodes."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from emulator.data import validate_target_timestamps
from emulator.training.metrics import _errors, _under, evaluate_metrics, gt_event_episodes


REQUIRED = ("tags", "target_timestamps", "y_true", "y_pred", "tau_physical", "exceedance_percentile",
            "metric_schema", "threshold_schema", "station", "split")
BRANCH_KEYS = ("body_phys", "excess_phys", "gate_probability", "gate_logits")


def scalar(arrays, key):
    value = np.asarray(arrays[key])
    if value.size != 1:
        raise ValueError(f"{key} must be scalar metadata.")
    return value.item()


def validate_export(arrays):
    missing = set(REQUIRED) - set(arrays)
    if missing:
        raise ValueError(f"Missing canonical prediction export fields: {sorted(missing)}")
    for key, expected in (("metric_schema", "hourly_q95_v1"), ("threshold_schema", "train_hourly_q95_v1")):
        if scalar(arrays, key) != expected:
            raise ValueError(f"Unsupported {key}; expected {expected}.")
    tau = float(scalar(arrays, "tau_physical"))
    if not np.isfinite(tau):
        raise ValueError("tau_physical must be the finite source TRAIN threshold.")
    percentile = float(scalar(arrays, "exceedance_percentile"))
    if not 0 < percentile < 100:
        raise ValueError("exceedance_percentile must lie strictly between zero and 100.")
    y, p = np.asarray(arrays["y_true"]), np.asarray(arrays["y_pred"])
    if y.ndim != 2 or y.shape != p.shape or not y.size or not np.isfinite(y).all() or not np.isfinite(p).all():
        raise ValueError("Finite y_true/y_pred must have identical nonempty [N,K] shapes.")
    tags = np.asarray(arrays["tags"])
    if tags.shape != (len(y),) or len(set(tags.tolist())) != len(y):
        raise ValueError("Prediction tags must be unique, one per forecast window.")
    times = validate_target_timestamps(arrays["target_timestamps"])
    if times.shape != y.shape or np.any(np.diff(times, axis=1) != 3600):
        raise ValueError("Target timestamps must have [N,K] shape with consecutive hourly horizons.")
    arrays = dict(arrays, target_timestamps=times, y_true=y.astype(np.float64), y_pred=p.astype(np.float64))
    if "split_ids" in arrays:
        splits = np.asarray(arrays["split_ids"])
        if splits.shape not in ((len(y),), y.shape):
            raise ValueError("split_ids must identify the split per window or per target hour.")
    branch_present = [key in arrays for key in BRANCH_KEYS]
    if any(branch_present) and not all(branch_present):
        raise ValueError("Dual exports require body, raw excess, gate probability and gate logits together.")
    if all(branch_present):
        body, raw = (np.asarray(arrays[key], dtype=float) for key in BRANCH_KEYS[:2])
        gate, logits = (np.asarray(arrays[key], dtype=float).reshape(-1) for key in BRANCH_KEYS[2:])
        if body.shape != y.shape or raw.shape != y.shape or gate.shape != (len(y),) or logits.shape != gate.shape:
            raise ValueError("Dual branch shapes must match horizons and one gate per window.")
        if not np.isfinite(body).all() or not np.isfinite(raw).all() or not np.isfinite(gate).all() or np.isnan(logits).any():
            raise ValueError("Nonfinite dual branch output.")
        if np.any(raw < 0) or np.any((gate < 0) | (gate > 1)) or np.any(body > tau + 2e-6):
            raise ValueError("Dual outputs violate body cap, excess non-negativity or probability bounds.")
        with np.errstate(over="ignore"):
            probability = 1 / (1 + np.exp(-logits))
        if not np.allclose(probability, gate, atol=2e-6, rtol=2e-5):
            raise ValueError("Gate logits disagree with gate probabilities.")
        if not np.allclose(body + gate[:, None] * raw, p, rtol=2e-5, atol=2e-6):
            raise ValueError("Dual reconstruction failed: prediction != body + gate * raw excess.")
    return arrays


def align_exports(exports):
    """Align full populations by exact IDs; reject any omitted/different target."""
    if not exports:
        raise ValueError("At least one named prediction export is required.")
    reference = validate_export(next(iter(exports.values())))
    tags = reference["tags"].tolist()
    aligned = {}
    for name, raw in exports.items():
        arrays = validate_export(raw)
        actual = arrays["tags"].tolist()
        if set(actual) != set(tags):
            raise ValueError(f"{name}: missing or extra prediction IDs; populations cannot be dropped.")
        lookup = {tag: index for index, tag in enumerate(actual)}
        order = [lookup[tag] for tag in tags]
        arrays = {key: value[order] if key in ("tags", "target_timestamps", "y_true", "y_pred", "sample_id", "split_ids", *BRANCH_KEYS, "severity_phys", "excess_shape") else value
                  for key, value in arrays.items()}
        for key in ("target_timestamps", "y_true"):
            if not np.array_equal(arrays[key], reference[key]):
                raise ValueError(f"{name}: {key} differs after exact window-ID alignment.")
        if not np.array_equal(arrays.get("split_ids"), reference.get("split_ids")):
            raise ValueError(f"{name}: split_ids differs after exact window-ID alignment.")
        for key in ("tau_physical", "exceedance_percentile", "metric_schema", "threshold_schema", "station", "split"):
            if scalar(arrays, key) != scalar(reference, key):
                raise ValueError(f"{name}: {key} differs; comparisons require the same source threshold and split.")
        aligned[name] = arrays
    return aligned


def shared_quantile_bins(values, count=4):
    """GT-only linear quantiles; collapse tied edges without splitting GT ties."""
    values = np.asarray(values, dtype=float)
    if count < 1:
        raise ValueError("Quantile bin count must be positive.")
    if not values.size:
        return np.empty(0, dtype=int), []
    edges = np.unique(np.quantile(values, np.linspace(0, 1, count + 1), method="linear"))
    if len(edges) == 1:
        edges = np.repeat(edges, 2)
    return np.searchsorted(edges[1:-1], values, side="right"), edges.tolist()


def utc(value):
    return str(np.datetime64(int(value), "s")) + "Z"


def target_split(arrays, row, lead):
    if "split_ids" not in arrays:
        return str(scalar(arrays, "split"))
    labels = np.asarray(arrays["split_ids"])
    return str(labels[row] if labels.ndim == 1 else labels[row, lead])


def branch_fields(arrays, row, lead, tau):
    if "body_phys" not in arrays:
        return {}
    body, raw = float(arrays["body_phys"][row, lead]), float(arrays["excess_phys"][row, lead])
    probability = float(np.asarray(arrays["gate_probability"]).reshape(-1)[row])
    logit = float(np.asarray(arrays["gate_logits"]).reshape(-1)[row])
    gated = probability * raw
    true_excess = float(arrays["y_true"][row, lead]) - tau
    return dict(body_prediction=body, raw_excess_prediction=raw, gate_logit=logit,
                gate_probability=probability, final_gated_excess=gated,
                reconstructed_final_prediction=body + gated,
                reconstruction_error=body + gated - float(arrays["y_pred"][row, lead]),
                raw_excess_error=raw - true_excess, final_excess_error=gated - true_excess,
                raw_excess_ratio=raw / true_excess, final_excess_ratio=gated / true_excess,
                gate_attenuation=raw - gated, body_deficit=tau - body)


def hour_rows(name, arrays, membership):
    y, p, times = (arrays[key] for key in ("y_true", "y_pred", "target_timestamps"))
    tau = float(scalar(arrays, "tau_physical"))
    rows = []
    for index, (i, h) in enumerate(zip(*np.nonzero(y > tau))):
        rows.append(dict(model=name, timestamp=utc(times[i, h]), station=scalar(arrays, "station"),
                         split=target_split(arrays, i, h), window_id=str(arrays["tags"][i]), lead=int(h),
                         y_true=float(y[i, h]), y_pred=float(p[i, h]), tau=tau,
                         true_excess=float(y[i, h]) - tau, true_excess_bin=int(membership[index]),
                         **branch_fields(arrays, i, h, tau)))
    return rows


def episode_rows(name, arrays, episodes, membership):
    y, p, times = (arrays[key].reshape(-1) for key in ("y_true", "y_pred", "target_timestamps"))
    tau = float(scalar(arrays, "tau_physical"))
    rows = []
    for episode_id, (indices, bin_id) in enumerate(zip(episodes, membership)):
        truth, pred = y[indices], p[indices]
        gt, predicted = int(truth.argmax()), int(pred.argmax())
        error = float(pred[gt] - truth[gt])
        row, lead = divmod(int(indices[0]), arrays["y_true"].shape[1])
        rows.append(dict(model=name, episode_id=episode_id, gt_severity_bin=int(bin_id),
                         station=scalar(arrays, "station"), split=target_split(arrays, row, lead),
                         start_timestamp=utc(times[indices[0]]), end_timestamp=utc(times[indices[-1]]),
                         duration_hours=len(indices), gt_peak=float(truth[gt]),
                         prediction_at_gt_peak=float(pred[gt]), predicted_episode_peak=float(pred[predicted]),
                         gt_peak_timestamp=utc(times[indices[gt]]), predicted_peak_timestamp=utc(times[indices[predicted]]),
                         episode_peak_error=float(pred[predicted] - truth[gt]),
                         episode_gt_aligned_peak_error=error,
                         peak_timing_error_hours=abs(int(times[indices[predicted]]) - int(times[indices[gt]])) / 3600,
                         detected=bool(np.any(pred > tau)),
                         excess_area_error=float(np.maximum(pred - tau, 0).sum() - np.maximum(truth - tau, 0).sum()),
                         **{f"under_{millimeters}mm": bool(pred[gt] < truth[gt] - millimeters / 1000) for millimeters in (0, 5, 10, 20)}))
    return rows


def window_rows(name, arrays, membership):
    y, p = arrays["y_true"], arrays["y_pred"]
    tau = float(scalar(arrays, "tau_physical"))
    rows = []
    for i, bin_id in zip(np.flatnonzero((y > tau).any(axis=1)), membership):
        h = int(y[i].argmax())
        rows.append(dict(model=name, window_id=str(arrays["tags"][i]), gt_peak_bin=int(bin_id),
                         station=scalar(arrays, "station"), split=target_split(arrays, i, h),
                         gt_peak=float(y[i, h]), lead=h, timestamp=utc(arrays["target_timestamps"][i, h]),
                         true_excess=float(y[i, h]) - tau, prediction_at_gt_peak=float(p[i, h]),
                         window_peak_error=float(p[i].max() - y[i, h]),
                         gt_aligned_peak_error=float(p[i, h] - y[i, h]),
                         **branch_fields(arrays, i, h, tau)))
    return rows


def summarize_branch(rows):
    if not rows or "raw_excess_error" not in rows[0]:
        return {}
    values = lambda key: np.asarray([row[key] for row in rows], dtype=float)
    result = {key: float(values(key).mean()) for key in ("raw_excess_ratio", "final_excess_ratio", "gate_attenuation", "body_deficit", "gate_probability")}
    for branch in ("raw", "final"):
        error = values(f"{branch}_excess_error")
        result[f"{branch}_excess_bias"] = float(error.mean())
        result[f"{branch}_excess_rmse"] = float(np.sqrt(np.mean(error ** 2)))
    result["reconstruction_max_abs_m"] = float(np.abs(values("reconstruction_error")).max())
    return result


def summarize_bins(rows, bin_key, population):
    summaries = []
    for model in dict.fromkeys(row["model"] for row in rows):
        for bin_id in sorted({row[bin_key] for row in rows}):
            selected = [row for row in rows if row["model"] == model and row[bin_key] == bin_id]
            values = lambda key: np.asarray([row[key] for row in selected], dtype=float)
            result = dict(model=model, population=population, quantile_bin=bin_id, n=len(selected))
            if population == "episode":
                for prefix in ("episode_peak", "episode_gt_aligned_peak"):
                    error = values(f"{prefix}_error")
                    result.update(_errors(error, prefix))
                area = _errors(values("excess_area_error"), "episode_excess_area")
                result.update({key: area[key] for key in ("episode_excess_area_mae", "episode_excess_area_bias")})
                result.update(_under(values("prediction_at_gt_peak"), values("gt_peak"), "episode_gt_aligned_peak"))
            elif population == "event_window_peak":
                for prefix in ("window_peak", "gt_aligned_peak"):
                    error = values(f"{prefix}_error")
                    result.update(_errors(error, prefix))
            result.update(summarize_branch(selected))
            summaries.append(result)
    return summaries


def compare(exports, quantile_bins=4):
    exports = align_exports(exports)
    reference = next(iter(exports.values()))
    y, times = reference["y_true"], reference["target_timestamps"]
    tau = float(scalar(reference, "tau_physical"))
    percentile = float(scalar(reference, "exceedance_percentile"))
    episodes = gt_event_episodes(y, times, tau, split_ids=reference.get("split_ids"))
    episode_membership, episode_edges = shared_quantile_bins([y.reshape(-1)[episode].max() for episode in episodes], quantile_bins)
    hour_membership, hour_edges = shared_quantile_bins(y[y > tau].astype(float) - tau, quantile_bins)
    window_membership, window_edges = shared_quantile_bins(y[(y > tau).any(axis=1)].max(axis=1), quantile_bins)
    frozen = [dict(episode_id=index, timestamps=[utc(value) for value in times.reshape(-1)[episode]],
                   split=target_split(reference, *divmod(int(episode[0]), y.shape[1])),
                   target_indices=episode.tolist(), gt_peak=float(y.reshape(-1)[episode].max()),
                   gt_severity_bin=int(episode_membership[index])) for index, episode in enumerate(episodes)]
    population_hash = hashlib.sha256(json.dumps(frozen, sort_keys=True).encode()).hexdigest()
    metrics, hours, windows, episode_records = {}, [], [], []
    for name, arrays in exports.items():
        metrics[name] = evaluate_metrics(arrays["y_pred"], y, tau, target_timestamps=times, split_ids=reference.get("split_ids"))
        episode_records.extend(episode_rows(name, arrays, episodes, episode_membership))
        hours.extend(hour_rows(name, arrays, hour_membership))
        windows.extend(window_rows(name, arrays, window_membership))
    return dict(metric_schema="hourly_q95_v1", threshold_schema="train_hourly_q95_v1", tau_physical=tau,
                exceedance_percentile=percentile,
                method=f"TRAIN hourly target Q{percentile:g}; tau={tau:.12g} m; strict y > tau",
                station=scalar(reference, "station"), split=scalar(reference, "split"),
                frozen_gt_episodes=frozen, episode_population_sha256=population_hash,
                bins=dict(episode_gt_peak_edges=episode_edges, event_window_gt_peak_edges=window_edges,
                          true_excess_edges=hour_edges, quantile_method="linear", requested_bin_count=quantile_bins),
                metrics=metrics, episode_rows=episode_records, extreme_hour_rows=hours, event_window_peak_rows=windows,
                episode_bins=summarize_bins(episode_records, "gt_severity_bin", "episode"),
                extreme_hour_bins=summarize_bins(hours, "true_excess_bin", "extreme_hour"),
                event_window_peak_bins=summarize_bins(windows, "gt_peak_bin", "event_window_peak"))


def write_csv(path, rows, metadata=None):
    metadata = metadata or {}
    with Path(path).open("w", newline="") as handle:
        fields = list(dict.fromkeys([*metadata, *(key for row in rows for key in row)]))
        if fields:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(dict(metadata, **row) for row in rows)


def plot_figures(report, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    models = list(report["metrics"])
    rows = report["episode_rows"]
    def save(fig, name):
        fig.suptitle(f"{report['station']} · {report['split']}\n{report['method']}", fontsize=10)
        for suffix in ("png", "pdf"):
            fig.savefig(output / f"{name}.{suffix}", dpi=180, bbox_inches="tight")
        plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    for model in models:
        selected = [row for row in rows if row["model"] == model]
        for ax, key in zip(axes, ("prediction_at_gt_peak", "predicted_episode_peak")):
            ax.scatter([row["gt_peak"] for row in selected], [row[key] for row in selected], s=15, alpha=.6, label=f"{model} (n={len(selected)})")
    for ax, title in zip(axes, ("At GT episode peak", "Maximum within GT episode")):
        limits = ax.get_xlim()
        ax.plot(limits, limits, "k--", linewidth=.7)
        ax.set(xlabel="GT episode peak (m)", ylabel="Predicted surge (m)", title=title)
        ax.legend()
    save(fig, "Figure6_episode_peak_comparison")
    fig, ax = plt.subplots(figsize=(7, 4), layout="constrained")
    truth = [episode["gt_peak"] for episode in report["frozen_gt_episodes"]]
    if truth:
        probability = np.arange(1, len(truth) + 1) / len(truth)
        ax.step(np.sort(truth), probability, where="post", linewidth=2, color="black", label=f"GT (n={len(truth)})")
        for model in models:
            pred = [row["predicted_episode_peak"] for row in rows if row["model"] == model]
            ax.step(np.sort(pred), probability, where="post", label=f"{model} (n={len(pred)})")
        ax.legend()
    ax.set(xlabel="Peak within fixed GT episode (m)", ylabel="Cumulative probability")
    save(fig, "Figure7_fixed_episode_peak_distribution")
    for name, keys, labels in (
        ("Figure8_episode_error_by_gt_severity", ("episode_peak_rmse", "episode_gt_aligned_peak_rmse"), ("EpisodePeakRMSE (m)", "EpisodeGTAlignedPeakRMSE (m)")),
        ("episode_bias_by_gt_severity", ("episode_peak_bias", "episode_gt_aligned_peak_bias"), ("Episode peak bias (m)", "At GT episode peak bias (m)")),
        ("episode_underprediction_by_gt_severity", ("episode_gt_aligned_peak_under_pct", "episode_gt_aligned_peak_under_10mm_pct"), ("Underprediction (%)", "Underprediction >10 mm (%)")),
        ("episode_area_error_by_gt_severity", ("episode_excess_area_mae", "episode_excess_area_bias"), ("Episode excess area MAE (m h)", "Episode excess area bias (m h)"))):
        fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
        for model in models:
            selected = [row for row in report["episode_bins"] if row["model"] == model]
            for ax, key, label in zip(axes, keys, labels):
                ax.plot([row["quantile_bin"] + 1 for row in selected], [row[key] for row in selected], "o-", label=model)
                ax.set(xlabel="Shared GT episode peak quantile bin", ylabel=label)
                ax.legend()
        save(fig, name)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", action="append", required=True, metavar="NAME=NPZ")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--quantile-bins", type=int, default=4)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args(argv)
    exports = {}
    for entry in args.predictions:
        if "=" not in entry:
            parser.error("--predictions must use NAME=NPZ")
        name, path = entry.split("=", 1)
        if not name or name in exports:
            parser.error("Prediction names must be nonempty and unique.")
        with np.load(path, allow_pickle=False) as archive:
            exports[name] = dict(archive)
    report = compare(exports, args.quantile_bins)
    args.output.mkdir(parents=True, exist_ok=True)
    metadata = {key: report[key] for key in ("station", "split", "metric_schema", "threshold_schema", "exceedance_percentile", "tau_physical", "method")}
    for key in ("episode_rows", "extreme_hour_rows", "event_window_peak_rows", "episode_bins", "extreme_hour_bins", "event_window_peak_bins"):
        write_csv(args.output / f"{key}.csv", report[key], metadata)
    compact = {key: value for key, value in report.items() if not key.endswith("_rows")}
    (args.output / "comparison.json").write_text(json.dumps(compact, indent=2, allow_nan=False) + "\n")
    (args.output / "frozen_gt_episodes.json").write_text(json.dumps(report["frozen_gt_episodes"], indent=2) + "\n")
    if not args.no_plots:
        plot_figures(report, args.output)
    print(f"{report['method']}\nCompared {len(exports)} models on {len(report['frozen_gt_episodes'])} identical GT episodes; wrote {args.output}")


if __name__ == "__main__":
    main()
