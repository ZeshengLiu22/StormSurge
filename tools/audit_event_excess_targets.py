#!/usr/bin/env python3
"""Audit physical excess targets only inside existing sampled GT event windows.

TRAIN membership comes from a saved checkpoint's split_tags. VAL/TEST targets
come directly from that run's exports and must exactly match the sampled graphs.
This tool reads data/metadata on CPU; it never constructs or executes a model.
"""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from emulator.data import ForcingGraphStore, fit_loss_thresholds, supervised_targets, validate_target_timestamps
from emulator.training.excess_amplitude import physical_excess_target
from emulator.training.metrics import evaluate_metrics


def require(condition, message):
    if not condition:
        raise ValueError(message)


def pct(count, total):
    return 100. * count / total if total else None


def distribution(values):
    values = np.asarray(values, dtype=np.float64)
    result = dict(n=int(values.size))
    result.update({name: None for name in ("min", "mean", "median", "q25", "q75", "q90", "q95", "max")})
    if values.size:
        result.update(min=float(values.min()), mean=float(values.mean()), max=float(values.max()))
        result.update(zip(("q25", "median", "q75", "q90", "q95"),
                          map(float, np.quantile(values, [.25, .5, .75, .9, .95], method="linear"))))
    return result


def audit_targets(labels, timestamps, tags, sample_ids, tau, station, split):
    """Every reported hour/horizon/window statistic is conditional on GT event=1."""
    labels = np.asarray(labels, dtype=np.float64)
    timestamps = validate_target_timestamps(timestamps)
    tags, sample_ids = np.asarray(tags, dtype=str), np.asarray(sample_ids)
    require(labels.ndim == 2 and labels.shape[1] > 0 and labels.shape == timestamps.shape,
            "Labels and target timestamps must have matching [windows, horizons] shapes.")
    require(np.isfinite(labels).all(), "Target labels must be finite.")
    require(tags.shape == sample_ids.shape == (len(labels),), "One tag and sample ID are required per window.")
    require(len(set(tags)) == len(tags) and len(set(sample_ids)) == len(sample_ids), "Repeated sampled window IDs.")
    require(np.all(np.diff(timestamps, axis=1) == 3600), "Targets must be consecutive hourly horizons.")

    # Reuse the actual production function, including its strict physical-tau
    # comparison. FP64 subtraction preserves the saved physical threshold.
    event, excess = physical_excess_target(None, None, None,
        target_phys=torch.from_numpy(labels), tau_physical=tau)
    selected = np.flatnonzero(event.numpy().reshape(-1))
    excess = excess.numpy()[selected]
    positive = excess > 0
    canonical = evaluate_metrics(labels, labels, tau, include_leadwise=False)
    require(len(selected) == canonical["event_window_n"] and int(positive.sum()) == canonical["extreme_hour_n"],
            "Production training/evaluation event populations disagree.")

    windows, width = excess.shape
    hours, positive_n = int(excess.size), int(positive.sum())
    zero_n = hours - positive_n
    counts = positive.sum(axis=1)
    zero_fractions = (width - counts) / width
    mixed = (counts > 0) & (counts < width)
    summary = dict(station=station, split=split.upper(), tau_physical_m=float(tau),
                   event_window_count=windows, target_hour_count=hours,
                   zero_excess_count=zero_n, zero_excess_pct=pct(zero_n, hours),
                   positive_excess_count=positive_n, positive_excess_pct=pct(positive_n, hours),
                   mixed_window_count=int(mixed.sum()), mixed_window_pct=pct(int(mixed.sum()), windows))
    positive_values = excess[positive]
    summary.update({f"positive_excess_{key}_mm": value for key, value in distribution(positive_values * 1000.).items()
                    if key != "n"})
    for limit in (5, 10, 20):
        n = int((positive_values < limit / 1000.).sum())
        summary[f"positive_excess_lt{limit}mm_count"] = n
        summary[f"positive_excess_lt{limit}mm_pct"] = pct(n, positive_n)

    by_horizon = []
    for lead in range(width):
        n = int(positive[:, lead].sum())
        by_horizon.append(dict(station=station, split=split.upper(), horizon_hours=lead,
                               target_hour_count=windows, zero_excess_count=windows - n,
                               zero_excess_pct=pct(windows - n, windows), positive_excess_count=n,
                               positive_excess_pct=pct(n, windows)))
    per_window = []
    for row, source_index in enumerate(selected):
        per_window.append(dict(station=station, split=split.upper(), tag=str(tags[source_index]),
            sample_id=int(sample_ids[source_index]),
            first_target_utc=str(np.datetime64(int(timestamps[source_index, 0]), "s")) + "Z",
            last_target_utc=str(np.datetime64(int(timestamps[source_index, -1]), "s")) + "Z",
            target_hour_count=width, zero_excess_count=int(width - counts[row]),
            zero_excess_fraction=float(zero_fractions[row]), positive_excess_count=int(counts[row]),
            mixed_zero_positive=bool(mixed[row])))
    histogram = [dict(station=station, split=split.upper(), positive_hours=k,
                      zero_excess_fraction=(width - k) / width,
                      event_window_count=int((counts == k).sum()),
                      event_window_pct=pct(int((counts == k).sum()), windows)) for k in range(width + 1)]
    return dict(summary=summary, by_horizon=by_horizon, per_event_window=per_window,
                positive_hours_per_window=dict(distribution=distribution(counts), histogram=histogram),
                zero_fraction_per_window=distribution(zero_fractions))


def fingerprint(*arrays):
    digest = hashlib.sha256()
    for value in arrays:
        value = np.ascontiguousarray(value)
        digest.update(str((value.dtype.str, value.shape)).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def file_record(path):
    path = Path(path)
    return dict(path=str(path.resolve()), bytes=path.stat().st_size,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def indices_from_tags(store, split_tags):
    """Use saved membership, failing on changed/missing/repeated sampled graphs."""
    lookup = {tag: i for i, tag in enumerate(store.graph_tags)}
    require(len(lookup) == len(store.graph_tags), "Repeated graph tags in source data.")
    require(set(split_tags) == {"train", "val", "test"}, "Checkpoint must retain TRAIN/VAL/TEST split tags.")
    all_tags = [tag for tags in split_tags.values() for tag in tags]
    require(len(all_tags) == len(set(all_tags)), "Checkpoint has repeated or cross-split window tags.")
    missing = set(all_tags) - lookup.keys()
    require(not missing, f"Checkpoint sampled graphs missing from source: {sorted(missing)[:3]}")
    return {split: [lookup[tag] for tag in tags] for split, tags in split_tags.items()}


def load_export(path, station, split, tau, expected):
    """Retain exported evaluation membership/order and verify every target."""
    with np.load(path, allow_pickle=False) as saved:
        require(str(saved["station"].item()) == station and str(saved["split"].item()) == split,
                f"{path}: station or split mismatch.")
        require(float(saved["tau_physical"]) == tau, f"{path}: TRAIN threshold mismatch.")
        arrays = {key: saved[key] for key in ("y_true", "target_timestamps", "tags", "sample_id")}
    for key, value in expected.items():
        require(np.array_equal(arrays[key], value), f"{path}: {key} differs from saved sampled graphs.")
    return arrays


def audit_run(run_dir, root_dir=None, role="overall"):
    run_dir = Path(run_dir).resolve()
    checkpoint_path = run_dir / f"best_{role}.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config, threshold = checkpoint["training_config"], checkpoint["threshold_metadata"]
    station, tau = checkpoint["station"], float(threshold["tau_physical"])
    require(not config.get("test_root_dir"), "This audit requires the saved held-out TEST split, not external OOD data.")
    require(threshold["fitted_on"] == "train", "Threshold must have been fitted on TRAIN.")
    if root_dir is None:
        root_dir = Path(config["root_dir"])
        if not root_dir.is_absolute():
            root_dir = REPO / root_dir
    print(f"{station}: loading sampled graph targets ...", flush=True)
    store = ForcingGraphStore(root_dir, station, targets_only=True)
    indices = indices_from_tags(store, checkpoint["split_tags"])
    require(indices == store.split(**checkpoint["split_config"]), "Saved split membership disagrees with the current pipeline.")
    require(all(int(store.graphs[i].source_history_steps) >= config["history_hours"] // 6 + 1
                for part in indices.values() for i in part), "Sampled graph history is insufficient for the saved run.")
    # Leakage/threshold verification only; the reported diagnostic is event-only.
    store.target_timestamps([i for part in indices.values() for i in part])
    fitted = fit_loss_thresholds(store, indices["train"], threshold["exceedance_percentile"])
    for key, value in fitted.items():
        if key != "tau_normalized":
            require(value == threshold[key], f"{station}: reconstructed TRAIN threshold metadata differs: {key}")
    provenance = dict(run_dir=str(run_dir), checkpoint=file_record(checkpoint_path),
                      graph_root=str(Path(root_dir).resolve()), threshold=threshold,
                      split_config=checkpoint["split_config"], splits={})
    audits = {}
    for split in ("train", "val", "test"):
        labels, times = supervised_targets(store, indices[split])
        arrays = dict(y_true=labels, target_timestamps=times,
                      tags=np.asarray(checkpoint["split_tags"][split]), sample_id=np.asarray(indices[split]))
        source = dict(sampled_window_count=len(labels),
                      year_groups=sorted({"_".join(tag.split("_")[:2]) for tag in arrays["tags"]}))
        if split != "train":
            export_path = run_dir / f"{split}_predictions_{role}.npz"
            arrays = load_export(export_path, station, split, tau, arrays)
            source["prediction_export"] = file_record(export_path)
            source["targets_ids_tags_timestamps_exactly_match_graphs"] = True
        source["sampled_target_sha256"] = fingerprint(arrays["y_true"].astype(np.float64),
            arrays["target_timestamps"], arrays["tags"], arrays["sample_id"])
        provenance["splits"][split] = source
        audits[split] = audit_targets(arrays["y_true"], arrays["target_timestamps"], arrays["tags"],
                                      arrays["sample_id"], tau, station, split)
        result = audits[split]["summary"]
        print(f"  {split.upper()}: {result['event_window_count']} event windows, "
              f"{result['target_hour_count']} target hours, {fmt(result['zero_excess_pct'])}% zero", flush=True)
    return dict(station=station, provenance=provenance, splits=audits)


def fmt(value):
    return "NA" if value is None else f"{value:.3f}"


def table(headers, rows):
    return ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)] + [
        "| " + " | ".join(map(str, row)) + " |" for row in rows]


def render_report(report):
    audits = [audit for run in report["stations"].values() for audit in run["splits"].values()]
    summaries = [audit["summary"] for audit in audits]
    lines = ["# Excess targets inside sampled GT event windows", "",
        f"Generated: {report['created_at']}", "",
        "Every statistic below is conditional on a sampled GT event window: "
        "`E_i = any_h(y_ih > tau_station)`. All hourly targets in each selected window are retained, "
        "including those with `y <= tau`. `excess_target = max(y - tau, 0)`; equality is zero excess.", "",
        "Event selection and excess targets call the current pipeline's "
        "`emulator.training.excess_amplitude.physical_excess_target` and are checked against "
        "`emulator.training.metrics.evaluate_metrics`. The saved TRAIN threshold is reused unchanged "
        "for VAL/TEST. Forecast blocks and GT event definitions are preserved.", "",
        "TRAIN windows are selected by the checkpoint's exact `split_tags`. VAL/TEST use the saved "
        "evaluation NPZ `y_true`, `sample_id`, `tags`, and `target_timestamps`, with exact equality "
        "checks against those sampled graphs. The production graph-store split and TRAIN threshold "
        "are also verified. Loader shuffling changes order only, `drop_last=False`, and the "
        "canonical metric population counts each sampled window once. No raw time series enters the report.", "",
        "Targets use physical meters; positive excess distributions below are in **mm**. "
        "Calculations promote saved physical labels to FP64 and preserve the saved physical threshold. "
        "Quantiles use NumPy's linear method. Undefined percentages/statistics are NA (JSON null).", "",
        "## Target counts within event windows", "",
        "Hour percentages divide by target hours **inside event windows**. Mixed-window percentages "
        "divide by event windows and require both zero and positive excess targets.", ""]
    lines += table(["Station", "Split", "Event windows", "Target hours", "Zero n (%)", "Positive n (%)", "Mixed windows n (%)"],
        [[s["station"], s["split"], s["event_window_count"], s["target_hour_count"],
          f"{s['zero_excess_count']} ({fmt(s['zero_excess_pct'])})",
          f"{s['positive_excess_count']} ({fmt(s['positive_excess_pct'])})",
          f"{s['mixed_window_count']} ({fmt(s['mixed_window_pct'])})"] for s in summaries])
    lines += ["", "## Positive excess distribution (mm)", "",
              "Only strictly positive excess targets inside the selected event windows enter this distribution.", ""]
    stats = ("mean", "median", "q25", "q75", "q90", "q95", "max")
    lines += table(["Station", "Split", "Mean", "Median", "Q25", "Q75", "Q90", "Q95", "Max"],
        [[s["station"], s["split"]] + [fmt(s[f"positive_excess_{name}_mm"]) for name in stats] for s in summaries])
    lines += ["", "## Small positive excess (%)", "",
              "Each denominator is the number of **positive** targets in event windows. Comparisons are strict `<`.", ""]
    lines += table(["Station", "Split", "<5 mm", "<10 mm", "<20 mm"],
        [[s["station"], s["split"]] + [fmt(s[f"positive_excess_lt{limit}mm_pct"]) for limit in (5, 10, 20)] for s in summaries])
    lines += ["", "## Zero / positive excess by forecast horizon (%)", "",
              "Cells are `zero % / positive %`. Each horizon denominator is the number of selected "
              "event windows for that station/split. Horizons start at the forecast center (`t+0`).", ""]
    widths = {len(a["by_horizon"]) for a in audits}
    require(len(widths) == 1, "A combined horizon table requires a common forecast width.")
    width = widths.pop()
    lines += table(["Station", "Split"] + [f"t+{h} h" for h in range(width)],
        [[a["summary"]["station"], a["summary"]["split"]] + [
            f"{fmt(h['zero_excess_pct'])} / {fmt(h['positive_excess_pct'])}" for h in a["by_horizon"]] for a in audits])
    lines += ["", "## Positive-excess hours per event window", "",
              "Cells are `window count (window %)`. The full histogram, including the zero-count bin, "
              "is in [positive_hours_per_window.csv](positive_hours_per_window.csv). "
              "No GT event window can have zero positive hours.", ""]
    lines += table(["Station", "Split"] + [str(k) + " positive h" for k in range(1, width + 1)],
        [[a["summary"]["station"], a["summary"]["split"]] + [
            f"{h['event_window_count']} ({fmt(h['event_window_pct'])})"
            for h in a["positive_hours_per_window"]["histogram"][1:]] for a in audits])
    lines += ["", "### Count distribution", ""]
    lines += table(["Station", "Split", "Min", "Mean", "Median", "Q25", "Q75", "Q90", "Q95", "Max"],
        [[a["summary"]["station"], a["summary"]["split"]] + [
            fmt(a["positive_hours_per_window"]["distribution"][k]) for k in ("min", *stats)] for a in audits])
    lines += ["", "## Zero-excess fraction per event window", "",
              "[per_event_window.csv](per_event_window.csv) contains every event window's station, split, "
              "saved sample ID/tag, target start/end, zero fraction, positive-hour count, and mixed flag. "
              "The fraction is `zero_excess_hours / target_hours_in_that_window` (range 0–1). "
              "The table summarizes those individual fractions with equal weight per event window.", ""]
    lines += table(["Station", "Split", "Min", "Mean", "Median", "Q25", "Q75", "Q90", "Q95", "Max"],
        [[a["summary"]["station"], a["summary"]["split"]] + [
            fmt(a["zero_fraction_per_window"][k]) for k in ("min", *stats)] for a in audits])
    lines += ["", "## Sources and validation", "",
              "All thresholds, split year groups, source paths/hashes, and sampled-target fingerprints "
              "are retained in [audit.json](audit.json). Counts and percentages reconcile across "
              "[summary.csv](summary.csv), [by_horizon.csv](by_horizon.csv), and the per-window rows.", ""]
    lines += table(["Station", "Saved TRAIN tau (m)", "Source run"],
        [[station, f"{run['provenance']['threshold']['tau_physical']:.12f}",
          f"`{run['provenance']['run_dir']}`"] for station, run in report["stations"].items()])
    lines += ["", "This diagnostic only reads saved graphs, checkpoint metadata, and evaluation exports. "
              "No training code/configuration was changed, no model was constructed, and no training "
              "or inference job was launched."]
    return "\n".join(lines) + "\n"


def write_outputs(report, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    audits = [audit for run in report["stations"].values() for audit in run["splits"].values()]
    outputs = {"summary.csv": [a["summary"] for a in audits],
               "by_horizon.csv": [row for a in audits for row in a["by_horizon"]],
               "per_event_window.csv": [row for a in audits for row in a["per_event_window"]],
               "positive_hours_per_window.csv": [row for a in audits for row in a["positive_hours_per_window"]["histogram"]]}
    for filename, rows in outputs.items():
        fields = list(rows[0]) if rows else ["station", "split", "tag", "sample_id", "first_target_utc", "last_target_utc",
            "target_hour_count", "zero_excess_count", "zero_excess_fraction", "positive_excess_count", "mixed_zero_positive"]
        with (output / filename).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    (output / "audit.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    (output / "REPORT.md").write_text(render_report(report))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", nargs="+", required=True, help="One completed source run per station.")
    parser.add_argument("--root-dir", help="Optional sampled-graph directory override; otherwise use saved config.")
    parser.add_argument("--role", default="overall", help="Saved checkpoint/export role, used only to read targets.")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    torch.set_num_threads(1)
    report = dict(schema="sampled_gt_event_excess_v1", created_at=datetime.now(timezone.utc).isoformat(),
                  event_definition="any_h(y_ih > saved TRAIN tau_station)",
                  event_function="emulator.training.excess_amplitude.physical_excess_target",
                  population="sampled GT event windows only; all horizons within each selected window",
                  percentage_scale="0 to 100; per-window zero_excess_fraction uses 0 to 1",
                  implementation_sources=[file_record(REPO / path) for path in (
                      "tools/audit_event_excess_targets.py", "emulator/training/excess_amplitude.py",
                      "emulator/training/metrics.py", "emulator/data/graph_store.py", "emulator/data/targets.py",
                      "emulator/data/stats.py", "emulator/data/loaders.py", "emulator/training/engine.py", "train.py")],
                  stations={})
    for path in args.run_dir:
        audit = audit_run(path, args.root_dir, args.role)
        require(audit["station"] not in report["stations"], "Supply exactly one source run per station.")
        report["stations"][audit["station"]] = audit
    write_outputs(report, args.output_dir)
    print(f"Wrote {Path(args.output_dir).resolve()}", flush=True)


if __name__ == "__main__":
    main()
