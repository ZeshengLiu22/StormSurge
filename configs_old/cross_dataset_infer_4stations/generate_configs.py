#!/usr/bin/env python3
"""Audit 48 source roles, then generate exactly 264 pinned pair configurations.

All paths are fixed to the requested production roots. --check compares rendered
files in memory: it does not write configs or run inference.
"""
import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import shlex
import sys

REPO = Path(__file__).resolve().parents[2]
PACT_ROOT = Path("/glade/u/home/zel/scratch/PACT-Derecho/PACT")
DATA = PACT_ROOT / "Data"
DESTINATION = REPO / "configs_old/cross_dataset_infer_4stations"
FORMAL = PACT_ROOT / "FormalRuns_0925"
RESULTS = FORMAL / "CrossDatasetInference_4Stations"
STATIONS = ("Boston", "CBBT", "Lewes", "Battery")
SOURCES = ("NCEP", "AWI", "CNRM", "EC_EARTH", "MPI", "MRI")
TARGETS = {"past_only": SOURCES, "future_year": SOURCES[1:]}
ORIGIN = "source_checkpoint_train"
CANDIDATE = "G0_T0500_EP0100"
sys.path.insert(0, str(REPO))


def years(first, last):
    """Build consecutive year-group labels from inclusive start-year bounds.

    Args:
        first: First integer start year to include.
        last: Last integer start year to include.

    Returns:
        list[str]: Labels such as ``1979_1980`` in ascending order; empty when
        ``last`` precedes ``first``.
    """
    return [f"{year}_{year + 1}" for year in range(first, last + 1)]


YEARS = {"past_only": years(2008, 2014), "future_year": years(2070, 2099)}


def require(condition, message):
    """Raise a validation error when a required condition is false.

    Args:
        condition: Value whose truthiness determines whether validation passes.
        message: Error message identifying the failed invariant.

    Returns:
        None.

    Raises:
        ValueError: ``condition`` is false.
    """
    if not condition:
        raise ValueError(message)


def dataset_root(group, name):
    """Select the graph directory for a dataset and evaluation period.

    Args:
        group: ``past_only`` selects historical-only CMIP6 data; ``future_year``
            selects the full historical/future tree.
        name: Dataset token from ``SOURCES``, such as ``NCEP`` or ``EC_EARTH``.

    Returns:
        str: Absolute graph-directory path under ``DATA``. NCEP uses ``Grid4_New/NCEP/graphs`` for both
        periods; CMIP6 names receive the ``CMIP6_`` prefix.
    """
    if name == "NCEP":
        return str(DATA / "Grid4_New/NCEP/graphs")
    tree = "Grid4_New_PastOnly" if group == "past_only" else "Grid4_New"
    return str(DATA / tree / f"CMIP6_{name}" / "graphs")


def shell_scalars(path):
    """Read literal scalar assignments from a resolved shell snapshot.

    Args:
        path: pathlib.Path to a training snapshot such as ``config_used.sh``.

    Returns:
        dict[str, str]: Identifier names mapped to shell-tokenized assignment
        values. Indexed-array declaration lines are skipped; later values replace
        earlier ones. Expansions are not evaluated and shell code is not executed.
    """
    values = {}
    for line in path.read_text().splitlines():
        if line.startswith("declare -a "):
            continue
        for word in shlex.split(line, comments=True):
            key, sep, value = word.partition("=")
            if sep and key.isidentifier():
                values[key] = value
    return values


def sha256(path):
    """Compute the SHA-256 digest of a file's bytes.

    Args:
        path: pathlib.Path to the file to read in full.

    Returns:
        str: Lowercase hexadecimal SHA-256 digest.
    """
    return hashlib.sha256(path.read_bytes()).hexdigest()


def discover(station, group, source):
    """Locate the unique active run for one station, period, and source.

    Args:
        station: Station token from ``STATIONS``.
        group: Experiment period, ``past_only`` or ``future_year``.
        source: Dataset token from ``SOURCES``.

    Returns:
        pathlib.Path: ``best_overall.pt`` within the sole run directory matching
        the fixed ``CANDIDATE``. Checkpoint existence is checked separately.

    Raises:
        ValueError: No active run matches or more than one matching run exists.
    """
    if source == "NCEP":
        area = FORMAL / (
            "single_tail_episodepeak_4x4" if group == "past_only" else "NCEP_future_transfer"
        )
        stem = f"NCEP_{station}" + ("" if group == "past_only" else "_future_transfer")
    else:
        area = FORMAL / f"CMIP6_{station}_QuickCheck_0929" / group
        stem = f"CMIP6_{source}_{station}_{group}"
    candidates = sorted(p for p in area.glob(f"{stem}_{CANDIDATE}__*") if p.is_dir())
    require(
        len(candidates) == 1,
        f"{station}/{group}/{source}: expected one active final run, found {list(map(str, candidates))}",
    )
    return candidates[0] / "best_overall.pt"


def training_provenance(run, checkpoint):
    """Verify full training completion and selection of the minimum VAL RMSE.

    Two explicitly named Lewes runs are permitted to have exit status 1 only
    when their logs show all epochs completed and the known missing-label error
    occurred during final reporting. Other nonzero exits fail validation.

    Args:
        run: pathlib.Path to a run directory containing metrics, exit status,
            and, when needed, the training log.
        checkpoint: Loaded checkpoint mapping with training settings, selected
            epoch, validation score, and checkpoint-selection metadata.

    Returns:
        dict: Recorded exit status, completed epoch count, completion explanation,
        and paths/hashes of the metrics and any required log evidence.

    Raises:
        ValueError: Epoch coverage, selection metadata, or completion evidence
            does not satisfy the accepted provenance conditions.
    """
    records = list(run.glob("metrics_*.jsonl"))
    require(len(records) == 1, f"{run}: ambiguous training metrics")
    epochs = [json.loads(line) for line in records[0].read_text().splitlines()]
    count = checkpoint["training_config"]["epochs"]
    require(
        [row["epoch"] for row in epochs] == list(range(1, count + 1)),
        f"{run}: incomplete training",
    )
    require(
        all(math.isfinite(row["val"]["all_rmse"]) for row in epochs), f"{run}: nonfinite VAL score"
    )
    best = min(epochs, key=lambda row: row["val"]["all_rmse"])
    require(
        checkpoint["epoch"] == best["epoch"],
        f"{run}: checkpoint is not full-run best overall epoch",
    )
    require(
        checkpoint["val"]["all_rmse"] == best["val"]["all_rmse"],
        f"{run}: saved VAL score mismatch",
    )
    require(
        checkpoint["selection_metric_key"] == "all_rmse"
        and checkpoint["selection_split"] == "val",
        f"{run}: wrong selection policy",
    )
    status = int((run / "exit_status").read_text().strip())
    note = "training_and_final_reporting_complete"
    evidence = {"training_metrics": str(records[0]), "training_metrics_sha256": sha256(records[0])}
    if status != 0:
        allowed = {
            "CMIP6_MRI_Lewes_past_only_G0_T0500_EP0100__20261003_051753",
            "CMIP6_EC_EARTH_Lewes_future_year_G0_T0500_EP0100__20261003_051753",
        }
        require(
            status == 1 and run.name in allowed, f"{run}: unvalidated nonzero exit status {status}"
        )
        logs = list(run.glob("train_*.log"))
        require(len(logs) == 1, f"{run}: ambiguous training log")
        log = logs[0].read_text()
        marker = "FINAL MULTI-CHECKPOINT RE-EVALUATION"
        require(marker in log, f"{run}: no completed-training final evaluation marker")
        before, after = log.split(marker, 1)
        require(
            f"Epoch {count:03d}/{count}" in before and "Traceback" not in before,
            f"{run}: training failed before completion",
        )
        require(
            after.rstrip().endswith("ValueError: Cannot evaluate nonfinite y_true."),
            f"{run}: unexpected final reporting failure",
        )
        note = "all_epochs_complete_best_overall_verified_final_test_reporting_failed_on_missing_labels"
        evidence.update(training_log=str(logs[0]), training_log_sha256=sha256(logs[0]))
    return dict(
        exit_status=status, training_epochs_completed=count, completion_provenance=note, **evidence
    )


def audit_checkpoint(station, group, source, path):
    """Validate one source checkpoint for the four-station transfer suite.

    Loads a checkpoint on CPU and verifies architecture, source identity,
    TRAIN-derived thresholds/normalization, chronological split provenance,
    finite model/station tensors, and training completion. Historical data paths
    are compared after relocation by their ``Data``-relative dataset identity.

    Args:
        station: Expected station token from ``STATIONS``.
        group: Source experiment period, ``past_only`` or ``future_year``.
        source: Source dataset token from ``SOURCES``.
        path: pathlib.Path to the pinned ``best_overall.pt`` checkpoint.

    Returns:
        dict: Serializable audit record with hashes, model and normalization
        settings, physical TRAIN Q95, split counts, and completion provenance.

    Raises:
        ValueError: Checkpoint identity, metadata, tensors, or provenance fail
            validation. Strict model-state loading may also raise RuntimeError.
    """
    import torch
    from emulator.models import build_model, model_config_from_checkpoint

    prefix = f"{station}/{group}/{source}: "
    require(path.is_file() and path.name == "best_overall.pt", prefix + f"missing {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model, training, threshold = (
        checkpoint[k] for k in ("model_config", "training_config", "threshold_metadata")
    )
    # Validate historical Single metadata through the shared compatibility adapter.
    model_config = model_config_from_checkpoint(checkpoint)
    for key, expected in (
        ("head_type", "single"),
        ("loss_mode", "mse"),
        ("exceedance_loss_mode", "mse"),
    ):
        require(training.get(key, expected) == expected, prefix + f"unsupported saved {key}")
    saved = shell_scalars(path.parent / "config_used.sh")
    require(checkpoint["station"] == station, prefix + "checkpoint station mismatch")
    require(checkpoint["checkpoint_role"] == "overall", prefix + "checkpoint role mismatch")
    for key, value in dict(
        model="pact",
        encoder_type="GraphSAGE",
        temporal_block="Transformer",
        history_steps=4,
        in_channels=5,
        out_channels=6,
    ).items():
        require(model[key] == value, prefix + f"model {key} mismatch: {model[key]!r}")
    for key, value in dict(
        station=station,
        model="perceiver3",
        encoder_type="GraphSAGE",
        temporal_block="Transformer",
        history_hours=24,
        use_site_elevation=0,
        use_bathymetry=0,
        exceedance_loss_weight=0.05,
        episode_gt_aligned_peak_weight=0.01,
        x_norm="zscore",
    ).items():
        require(training[key] == value, prefix + f"training {key} mismatch: {training[key]!r}")
        if key.upper() in saved:
            observed = saved[key.upper()]
            require(
                float(observed) == value if isinstance(value, (int, float)) else observed == value,
                prefix + f"config_used.sh {key} mismatch",
            )
        else:
            require(
                key in ("history_hours", "loss_mode"), prefix + f"config_used.sh missing {key}"
            )
    source_root = dataset_root(group, source)
    # Training snapshots retain their original filesystem paths and hashes.
    # Match the exact Data-relative dataset identity after relocation to GLADE.
    for value in (training["root_dir"], saved["ROOT_DIR"]):
        parts = Path(value).parts
        require("Data" in parts, prefix + f"unrecognized training data path: {value}")
        data_index = max(i for i, part in enumerate(parts) if part == "Data")
        relocated = DATA.joinpath(*parts[data_index + 1 :])
        require(
            relocated.resolve() == Path(source_root).resolve(), prefix + "source root mismatch"
        )
    require(threshold["fitted_on"] == "train", prefix + "threshold not fitted on TRAIN")
    require(threshold["exceedance_percentile"] == 95.0, prefix + "threshold is not Q95")
    require(
        threshold["threshold_schema"] == "train_hourly_q95_v1",
        prefix + "threshold schema mismatch",
    )
    require(threshold["metric_schema"] == "hourly_q95_v1", prefix + "metric schema mismatch")
    require(
        math.isfinite(threshold["tau_physical"])
        and checkpoint["tau_physical"] == threshold["tau_physical"],
        prefix + "invalid or inconsistent saved tau",
    )
    groups = {
        split: sorted({"_".join(tag.split("_")[:2]) for tag in tags})
        for split, tags in checkpoint["split_tags"].items()
    }
    expected = (
        dict(train=years(1979, 2000), val=years(2001, 2007), test=YEARS["past_only"])
        if group == "past_only"
        else dict(
            train=years(1979, 2008),
            val=years(2009, 2014),
            test=[] if source == "NCEP" else YEARS["future_year"],
        )
    )
    require(groups == expected, prefix + f"split provenance mismatch: {groups}")
    tags = [tag for values in checkpoint["split_tags"].values() for tag in values]
    require(len(tags) == len(set(tags)), prefix + "overlapping split samples")
    require(all(tag.split("_")[2] == station for tag in tags), prefix + "split station mismatch")
    require(
        threshold["train_windows"] == len(checkpoint["split_tags"]["train"]),
        prefix + "TRAIN population mismatch",
    )
    # Current checkpoints may omit missing target hours; require the saved valid population.
    require(
        0 < threshold["train_target_hour_count"] <= 6 * threshold["train_windows"],
        prefix + "invalid TRAIN hourly count",
    )
    normalization = checkpoint["normalization"]
    for key, width in dict(x_center=5, x_scale=5, y_mean=6, y_std=6).items():
        value = normalization[key]
        require(
            tuple(value.shape) == (width,) and bool(torch.isfinite(value).all()),
            prefix + f"invalid {key}",
        )
        if key in ("x_scale", "y_std"):
            require(bool((value > 0).all()), prefix + f"nonpositive {key}")
    network = build_model(model_config)
    network.load_state_dict(checkpoint["model_state"], strict=True)
    require(
        all(bool(torch.isfinite(v).all()) for v in checkpoint["model_state"].values()),
        prefix + "nonfinite weights",
    )
    station_feat = checkpoint["station_feat"]
    require(
        station_feat is not None
        and station_feat.numel() == model["station_feat_dim"]
        and bool(torch.isfinite(station_feat).all()),
        prefix + "invalid source station features",
    )
    provenance = training_provenance(path.parent, checkpoint)
    return dict(
        station=station,
        group=group,
        source=source,
        checkpoint=str(path),
        sha256=sha256(path),
        epoch=checkpoint["epoch"],
        model="perceiver3",
        encoder_type=model["encoder_type"],
        temporal_block=model["temporal_block"],
        head_type="single",
        history_steps=model["history_steps"],
        history_hours=24,
        candidate=CANDIDATE,
        normalization_type=training["x_norm"],
        normalization_origin=ORIGIN,
        normalization={k: v.tolist() for k, v in normalization.items()},
        tau_physical=threshold["tau_physical"],
        source_tau_physical=threshold["tau_physical"],
        source_exceedance_percentile=threshold["exceedance_percentile"],
        threshold_schema=threshold["threshold_schema"],
        threshold_metadata=threshold,
        evaluation_threshold_origin=ORIGIN,
        source_root=source_root,
        split_years=groups,
        train_group_count=len(groups["train"]),
        val_group_count=len(groups["val"]),
        test_group_count=len(groups["test"]),
        model_state_strict_load=True,
        loss_mode="mse",
        exceedance_loss_weight=training["exceedance_loss_weight"],
        episode_gt_aligned_peak_weight=training["episode_gt_aligned_peak_weight"],
        config_used=str(path.parent / "config_used.sh"),
        config_used_sha256=sha256(path.parent / "config_used.sh"),
        **provenance,
    )


def audit_targets():
    """Validate station metadata and requested target-year inventories.

    Reads station JSON and graph filenames under the configured roots. Graph
    tensors are not loaded and no statistics are fitted.

    Returns:
        list[dict]: Inventory records for each station, period, and target, with
        its root plus requested and available year-group labels.

    Raises:
        ValueError: A station token is inconsistent or requested years are absent.
    """
    rows = []
    for station in STATIONS:
        metadata = json.loads((REPO / "station_json" / f"{station}.json").read_text())
        require(metadata["station_key"] == station, f"{station}: station JSON token mismatch")
        for group, targets in TARGETS.items():
            for target in targets:
                root = REPO / dataset_root(group, target)
                files = sorted(root.glob(f"*_{station}_*graphs.pt"))
                available = sorted({"_".join(p.name.split("_")[:2]) for p in files})
                require(
                    set(YEARS[group]) <= set(available),
                    f"{station}/{group}/{target}: missing years {sorted(set(YEARS[group]) - set(available))}",
                )
                rows.append(
                    dict(
                        station=station,
                        group=group,
                        target=target,
                        target_root=str(root),
                        requested_years=YEARS[group],
                        available_years=available,
                    )
                )
    return rows


def csv_text(rows):
    """Serialize heterogeneous row mappings as CSV with stable column order.

    Args:
        rows: Reiterable collection of mappings. Columns are the ordered union
            of keys; nested dictionary/list values are encoded as sorted-key JSON.

    Returns:
        str: CSV header and rows using Unix line endings. Missing fields are empty.
    """
    output = io.StringIO()
    writer = csv.DictWriter(
        output, fieldnames=list(dict.fromkeys(k for row in rows for k in row)), lineterminator="\n"
    )
    writer.writeheader()
    writer.writerows(
        {
            k: json.dumps(v, sort_keys=True) if isinstance(v, (dict, list)) else v
            for k, v in row.items()
        }
        for row in rows
    )
    return output.getvalue()


def render(checkpoints):
    """Render the four-station transfer suite without writing files.

    Args:
        checkpoints: Mapping from ``(station, group, source)`` to an absolute
            ``best_overall.pt`` path without wildcard characters.

    Returns:
        tuple[dict[str, str], list[dict]]: Relative output paths mapped to their
        complete text, and the 264 pair-manifest rows. Includes shared shell
        configuration files and ``manifest.csv``.

    Raises:
        ValueError: A checkpoint path is not pinned or the pair count is not 264.
    """
    files, rows = {}, []
    source_common = '#!/usr/bin/env bash\nsource "$(dirname "${BASH_SOURCE[0]}")/../common.sh"\n\n'
    for station in STATIONS:
        files[f"{station}/common.sh"] = source_common + f'STATION="{station}"\n'
        for group, targets in TARGETS.items():
            year_list = ",".join(YEARS[group])
            files[f"{station}/{group}/common.sh"] = (
                source_common + f'EXPERIMENT_GROUP="{group}"\nYEARS="{year_list}"\n'
                f"EXPECTED_YEAR_COUNT={len(YEARS[group])}\n"
                f'INFERENCE_RESULTS_ROOT="{RESULTS}/{station}/{group}"\n'
            )
            for source in SOURCES:
                for target in targets:
                    checkpoint = str(checkpoints[station, group, source])
                    require(
                        Path(checkpoint).is_absolute()
                        and Path(checkpoint).name == "best_overall.pt"
                        and not any(c in checkpoint for c in "*?["),
                        f"unpinned checkpoint: {checkpoint}",
                    )
                    relative = f"{station}/{group}/{source}/{source}_to_{target}.sh"
                    source_root, target_root = dataset_root(group, source), dataset_root(
                        group, target
                    )
                    files[relative] = (
                        source_common + f'NAME="{group}_{station}_{source}_to_{target}"\n'
                        f'SOURCE_NAME="{source}"\nTARGET_NAME="{target}"\n\n'
                        f"CKPT_PATH={shlex.quote(checkpoint)}\n\n"
                        f'SOURCE_ROOT="{source_root}"\nTARGET_ROOT="{target_root}"\n\n'
                        'ROOT_DIR="${SOURCE_ROOT}"\nTEST_ROOT_DIR="${TARGET_ROOT}"\n'
                    )
                    rows.append(
                        dict(
                            station=station,
                            group=group,
                            source=source,
                            target=target,
                            checkpoint=checkpoint,
                            source_root=source_root,
                            target_root=target_root,
                            years=year_list,
                            year_count=len(YEARS[group]),
                            config_path=str((DESTINATION / relative).relative_to(REPO)),
                            threshold_origin=ORIGIN,
                        )
                    )
    require(len(rows) == 264, "Expected 264 configs")
    files["manifest.csv"] = csv_text(rows)
    return files, rows


def main():
    """Audit 48 source roles and write or verify the four-station transfer suite.

    Parses ``--check``, limits PyTorch to one thread, and validates source
    checkpoints and target files. Normal mode writes rendered configurations,
    manifest, and checkpoint audit files under ``DESTINATION``. Check mode
    compares their current text without writing those artifacts.

    Returns:
        None. Prints per-source validation details and the suite totals.

    Raises:
        ValueError: An audit fails, an unexpected shell configuration exists,
            or check mode finds generated-file drift.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    import torch

    torch.set_num_threads(1)
    checkpoints, audits = {}, []
    for station in STATIONS:
        for group in TARGETS:
            for source in SOURCES:
                path = discover(station, group, source)
                row = audit_checkpoint(station, group, source, path)
                checkpoints[station, group, source] = path
                audits.append(row)
                print(
                    f"PASS {station}/{group}/{source}: epoch={row['epoch']} exit={row['exit_status']} "
                    f"split={row['train_group_count']}/{row['val_group_count']}/{row['test_group_count']}",
                    flush=True,
                )
    targets = audit_targets()
    files, rows = render(checkpoints)
    files["checkpoint_audit.csv"] = csv_text(audits)
    files["checkpoint_audit.json"] = (
        json.dumps(dict(checkpoints=audits, target_years=targets), indent=2) + "\n"
    )
    for station in STATIONS:
        actual = {str(p.relative_to(DESTINATION)) for p in (DESTINATION / station).rglob("*.sh")}
        require(
            actual <= set(files),
            f"{station}: unexpected config files: {sorted(actual - set(files))}",
        )
    if args.check:
        for name, content in files.items():
            require(
                (DESTINATION / name).is_file() and (DESTINATION / name).read_text() == content,
                f"Generated file drift: {DESTINATION / name}",
            )
    else:
        for name, content in files.items():
            path = DESTINATION / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)
    print(
        f"PASS: 48 source roles, {len(targets)} target year audits, {len(rows)} configs; "
        f'{"checked without writes" if args.check else "written under " + str(DESTINATION)}'
    )


if __name__ == "__main__":
    main()
