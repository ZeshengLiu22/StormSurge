#!/usr/bin/env python3
"""Audit pinned source checkpoints and deterministically generate 66 pair configs.

No inference is run. All source audits finish before any generated file is written.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shlex
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
FORMAL = Path("/home/exouser/media/share/PACT/FormalRuns_0925")
DESTINATION = Path(__file__).resolve().parent
SOURCES = ("NCEP", "AWI", "CNRM", "EC_EARTH", "MPI", "MRI")
TARGETS = {"past_only": SOURCES, "future_year": SOURCES[1:]}
# Pinned after inspecting the completed runs in the requested active areas.
# Archived runs under FormalRuns_0925/Achieved are outside these source areas.
RUNS = {
    "past_only": {
        "NCEP": "single_tail_episodepeak_4x4/NCEP_Boston_G0_T0500_EP0100__20260927_191110",
        "AWI": "CMIP6_Boston_QuickCheck_0929/past_only/CMIP6_AWI_Boston_past_only_G0_T0500_EP0100__20260929_054511",
        "CNRM": "CMIP6_Boston_QuickCheck_0929/past_only/CMIP6_CNRM_Boston_past_only_G0_T0500_EP0100__20260929_054511",
        "EC_EARTH": "CMIP6_Boston_QuickCheck_0929/past_only/CMIP6_EC_EARTH_Boston_past_only_G0_T0500_EP0100__20260929_054511",
        "MPI": "CMIP6_Boston_QuickCheck_0929/past_only/CMIP6_MPI_Boston_past_only_G0_T0500_EP0100__20260929_054511",
        "MRI": "CMIP6_Boston_QuickCheck_0929/past_only/CMIP6_MRI_Boston_past_only_G0_T0500_EP0100__20260929_054511",
    },
    "future_year": {
        "NCEP": "NCEP_future_transfer/NCEP_Boston_future_transfer_G0_T0500_EP0100__20261001_165726",
        "AWI": "CMIP6_Boston_QuickCheck_0929/future_year/CMIP6_AWI_Boston_future_year_G0_T0500_EP0100__20260929_054511",
        "CNRM": "CMIP6_Boston_QuickCheck_0929/future_year/CMIP6_CNRM_Boston_future_year_G0_T0500_EP0100__20260929_054511",
        "EC_EARTH": "CMIP6_Boston_QuickCheck_0929/future_year/CMIP6_EC_EARTH_Boston_future_year_G0_T0500_EP0100__20260930_142355",
        "MPI": "CMIP6_Boston_QuickCheck_0929/future_year/CMIP6_MPI_Boston_future_year_G0_T0500_EP0100__20260929_054511",
        "MRI": "CMIP6_Boston_QuickCheck_0929/future_year/CMIP6_MRI_Boston_future_year_G0_T0500_EP0100__20260930_142355",
    },
}


def years(first, last):
    return [f"{year}_{year + 1}" for year in range(first, last + 1)]


YEARS = {"past_only": years(2008, 2014), "future_year": years(2070, 2099)}


def dataset_root(group, name):
    if name == "NCEP":
        return "./Data/Grid4_New/NCEP/graphs"
    tree = "Grid4_New_PastOnly" if group == "past_only" else "Grid4_New"
    return f"./Data/{tree}/CMIP6_{name}/graphs"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def shell_scalars(path):
    """Read literal scalar assignments from resolved snapshots without executing them."""
    values = {}
    for line in path.read_text().splitlines():
        if line.startswith("declare -a "):
            continue
        for word in shlex.split(line, comments=True):
            name, sep, value = word.partition("=")
            if sep and name.isidentifier():
                values[name] = value
    return values


def audit_checkpoint(group, source, path):
    import torch
    from emulator.models import ModelConfig, build_model

    prefix = f"{group}/{source}: "
    require(path.is_file() and path.name == "best_overall.pt", prefix + f"missing checkpoint {path}")
    run = path.parent
    candidates = sorted(p.name for p in run.parent.glob(run.name.split("__")[0] + "__*") if p.is_dir())
    require(candidates == [run.name], prefix + f"ambiguous active source runs: {candidates}")
    require((run / "exit_status").read_text().strip() == "0", prefix + "training did not complete successfully")
    saved = shell_scalars(run / "config_used.sh")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model, training, threshold = (checkpoint[key] for key in ("model_config", "training_config", "threshold_metadata"))
    for key, value in dict(model="pact", encoder_type="GraphSAGE", temporal_block="Transformer",
                           head_type="single", history_steps=4, in_channels=5, out_channels=6).items():
        require(model[key] == value, prefix + f"model {key}={model[key]!r}; expected {value!r}")
    require(checkpoint["station"] == "Boston", prefix + "checkpoint station mismatch")
    require(checkpoint["checkpoint_role"] == "overall", prefix + "checkpoint role mismatch")
    for key, value in dict(station="Boston", model="perceiver3", encoder_type="GraphSAGE",
                           temporal_block="Transformer", head_type="single", history_hours=24,
                           use_site_elevation=0, use_bathymetry=0, loss_mode="mse",
                           exceedance_loss_mode="mse", exceedance_loss_weight=.05,
                           episode_gt_aligned_peak_weight=.01, x_norm="zscore").items():
        require(training[key] == value, prefix + f"training {key} mismatch")
        if key.upper() in saved:
            observed = saved[key.upper()]
            require((float(observed) == value) if isinstance(value, (float, int)) else observed == value,
                    prefix + f"config_used.sh {key} mismatch")
        elif key not in ("history_hours", "loss_mode"):
            raise ValueError(prefix + f"config_used.sh missing {key}")
    source_root = dataset_root(group, source)
    for value in (training["root_dir"], saved["ROOT_DIR"]):
        require((REPO / value).resolve() == (REPO / source_root).resolve(), prefix + "source root mismatch")
    require(threshold["fitted_on"] == "train", prefix + "threshold was not fitted on TRAIN")
    require(threshold["exceedance_percentile"] == 95., prefix + "threshold is not Q95")
    require(threshold["threshold_schema"] == "train_hourly_q95_v1", prefix + "threshold schema mismatch")
    require(threshold["metric_schema"] == "hourly_q95_v1", prefix + "metric schema mismatch")
    require(checkpoint["tau_physical"] == threshold["tau_physical"], prefix + "inconsistent saved tau")
    groups = {split: sorted({"_".join(tag.split("_")[:2]) for tag in tags})
              for split, tags in checkpoint["split_tags"].items()}
    expected = (dict(train=years(1979, 2000), val=years(2001, 2007), test=YEARS["past_only"])
                if group == "past_only" else
                dict(train=years(1979, 2008), val=years(2009, 2014),
                     test=[] if source == "NCEP" else YEARS["future_year"]))
    require(groups == expected, prefix + f"saved split groups mismatch: {groups}")
    all_tags = [tag for tags in checkpoint["split_tags"].values() for tag in tags]
    require(len(all_tags) == len(set(all_tags)), prefix + "duplicate / overlapping split samples")
    require(threshold["train_windows"] == len(checkpoint["split_tags"]["train"]), prefix + "TRAIN threshold population mismatch")
    require(threshold["train_target_hour_count"] == 6 * threshold["train_windows"], prefix + "TRAIN hourly count mismatch")
    normalization = checkpoint["normalization"]
    for key, width in dict(x_center=5, x_scale=5, y_mean=6, y_std=6).items():
        value = normalization[key]
        require(tuple(value.shape) == (width,) and bool(torch.isfinite(value).all()), prefix + f"invalid {key}")
        if key in ("x_scale", "y_std"):
            require(bool((value > 0).all()), prefix + f"nonpositive {key}")
    network = build_model(ModelConfig(**model))
    network.load_state_dict(checkpoint["model_state"], strict=True)
    return dict(group=group, source=source, checkpoint=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                active_candidate_runs=candidates, exit_status=0, config_used=str(run / "config_used.sh"),
                config_used_sha256=hashlib.sha256((run / "config_used.sh").read_bytes()).hexdigest(),
                epoch=checkpoint["epoch"], station=checkpoint["station"], model="perceiver3",
                encoder_type=model["encoder_type"], temporal_block=model["temporal_block"], head_type=model["head_type"],
                history_steps=model["history_steps"], history_hours=model["history_steps"] * 6,
                normalization_type=training["x_norm"], normalization_origin="source_checkpoint_train",
                normalization={key: value.tolist() for key, value in normalization.items()},
                tau_physical=threshold["tau_physical"], exceedance_percentile=threshold["exceedance_percentile"],
                threshold_schema=threshold["threshold_schema"], threshold_metadata=threshold,
                evaluation_threshold_origin="source_checkpoint_train",
                source_root=source_root, split_years=groups,
                split_group_counts={key: len(value) for key, value in groups.items()},
                split_sample_counts={key: len(value) for key, value in checkpoint["split_tags"].items()},
                model_state_strict_load=True, loss_mode=training["loss_mode"],
                exceedance_loss_weight=training["exceedance_loss_weight"],
                episode_gt_aligned_peak_weight=training["episode_gt_aligned_peak_weight"])


def audit_targets():
    """Check requested station filenames without loading labels or fitting statistics."""
    rows = []
    for group, targets in TARGETS.items():
        for target in targets:
            root = REPO / dataset_root(group, target)
            available = sorted({"_".join(p.name.split("_")[:2]) for p in root.glob("*_Boston_*graphs.pt")})
            missing = set(YEARS[group]) - set(available)
            require(not missing, f"{group}/{target}: target year files missing: {sorted(missing)}")
            rows.append(dict(group=group, target=target, target_root=dataset_root(group, target),
                             requested_years=YEARS[group], requested_year_count=len(YEARS[group]),
                             available_years=available))
    return rows


def generate(destination, checkpoints):
    destination = Path(destination)
    rows = []
    for group, targets in TARGETS.items():
        directory = destination / group
        directory.mkdir(parents=True, exist_ok=True)
        year_list = ",".join(YEARS[group])
        (directory / "common.sh").write_text(
            '#!/usr/bin/env bash\nsource "$(dirname "${BASH_SOURCE[0]}")/../common.sh"\n\n'
            f'EXPERIMENT_GROUP="{group}"\nYEARS="{year_list}"\nEXPECTED_YEAR_COUNT={len(YEARS[group])}\n'
            f'INFERENCE_RESULTS_ROOT="{FORMAL}/CrossDatasetInference_dev/{group}"\n')
        for source in SOURCES:
            source_directory = directory / source
            source_directory.mkdir(exist_ok=True)
            expected_names = {f"{source}_to_{target}.sh" for target in targets}
            unexpected = {p.name for p in source_directory.iterdir()} - expected_names
            require(not unexpected, f"{source_directory}: unexpected files: {sorted(unexpected)}")
            for target in targets:
                checkpoint = str(checkpoints[group, source])
                require(Path(checkpoint).is_absolute() and Path(checkpoint).name == "best_overall.pt",
                        f"Expected an absolute best_overall.pt path: {checkpoint}")
                source_root, target_root = dataset_root(group, source), dataset_root(group, target)
                relative = Path(group) / source / f"{source}_to_{target}.sh"
                (destination / relative).write_text(
                    '#!/usr/bin/env bash\nsource "$(dirname "${BASH_SOURCE[0]}")/../common.sh"\n\n'
                    f'NAME="{source}_to_{target}_{group}"\n'
                    f'SOURCE_NAME="{source}"\nTARGET_NAME="{target}"\n\n'
                    f'CKPT_PATH={shlex.quote(checkpoint)}\n\n'
                    f'SOURCE_ROOT="{source_root}"\nTARGET_ROOT="{target_root}"\n\n'
                    'ROOT_DIR="${SOURCE_ROOT}"\nTEST_ROOT_DIR="${TARGET_ROOT}"\n')
                rows.append(dict(group=group, source=source, target=target, checkpoint=checkpoint,
                                 source_root=source_root, target_root=target_root, years=year_list,
                                 year_count=len(YEARS[group]), config_path=f"configs/cross_dataset_infer/{relative}",
                                 threshold_origin="source_checkpoint_train"))
    with (destination / "manifest.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    require(len(rows) == 66, "Expected 66 configs")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Audit checkpoints/targets and compare generated files without writing.")
    args = parser.parse_args()
    import torch
    torch.set_num_threads(1)
    checkpoints = {(group, source): (FORMAL / run / "best_overall.pt").resolve()
                   for group, sources in RUNS.items() for source, run in sources.items()}
    audits = [audit_checkpoint(group, source, path) for (group, source), path in checkpoints.items()]
    target_audit = audit_targets()
    evidence = json.dumps(dict(sources=audits, target_year_files=target_audit), indent=2) + "\n"
    if args.check:
        import tempfile
        with tempfile.TemporaryDirectory() as temporary:
            generate(temporary, checkpoints)
            for path in sorted(Path(temporary).rglob("*")):
                if path.is_file():
                    actual = DESTINATION / path.relative_to(temporary)
                    require(actual.is_file() and actual.read_bytes() == path.read_bytes(), f"Generated file differs: {actual}")
        require((DESTINATION / "checkpoint_audit.json").read_text() == evidence, "Checkpoint audit differs; inspect before regenerating")
    else:
        generate(DESTINATION, checkpoints)
        (DESTINATION / "checkpoint_audit.json").write_text(evidence)
    for row in audits:
        counts = "/".join(str(row["split_group_counts"][s]) for s in ("train", "val", "test"))
        print(f'{row["group"]}/{row["source"]}: epoch={row["epoch"]} Boston GraphSAGE Transformer single '
              f'24h zscore tau={row["tau_physical"]:.17g} Q95 groups={counts}\n  {row["checkpoint"]}')
    print("PASS: 12 audited checkpoints; 36 past-only + 30 future-year = 66 configs; years 7 / 30. No inference run.")


if __name__ == "__main__":
    main()
