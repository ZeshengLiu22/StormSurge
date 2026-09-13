#!/usr/bin/env python3
"""Validate P1 declarations and production DRY_RUN commands without training."""

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import tempfile


HERE = Path(__file__).resolve().parent
STATIONS = ("CBBT", "Lewes", "Battery", "Boston")
SETTINGS = {
    "ex5_notail": ("5", "0", "mse"),
    "ex10_notail": ("10", "0", "mse"),
    "ex2_tail005": ("2", "0.05", "mse_tail"),
    "ex2_tail010": ("2", "0.10", "mse_tail"),
    "ex5_tail005": ("5", "0.05", "mse_tail"),
    "ex5_tail010": ("5", "0.10", "mse_tail"),
    "ex10_tail005": ("10", "0.05", "mse_tail"),
    "ex10_tail010": ("10", "0.10", "mse_tail"),
}
COLUMNS = ["config", "station", "excess_weight", "tail_lambda", "loss_mode",
           "history", "lr", "run_name", "recommended_qsub_label", "qsub_command"]
STAMP = "20000101_000000"
PYTHON = "/media/volume/PACT-Data/conda_envs/torchpyg-cu124/bin/python"


def run(command, repo, env=None):
    result = subprocess.run(command, cwd=repo, env=env, text=True,
                            capture_output=True, timeout=30)
    assert result.returncode == 0, (command, result.stdout, result.stderr)
    return result


def declarations(config):
    """Require standalone assignments, with no source statements or other code."""
    result = {}
    for line in config.read_text().splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        match = re.fullmatch(r"(?:export )?([A-Za-z_]\w*)=.+", line)
        assert match, (config, "Expected a self-contained assignment", line)
        name = match.group(1)
        assert name not in result, (config, "Duplicate setting", name)
        result[name] = line
    return result


def source_values(config, names, repo):
    script = r'''set -euo pipefail
source "$1"
shift
for name in "$@"; do
    declare -p "$name" >/dev/null
    declare -n value="$name"
    items=("${value[@]}")
    printf '%s\0' "$name" "${#items[@]}" "${items[@]}"
    unset -n value
done
'''
    result = run(["bash", "--noprofile", "--norc", "-c", script,
                  "bash", str(config), *names], repo, env={"PATH": "/usr/bin:/bin"})
    words = iter(result.stdout.removesuffix("\0").split("\0"))
    values = {}
    for name in words:
        values[name] = [next(words) for _ in range(int(next(words)))]
    return values


def dry_run(config, repo, parse_args):
    # No inherited shell functions, scheduler variables or user overrides.
    environment = dict(PATH="/usr/bin:/bin", DRY_RUN="1", SESSION_NAME="P1_Validation",
                       PACT_RUN_NAME="must_not_replace_config", PACT_RUNSTAMP=STAMP,
                       PYTHONDONTWRITEBYTECODE="1")
    argument = str(config.relative_to(repo)) if config.is_relative_to(repo) else str(config)
    result = run(["bash", "train.sh", argument], repo, environment)
    commands = [shlex.split(line[5:]) for line in result.stdout.splitlines()
                if line.startswith("CMD: ")]
    assert len(commands) == 1 and commands[0][0] == "train.py", (config, commands)
    assert "launching inside tmux" not in result.stdout, config
    # Import only the production argument parser; never import/execute train.py.
    return vars(parse_args(commands[0][1:])), commands[0][1:], result.stdout


def result_snapshot(root):
    return dict(exists=root.exists(), entries=sorted(
        (str(p.relative_to(root)), p.stat().st_size, p.stat().st_mtime_ns)
        for p in root.rglob("*")
    ))


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=HERE.parents[1],
                        help="Production repository (override when validating staged configs)")
    parser.add_argument("--report-dir", type=Path,
                        help="Evidence destination; defaults to a temporary directory")
    options = parser.parse_args()
    repo = options.repo.resolve()
    evidence = options.report_dir or Path(tempfile.mkdtemp(prefix="p1-config-validation-"))
    evidence.mkdir(parents=True, exist_ok=True)
    sys.dont_write_bytecode = True
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    sys.path.insert(0, str(repo))
    from emulator.training.arguments import parse_args

    configs = sorted(HERE.glob("train_config_*.sh"))
    assert len(configs) == len(list(HERE.glob("*.sh"))) == 32
    with (HERE / "manifest.tsv").open() as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        assert reader.fieldnames == COLUMNS, reader.fieldnames
        manifest = list(reader)
    assert len(manifest) == 32, len(manifest)
    by_config = {row["config"]: row for row in manifest}
    assert len(by_config) == 32
    assert set(by_config) == {f"experiment_config/P1_WeightProbe/{p.name}" for p in configs}
    assert len({row["run_name"] for row in manifest}) == 32
    assert len({row["recommended_qsub_label"] for row in manifest}) == 32
    for path in [repo / "train.sh", *configs]:
        run(["bash", "-n", str(path)], repo)
    compile(Path(__file__).read_text(), __file__, "exec")

    output_root = repo / "All_Results/P1_WeightProbe"
    before = result_snapshot(output_root)
    protocol_paths = [repo / "train.sh", repo / "train.py", *(repo / "emulator").rglob("*.py")]
    protocol_hashes = {str(p.relative_to(repo)): sha256(p) for p in sorted(protocol_paths)}
    references, reference_logs = {}, []
    for station in STATIONS:
        for variant, mode, active_tail in (("mse", "mse", 0), ("tail", "mse_tail", .025)):
            path = repo / f"experiment_config/P0_QuickRun/train_config_NCEP_{station}_24h_dual_{variant}.sh"
            declared = declarations(path)
            values = source_values(path, list(declared), repo)
            args, _, log = dry_run(path, repo, parse_args)
            assert args["head_type"] == "dual" and args["excess_loss_weight"] == 2
            assert args["loss_mode"] == mode
            if active_tail:
                assert args["tail_lambda"] == active_tail
            references[station, mode] = dict(path=path, declared=declared, values=values,
                                             args=args, sha256=sha256(path))
            reference_logs.append(f"### {path.relative_to(repo)} (reference DRY_RUN only)\n{log}")

    reports, dry_logs = [], []
    for config in configs:
        relative = f"experiment_config/P1_WeightProbe/{config.name}"
        row = by_config[relative]
        station = row["station"]
        assert station in STATIONS, station
        prefix = f"train_config_NCEP_{station}_24h_dual_"
        assert config.name.startswith(prefix), config
        tag = config.name.removeprefix(prefix).removesuffix(".sh")
        assert tag in SETTINGS, tag
        excess, tail, mode = SETTINGS[tag]
        name = f"NCEP_{station}_24h_dual_{tag}"
        label = f"P1_{station}_dual_{tag}"
        expected_row = dict(config=relative, station=station, excess_weight=excess,
                            tail_lambda=tail, loss_mode=mode, history="24", lr="5e-3",
                            run_name=name, recommended_qsub_label=label,
                            qsub_command=f"qsub_local train.sh {label} {relative}")
        assert row == expected_row, (row, expected_row)
        assert os.access(config, os.X_OK), config

        reference = references[station, mode]
        # These are the only declarations allowed to differ from the matching P0 anchor.
        overrides = dict(EXCESS_LOSS_WEIGHT=excess, TAIL_LAMBDA_LIST=f'("{tail}")',
                         SLOPE_LAMBDA_LIST='("0")', SESSION_NAME=f'"${{SESSION_NAME:-{label}}}"',
                         PACT_RUN_NAME=f'"{name}"', ALL_RESULTS_ROOT='"./All_Results/P1_WeightProbe"')
        expected_declarations = dict(reference["declared"])
        expected_declarations.update({key: f"{key}={value}" for key, value in overrides.items()})
        assert declarations(config) == expected_declarations, (config, "P0 protocol drift")
        expected_values = dict(reference["values"])
        expected_values.update(EXCESS_LOSS_WEIGHT=[excess], TAIL_LAMBDA_LIST=[tail],
                               SLOPE_LAMBDA_LIST=["0"], SESSION_NAME=[label], PACT_RUN_NAME=[name],
                               ALL_RESULTS_ROOT=["./All_Results/P1_WeightProbe"])
        values = source_values(config, list(expected_declarations), repo)
        assert values == expected_values, (config, "Standalone values differ from P0 protocol")
        for key, value in dict(HEAD_TYPE="dual", GATE_MODE="window", DUAL_MODE="exceedance",
                               DUAL_LOSS="1", DUAL_ABLATION="none", EXCEEDANCE_PERCENTILE="95",
                               MODEL="perceiver3", ENCODER_TYPE="GraphSAGE", TEMPORAL_BLOCK="Transformer",
                               HISTORY_HOURS_LIST="24", LR_LIST="5e-3", BODY_LOSS_WEIGHT="1",
                               GATE_LOSS_WEIGHT="0.5", TAIL_FRAC="0.05", SLOPE_LAMBDA_LIST="0",
                               DISABLE_OOD="1", X_NORM="zscore", X_CLIP="0", X_AUG="0",
                               X_AUG_PROB="0", X_AUG_SCALE="0", X_AUG_BIAS="0").items():
            assert values[key] == [value], (config, key, values[key])
        assert all(len(value) == 1 for key, value in values.items() if key.endswith("_LIST")), config

        args, command, log = dry_run(config, repo, parse_args)
        expected_args = dict(reference["args"])
        expected_args.update(excess_loss_weight=float(excess),
                             output_dir=f"{repo}/./All_Results/P1_WeightProbe/{name}__{STAMP}")
        if mode == "mse_tail":
            expected_args["tail_lambda"] = float(tail)
            expected_args["run_tag"] = expected_args["run_tag"].replace("_tl0.025_", f"_tl{tail}_")
            assert "--tail_lambda" in command, config
        else:
            assert "--tail_lambda" not in command, config
        assert args == expected_args, (config, {key: (args.get(key), value)
                                              for key, value in expected_args.items() if args.get(key) != value})
        assert args["loss_mode"] in ("mse", "mse_tail")
        assert not any(flag.startswith("--slope_") for flag in command), config
        assert Path(args["output_dir"]).resolve() == output_root / f"{name}__{STAMP}"
        for marker in (f"Run name:      {name}", "num_gpus:      1",
                       "USE_TMUX=1 SESSION_NAME=P1_Validation DRY_RUN=1",
                       "DISABLE_OOD:   1 (x_norm=zscore, x_clip=0, x_aug=0)",
                       f"Loss weights:  body=1 excess={excess} gate=0.5 tail={tail} slope=0",
                       f"Python:        {PYTHON} (DO_CONDA=0)", f"LAUNCHER:      {PYTHON} -u",
                       "Run dir style: runname_timestamp"):
            assert marker in log, (config, marker)
        dry_logs.append(f"### {relative}\n{log}")
        reports.append(dict(**row, tag=tag, sha256=sha256(config), commands=1, standalone=True,
                            frozen_p0_protocol="PASS", p0_reference=str(reference["path"].relative_to(repo)),
                            configured_weights=dict(body=1, excess=float(excess), gate=.5,
                                                    tail=float(tail), slope=0),
                            active_terms=dict(dual=True, tail=mode == "mse_tail", slope=False),
                            resolved_args=args))

    assert Counter(r["station"] for r in reports) == Counter({station: 8 for station in STATIONS})
    for station in STATIONS:
        station_reports = [r for r in reports if r["station"] == station]
        assert {r["tag"] for r in station_reports} == set(SETTINGS)
        assert {(r["excess_weight"], r["tail_lambda"], r["loss_mode"]) for r in station_reports} == set(SETTINGS.values())
    assert result_snapshot(output_root) == before, "DRY_RUN changed P1 result artifacts"
    assert all(sha256(repo / path) == digest for path, digest in protocol_hashes.items())
    assert all(sha256(ref["path"]) == ref["sha256"] for ref in references.values())
    report = dict(status="PASS", validated_utc=datetime.now(timezone.utc).isoformat(),
                  configs=32, manifest_rows=32, command_count=32, standalone_configs=32,
                  stations=dict(Counter(r["station"] for r in reports)),
                  settings={tag: dict(excess_weight=ex, tail_lambda=tail, loss_mode=mode)
                            for tag, (ex, tail, mode) in SETTINGS.items()},
                  shell_syntax="PASS", python_syntax="PASS", frozen_p0_protocol="PASS",
                  unique_run_directories=32, slope_disabled=True, ood_disabled=True,
                  p0_anchor_runs_duplicated=0, p0_reference_dry_commands=8,
                  dry_run_artifacts_created=False, real_training_started=False,
                  inactive_parameter_note=("Production train.sh omits inactive tail/slope flags. "
                                           "Parser defaults in resolved_args are inactive because loss_mode "
                                           "is mse or mse_tail; configured_weights records the explicit P1 zeros."),
                  protocol_sha256=protocol_hashes,
                  p0_reference_sha256={str(ref["path"].relative_to(repo)): ref["sha256"]
                                       for ref in references.values()},
                  experiments=reports)
    (evidence / "config_validation.json").write_text(json.dumps(report, indent=2, default=str) + "\n")
    (evidence / "dry_runs.txt").write_text("\n".join(dry_logs))
    (evidence / "p0_reference_dry_runs.txt").write_text("\n".join(reference_logs))
    print(f"PASS: 32 standalone P1 configs, 8 per station, 32 production dry commands; "
          f"frozen P0 protocol preserved; no training. Evidence: {evidence}")


if __name__ == "__main__":
    main()
