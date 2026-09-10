#!/usr/bin/env python3
"""Validate all P0 configs through train.sh DRY_RUN=1; never call train.main."""

import argparse
from collections import Counter
import csv
from dataclasses import fields
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import tempfile
from unittest.mock import patch


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
from emulator.models.heads import SingleHead
from emulator.training.arguments import parse_args
from emulator.training.losses import ForecastLoss, LossConfig
import torch


PYTHON = "/media/volume/PACT-Data/conda_envs/torchpyg-cu124/bin/python"
STATIONS = ("CBBT", "Lewes", "Battery", "Boston")
MODES = {"mse": "mse", "tail": "mse_tail", "slope": "mse_slope", "tail_slope": "mse_tail_slope"}
SCALARS = dict(
    TRAIN_PY="train.py", ROOT_DIR="/media/share/PACT/Data/Grid4_New/NCEP/graphs",
    STATION_JSON_DIR="/media/volume/PACT-Data/StormSurge/station_json", TEST_ROOT_DIR="",
    MODEL="perceiver3", ENCODER_TYPE="GraphSAGE", CNN_INTERMEDIATE_CHANNEL="29",
    TEMPORAL_BLOCK="Transformer", HIDDEN_CHANNELS="128", NUM_LAYERS="2", DROPOUT="0.05",
    HEAD_DROPOUT="0.05", NODE_READ_HEADS="8", TIME_READ_HEADS="8", TRANSFORMER_LAYERS="2",
    TRANSFORMER_FF_MULT="4.0", TRANSFORMER_DROPOUT="0.0", MAX_TIME_STEPS="32",
    TRAIN_RATIO="0.6", VAL_RATIO="0.2", SHUFFLE_YEARS="0", FUTURE_ONLY="0",
    FUTURE_YEAR_THRESHOLD="2030", SEED="42", USE_SITE_ELEVATION="0", USE_BATHYMETRY="0",
    GATE_MODE="window", DUAL_MODE="exceedance", DUAL_ABLATION="none", EXCEEDANCE_PERCENTILE="95",
    BODY_LOSS_WEIGHT="1", EXCESS_LOSS_WEIGHT="2", GATE_LOSS_WEIGHT="0.5", TAIL_FRAC="0.05",
    SLOPE_ROBUST="charb", SLOPE_CHARB_EPS="1e-3", SLOPE_HUBER_DELTA="0.05",
    WMSE_ALPHA="4.0", WMSE_S="0.10", WMSE_USE_ABS="1",
    DISABLE_OOD="1", X_NORM="zscore", X_CLIP="0", X_AUG="0", X_AUG_PROB="0",
    X_AUG_SCALE="0", X_AUG_BIAS="0", X_P_LO="1.0", X_P_HI="99.0", X_NODES_PER_GRAPH="256",
    BATCH_SIZE="256", GRAD_ACCUM_STEPS="4", EPOCHS="300", SCHEDULER="cosine",
    WARMUP_EPOCHS="5", WARMUP_START_FACTOR="0.1", MIN_LR="1e-6", MAX_GRAD_NORM="0",
    DETERMINISTIC="0", ROP_METRIC="val_rmse_phys", ROP_FACTOR="0.5", ROP_PATIENCE="20",
    ROP_THRESHOLD="1e-4", ROP_COOLDOWN="0", ROP_MIN_LR="1e-6",
    num_gpus="1", CUDA_VISIBLE_DEVICES="0", USE_AMP="1", AMP_DTYPE="bf16", USE_TF32="1",
    TORCH_THREADS="1", NUM_WORKERS="0", PIN_MEMORY="0", PERSISTENT_WORKERS="0",
    PREFETCH_FACTOR="0", MP_CONTEXT="fork", PYTHON_BIN=PYTHON, DO_CONDA="0",
    CONDA_ENV="/media/volume/PACT-Data/conda_envs/torchpyg-cu124", PYTHONDONTWRITEBYTECODE="1",
    USE_TMUX="1", ALL_RESULTS_ROOT="./All_Results/P0_QuickRun", RUN_DIR_NAME_STYLE="runname_timestamp",
)
ARRAYS = dict(HISTORY_HOURS_LIST=["24"], LR_LIST=["5e-3"], TAIL_LAMBDA_LIST=["0.025"],
              SLOPE_LAMBDA_LIST=["0.01"], SLOPE_MASK_S_LIST=["0.10"], WMSE_Q_LIST=["95"])


def run(command, *, env=None, cwd=REPO, check=True):
    result = subprocess.run(command, cwd=cwd, env=env, text=True, capture_output=True, timeout=30)
    if check and result.returncode:
        raise AssertionError(f"Command failed: {command}\n{result.stdout}\n{result.stderr}")
    return result


def source_values(config, names):
    # Empty environment proves the config does not need an activated shell or another config.
    script = '''set -euo pipefail
source "$1"
shift
for name in "$@"; do
    declare -p "$name" >/dev/null
    declare -n value="$name"
    items=("${value[@]}")
    printf '%s\\0' "$name" "${#items[@]}" "${items[@]}"
    unset -n value
done
'''
    result = run(["bash", "--noprofile", "--norc", "-c", script, "bash", str(config), *names],
                 env={"PATH": "/usr/bin:/bin", "SESSION_NAME": "P0_Test"})
    words = iter(result.stdout.rstrip("\0").split("\0"))
    values = {}
    for name in words:
        count = int(next(words))
        values[name] = [next(words) for _ in range(count)]
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", type=Path, help="Save evidence here; defaults to a temporary directory")
    options = parser.parse_args()
    evidence = options.report_dir or Path(tempfile.mkdtemp(prefix="p0-config-validation-"))
    evidence.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    configs = sorted(HERE.glob("train_config_*.sh"))
    assert len(configs) == len(list(HERE.glob("*.sh"))) == 32
    with (HERE / "manifest.tsv").open() as handle:
        manifest = list(csv.DictReader(handle, delimiter="\t"))
    assert len(manifest) == 32
    by_config = {row["config"]: row for row in manifest}
    assert len(by_config) == 32
    assert set(by_config) == {str(p.relative_to(REPO)) for p in configs}
    for path in [REPO / "train.sh", *configs, Path.home() / ".local/bin/qsub_local_worker"]:
        run(["bash", "-n", str(path)])
    for path in HERE.glob("*.py"):
        compile(path.read_text(), str(path), "exec")
    run(["bash", "-n", str(Path.home() / ".bashrc")])
    output_root = REPO / "All_Results/P0_QuickRun"
    before = sorted(str(p) for p in output_root.rglob("*")) if output_root.exists() else None
    environment = {k: v for k, v in os.environ.items()
                   if not k.startswith(("SLURM_", "BASH_FUNC_"))}
    environment.update(DRY_RUN="1", SESSION_NAME="P0_Test", PACT_RUN_NAME="must_not_replace_config",
                       PACT_RUNSTAMP="20000101_000000", PYTHONDONTWRITEBYTECODE="1")
    reports, dry_logs = [], []
    for config in configs:
        relative = str(config.relative_to(REPO))
        row = by_config[relative]
        match = re.fullmatch(r"train_config_NCEP_(CBBT|Lewes|Battery|Boston)_24h_(single|dual)_(mse|tail|slope|tail_slope)\.sh", config.name)
        assert match, config
        station, head, variant = match.groups()
        mode = MODES[variant]
        name = f"NCEP_{station}_24h_{head}_{variant}"
        label = f"P0_{station}_{head}_{variant}"
        expected_row = dict(config=relative, station=station, head=head, loss_mode=mode,
                            tail=str(int("tail" in variant)), slope=str(int("slope" in variant)),
                            history="24", lr="5e-3", run_name=name, recommended_qsub_label=label,
                            qsub_command=f"qsub_local train.sh {label} {relative}")
        assert row == expected_row, row
        content = config.read_text()
        assert os.access(config, os.X_OK)
        assert not re.search(r"^\s*(?:source|\.)\s", content, re.M)
        assert not re.search(r"STABILITY_GUARD|STABLE_ARCH|ALPHA_INIT_LOGIT|TAIL_TANH_CLIP|GATE_BIAS_INIT|no_gate_bce|no_excess_loss|no_branch_supervision|fixed_gate", content)
        expected = {key: [value] for key, value in SCALARS.items()}
        expected.update(ARRAYS, STATION=[station], HEAD_TYPE=[head], DUAL_LOSS=[str(int(head == "dual"))],
                        LOSS_MODE_LIST=[mode], PACT_RUN_NAME=[name], SESSION_NAME=["P0_Test"])
        values = source_values(config, list(expected))
        assert values == expected, {k: (values.get(k), v) for k, v in expected.items() if values.get(k) != v}
        result = run(["bash", "train.sh", relative], env=environment)
        dry_logs.append(f"### {relative}\n{result.stdout}")
        commands = [shlex.split(line[5:]) for line in result.stdout.splitlines() if line.startswith("CMD: ")]
        assert len(commands) == 1 and commands[0][0] == "train.py", commands
        command = commands[0][1:]
        args = parse_args(command)  # Production parser only; no train.py entry point.
        expected_args = dict(root_dir=SCALARS["ROOT_DIR"], station_json_dir=Path(SCALARS["STATION_JSON_DIR"]),
                             test_root_dir="", station=station, model="perceiver3", encoder_type="GraphSAGE",
                             cnn_intermediate_channel=29, temporal_block="Transformer", head_type=head,
                             history_hours=24, hidden_channels=128, num_layers=2, dropout=.05, head_dropout=.05,
                             node_read_heads=8, time_read_heads=8, transformer_layers=2, transformer_ff_mult=4.,
                             transformer_dropout=0., max_time_steps=32, train_ratio=.6, val_ratio=.2,
                             shuffle_years=0, future_only=0, future_year_threshold=2030, seed=42,
                             use_site_elevation=0, use_bathymetry=0, batch_size=256, grad_accum_steps=4,
                             lr=.005, epochs=300, scheduler="cosine", warmup_epochs=5, warmup_start_factor=.1,
                             min_lr=1e-6, max_grad_norm=0., deterministic=0, torch_threads=1,
                             num_workers=0, pin_memory=False, persistent_workers=False, prefetch_factor=0,
                             mp_context="fork", amp=True, amp_dtype="bf16", tf32=True,
                             x_norm="zscore", x_clip=0., x_aug=0, x_aug_prob=0., x_aug_scale=0., x_aug_bias=0.,
                             loss_mode=mode, tail_frac=.05, wmse_q=95., wmse_alpha=4., wmse_s=.1, wmse_use_abs=1,
                             dual_mode="exceedance", gate_mode="window", dual_ablation="none",
                             exceedance_percentile=95., dual_loss=int(head == "dual"))
        if "tail" in variant:
            expected_args["tail_lambda"] = .025
        if "slope" in variant:
            expected_args.update(slope_lambda=.01, slope_mask_s=.1, slope_robust="charb",
                                 slope_charb_eps=.001, slope_huber_delta=.05)
        if head == "dual":
            expected_args.update(body_loss_weight=1., excess_loss_weight=2., gate_loss_weight=.5)
        for key, value in expected_args.items():
            assert getattr(args, key) == value, (relative, key, getattr(args, key), value)
        assert Path(args.output_dir).resolve() == output_root / f"{name}__20000101_000000"
        for marker in ("TRAIN_DATA_TAG:NCEP", "num_gpus:      1", "USE_TMUX=1 SESSION_NAME=P0_Test DRY_RUN=1",
                       "DISABLE_OOD:   1 (x_norm=zscore, x_clip=0, x_aug=0)",
                       f"Python:        {PYTHON} (DO_CONDA=0)", f"LAUNCHER:      {PYTHON} -u",
                       "Loss weights:  body=1 excess=2 gate=0.5 tail=0.025 slope=0.01",
                       "Run dir style: runname_timestamp"):
            assert marker in result.stdout, (relative, marker)
        assert "launching inside tmux" not in result.stdout
        if head == "single":
            assert all(flag not in command for flag in ("--dual_loss", "--body_loss_weight", "--excess_loss_weight", "--gate_loss_weight"))
            loss_config = LossConfig(**{field.name: getattr(args, field.name) for field in fields(LossConfig)})
            # Fixed requested weights remain harmless on a real SingleHead output.
            loss_config.body_loss_weight, loss_config.excess_loss_weight, loss_config.gate_loss_weight = 1., 2., .5
            criterion = ForecastLoss(loss_config, dict(y_mean=torch.zeros(6), y_std=torch.ones(6)), .8, .7)
            with torch.no_grad(), patch("emulator.training.losses.dual_loss_terms", side_effect=AssertionError("Single head entered dual auxiliary losses")):
                output = SingleHead(128, .05).eval()(torch.zeros(2, 6, 128))
                assert output.body is output.excess is output.gate_logits is None
                assert torch.isfinite(criterion(output, output.prediction, torch.zeros(2, 6)))
        reports.append(dict(**row, sha256=hashlib.sha256(config.read_bytes()).hexdigest(),
                            commands=1, standalone=True, configured_weights=dict(body=1, excess=2, gate=.5, tail=.025, slope=.01),
                            active_terms=dict(dual=head == "dual", tail="tail" in variant, slope="slope" in variant),
                            resolved_args=vars(args)))
    after = sorted(str(p) for p in output_root.rglob("*")) if output_root.exists() else None
    assert after == before, "DRY_RUN created or changed result artifacts"
    assert Counter(r["station"] for r in reports) == Counter({s: 8 for s in STATIONS})
    assert Counter(r["head"] for r in reports) == Counter(single=16, dual=16)
    designs = Counter(f'{r["head"]}/{r["loss_mode"]}' for r in reports)
    assert len(designs) == 8 and set(designs.values()) == {4}
    # Naming fallback and DDP guards use only temporary config text and dry commands.
    with tempfile.TemporaryDirectory(prefix="p0-launcher-checks-") as temporary:
        legacy = Path(temporary) / "legacy.sh"
        legacy.write_text('MODEL=perceiver3\nDO_CONDA=0\nUSE_TMUX=0\nnum_gpus=1\n')
        clean = dict(environment)
        clean.pop("PACT_RUN_NAME")
        result = run(["bash", "train.sh", str(legacy)], env=clean)
        assert "20000101_000000_P0_Test" in result.stdout and "Run dir style: timestamp_runname" in result.stdout
        legacy.write_text('MODEL=perceiver3\nDO_CONDA=0\nUSE_TMUX=0\nnum_gpus=2\nGRAD_ACCUM_STEPS=4\n')
        clean.pop("CUDA_VISIBLE_DEVICES", None)
        result = run(["bash", "train.sh", str(legacy)], env=clean, check=False)
        assert result.returncode != 0 and "supported only with num_gpus=1" in result.stdout
    report = dict(status="PASS", configs=32, command_count=32, standalone_configs=32,
                  stations=dict(Counter(r["station"] for r in reports)), heads=dict(Counter(r["head"] for r in reports)),
                  designs=dict(designs), single_head_auxiliary_path_checks=16, shell_syntax="PASS",
                  helper_python_syntax="PASS", legacy_naming="PASS", accumulation_single_gpu_guard="PASS",
                  dry_run_artifacts_created=False, real_training_started=False,
                  inactive_parameter_note="Production launcher omits inactive branch weights and inactive tail/slope flags. Parser defaults for these unused fields are retained; config_used.sh and configured_weights record the frozen P0 settings. Active losses use exactly the frozen weights.",
                  experiments=reports)
    (evidence / "config_validation.json").write_text(json.dumps(report, indent=2, default=str) + "\n")
    (evidence / "dry_runs.txt").write_text("\n".join(dry_logs))
    print(f"PASS: 32 standalone configs, 32 production dry commands, 16 single-head loss-path checks; no training. Evidence: {evidence}")


if __name__ == "__main__":
    main()
