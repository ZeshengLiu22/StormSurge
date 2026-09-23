#!/usr/bin/env python3
"""Generate matched current, WQE placement, or fresh WQE/Tail factorial configs without training."""

import argparse
import csv
import itertools
import shlex
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
WQE_RESULTS_ROOT = "/home/exouser/media/share/PACT/WQE_Results"
FACTORIAL_RESULTS_ROOT = "/home/exouser/media/share/PACT/All_results_0922_wqe_factorial_multickpt"
FACTORIAL_CONFIG_DIR = "0922_wqe_factorial_multickpt"
FACTORIAL_STATIONS = ("CBBT", "Boston", "Battery", "Lewes")
STATIONS = ("CBBT", "Lewes", "Battery", "Boston")
VARIANTS = (
    ("S0", "Single", "single", 0.0, 0.0),
    ("D0", "DualBase", "dual", 0.0, 0.0),
    ("D1", "Exceedance", "dual", 0.025, 0.0),
    ("D2", "Amp", "dual", 0.0, 0.003),
    ("D3", "ExceedanceAmp", "dual", 0.025, 0.003),
)
WQE_VARIANTS = (
    ("W1", "GlobalWQE", "wqe", "mse"),
    ("W2", "ExcessWQE", "mse", "wqe"),
    ("W3", "BothWQE", "wqe", "wqe"),
)

TEMPLATE = '''#!/usr/bin/env bash
# Matched hourly exceedance formulation: {station} / {condition}.

TRAIN_PY="train.py"
DO_CONDA=0
num_gpus=1
USE_TMUX="${{USE_TMUX:-1}}"

ROOT_DIR="./Data/Grid4_New/NCEP/graphs"
TEST_ROOT_DIR=""
STATION="{station}"
MODEL="perceiver3"
ENCODER_TYPE="GraphSAGE"
CNN_INTERMEDIATE_CHANNEL=29
TEMPORAL_BLOCK="Transformer"
HEAD_TYPE="{head}"
HISTORY_HOURS_LIST=(24)
HIDDEN_CHANNELS=128
NUM_LAYERS=2
DROPOUT=0.05
HEAD_DROPOUT=0.05
NODE_READ_HEADS=8
TIME_READ_HEADS=8
TRANSFORMER_LAYERS=2
TRANSFORMER_FF_MULT=4.0
TRANSFORMER_DROPOUT=0.0
MAX_TIME_STEPS=32

TRAIN_RATIO="0.6"
VAL_RATIO="0.2"
SHUFFLE_YEARS=0
FUTURE_ONLY=0
FUTURE_YEAR_THRESHOLD=2030
SEED=42
STATION_JSON_DIR="./station_json"
USE_SITE_ELEVATION=0
USE_BATHYMETRY=0

X_NORM="zscore"
DISABLE_OOD=1
X_P_LO="1.0"
X_P_HI="99.0"
X_NODES_PER_GRAPH=0
X_CLIP=0
X_AUG=0
X_AUG_PROB=0
X_AUG_SCALE=0
X_AUG_BIAS=0

BATCH_SIZE=256
GRAD_ACCUM_STEPS=4
LR_LIST=("5e-3")
EPOCHS=300
SCHEDULER="cosine"
WARMUP_EPOCHS=5
WARMUP_START_FACTOR=0.1
MIN_LR=1e-6
MAX_GRAD_NORM=0
DETERMINISTIC=0
# Inactive ReduceLROnPlateau defaults.
ROP_METRIC="val_all_rmse"
ROP_FACTOR=0.5
ROP_PATIENCE=20
ROP_THRESHOLD=1e-4
ROP_COOLDOWN=0
ROP_MIN_LR=1e-6

USE_AMP=1
AMP_DTYPE="bf16"
USE_TF32=1
NUM_WORKERS=0
TORCH_THREADS=1
PIN_MEMORY=0
PERSISTENT_WORKERS=0
PREFETCH_FACTOR=0
MP_CONTEXT="fork"

LOSS_MODE_LIST=("mse")
EXCEEDANCE_LOSS_WEIGHT={exceedance_weight}
EXCESS_AMP_LOSS_WEIGHT={amp_weight}

# These head/branch settings are inactive for Single.
DUAL_MODE="exceedance"
EXCESS_FORMULATION="{formulation}"
EXCEEDANCE_HEAD_EXPERIMENT=""
EXCEEDANCE_GATE_POOLING="mean"
GATE_MODE="window"
DUAL_ABLATION="none"
DUAL_LOSS={dual_loss}
EXCEEDANCE_PERCENTILE=95
BODY_LOSS_WEIGHT=1
EXCESS_LOSS_WEIGHT={excess_weight}
GATE_LOSS_WEIGHT={gate_weight}
SHAPE_LOSS_WEIGHT={shape_weight}
SEVERITY_SHAPE_EPS=1e-6

CHECKPOINT_SELECTION="overall"
CKPT_W_ALL=0.65
CKPT_W_EXCEEDANCE=0.20
CKPT_W_PEAK=0.15

ALL_RESULTS_ROOT="{results_root}"
RUN_DIR_NAME_STYLE="runname_timestamp"
PACT_RUN_NAME="{run_name}"
PYTHON_RUN_TAG_BASE="{run_name}"
'''

WQE_TEMPLATE = TEMPLATE.replace(
    "# Matched hourly exceedance formulation:",
    "# WQE placement experiment (existing D0 is the MSE/MSE control):",
).replace('LOSS_MODE_LIST=("mse")', '''LOSS_MODE_LIST=("{loss_mode}")
EXCESS_LOSS_MODE="{excess_loss_mode}"
WQE_QUANTILE_TAU=0.25
WQE_EXPECTILE_TAU=0.82
WQE_QUANTILE_WEIGHT=0.16666666666666667
WQE_EXPECTILE_WEIGHT=0.83333333333333333''')


def generate_wqe(output, *, results_root=WQE_RESULTS_ROOT):
    """Use unchanged D0 settings to isolate global versus raw-excess WQE."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for station in STATIONS:
        for condition, label, loss_mode, excess_loss_mode in WQE_VARIANTS:
            name = f"NCEP_{station}_{condition}_{label}"
            config = output / f"train_config_{name}.sh"
            config.write_text(WQE_TEMPLATE.format(
                station=station, condition=condition, head="dual",
                loss_mode=loss_mode, excess_loss_mode=excess_loss_mode,
                exceedance_weight=0.0, amp_weight=0.0, formulation="direct",
                shape_weight=0, dual_loss=1, excess_weight=2, gate_weight=0.5,
                results_root=results_root, run_name=name))
            rows.append(dict(station=station, variant=condition, head_type="dual",
                loss_mode=loss_mode, excess_loss_mode=excess_loss_mode,
                wqe_quantile_tau=0.25, wqe_expectile_tau=0.82,
                wqe_quantile_weight=1.0 / 6.0, wqe_expectile_weight=5.0 / 6.0,
                excess_formulation="direct", exceedance_loss_weight=0.0,
                excess_amp_loss_weight=0.0, shape_loss_weight=0,
                body_loss_weight=1, excess_loss_weight=2, gate_loss_weight=0.5,
                config=config.name))
    with (output / "manifest.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0], lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return rows


def generate_wqe_factorial_multickpt(output, *, results_root=FACTORIAL_RESULTS_ROOT):
    """Fresh 4-station × global WQE × excess WQE × strict Tail-MSE factorial."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    template = WQE_TEMPLATE.replace(
        "WQE placement experiment (existing D0 is the MSE/MSE control)",
        "Fresh WQE/Tail factorial (six VAL-only checkpoint roles)")
    rows, all_commands = [], []
    # qsub_local is a shell function on the local host; load the interactive
    # shell initialization only when it is unavailable to the script.
    header = ("#!/usr/bin/env bash\n"
              'if ! command -v qsub_local >/dev/null 2>&1 && [[ $- != *i* ]]; then\n'
              '  exec bash -i "$0" "$@"\n'
              'fi\n'
              "set -euo pipefail\n"
              f"cd {shlex.quote(str(REPO))}\n\n")

    def write_launcher(name, commands):
        path = output / name
        path.write_text(header + "\n".join(commands) + "\n")
        path.chmod(0o755)

    for g, e, t in itertools.product((0, 1), repeat=3):
        cell = f"G{g}_E{e}_T{t}"
        global_loss, excess_loss = ("mse", "wqe")[g], ("mse", "wqe")[e]
        tail_weight = 0.025 if t else 0
        commands = []
        for station in FACTORIAL_STATIONS:
            name = f"NCEP_{station}_{cell}"
            config = output / f"train_config_{name}.sh"
            config.write_text(template.format(
                station=station, condition=cell, head="dual",
                loss_mode=global_loss, excess_loss_mode=excess_loss,
                exceedance_weight=tail_weight, amp_weight=0, formulation="direct",
                shape_weight=0, dual_loss=1, excess_weight=2, gate_weight=0.5,
                results_root=results_root, run_name=name))
            rows.append(dict(station=station, cell=cell, global_loss=global_loss,
                excess_loss=excess_loss, tail_enabled=t, exceedance_loss_weight=tail_weight,
                body_loss_weight=1, excess_loss_weight=2, gate_loss_weight=0.5,
                excess_amp_loss_weight=0, shape_loss_weight=0,
                config=config.name, result_root=str(results_root)))
            resolved = config.resolve()
            config_arg = resolved.relative_to(REPO) if resolved.is_relative_to(REPO) else resolved
            commands.append(f"qsub_local train.sh WQEF_{station}_G{g}E{e}T{t} {shlex.quote(str(config_arg))}")
        write_launcher(f"launch_G{g}E{e}T{t}.sh", commands)
        all_commands.extend(commands)
    write_launcher("launch_all.sh", all_commands)
    with (output / "manifest.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0], lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return rows


def generate(output, *, include_severity_shape=False, results_root=None, family="current"):
    if results_root is None:
        results_root = {"wqe": WQE_RESULTS_ROOT,
                        "wqe_factorial_multickpt": FACTORIAL_RESULTS_ROOT}.get(family, "./All_Results")
    if family == "wqe_factorial_multickpt":
        if include_severity_shape:
            raise ValueError("WQE factorial configs use the direct D0 excess formulation only.")
        return generate_wqe_factorial_multickpt(output, results_root=results_root)
    if family == "wqe":
        if include_severity_shape:
            raise ValueError("WQE placement configs use the direct D0 excess formulation only.")
        return generate_wqe(output, results_root=results_root)
    if family != "current":
        raise ValueError(f"Unknown config family: {family!r}")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    formulations = ("direct", "severity_shape") if include_severity_shape else ("direct",)
    for station in STATIONS:
        for formulation in formulations:
            for condition, label, head, exceedance, amplitude in VARIANTS:
                if formulation == "severity_shape" and head == "single":
                    continue
                suffix = "_SeverityShape" if formulation == "severity_shape" else ""
                name = f"NCEP_{station}_{condition}_{label}{suffix}"
                config = output / f"train_config_{name}.sh"
                dual = head == "dual"
                config.write_text(TEMPLATE.format(station=station, condition=condition,
                    head=head, exceedance_weight=exceedance, amp_weight=amplitude,
                    formulation=formulation, shape_weight=0, dual_loss=int(dual),
                    excess_weight=2 if dual else 1, gate_weight=0.5 if dual else 1,
                    results_root=results_root, run_name=name))
                rows.append(dict(station=station, variant=condition, head_type=head,
                    excess_formulation=formulation, exceedance_loss_weight=exceedance,
                    excess_amp_loss_weight=amplitude, shape_loss_weight=0,
                    body_loss_weight=1 if dual else 0, excess_loss_weight=2 if dual else 0,
                    gate_loss_weight=0.5 if dual else 0, config=config.name))
    with (output / "manifest.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0], lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=("current", "wqe", "wqe_factorial_multickpt"), default="current",
                        help="WQE adds W1/W2/W3; wqe_factorial_multickpt adds 32 fresh WQE/Tail runs.")
    parser.add_argument("--output", type=Path,
                        help="Defaults to configs/current, configs/wqe, or configs/0922_wqe_factorial_multickpt.")
    parser.add_argument("--include-severity-shape", action="store_true")
    parser.add_argument("--results-root",
                        help=f"Defaults to {FACTORIAL_RESULTS_ROOT} for the factorial, {WQE_RESULTS_ROOT} for WQE, otherwise ./All_Results.")
    args = parser.parse_args(argv)
    if args.family != "current" and args.include_severity_shape:
        parser.error(f"--family {args.family} cannot be combined with --include-severity-shape.")
    directory = FACTORIAL_CONFIG_DIR if args.family == "wqe_factorial_multickpt" else args.family
    output = args.output or REPO / "configs" / directory
    rows = generate(output, include_severity_shape=args.include_severity_shape,
                    results_root=args.results_root, family=args.family)
    print(f"Generated {len(rows)} configs in {output}. No training launched.")


if __name__ == "__main__":
    main()
