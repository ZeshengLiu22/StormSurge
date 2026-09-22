#!/usr/bin/env python3
"""Generate matched S0/D0/D1/D2/D3 configs without launching training."""

import argparse
import csv
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
STATIONS = ("CBBT", "Lewes", "Battery", "Boston")
VARIANTS = (
    ("S0", "Single", "single", 0.0, 0.0),
    ("D0", "DualBase", "dual", 0.0, 0.0),
    ("D1", "Exceedance", "dual", 0.025, 0.0),
    ("D2", "Amp", "dual", 0.0, 0.003),
    ("D3", "ExceedanceAmp", "dual", 0.025, 0.003),
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


def generate(output, *, include_severity_shape=False, results_root="./All_Results"):
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
    parser.add_argument("--output", type=Path, default=REPO / "configs/current")
    parser.add_argument("--include-severity-shape", action="store_true")
    parser.add_argument("--results-root", default="./All_Results")
    args = parser.parse_args(argv)
    rows = generate(args.output, include_severity_shape=args.include_severity_shape,
                    results_root=args.results_root)
    print(f"Generated {len(rows)} configs in {args.output}. No training launched.")


if __name__ == "__main__":
    main()
