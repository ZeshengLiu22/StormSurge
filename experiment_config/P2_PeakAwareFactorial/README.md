# P2 — Peak-Aware Factorial

This in-domain, seed-42 mechanism screen tests tail trajectory supervision,
excess amplitude supervision, direct versus severity-shape parameterization,
explicit shape supervision, and final physical peak supervision. It precedes
the larger architecture sweep and final multi-seed evaluation.

There are **96 fully standalone configs**, with no runtime config inheritance:

| Station | Direct | Severity-shape | Total | Experiment IDs |
| --- | ---: | ---: | ---: | --- |
| CBBT | 8 | 16 | 24 | 1–24 |
| Lewes | 8 | 16 | 24 | 25–48 |
| Battery | 8 | 16 | 24 | 49–72 |
| Boston | 8 | 16 | 24 | 73–96 |
| Total | 32 | 64 | 96 | 1–96 |

Direct uses all `2 Tail × 2 Amp × 2 Peak` combinations, with Shape always OFF.
Severity-shape uses all `2 Tail × 2 Amp × 2 Shape × 2 Peak` combinations.
Every valid interaction combination occurs exactly once per station.

## Experimental weights

| Factor | Direct OFF / ON | Severity-shape OFF / ON |
| --- | --- | --- |
| Tail | MSE / MSE + active tail 0.025 | MSE / MSE + active tail 0.025 |
| Excess amplitude | 0 / 0.0007 | 0 / 0.0025 |
| Shape supervision | always 0 | 0 / 0.00003 |
| Final physical peak | 0 / 0.0035 | 0 / 0.0035 |

The different amplitude coefficients intentionally approximately match a
conservative initial output-gradient scale across formulations, based on the
TRAIN-only diagnostic. The peak coefficient is shared because the diagnostic
found nearly equal output-gradient scale for that objective. No weight scans
beyond the specified binary factors are included.

All configs declare `TAIL_LAMBDA_LIST=("0.025")`, following frozen P0 syntax.
Tail OFF is `LOSS_MODE_LIST=("mse")`; Tail ON is `LOSS_MODE_LIST=("mse_tail")`.
MSE never enters the tail objective. The launcher omits `--tail_lambda` for
MSE, so the parsed value remains an **inactive** production default, currently
0.10. Validation checks active semantics and does not require that field to
be zero. Tail ON must resolve an active coefficient of exactly 0.025.

`manifest.csv` uses `tail_lambda` for the **effective coefficient**: 0 for
inactive MSE and 0.025 for `mse_tail`. `configured_tail_lambda` records the
literal 0.025 declared by every config. `tail_enabled` is authoritative for
factor grouping. Full parsed values are recorded in the validation JSON.

The severity-shape formulation uses the current production
`severity_phys × normalized_shape` implementation. `SEVERITY_SHAPE_EPS=1e-6`.
Both amplitude and final-peak pools use hard `max`; both inactive smoothmax
betas remain 20. No direct config enables shape loss.

## Frozen P0 protocol

| Control group | Values |
| --- | --- |
| Model | `perceiver3`, GraphSAGE, Transformer; history 24 h |
| Spatial model | hidden channels 128; layers 2; dropout 0.05; inactive CNN intermediate channels 29 |
| Head | dual exceedance, window gate; dual loss 1; ablation `none`; head dropout 0.05 |
| Temporal model | node/time read heads 8/8; Transformer layers 2; FF multiplier 4.0; dropout 0; max time steps 32 |
| Frozen branch weights | body 1; excess 2; gate 0.5 |
| Events | TRAIN window-maximum exceedance percentile 95; tail fraction 0.05 |
| Split | train 0.6; validation 0.2; remaining years test; chronological years; shuffle 0; future-only 0; inactive future-year threshold 2030 |
| Seed | 42 for every config |
| Optimization | unchanged production Adam; LR 0.005; weight decay 1e-5; epochs 300 |
| Batch | 256; gradient accumulation 4; nominal effective batch 1024 |
| Scheduler | cosine; warmup 5; **warmup start factor 0.1**; min LR 1e-6 |
| Stability | max gradient norm 0; deterministic kernels 0 |
| Runtime | one GPU; CUDA device 0; AMP enabled, bf16; TF32 enabled; Torch threads 1 |
| Loader | workers 0; pin memory 0; persistent workers 0; prefetch 0; multiprocessing `fork` |
| OOD | disabled; zscore; clipping 0; augmentation, probability, scale and bias all 0 |
| Station features | site elevation 0; bathymetry 0; existing lat/lon-derived metadata retained |
| Checkpoints | `overall`: minimum VAL `rmse_all`; auxiliary checkpoints 0; overall tolerance 0.01 |

The P0 inactive controls are also explicit and unchanged: WMSE percentile 95,
alpha 4.0, softness 0.10, absolute weighting 1; slope lambda 0.01, mask 0.10,
Charbonnier, epsilon 1e-3, Huber delta 0.05; ROP metric `val_rmse_phys`, factor
0.5, patience 20, threshold 1e-4, cooldown 0, minimum LR 1e-6; robust-statistics
percentiles 1/99 and nodes per graph 256. No P2 loss mode enables slope or WMSE,
and cosine leaves ROP inactive.

Input paths and environment are copied explicitly from the validated P0 protocol:

```text
ROOT_DIR=/media/share/PACT/Data/Grid4_New/NCEP/graphs
TEST_ROOT_DIR=""
STATION_JSON_DIR=/media/volume/PACT-Data/StormSurge/station_json
PYTHON_BIN=/media/volume/PACT-Data/conda_envs/torchpyg-cu124/bin/python
CONDA_ENV=/media/volume/PACT-Data/conda_envs/torchpyg-cu124
DO_CONDA=0
```

Station metadata keeps the current six coordinate features: latitude/90,
longitude/180, and the sine/cosine of each coordinate in radians. The production
defaults `use_station_meta=1` and `device=auto` are unchanged; the launcher has
no config switch for these fields. All four station feature vectors are
validated. Python bytecode writing remains disabled, as in P0.

## Names, results and deferred commands

Config names follow:

```text
train_config_NCEP_<Station>_24h_dual_<direct|severity>_T<t>_A<a>_S<s>_P<p>.sh
```

`severity` in a filename denotes `EXCESS_FORMULATION="severity_shape"`.
Direct always has `S0`. Filenames encode factor levels, without floating weights.

Every config sets the semantic name in `PACT_RUN_NAME` and then declares:

```bash
PYTHON_RUN_TAG_BASE="${PACT_RUN_NAME}"
ALL_RESULTS_ROOT="./All_Results/P2_PeakAwareFactorial"
RUN_DIR_NAME_STYLE="runname_timestamp"
```

The result folder is `<semantic-name>__<execution-timestamp>` below that root.
Python metadata receives `<semantic-name>_<execution-timestamp>`. The launcher
refuses to overwrite an existing run directory. The validation timestamp
`20000101_000000` is supplied only by the validator; it is never frozen into
an experiment config or launch command.

`PYTHON_RUN_TAG_BASE` is an opt-in launcher setting with an empty default.
When absent or empty, it preserves the historical automatically generated tag
exactly. The optional setting is included in resolved launcher snapshots.
P0/P1 configs and their full Python command lines remain unchanged.

`commands_all.txt` contains 96 deferred commands. Each `commands_<Station>.txt`
contains 24 commands, using the locally supported syntax:

```text
qsub_local train.sh <label> <config_path>
```

These lists assume the repository root as the working directory and the existing
shell that defines `qsub_local`. They have **not been executed**. The existing
queue/tmux settings remain in the runnable configs for later authorized use;
`DRY_RUN=1` bypasses all launch, tmux and artifact-creation branches during validation.
See [RUN_ORDER.md](RUN_ORDER.md) for deterministic station grouping.

## Validation and reproducibility

[validation_report.md](validation_report.md) gives the complete result;
[validation_report.json](validation_report.json) records all 96 resolved CLI
namespaces, config hashes, coverage checks and P0 differences. Additional raw
dry-run and launcher-regression evidence is in `validation/`.

From the repository root, repeat config validation without training:

```bash
/media/volume/PACT-Data/conda_envs/torchpyg-cu124/bin/python -B \
  experiment_config/P2_PeakAwareFactorial/validate_configs.py
```

The validator writes fresh evidence to a temporary directory by default and
requires the saved passing launcher regression to match the current launcher.
`validate_launcher.py` can repeat all 64 P0/P1 command comparisons against
baseline commit `3bfe08c0978a4b9278e5536debf1852e9fceb54f`, plus the semantic-tag,
Tail OFF/ON and resolved-snapshot checks. It requires `--report <output.json>`.

`generate_configs.py` contains the complete frozen literal template. It
recreates configs, manifest, command lists and run order after a passing
launcher regression, and refuses to overwrite changed generated content.
Neither generation nor validation calls `train.py`, `qsub_local` or tmux.
