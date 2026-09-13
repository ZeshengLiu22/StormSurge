# P1_WeightProbe

P1 probes whether **stronger excess supervision and stronger tail supervision
can materially improve extreme/peak behavior**, before introducing new
architecture or loss formulations. It adds exactly **eight new settings per
station: CBBT, Lewes, Battery and Boston**, for **32 new runs**. This is a small,
structured weight probe, not a broad hyperparameter search.

## Design and P0 anchors

P0 already provides the two dual-head reference anchors at **EXCESS=2, TAIL=0**
(`dual_mse`) and **EXCESS=2, TAIL=0.025** (`dual_tail`). P1 does not duplicate
either anchor. Keep their P0 results in the eventual analysis.

Each station has exactly this P1 matrix:

| Setting | Tag | Excess weight | Tail lambda | Loss mode |
| --- | --- | --- | --- | --- |
| P1-A | `ex5_notail` | 5 | 0 | `mse` |
| P1-B | `ex10_notail` | 10 | 0 | `mse` |
| P1-C | `ex2_tail005` | 2 | 0.05 | `mse_tail` |
| P1-D | `ex2_tail010` | 2 | 0.10 | `mse_tail` |
| P1-E | `ex5_tail005` | 5 | 0.05 | `mse_tail` |
| P1-F | `ex5_tail010` | 5 | 0.10 | `mse_tail` |
| P1-G | `ex10_tail005` | 10 | 0.05 | `mse_tail` |
| P1-H | `ex10_tail010` | 10 | 0.10 | `mse_tail` |

The excess values are **5 and 10**, plus the P0 baseline **2**; the stronger
tail values are **0.05 and 0.10**. Together with P0, these comparisons separate:

- Stronger excess: 2 → 5 → 10 at a fixed tail weight.
- Stronger tail: 0 / 0.025 → 0.05 → 0.10 at excess 2; 0 → 0.05 → 0.10
  at excess 5 and 10.
- Interaction between stronger excess and stronger tail supervision across
  the shared tail levels 0, 0.05 and 0.10.

The purpose is to test **amplitude-supervision strength before introducing
new methodology**. The [manifest](manifest.tsv) contains exactly 32 data rows
(plus its header), with config paths, weights, loss mode, history, LR, semantic
run names, recommended queue labels and submission commands.

## Frozen P0 protocol

Every final shell config is fully self-contained. It does not source P0,
legacy configs, another P1 config or a shared common config. Each declares
all settings directly, with one value in every sweep list, so each invocation
resolves to exactly one training command.

All P1 configs keep:

```bash
BODY_LOSS_WEIGHT=1
GATE_LOSS_WEIGHT=0.5
TAIL_FRAC="0.05"
SLOPE_LAMBDA_LIST=("0")
HEAD_TYPE="dual"
GATE_MODE="window"
DUAL_MODE="exceedance"
DUAL_LOSS=1
DUAL_ABLATION="none"
EXCEEDANCE_PERCENTILE=95
```

Body and gate supervision remain frozen; **slope is disabled in all 32 runs**.
Architecture, head formulation and all other training settings match the
corresponding P0 dual anchor:

- `perceiver3`, GraphSAGE + Transformer, **24 h history**, hidden width 128,
  two GraphSAGE layers, two Transformer layers, eight node/time read heads,
  feedforward multiplier 4, maximum time steps 32; spatial/head dropout 0.05
  and Transformer dropout 0.
- Chronological year-group split: train 0.6, validation 0.2, remaining held out;
  seed 42, no year shuffle or future-only filtering. Station metadata behavior
  is unchanged; elevation and bathymetry are off.
- Batch 256, accumulation 4, one GPU; **LR `5e-3`**, 300 epochs, Adam with
  weight decay `1e-5`, cosine scheduler, five warmup epochs, start factor 0.1,
  minimum LR `1e-6`, no gradient clipping, deterministic kernels off.
- One local H100 (`CUDA_VISIBLE_DEVICES=0`, `num_gpus=1`), AMP bf16, TF32 on,
  one Torch thread. Loader workers, pinning, persistent workers and prefetch
  remain zero; multiprocessing context is `fork`.
- **Checkpoint selection remains minimum validation All RMSE**, using the
  unchanged production training code. No peak-based checkpoint selection.
- **OOD remains disabled**: `DISABLE_OOD=1`, `X_NORM="zscore"`, `X_CLIP="0"`,
  `X_AUG=0`, and augmentation probability, scale and bias all zero.

Inactive slope-shape and WMSE threshold settings retain their P0 declarations.
The production launcher passes tail flags only for `mse_tail`, and passes no
slope flags for either P1 loss mode. Thus parsed Python arguments can retain
inactive defaults (`tail_lambda=0.10` for `mse`, `slope_lambda=0.01` for both
modes). Those defaults do not activate these terms. The configs explicitly
declare tail zero for `notail` and slope zero for every setting; the validator
checks declarations, selected modes and active weights.

## Paths, names and queue commands

Run commands from `/media/volume/PACT-Data/StormSurge`. P0 data and interpreter
paths are retained:

| Setting | Value |
| --- | --- |
| `ROOT_DIR` | `/media/share/PACT/Data/Grid4_New/NCEP/graphs` |
| `STATION_JSON_DIR` | `/media/volume/PACT-Data/StormSurge/station_json` |
| `TEST_ROOT_DIR` | empty; held-out NCEP year groups |
| `PYTHON_BIN` | `/media/volume/PACT-Data/conda_envs/torchpyg-cu124/bin/python` |
| `DO_CONDA` | `0` |
| `USE_TMUX` | `1` |
| `ALL_RESULTS_ROOT` | `./All_Results/P1_WeightProbe` |
| `RUN_DIR_NAME_STYLE` | `runname_timestamp` |

Filenames follow `train_config_NCEP_<station>_24h_dual_<tag>.sh`, with semantic
`PACT_RUN_NAME=NCEP_<station>_24h_dual_<tag>`. Results will use:

```text
All_Results/P1_WeightProbe/NCEP_CBBT_24h_dual_ex5_notail__<timestamp>/
```

P0 queue/tmux behavior and artifact layout are retained. Queue labels such as
`P1_CBBT_dual_ex5_notail` remain separate from semantic run names. The manifest
provides all commands for future submission; for example:

```bash
qsub_local train.sh P1_CBBT_dual_ex5_notail \
  experiment_config/P1_WeightProbe/train_config_NCEP_CBBT_24h_dual_ex5_notail.sh
```

## Validation without training

The [validation report](validation/config_validation.json) and
[32 dry-run logs](validation/dry_runs.txt) record syntax checks and
`DRY_RUN=1` validation for all 32 P1 configs. **No model training or queue
submission was started.** The P1 result root contains no training runs.

Repeat validation from the repository directory:

```bash
/media/volume/PACT-Data/conda_envs/torchpyg-cu124/bin/python -B \
  experiment_config/P1_WeightProbe/validate_configs.py
```

The validator checks the exact eight-setting matrix at every station, all
manifest fields, unique names and result paths, shell syntax, standalone
sourcing in an empty environment, and exactly one production command per
config. It compares every declaration and parsed argument against the matching
P0 anchor, permitting only the requested weights and P1 identity/output changes.
It also verifies the frozen settings, slope/OOD disabled, and unchanged P1
result contents. The eight P0 reference commands are inspected using
`DRY_RUN=1` only; no anchor training is repeated. Validation evidence defaults
to a temporary directory and can be saved with `--report-dir <directory>`.

The validator is a development tool; none of the final configs loads it or
depends on P0 files at runtime.
