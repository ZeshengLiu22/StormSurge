# P0_QuickRun

P0 compares the single regression head and the current supervised exceedance
dual head under four prediction losses, with a fixed architecture and training
protocol. There are **32 independent configurations: four NCEP stations × eight
designs**. Each config contains all relevant settings; none sources another
config. Run commands from `/media/volume/PACT-Data/StormSurge`.

The stations are CBBT, Lewes, Battery and Boston. Each has these eight designs:

| Head | Filename suffix | Production loss mode | Tail term | Slope term |
| --- | --- | --- | --- | --- |
| single | `single_mse` | `mse` | off | off |
| single | `single_tail` | `mse_tail` | on | off |
| single | `single_slope` | `mse_slope` | off | on |
| single | `single_tail_slope` | `mse_tail_slope` | on | on |
| dual | `dual_mse` | `mse` | off | off |
| dual | `dual_tail` | `mse_tail` | on | off |
| dual | `dual_slope` | `mse_slope` | off | on |
| dual | `dual_tail_slope` | `mse_tail_slope` | on | on |

The [manifest](manifest.tsv) lists all 32 filenames, settings, semantic run
names, queue labels and exact submission commands. Each invocation produces
exactly one training command. There is no LR sweep, weight sensitivity study,
architecture selection or mechanism ablation in P0.

## Frozen protocol

- `perceiver3`, GraphSAGE, Transformer, **24 h history**, hidden width 128;
  two GraphSAGE layers, two Transformer layers, eight node/time read heads,
  feedforward multiplier 4, maximum time steps 32. Spatial/head dropout is
  0.05; Transformer dropout is 0. CNN width 29 is an inactive parser setting.
- Chronological year-group split: train 0.6, validation 0.2, remaining held out;
  seed 42, no year shuffle or future-only filtering. Elevation and bathymetry
  are both off. Existing station metadata behavior otherwise remains unchanged.
- Batch size 256, accumulation 4, one GPU: nominal effective batch 1024.
  LR `5e-3`, 300 epochs, cosine schedule, five warmup epochs, start factor 0.1,
  minimum LR `1e-6`, no gradient clipping, deterministic kernels off.
  Production `train.py` fixes Adam and weight decay `1e-5`; the current launcher
  has no independent optimizer/weight-decay config option.
- One local H100 (`CUDA_VISIBLE_DEVICES=0`, `num_gpus=1`), AMP bf16, TF32 on,
  one Torch thread. Loader workers, pinning, persistent workers and prefetch
  are all zero; multiprocessing context is `fork`.

All configs explicitly freeze these values:

```bash
BODY_LOSS_WEIGHT=1
EXCESS_LOSS_WEIGHT=2
GATE_LOSS_WEIGHT=0.5
TAIL_FRAC="0.05"
TAIL_LAMBDA_LIST=("0.025")
SLOPE_LAMBDA_LIST=("0.01")
SLOPE_MASK_S_LIST=("0.10")
SLOPE_ROBUST="charb"
SLOPE_CHARB_EPS="1e-3"
SLOPE_HUBER_DELTA="0.05"
```

Dual runs use `GATE_MODE=window`, `DUAL_MODE=exceedance`, `DUAL_LOSS=1`,
`DUAL_ABLATION=none` and `EXCEEDANCE_PERCENTILE=95`. Single configs set
`DUAL_LOSS=0`; the production parser also enforces zero for a single head,
and a single head returns no body/excess/gate branch outputs.

The production launcher passes branch weights only for dual heads and tail/slope
flags only for their selected loss modes. Unused Python argument fields can
therefore retain parser defaults (for example, `tail_lambda=0.1` for `mse`);
they do not activate a loss. `config_used.sh` and the launcher log record the
frozen P0 declarations, while the Python JSON records parsed arguments. This
existing dispatch is unchanged. All active loss weights are validated against
the frozen values. `WMSE_Q=95`, alpha 4, softness 0.10 and absolute-value mode
remain solely for the existing threshold infrastructure; no P0 loss is WMSE.

## Preprocessing and local paths

OOD functionality is **completely off** in every config and resolved command:

```bash
DISABLE_OOD=1
X_NORM="zscore"
X_CLIP="0"
X_AUG=0
X_AUG_PROB="0"
X_AUG_SCALE="0"
X_AUG_BIAS="0"
```

The launcher applies its production OOD override, and validation checks
`x_norm=zscore`, `x_clip=0` and `x_aug=0`. Percentile/sample settings passed by
the launcher are unused by zscore normalization and cannot enable clipping or
augmentation.

Verified on 2026-09-10:

| Setting | Verified absolute path |
| --- | --- |
| `ROOT_DIR` | `/media/share/PACT/Data/Grid4_New/NCEP/graphs` |
| `STATION_JSON_DIR` | `/media/volume/PACT-Data/StormSurge/station_json` |
| `TEST_ROOT_DIR` | empty; use the held-out NCEP year groups |
| `PYTHON_BIN` | `/media/volume/PACT-Data/conda_envs/torchpyg-cu124/bin/python` |

The graph path is the canonical target of the repository's
`Data/Grid4_New/NCEP/graphs` link. All four stations have 36 readable graph
files and a station JSON. The Python environment imports PyTorch
`2.6.0+cu124` and torch-geometric `2.7.0`; the host query reported CUDA
available and `NVIDIA H100 80GB HBM3`. The sandbox hides CUDA, so that query
was verified on the host without any training or GPU computation.

Every config sets `DO_CONDA=0` and the absolute `PYTHON_BIN`. No interactive
Conda activation or HPC module loading is required by P0.

## Results and tmux

Every config sets `USE_TMUX=1`,
`ALL_RESULTS_ROOT="./All_Results/P0_QuickRun"` and
`RUN_DIR_NAME_STYLE="runname_timestamp"`. A Battery example is:

```text
All_Results/P0_QuickRun/
└── NCEP_Battery_24h_dual_tail_slope__20260910_021530/
    ├── config_used.sh
    ├── launcher.log
    ├── exit_status
    ├── train_*.log
    ├── config_*.json
    ├── best_*.pth
    ├── metrics_*.jsonl
    ├── summary_*.json
    └── test_preds_*.npz
```

The final five training artifacts appear as training progresses/completes.
All production writes in `train.py` use the exact supplied `--output_dir`.
This invocation does not run a separate plotting or inference/export program.

The outer launcher creates one detached tmux session; the inner invocation
alone creates the result directory. The semantic name and timestamp are passed
explicitly into the inner invocation. The queue label sets `SESSION_NAME` and
does not replace the config's `PACT_RUN_NAME`. Session labels contain no dot or
colon. Reusing an active session label is rejected by tmux. A collision with
an existing P0 name/timestamp is rejected before overwriting its artifacts.

Completed and failed tmux sessions close. The inner launcher saves its final
status in `exit_status`; the wrapper also hands the status to the local queue
worker. The worker waits for the session to finish before releasing its slot.
For a direct detached invocation, the outer shell returns the session-creation
status immediately; inspect `exit_status` for the eventual training status.
Configs without `RUN_DIR_NAME_STYLE` keep the historical
`<TIMESTAMP>_<RUN_NAME>` directory naming.

## Local queue commands

These instructions follow the current functions in `~/.bashrc` and
`~/.local/bin/qsub_local_worker`. Neither file was modified for P0.

```text
qsub_local <script.sh> [label] [script_arg...]
```

Submit one experiment from the repository directory:

```bash
qsub_local train.sh P0_Battery_dual_tail_slope \
  experiment_config/P0_QuickRun/train_config_NCEP_Battery_24h_dual_tail_slope.sh
```

| Command | Current behavior |
| --- | --- |
| `qstat_local` | Show active tracked jobs; `-a` includes history, `--raw` shows task-spooler output. |
| `qlog_local P0_Battery_dual_tail_slope` | Tail the recorded `LOG_FILE` if present, otherwise task-spooler output. |
| `qattach_local P0_Battery_dual_tail_slope` | Attach to the tracked tmux session; detach with Ctrl-b then d. |
| `qdel_local P0_Battery_dual_tail_slope` | Remove a queued task, or stop a running task and its tracked tmux process tree. |

With the direct submission above, `qlog_local` normally shows the outer launcher
and worker lifecycle through task-spooler. Full training output is visible in
tmux and the result directory's `launcher.log` / `train_*.log`. Task-spooler's
own queue output and bookkeeping remain in its normal service locations;
they are separate from the contained training artifacts.

`QSUB_LOCAL_SLOTS` defaults to **1**, and `qsub_local` sets `TS_SLOTS` to that
value. Keep that default for this one-H100 protocol. The worker executes
`bash train.sh <config>` directly and needs no `qsub_local` function in its
noninteractive shell. The helper also supports an explicit
`QSUB_LOCAL_SESSION_NAME` override; normally use the supplied manifest label.

## Validation and maintenance

The [audit](AUDIT.md) and [validation evidence](validation/config_validation.json)
record the completed checks. No real training, LR sweep, weight diagnostic or
architecture search was started.

Repeat production dry validation without training:

```bash
/media/volume/PACT-Data/conda_envs/torchpyg-cu124/bin/python -B \
  experiment_config/P0_QuickRun/validate_configs.py
```

It checks the manifest, exact constants, shell syntax, standalone sourcing in
an empty environment, the actual production parser, one command per config,
OOD, active loss weights, single-head loss dispatch, naming and the accumulation
GPU guard. It creates no result folders; evidence defaults to `/tmp`.

The controlled queue compatibility test is also safe to repeat:

```bash
/media/volume/PACT-Data/conda_envs/torchpyg-cu124/bin/python -B \
  experiment_config/P0_QuickRun/validate_queue.py
```

It uses private queue/tmux sockets and temporary metadata, submits one actual
P0 dry run, then uses two stdlib-only process stubs through the production
launcher. It verifies serialization, exits 23/0, session/folder counts, name
and timestamp handoff, paths with spaces, collision protection and cleanup.
It never calls `train.py`, imports Torch in the stub, or touches the normal queue.

`generate_configs.py` is a development-only generator with its complete
template embedded. Existing edited configs are protected from overwrite.
The generated shell files never load the generator or either validator.

## All 32 configurations

| Station | Config |
| --- | --- |
| CBBT | [train_config_NCEP_CBBT_24h_single_mse.sh](train_config_NCEP_CBBT_24h_single_mse.sh) |
| CBBT | [train_config_NCEP_CBBT_24h_single_tail.sh](train_config_NCEP_CBBT_24h_single_tail.sh) |
| CBBT | [train_config_NCEP_CBBT_24h_single_slope.sh](train_config_NCEP_CBBT_24h_single_slope.sh) |
| CBBT | [train_config_NCEP_CBBT_24h_single_tail_slope.sh](train_config_NCEP_CBBT_24h_single_tail_slope.sh) |
| CBBT | [train_config_NCEP_CBBT_24h_dual_mse.sh](train_config_NCEP_CBBT_24h_dual_mse.sh) |
| CBBT | [train_config_NCEP_CBBT_24h_dual_tail.sh](train_config_NCEP_CBBT_24h_dual_tail.sh) |
| CBBT | [train_config_NCEP_CBBT_24h_dual_slope.sh](train_config_NCEP_CBBT_24h_dual_slope.sh) |
| CBBT | [train_config_NCEP_CBBT_24h_dual_tail_slope.sh](train_config_NCEP_CBBT_24h_dual_tail_slope.sh) |
| Lewes | [train_config_NCEP_Lewes_24h_single_mse.sh](train_config_NCEP_Lewes_24h_single_mse.sh) |
| Lewes | [train_config_NCEP_Lewes_24h_single_tail.sh](train_config_NCEP_Lewes_24h_single_tail.sh) |
| Lewes | [train_config_NCEP_Lewes_24h_single_slope.sh](train_config_NCEP_Lewes_24h_single_slope.sh) |
| Lewes | [train_config_NCEP_Lewes_24h_single_tail_slope.sh](train_config_NCEP_Lewes_24h_single_tail_slope.sh) |
| Lewes | [train_config_NCEP_Lewes_24h_dual_mse.sh](train_config_NCEP_Lewes_24h_dual_mse.sh) |
| Lewes | [train_config_NCEP_Lewes_24h_dual_tail.sh](train_config_NCEP_Lewes_24h_dual_tail.sh) |
| Lewes | [train_config_NCEP_Lewes_24h_dual_slope.sh](train_config_NCEP_Lewes_24h_dual_slope.sh) |
| Lewes | [train_config_NCEP_Lewes_24h_dual_tail_slope.sh](train_config_NCEP_Lewes_24h_dual_tail_slope.sh) |
| Battery | [train_config_NCEP_Battery_24h_single_mse.sh](train_config_NCEP_Battery_24h_single_mse.sh) |
| Battery | [train_config_NCEP_Battery_24h_single_tail.sh](train_config_NCEP_Battery_24h_single_tail.sh) |
| Battery | [train_config_NCEP_Battery_24h_single_slope.sh](train_config_NCEP_Battery_24h_single_slope.sh) |
| Battery | [train_config_NCEP_Battery_24h_single_tail_slope.sh](train_config_NCEP_Battery_24h_single_tail_slope.sh) |
| Battery | [train_config_NCEP_Battery_24h_dual_mse.sh](train_config_NCEP_Battery_24h_dual_mse.sh) |
| Battery | [train_config_NCEP_Battery_24h_dual_tail.sh](train_config_NCEP_Battery_24h_dual_tail.sh) |
| Battery | [train_config_NCEP_Battery_24h_dual_slope.sh](train_config_NCEP_Battery_24h_dual_slope.sh) |
| Battery | [train_config_NCEP_Battery_24h_dual_tail_slope.sh](train_config_NCEP_Battery_24h_dual_tail_slope.sh) |
| Boston | [train_config_NCEP_Boston_24h_single_mse.sh](train_config_NCEP_Boston_24h_single_mse.sh) |
| Boston | [train_config_NCEP_Boston_24h_single_tail.sh](train_config_NCEP_Boston_24h_single_tail.sh) |
| Boston | [train_config_NCEP_Boston_24h_single_slope.sh](train_config_NCEP_Boston_24h_single_slope.sh) |
| Boston | [train_config_NCEP_Boston_24h_single_tail_slope.sh](train_config_NCEP_Boston_24h_single_tail_slope.sh) |
| Boston | [train_config_NCEP_Boston_24h_dual_mse.sh](train_config_NCEP_Boston_24h_dual_mse.sh) |
| Boston | [train_config_NCEP_Boston_24h_dual_tail.sh](train_config_NCEP_Boston_24h_dual_tail.sh) |
| Boston | [train_config_NCEP_Boston_24h_dual_slope.sh](train_config_NCEP_Boston_24h_dual_slope.sh) |
| Boston | [train_config_NCEP_Boston_24h_dual_tail_slope.sh](train_config_NCEP_Boston_24h_dual_tail_slope.sh) |
