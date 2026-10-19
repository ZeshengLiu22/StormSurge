# PACT storm-surge forecasting

PACT predicts six hourly storm-surge residuals in meters from atmospheric forcing
on a spatial graph, with forcing history sampled every six hours.

The `perceiver3` model uses a GraphSAGE or CNN spatial encoder. A station query
reads the spatial nodes through cross-attention; a Transformer, MLP, LSTM, or GRU
models the resulting history. Learned horizon embeddings query temporal memory
through `forecast_readout` cross-attention. A Single regression head produces
one prediction per horizon. A spatial-mean baseline is also available.

Training minimizes physical Global MSE plus two optional terms:

```text
GlobalMSE + EXCEEDANCE_LOSS_WEIGHT * TailMSE
          + EPISODE_GT_ALIGNED_PEAK_WEIGHT * EpisodeGTAlignedPeakMSE
```

Tail MSE uses strict `GT > TRAIN Q95` and fixed TRAIN extreme-hour prevalence.
Episode peak MSE supervises the GT peak timestamp of each physical episode,
using fixed TRAIN peak prevalence. Global MSE masks missing targets pointwise;
Tail and episode terms exclude forecast windows containing missing targets.
Normalization and thresholds are fitted only on TRAIN. Both overall and
exceedance checkpoints are retained using VAL AllRMSE and VAL ExceedanceRMSE;
the chosen primary role is controlled by the experiment config.

Current experiment references remain under [configs_old](configs_old/README.md):
NCEP weight sweeps, four-station CMIP6 comparisons, historical NCEP training for
future transfer, and cross-dataset inference. Each family retains its original
data, split, checkpoint, and result provenance.

Use [environment.yml](environment.yml), exported from `derecho-ml`, for model
execution. Activate `derecho-ml` before running either shell launcher. The
launchers use the active environment's Python, including inside tmux; set
`PYTHON_BIN` explicitly only when a different interpreter is needed.
Config arguments are relative to the caller's working directory. Paths inside
configs are relative to the repository root; the default graph directory is
`./dataset/NCEP/graphs`. Provide the data, checkpoint, and output paths required
by the selected config.
Preview a training command before launching it:

```bash
DRY_RUN=1 USE_TMUX=0 bash train.sh \
  configs_old/single_tail_episodepeak_4x4/train_config_NCEP_Boston_G0_T0250_EP0050.sh

USE_TMUX=0 bash train.sh \
  configs_old/single_tail_episodepeak_4x4/train_config_NCEP_Boston_G0_T0250_EP0050.sh
```

Evaluate a checkpoint on its saved source split:

```bash
python infer.py --ckpt /path/to/run/best_overall.pt \
  --root_dir ./Data/Grid4_New/NCEP/graphs --save_npz \
  --out_dir ./All_Inference_Results/example
```

For cross-dataset evaluation, provide `--test_root_dir` and explicit `--years`
with `--strict_years`. Inference reuses source-checkpoint normalization and source
TRAIN Q95. Use `bash infer.sh <config.sh>` for the retained assignment configs,
or their `run_group.sh` launchers for complete experiment groups. Dry runs submit
no jobs.

Existing final Single checkpoints load through a small metadata compatibility
adapter, with strict model-state loading. Legacy Dual checkpoints are rejected.

## Shallow neural-network baselines

Seven independent models share the existing data, normalization, optimization,
ForecastLoss, checkpoint selection, and inference pipeline:

| Model key | Spatial encoder | Temporal module |
| --- | --- | --- |
| `baseline_mlp` | Two pointwise linear layers | None; current forcing only |
| `baseline_cnn` | Two 3×3 convolutions | None; current forcing only |
| `baseline_mlp_lstm` | Two pointwise linear layers | One-layer LSTM |
| `baseline_mlp_gru` | Two pointwise linear layers | One-layer GRU |
| `baseline_cnn_lstm` | Two 3×3 convolutions | One-layer LSTM |
| `baseline_cnn_gru` | Two 3×3 convolutions | One-layer GRU |
| `baseline_convlstm` | One shared 3×3 input convolution | One ConvLSTM layer |

The pointwise MLP shares weights across grid cells. CNN/MLP weights are also
shared across forcing timestamps. The four separable recurrent models pool and
project each timestamp before recurrence; the final recurrent output feeds a
single direct multi-horizon regression head. All models use LeakyReLU spatial
activations and modest configurable dropout, with no attention or autoregression.

`baseline_cnn_gru` is a traditional shallow CNN+GRU. It is distinct from
`perceiver3` configured with CNN and GRU, which includes PACT station-conditioned
and horizon-aware cross-attention. The original `baseline` model remains
available unchanged; its key never selects one of the seven new architectures.

ConvLSTM first applies a shared 3×3 convolution from input features to
`BASELINE_STEM_CHANNELS` and LeakyReLU. A single 3×3 convolution computes its four
input/forget/output/candidate gates from the projected input and previous hidden
map. Hidden and cell states retain their full 2D spatial dimensions and
`BASELINE_STEM_CHANNELS` channels throughout recurrence. The final hidden map is
pooled, flattened, projected to `HIDDEN_CHANNELS`, passed through LeakyReLU and
dropout, and mapped linearly to all output horizons.

The new options, used only by these seven models, are:

| CLI option | Shell variable | Default |
| --- | --- | --- |
| `--baseline_pooling` | `BASELINE_POOLING` | `adaptive` |
| `--baseline_pool_grid` | `BASELINE_POOL_GRID` | `4` |
| `--baseline_stem_channels` | `BASELINE_STEM_CHANNELS` | `32` |

Adaptive average pooling preserves a configurable coarse square grid (4×4 by
default); `global_mean` averages the entire spatial grid. Any positive adaptive
pool size is supported. Models accept different grid resolutions in separate
batches; each batch must contain a consistent rectangular grid with row-major
nodes and valid grid metadata. Example configs use `HIDDEN_CHANNELS=128` and
`DROPOUT=0.05`; legacy PACT defaults are unchanged.

Use `HISTORY_HOURS_LIST=(0)` for static MLP/CNN; positive history is rejected.
Recurrent examples use an editable 24 hours. Values such as 12, 24, or 48 hours
select `history_hours // 6 + 1` chronological snapshots, subject to available
stored history. A recurrent model also accepts zero hours (one snapshot).
History length is set by the experiment configuration, with no fixed seven-frame
assumption. All three pressure modes below work through the shared loader, which
infers five or six input features. Loss remains physical Global MSE plus editable
Tail and episode peak terms. Example weights are zero for MSE-only training;
nonzero values retain the existing TRAIN-only Q95 and missing-label semantics.

Assignment-only examples for all seven models are in [configs/baselines](configs/baselines).
Preview an example without starting training or submitting a job:

```bash
DRY_RUN=1 USE_TMUX=0 bash train.sh configs/baselines/train_config_baseline_cnn_gru.sh
```

Automatic names identify model, station, source dataset, history, pooling,
pressure mode, and seed, while retaining a timestamp. Multi-history or
multi-learning-rate baseline launches place each combination in its own
subdirectory. Explicit run names retain their existing meaning. Resolved
configuration snapshots and checkpoint metadata retain architecture parameters
and loss weights; checkpoints reconstruct models without the original shell file.

Evaluate any new baseline using the same inference entry point and an optional
exact model assertion:

```bash
python infer.py --ckpt /path/to/run/best_overall.pt \
  --root_dir ./dataset/NCEP/graphs --model baseline_cnn_gru \
  --save_npz --out_dir ./All_Inference_Results/baseline_example
```

For cross-dataset evaluation, add `--test_root_dir`, `--years`, and
`--strict_years` as needed. Inference preserves the exact model key, effective
spatial/temporal modules, pooling, history, pressure, parameter count, and
source/target provenance. It reuses source normalization and loads weights with
`strict=True`. Legacy Single-head PACT and Baseline checkpoint handling remains
in place, including the existing rejection of retired Dual-head checkpoints.

## Atmospheric pressure representations

Training accepts `--pressure_mode` (or `PRESSURE_MODE` in shell configs):

| Mode | Loaded node features |
| --- | --- |
| `anomaly` (default) | `[u, v, p_anomaly, lon, lat]`, unchanged |
| `raw` | `[u, v, p_anomaly + p_mean, lon, lat]` |
| `anomaly_plus_mean` | `[u, v, p_anomaly, lon, lat, p_mean]` |

Non-default modes require existing graph metadata `p_mean_hist: [T]` containing
the absolute spatial pressure mean for every stored forcing timestep. When
present, `p_mean_curr` must agree with its final value. Transformations happen
while loading the entire stored history, before TRAIN normalization and before
selecting shorter history windows. Graph files remain unchanged; preprocessing
does not need to be rerun. Missing or invalid pressure metadata raises an error
for non-default modes; `anomaly` requires no pressure-mean metadata.

For `zscore`, absolute-pressure channels use a two-pass FP64 variance calculation
on TRAIN to retain small variations around large pressure means. Default `anomaly`
statistics, other feature channels, and `robust`/`mag` normalization are unchanged.

Existing configs retain the default anomaly features and run names. Automatic
training run names gain `_praw` or `_panomaly_mean` for the other modes;
explicit run names and output roots are preserved. New checkpoints record the
mode in `training_config`. Inference automatically uses that mode and the saved
source normalization; old Single checkpoints without it use `anomaly` and keep
strict state loading. An inference `--pressure_mode` or shell `PRESSURE_MODE`
is optional and only asserts a match with the checkpoint; mismatches fail.

```bash
PRESSURE_MODE=raw DRY_RUN=1 USE_TMUX=0 bash train.sh path/to/train_config.sh
python train.py --root_dir ./dataset/NCEP/graphs --pressure_mode anomaly_plus_mean
python infer.py --ckpt /path/to/raw_run/best_overall.pt \
  --root_dir ./dataset/NCEP/graphs --out_dir ./All_Inference_Results/raw_example
```

Check Python and shell syntax without loading data or starting a job:

```bash
python -m compileall -q train.py infer.py emulator preprocessing
bash -n train.sh infer.sh
```

## Code documentation

Python functions and methods document their purpose, inputs, and returned
values in English docstrings. Parameter descriptions should identify units,
array dimensions and axis order, accepted sentinel values, and required
metadata where applicable. File-writing routines describe their artifacts;
functions that modify arguments state those side effects. Constructors document
configuration inputs, and model forward methods specify tensor shapes.

Inline comments explain scientific assumptions or implementation constraints,
such as hourly target alignment, TRAIN-only fitting, and missing-value masks.
Keep these comments consistent with executable behavior; omit conversational
notes, edit history, and statements that merely repeat the next line of code.
Bash function comments describe positional arguments, required shell state,
output, and exit behavior. Preserve scheduler and static-analysis directives.
