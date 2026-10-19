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
python -m compileall -q train.py infer.py emulator
bash -n train.sh infer.sh
```
