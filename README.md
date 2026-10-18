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

Use [environment_training.yml](environment_training.yml) for model execution.
Set the config's interpreter override for your machine, and provide the
preprocessed graph data and station metadata referenced by that config.
Preview a training command before launching it:

```bash
S0_PYTHON_BIN="$(command -v python)" DRY_RUN=1 USE_TMUX=0 bash train.sh \
  configs_old/single_tail_episodepeak_4x4/train_config_NCEP_Boston_G0_T0250_EP0050.sh

S0_PYTHON_BIN="$(command -v python)" USE_TMUX=0 bash train.sh \
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

Check Python and shell syntax without loading data or starting a job:

```bash
python -m compileall -q train.py infer.py emulator
bash -n train.sh infer.sh
```
