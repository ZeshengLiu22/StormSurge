# StormSurge

Independent experimental repository for PACT, derived from [PACT_Storm_Surge_Emulator](https://github.com/BinaLab/PACT_Storm_Surge_Emulator) at commit `3d4be39c48214cbcd3afbdc51ec48f7fb839e0de`. This repository starts with a standalone source snapshot and has its own Git history.

The [original-versus-current comparison](docs/ORIGINAL_COMPARISON.md) lists the old behavior, current behavior, impact, and restoration recommendation for each change. Original experiment combinations are preserved; input data paths in the shell profiles still refer to external datasets and should be set for the machine running them.

The [archived Emulator changelog](docs/history/Emulator_changelog.md) is an unmodified copy of `changelog.md` from original commit `bb62a2297a0d37a35db5bff352ee815e05676c93`. It records that version's behavior and validation; use the current documentation for this repository. Existing config files remain in place for a separate review.

This folder is the experimental training implementation for **fresh runs**. It has one current backbone, a single head or a supervised body/exceedance dual head, and no old checkpoint loader or recovery guard.

## Model flow

`forcing history → GraphSAGE/CNN → station attention → temporal block → horizon attention → single/dual head`

- [`architectures.py`](emulator/models/architectures.py) composes PACT and the mean-pooling baseline.
- [`spatial.py`](emulator/models/spatial.py) contains the two spatial encoders.
- [`temporal.py`](emulator/models/temporal.py) contains MLP, LSTM, GRU and Transformer blocks.
- [`heads.py`](emulator/models/heads.py) contains the head computations and `ForecastOutput`.
- [`losses.py`](emulator/training/losses.py) contains supervision, including the three dual-head losses.
- [`engine.py`](emulator/training/engine.py) performs one forward per batch for train, validation and inference.

The forward pass consumes inputs and station features only. `ForecastOutput.prediction` is normalized; body, excess, gate logits and threshold are returned only by the dual head because its loss needs them. Attention weights and diagnostic statistics are never constructed.

See [the dual-head derivation](DUAL_HEAD_EXPLAINED.md) and [the stability changes](STABILITY_AND_DUAL_HEAD.md).

## Train

The original shell configuration tree and conditional sweeps are retained. All dual-head experiments now use the supervised exceedance head; single-head and baseline controls remain available.

```bash
bash train.sh configs/configs_train/NCEP/train_config_NCEP_Battery_P3_Best.sh
bash train.sh configs/train_config_NCEP_Battery_Stable_Single.sh
DRY_RUN=1 bash train.sh configs/configs_train/NCEP/train_config_NCEP_Battery_P3_Best.sh
```

`configs/configs_train/` holds the original training profiles, `configs/configs_train_single/` holds the **single-GPU / gradient-accumulation** profiles, and `configs/configs_infer/` holds inference profiles. The word `single` in the directory name refers to the GPU count. Select the prediction head with `HEAD_TYPE=single` or `HEAD_TYPE=dual`.

Keep experiment overrides in a shell config, after sourcing the desired profile. For example:

```bash
source /path/to/PACT-stability-v2/configs/configs_train/NCEP/train_config_NCEP_Battery_P3_Best.sh
ENCODER_TYPE=CNN
TEMPORAL_BLOCK=MLP
HEAD_TYPE=dual
HISTORY_HOURS_LIST=(0 6 12 18 24 30 36 42 48)
LR_LIST=(5e-3)
LOSS_MODE_LIST=(mse mse_tail_slope)
num_gpus=1
GRAD_ACCUM_STEPS=4
USE_TMUX=0
DO_CONDA=0
```

The launcher preserves tmux, optional conda activation, torchrun, per-combination logs, and resolved shell snapshots. Set `PYTHON_BIN` to an explicit interpreter if needed. `num_gpus` is the original shell setting; `batch_size` is per process. As before, the launcher permits gradient accumulation with one process. Microbatch losses have equal weight, including a short final microbatch; a short accumulation group uses its actual microbatch count.

The sweep axes are `LOSS_MODE_LIST`, `LR_LIST`, `HISTORY_HOURS_LIST`, `WMSE_Q_LIST`, `TAIL_LAMBDA_LIST`, `SLOPE_LAMBDA_LIST` and `SLOPE_MASK_S_LIST`. Irrelevant axes are reduced exactly as in the original launcher. In particular, `WMSE_Q_LIST` varies for `wmse*`/`*wtail*`; other modes use its first entry, including plain-MSE slope modes. Spatial encoder, temporal block and head type are selected per profile. `DRY_RUN=1` prints all generated commands without training or creating result folders.

The established Python CLI remains available:

```bash
python train.py --root_dir /path/to/graphs --station Battery \
  --model perceiver3 --encoder_type CNN --temporal_block MLP \
  --head_type dual --history_hours 48 --loss_mode mse_tail_slope \
  --amp --amp_dtype bf16 --tf32 --output_dir All_Results/example
```

PACT is selected with `--model perceiver3`; `--model baseline` uses spatial mean pooling at H=0 and adds one LSTM at H>0. H is in hours and must be a multiple of six. H=48 consumes nine forcing slices including the present. `--max_time_steps` controls lag-embedding capacity. Temporal settings keep their original names: `--transformer_layers`, `--transformer_ff_mult` and `--transformer_dropout`, including for MLP/LSTM/GRU.

All ten loss modes are retained: `mse`, `wmse`, `mse_tail`, `wmse_tail`, `mse_wtail`, and each with `_slope`. Weighted-tail modes retain weighted tail errors. The slope soft mask uses the TRAIN pointwise `wmse_q` threshold; the tail loss uses the TRAIN window-peak threshold. The old residual head's alpha, gate-bias and tanh-clip controls are removed.

Tail loss is sample-additive. With `q = tail_frac`, the threshold is still fitted once as the TRAIN `(1-q)` quantile of each window's signed maximum over forecast horizons:

```text
event[b] = 1[max_h target[b,h] >= peak_threshold]
tail_loss = sum_b(event[b] * mean_h(tail_error[b,h])) / (q * B)
prediction_loss = base_loss + tail_lambda * tail_loss  # plus the existing slope term when selected
```

`mse_tail` uses ordinary MSE for both base and tail; `wmse_tail` uses the existing weighted MSE for both; `mse_wtail` uses ordinary base MSE and weighted tail MSE. Their `_slope` variants use the same tail formula and unchanged slope term. Each event has a fixed weight for a given batch size; an empty-tail batch contributes differentiable zero. Averaging equal-sized batch partitions gives the concatenated loss, including equal-sized microbatches and DDP local batches with normal gradient averaging. Existing accumulation normalization is unchanged. `LossConfig.tail_frac` defaults to `0.05` and receives the existing `--tail_frac`; all existing config values remain unchanged. There is no conditional-reduction compatibility option.

**Dual loss** supervises body, excess and the window-event gate in addition to the selected prediction loss. With `DUAL_ABLATION=none` (default), `DUAL_LOSS=0` and zero branch weights are corrected to 1 with one timestamped warning. Positive weights are retained. Effective settings are saved before training. Baseline/single runs have no dual loss. Excess loss masks non-event windows and averages over the whole `B × K`, without dividing by event count or prevalence.

`EXCEEDANCE_PERCENTILE=95` / `--exceedance_percentile 95` defines the dual threshold from TRAIN window maxima. `TAIL_FRAC=0.05` controls the final prediction's tail auxiliary loss independently: dual events use `max(Y) > tau`, while tail loss keeps `max(Y) >= tail_threshold`. The gate is initialized from the actual strict-event TRAIN prevalence, including ties. Checkpoints and summaries save `dual_metadata` with `tau_phys`, `event_prior`, `gate_init_prior`, event count, TRAIN window count, percentile and ablation. Only learned-logit initialization clips prevalence to `[1e-6, 1-1e-6]`; the empirical value is preserved.

Use a named experiment to disable supervision deliberately:

| `DUAL_ABLATION` / `--dual_ablation` | Active branch losses | Gate |
|---|---|---|
| `none` | body, excess, BCE | learned |
| `no_gate_bce` | body, excess | learned through prediction loss |
| `no_excess_loss` | body, BCE | learned |
| `no_branch_supervision` | none; prediction loss only | learned |
| `fixed_gate` | body, excess | constant empirical TRAIN prevalence |

Inactive weights are set to 0; active zero weights among the original body/excess/gate terms are restored to 1. These modes require a PACT dual head. Matched ready-to-run profiles are in [`configs/dual_ablations`](configs/dual_ablations), each sourcing Stable Dual. For example: `bash train.sh configs/dual_ablations/train_config_NCEP_Battery_Fixed_Gate.sh`.

Optional [physical excess peak-amplitude supervision](docs/EXCESS_AMPLITUDE.md) adds `EXCESS_AMP_LOSS_WEIGHT` (default `0`), `EXCESS_AMP_POOL` (default `max`) and `EXCESS_AMP_BETA` (default `20.0`, inverse meters for active smoothmax only). It converts each horizon to meters before pooling and uses the exact saved strict-event TRAIN `event_prior` as a fixed normalization. Zero weight skips the new computation; the optional weight is never forced to one. `no_excess_loss` and `no_branch_supervision` disable amplitude supervision too; `no_gate_bce` and `fixed_gate` can retain it. Architecture, inference and checkpoint selection remain unchanged. Dual diagnostic exports also report true-event excess-amplitude RMSE, MAE, bias, and predicted/target amplitude means.

Optional [severity-shape excess decomposition](docs/SEVERITY_SHAPE.md) is selected with `EXCESS_FORMULATION="severity_shape"`; the default `direct` keeps the post-#2 head, state dict and predictions. The new head predicts physical severity in meters and a nonnegative temporal shape with per-window maximum one, then divides their product by each horizon's saved TRAIN `y_std`. It reuses #2 amplitude supervision and adds only `SHAPE_LOSS_WEIGHT` (default `0`, dimensionless loss normalized by the fixed TRAIN event prior) and `SEVERITY_SHAPE_EPS` (default `1e-6`). Original trajectory supervision remains available. Both excess-removing ablations also disable shape supervision. New checkpoints record the formulation/scale/epsilon, old checkpoints load as direct, and optional diagnostics export severity and shape only when present.

Normalization (`zscore`, `robust`, `mag`), augmentation, year splits/shuffling/future filtering, external test roots and station feature switches are retained. `mag` only checks its used upper percentile (`0 < x_p_hi <= 100`) and ignores `x_p_lo`; `robust` still requires `0 <= x_p_lo < x_p_hi <= 100`. The statistics formulas are unchanged. Training and inference strip surrounding whitespace from head and temporal names before recognizing their existing values and the `attn` alias. Forcing inputs use the five-channel `[u,v,p',lon,lat]` layout, with spatial-mean pressure removal during preprocessing.

The Python data interface is `ForcingGraphStore(root_dir, station_filter=None, *, pattern="*graphs.pt")`. The original default filename glob is restored; use a keyword pattern such as `pattern="*_fixed315_hist48_graphs.pt"` to select a subset before loading. Filenames retain the year/year/station convention; standard `_graphs.pt` suffixes and custom matched extensions are parsed as before. Data loads onto CPU with strict station filtering. Training derives input/output dimensions from the first graph in the actual TRAIN split, including after year filtering or shuffling.

PACT station metadata keeps two independent switches: `--use_site_elevation` defaults to `1`, and `--use_bathymetry` to `0` (shell: `USE_SITE_ELEVATION` and `USE_BATHYMETRY`). JSON filenames are tried in exact, lowercase, then uppercase station-name order. Coordinate aliases are `lat`/`latitude`/`Latitude` and `lon`/`longitude`/`Longitude`; elevation accepts `elevation_m`/`elevation`/`elev_m`. The first present alias is used. Coordinates and each enabled optional field must contain finite numeric values; missing or invalid selected fields raise an error. Disabled optional fields are ignored. Missing station JSON raises an error when metadata is requested for a station; `--use_station_meta 0` explicitly selects the learned station token. Inference for the training station uses saved checkpoint features; changing stations loads JSON with the saved feature switches.

The original training protocol is restored: DDP seeds each process with `seed + rank`; its samplers use the original default seed 0. Z-score statistics use the training device and distributed sums, preserving the original FP32/FP64 arithmetic order. Robust/mag statistics use TRAIN node samples; `x_nodes_per_graph <= 0` means at most 256 nodes per graph, without reducing the model's input graph. Augmentation at probability 1 skips the probability draw, preserving the original RNG sequence. The original lower bounds remain `1e-6` for `wmse_s`/`slope_mask_s` and `1e-12` for `slope_charb_eps`/`slope_huber_delta`; normal configured values are unaffected.

Cosine with linear warmup remains the configured scheduler. The original `rop` option and its controls are also available. `--deterministic 0` and `--max_grad_norm 0` are defaults; positive `max_grad_norm` explicitly requests clipping. There is no retry or rollback guard. The redundant `--stable_arch` / `STABLE_ARCH` setting is removed; the current stability architecture always applies. `--dual_mode` remains and accepts only `exceedance`. Bare switches such as `--amp`, `--tf32`, `--pin_memory` and `--persistent_workers` retain their original argparse behavior. Shipped profiles enable BF16 and TF32.

## Metrics and artifacts

Each epoch prints `[YYYY-MM-DD|HH:MM:SS]`, followed by exactly eight errors in the target's physical units (surge **meters**):

| Split | All windows | Top 5% peak windows |
|---|---|---|
| Train | RMSE, MAE | RMSE, MAE |
| Val | RMSE, MAE | RMSE, MAE |

Top 5% means the `ceil(0.05 × N)` windows with the largest true maximum over the forecast horizon, within the whole split. Errors include all forecast positions in those windows. Validation restores dataset order and the original NumPy argsort/FP32 peak calculation; online Train metrics use stable sample-ID ordering for ties. This reporting subset is independent of the TRAIN threshold used by the losses.

Train metrics reuse online training predictions, including dropout/augmentation and changing weights; Val uses eval mode. There is no second diagnostic validation pass. DDP validation restores DistributedSampler padding: Val All includes duplicate padding samples, as in the original trainer; Val Peak counts each sample once. Val All also retains the original FP32 batch means and FP64 weighted accumulation. Training uses padding for synchronized updates, but its reported metrics count each sample once.

After training, rank 0 prints the best epoch's complete **Val and Test** metrics together. Val is read from that checkpoint, using the same metrics and sampler-padding convention that selected it; it is not the last epoch's Val. Test evaluates the reloaded best weights once over each actual test sample and exports those predictions. Reporting Val adds no forward pass or diagnostics.

A launcher run can contain many combinations. Each has a unique stem in its artifact filenames:

- `metrics_<stem>.jsonl`: epoch plus eight errors; best model selection uses **Val All RMSE**.
- `best_<stem>.pth`: model config/weights, normalization, station features, exact split tags and training config.
- `config_<stem>.json`: resolved Python arguments; `config_used.sh` retains the resolved shell sweep.
- `summary_<stem>.json`: best epoch, training/validation-loop seconds, total `wall_seconds`, complete best-checkpoint `val` and `test` errors, and test scope.
- `test_preds_<stem>.npz`: physical `y_true`, `y_pred` and tags for the best checkpoint.
- `train_<tag>.log`: launcher command, timestamped epoch metrics, best-checkpoint Val/Test metrics and final wall time. Long log names are shortened with a hash; the complete tag remains inside the log.

`training_seconds` includes epoch metrics/checkpoint overhead. `wall_seconds` also includes data preparation and the final test. Each training process prints its final wall time; the launcher ends with wall time for the whole sweep.

## Inference

```bash
bash infer.sh configs/configs_infer/infer_config_NCEP.sh
bash infer_multi.sh configs/configs_infer/infer_multi_config_NCEP.sh
python infer.py --ckpt /path/to/best_<stem>.pth \
  --root_dir /path/to/NCEP/graphs --out_dir predictions/Battery --save_npz
```

Set `CKPT_PATH` in the inference config to a newly trained checkpoint or glob. Inference loads only the current checkpoint format. The original model/head/temporal/feature arguments validate the saved architecture; they do not silently change it. Omitted station features are read from the checkpoint.

The default evaluates exact held-out test tags saved during training. `--test_root_dir` evaluates all years in an external directory. `--scope all` is an explicit alternative for evaluating the supplied root, and `--years` filters the selected scope. The scope is recorded in `metrics.json`. `--save_npz` controls prediction export. Loader, AMP and TF32 options use their inference CLI settings.

For dual checkpoints with TRAIN event metadata, `--dual_diagnostics` (shell: `DUAL_DIAGNOSTICS=1`) additionally writes `dual_diagnostics.npz` and `dual_diagnostics.json`. Arrays contain aligned `y_true`, `y_pred`, `tags`, `gate_probability`, `body_phys`, `excess_phys`, `contribution_phys`, `event`, and the TRAIN threshold/prior. JSON reports overall/per-year Brier, stepwise average precision, trapezoidal PR-AUC, reliability bins, event/non-event gate histograms and conditional errors. Undefined metrics and empty bins are `null`; no threshold is fitted on evaluation data. This explicit export also works without `--save_npz`; regular prediction files retain their original three fields, and epoch training/validation logs do not collect these diagnostics.

Inference reports each year's physical RMSE, MAE, sample count and runtime, then sample-weighted ALL/past/future errors. Past is 1979–2014 and future is 2070–2099, classified by the starting year. The average annual runtime excludes `2014_2015`, while accuracy still includes its samples. Other years contribute to ALL only. Reports are computed even without `--save_npz`.

The original files are restored: `metrics_per_year_<target>_<station>_<model><architecture_suffix>.json` and, when requested, `preds_<target>_<station>_<model><architecture_suffix>_ALLYEARS.npz` with `y_true`, `y_pred`, `tags`. The automatic directory again uses `<station>_<label>_<source>_To_<target>_<YYYYMMDD_HHMMSS>/outputs/`. `metrics.json` and `predictions.npz` are also retained for current v2 tools. Every batch still has one model forward, with timestamped logs and a final wall time.

`infer.sh` preserves tmux and also supports `USE_TMUX=0` for direct execution. `infer_multi.sh` preserves `RUNS=("Name|test_root" ...)`. Both save the resolved config and exact replay command.

## Preprocessing and checks

The original five `preprocessing_forcing_CMIP6_{AWI,CNRM,EC_EARTH,MPI,MRI}_Mean_Removal.py` entrypoints and their per-model default paths are restored. They preserve NPY and NPZ outputs, pressure-mean removal and 3h→6h subsampling. The additional `preprocessing/forcing_cmip6.py` CLI remains available with explicit `--model`, `--forcing_dir`, `--grid_netcdf` and `--out_dir`; it now saves NPY as well. NCEP preprocessing also restores NPY alongside NPZ.

Simulation preprocessing restores the original mesh reader, station mapping, default paths and default year **2005**. Use `--years` to process other years explicitly. Time alignment restores both `fixed315_hist48` and `peryear_hist48` builders, `--out_root_fixed`/`--out_root_peryear`, missing-year skip behavior and the explicit fallback for legacy CSVs without timestamps. `--out_root` remains an alias for `--out_root_peryear`. As in the original file, the default main loop runs only the per-year builder; the fixed builder is available through `process_one_pair(version="fixed", ...)`.

```bash
python -m unittest discover -s tests
```

Tests cover architecture variants, label independence, gradients, dual supervision, physical metrics, accumulation, station features, year splits, preprocessing alignment and fresh checkpoint inference. Old diagnostic scripts and compatibility tests have been removed from this folder.

`environment_training.yml` and `environment_dataprep.yml` are restored byte-for-byte from the original Emulator, including the training recipe's PyTorch 2.8+cu128 and original data-preparation versions. Restoring these files does not modify an installed environment. The full test suite needs the data-preparation packages (`pandas`, `netCDF4`, `xarray`) as well as torch/PyG.
