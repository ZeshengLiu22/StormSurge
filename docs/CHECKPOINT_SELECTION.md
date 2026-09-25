# Checkpoint selection

[`CheckpointTracker`](../emulator/training/checkpoint_selection.py) always retains four checkpoint roles from **one training trajectory**. Each role keeps the earliest strict minimum (`<`) of its score; ties never use a secondary ranking.

Selection observes the completed VAL pass once per epoch. TEST is evaluated after training and never enters checkpoint selection. Every metric uses the same physical threshold fitted on TRAIN target hours; see [Formulation](FORMULATION.md) and [Metrics](METRICS.md).

## Exact scores

Let A = VAL AllRMSE, E = VAL ExceedanceRMSE, and P = VAL GTAlignedPeakRMSE, all in physical meters.

| Role | Checkpoint | Selection score to minimize |
| --- | --- | --- |
| `overall` | `best_overall.pt` | A |
| `exceedance` | `best_exceedance.pt` | E |
| `aligned_peak` | `best_aligned_peak.pt` | P |
| `bea` (Balanced Event-Aware) | `best_bea.pt` | 0.50A + 0.25E + 0.25P |

Exceedance minimizes error over strict hourly exceedances above the fixed TRAIN Q95 threshold. Aligned peak minimizes prediction error at the ground-truth peak-aligned time in each GT Event Window.

BEA uses **raw physical RMSE values** with fixed coefficients 0.50/0.25/0.25. Its weights are not configurable. There is no normalization, reference-model scaling, running minimum, or future-epoch information. Each score can be calculated immediately.

All three VAL components must be present, finite, and nonnegative. An empty VAL extreme population produces undefined conditional RMSE and an informative selection error. There is no automatic fallback.

Bias, underprediction percentages, precision, recall, F1, timing, and episode metrics remain diagnostics outside checkpoint selection.

## Primary role and saved artifacts

<!-- choices checkpoint_selection: overall,exceedance,aligned_peak,bea -->

`CHECKPOINT_SELECTION` / `--checkpoint_selection` supports exactly these four roles, with default `overall`. It selects which retained role is copied into `best.pt` and `best_<run-stem>.pth`, and which supplies `best_epoch`, the top-level final VAL/TEST summary, and `test_preds_<run-stem>.npz`.

All four role files are retained even when several select the same epoch. The tracker invokes its checkpoint factory once for an improving epoch, so all winning roles contain the same training state. Each role records its selected epoch, VAL metrics, selection metric key and value, and artifact path. Thresholds, normalization, model configuration, and split tags travel with the model. Role replacement uses [atomic persistence](../emulator/training/checkpoints.py); a run requires a fresh output directory.

Epoch JSONL records contain TRAIN metrics, VAL metrics, and `bea_score`. Checkpoints record `bea_score`, `bea_settings` with the fixed weights, and `selection_split="val"`. The selection summary records `selected_bea_score`, `bea_settings`, and each role's `bea_score`; `selection_metric_value` is the actual score minimized by that role.

Checkpoint bookkeeping does not change model construction, losses, optimizer updates, data order, scheduler configuration, or training RNG. The primary role changes aliases and reporting only.

## Final evaluation

After training, [`train.py`](../train.py) loads each retained model and evaluates complete VAL and TEST datasets: **eight final evaluation passes**, including when roles selected identical epochs. These frozen-model passes include all canonical metrics, timestamp-based episodes and excess-area metrics, and per-lead exceedance diagnostics. Final evaluations never feed observations back into the tracker.

Every role gets `val_predictions_<role>.npz` and `test_predictions_<role>.npz`; dual models include branch diagnostics. `checkpoint_comparison.json` contains all four roles' epochs, paths, selection metadata, complete final metrics, and threshold metadata. `checkpoint_comparison.md` lists each selected epoch and groups complete metric tables by split and metric family, with four role columns. The run summary contains the primary role's fresh VAL/TEST metrics and the full comparison.

See the [tracker tests](../tests/test_checkpoint_selection.py) and [pipeline tests](../tests/test_checkpoint_pipeline.py) for exact scores, four distinct winners, earliest ties, trajectory parity, aliases, and TEST isolation.

## Fresh WQE/Tail factorial

```bash
python tools/generate_configs.py --family wqe_factorial_multickpt
DRY_RUN=1 bash train.sh configs/wqe_factorial_multickpt/train_config_NCEP_CBBT_G0_E0_T0.sh
```

This creates 32 matched configs: CBBT, Boston, Battery, and Lewes × global MSE/WQE × raw-excess MSE/WQE × strict Q95 Tail-MSE off/on (0/0.025). Every config uses direct Dual, the existing WQE parameters and D0 training settings, zero amplitude/shape losses, and primary `overall`. Body remains MSE and Tail remains the existing extra MSE.

All configs, the [manifest](../configs/wqe_factorial_multickpt/manifest.csv), the [full launch script](../configs/wqe_factorial_multickpt/launch_all.sh), and eight four-station launch subsets live in `configs/wqe_factorial_multickpt`. Jobs are ordered by cell then CBBT, Boston, Battery, Lewes. Results use fresh `NCEP_<station>_Gg_Ee_Tt__<timestamp>` directories under `/home/exouser/media/share/PACT/All_results_0922_wqe_factorial_multickpt`. Generation and dry runs submit no jobs. Invoke a launch script explicitly to submit its jobs.

The factorial configs explicitly select `/home/exouser/.conda/envs/torchpyg-cu12x/bin/python`, matching the successful WQE runs. This prevents queue/tmux shell initialization from selecting the base Anaconda Python. Export `WQEF_PYTHON_BIN` to deliberately use another training interpreter. Each batch launcher checks training imports and CUDA availability before submitting any jobs; a failed check submits none.
