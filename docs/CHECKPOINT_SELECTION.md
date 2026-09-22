# Checkpoint selection

[`EventAwareTracker`](../emulator/training/eventaware_checkpoints.py) always retains two checkpoint roles from **one training trajectory**:

- `best_overall.pt`: earliest epoch with the smallest VAL AllRMSE.
- `best_eventaware.pt`: earliest epoch with the smallest VAL EventAwareScore.

Selection observes the completed VAL pass once per epoch. TEST is evaluated after training and never enters checkpoint selection. Every metric uses the same physical threshold fitted on TRAIN target hours; see [Formulation](FORMULATION.md) and [Metrics](METRICS.md).

## Exact scores

$$
e_{\mathrm{overall}}=\operatorname*{argmin}_{e}\operatorname{ValAllRMSE}(e).
$$

$$
S_{\mathrm{event}}(e)=w_{\mathrm{all}}\operatorname{ValAllRMSE}(e)
+w_{\mathrm{exc}}\operatorname{ValExceedanceRMSE}(e)
+w_{\mathrm{peak}}\operatorname{ValGTAlignedPeakRMSE}(e),
\qquad e_{\mathrm{eventaware}}=\operatorname*{argmin}_{e}S_{\mathrm{event}}(e).
$$

This is the **raw weighted sum of physical RMSE values**, in meters. The default weights are 0.65, 0.20, and 0.15. There is no reference normalization or score rescaling.

| Term | Canonical metric key | CLI argument | Shell variable | Default |
| --- | --- | --- | --- | ---: |
| Overall | `all_rmse` | `--ckpt_w_all` | `CKPT_W_ALL` | 0.65 |
| Extreme hours | `exceedance_rmse` | `--ckpt_w_exceedance` | `CKPT_W_EXCEEDANCE` | 0.20 |
| GT-aligned event-window peak | `gt_aligned_peak_rmse` | `--ckpt_w_peak` | `CKPT_W_PEAK` | 0.15 |

Weights must be finite, nonnegative, and sum to one within absolute tolerance $10^{-12}$. All three VAL components must be present, finite, and nonnegative, even if a weight is zero. An empty VAL extreme population produces undefined conditional RMSE and therefore an informative selection error. There is no automatic fallback to another score or checkpoint policy.

Bias and underprediction percentages remain diagnostics: signed bias can cancel across examples, and reducing underprediction alone can reward overprediction. Precision, recall, F1, timing, and episode metrics also remain outside the checkpoint score.

## Ties and a three-epoch example

Each role updates only on strict improvement (`<`). Equal overall RMSE retains the earliest overall epoch; equal EventAwareScore retains the earliest event-aware epoch. Neither rule introduces a secondary tie-breaking metric.

| Epoch | VAL AllRMSE (m) | VAL ExceedanceRMSE (m) | VAL GTAlignedPeakRMSE (m) | EventAwareScore (m) | Retained roles after epoch |
| --- | ---: | ---: | ---: | ---: | --- |
| 1 | 0.100 | 0.200 | 0.400 | 0.1650 | Overall: 1; event-aware: 1 |
| 2 | 0.102 | 0.100 | 0.200 | 0.1163 | Overall: 1; event-aware: 2 |
| 3 | 0.102 | 0.100 | 0.200 | 0.1163 | Overall: 1; event-aware: 2 |

For epoch 2, $0.65(0.102)+0.20(0.100)+0.15(0.200)=0.1163$. The two roles therefore select different epochs of the same training run. Epoch 3 changes neither artifact because it ties epoch 2.

## Primary role and saved artifacts

<!-- choices checkpoint_selection: overall,eventaware -->

`CHECKPOINT_SELECTION` / `--checkpoint_selection` supports exactly `overall` and `eventaware`, with default `overall`. It selects which retained role is copied into the primary aliases `best.pt` and `best_<run-stem>.pth`, and which role supplies the top-level final summary and primary TEST prediction export. It does not change model construction, losses, optimizer updates, data order, or scheduler configuration.

Both role files are always retained, including when they point to the same epoch. When both roles improve at an epoch, the tracker invokes its checkpoint factory once, so both artifacts contain identical model state from that epoch. Role metadata identifies its selection metric and value. The saved threshold, normalization, model configuration, split tags, and validation metrics travel with the model. Role replacement uses [atomic persistence](../emulator/training/checkpoints.py); a run requires a fresh output directory.

Epoch JSONL records include TRAIN metrics, VAL metrics, and `eventaware_score`. Checkpoints record `eventaware_settings`, including all weights and `selection_split="val"`. The summary records the selected epoch and both retained role paths.

## Final evaluation

After training, [`train.py`](../train.py) loads each retained model and evaluates the complete VAL and TEST datasets: **four final evaluation passes**, including when both roles chose the same epoch. These frozen-model passes include timestamp-based episode metrics and per-lead exceedance diagnostics. The passes never feed new observations back into the tracker.

Each role gets `val_predictions_<role>.npz` and `test_predictions_<role>.npz`; dual models also include branch diagnostics in these exports. `checkpoint_comparison.json` contains the two roles' epochs, paths, complete final metric dictionaries, and explicit threshold metadata and method label. `checkpoint_comparison.md` groups metrics and reports event-aware minus overall deltas. The run summary exposes the selected primary role's fresh VAL/TEST metrics and the complete two-role comparison. The primary TEST export also uses the `test_preds_<run-stem>.npz` name.

The comparison reports differences without changing either selected epoch. See [the checkpoint tests](../tests/test_eventaware_checkpoints.py) and [pipeline tests](../tests/test_checkpoint_pipeline.py) for exact score, earliest-tie, unchanged-training-trajectory, and TEST-isolation checks.
