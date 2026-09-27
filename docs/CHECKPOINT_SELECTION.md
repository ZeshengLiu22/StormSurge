# Checkpoint selection

[CheckpointTracker](../emulator/training/checkpoint_selection.py) retains two roles
from one training trajectory. Each keeps the earliest strict minimum (`<`);
equal scores keep the first epoch. Selection reads only the completed VAL pass.

| Role | Checkpoint | Score to minimize | Use |
| --- | --- | --- | --- |
| `exceedance` | `best_exceedance.pt` | VAL ExceedanceRMSE | Primary research result |
| `overall` | `best_overall.pt` | VAL AllRMSE | Conventional sensitivity result |

Both errors use physical meters and the saved TRAIN threshold. Required VAL
scores must be finite and nonnegative. An empty VAL exceedance population fails
clearly. Episode metrics are used for final evaluation.

<!-- choices checkpoint_selection: overall,exceedance -->

`CHECKPOINT_SELECTION` / `--checkpoint_selection` defaults to `exceedance`.
It controls the `best.pt` and `best_<run-stem>.pth` aliases, top-level final
VAL/TEST summary, and `test_preds_<run-stem>.npz` export. Both role files are
always retained, including when their selected epoch is the same.

A checkpoint factory is called once per improving epoch. Each saved role records
its epoch, VAL metrics, `selection_metric_key`, `selection_metric_value`, and
`selection_split="val"`. Model, normalization, split tags, TRAIN threshold,
and Single episode peak metadata travel together. Replacement is atomic and
requires a fresh run output directory. Epoch logs contain only the two
`checkpoint_scores`, plus hourly metrics and training loss components.

## Final evaluation

After training, each role is loaded independently and reevaluated on VAL and TEST
with explicit target timestamps. This produces exactly the eleven corrected
[metrics](METRICS.md), `val_predictions_<role>.npz`, and
`test_predictions_<role>.npz`. TEST never selects an epoch.

`checkpoint_comparison.json`, `checkpoint_comparison.md`, and the run summary
report both roles using the same metric definitions. The compact console displays
water-level errors in mm and timing in hours. Saved errors remain in meters.

The active [Single Tail × EpisodePeak 4×4](../configs/single_tail_episodepeak_4x4/README.md)
uses these two roles, with primary `exceedance` and secondary `overall`.
The completed [Single G/T](../configs/s0_refresh/README.md) and
[Dual G/E/T](../configs/wqe_factorial_multickpt/README.md) specifications remain
available for reproduction with the same current roles. Config generation and
dry runs do not start training.

Historical `aligned_peak` and `bea` selectors are retired. Saved results from
completed studies retain their original selection and metric schemas; current
reruns use the two roles and eleven final metrics above. Git history retains
the previous selection implementation.
