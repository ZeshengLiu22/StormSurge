# Validation-only checkpoint selection

## Controls and metric mapping

| Shell setting | CLI option | Default |
| --- | --- | --- |
| `CHECKPOINT_SELECTION` | `--checkpoint_selection` | `overall` |
| `CHECKPOINT_OVERALL_TOL` | `--checkpoint_overall_tol` | `0.01` |
| `SAVE_AUX_CHECKPOINTS` | `--save_aux_checkpoints` | `0` |

Tolerance is a finite, nonnegative relative fraction. Auxiliary saving uses
`0`/`1`, `false`/`true`, or `no`/`yes`. These settings appear in resolved config
JSON, `config_used.sh`, and checkpoint `training_config`, and contribute to
the config hash.

Exactly five modes are supported:

| Mode | Exact VAL source key | Feasible checkpoints |
| --- | --- | --- |
| `overall` | `rmse_all` | All completed epochs |
| `peak5` | `rmse_peak5` | All completed epochs |
| `true_peak` | `true_peak_rmse_top5` | All completed epochs |
| `constrained_peak5` | `rmse_peak5` | Within the final overall limit |
| `constrained_true_peak` | `true_peak_rmse_top5` | Within the final overall limit |

The trajectory metrics AllRMSE, AllMAE, Top5RMSE, and Top5MAE evaluate every
forecast horizon. The top-5% population contains
`max(1, ceil(0.05 * N))` windows selected by their true physical window maxima,
with the existing validation/inference sorting and tie behavior.

For canonical amplitude metrics, define the true peak horizon as
`h* = argmax_h y[h]`, with the first occurrence winning ties. The signed
error is `prediction[h*] - y[h*]`. TruePeakRMSE, TruePeakMAE, TruePeakBias,
and TruePeakUnder% use this same-horizon error. TruePeakUnder% is the
percentage of errors strictly below zero. TimingSteps separately measures
`abs(argmax_h prediction[h] - h*)`. Each metric is available for all, top5,
and strict event populations; event membership uses the saved TRAIN threshold.
The difference `max_h prediction[h] - max_h y[h]` is no longer canonical
amplitude error because those maxima may occur at different forecast horizons.

The selector receives only aggregated VAL metrics and epoch numbers. TEST
never participates in selection. Selection does not depend on enabling an
auxiliary amplitude loss, and enabling that loss does not change the default
`overall` selection mode.

## Exact constrained selection and ties

After all epochs, let `r_best = min_e VAL_e["rmse_all"]`. The admissible set is

```text
VAL_e["rmse_all"] <= (1 + CHECKPOINT_OVERALL_TOL) * r_best
```

The configured metric is minimized over that set. For `0.01`, the limit is
`1.01 * r_best`; for `0`, only exact overall minima qualify. No numerical
slack is added. A finite tolerance so large that the product overflows admits
all finite RMSE candidates; JSON records the limit as `"unbounded"`.

`overall` retains the first epoch reaching the minimum overall RMSE, preserving
the original strict `<` save rule. Every peak-based role uses the ascending
tuple `(target_metric, rmse_all, epoch)`. Missing or nonfinite selection
metrics fail explicitly. The evaluation pipeline supplies all three required
selection keys.

`CheckpointSelector` in `emulator/training/checkpoints.py` maintains exact
two-dimensional Pareto frontiers for `(rmse_all, rmse_peak5)` and
`(rmse_all, true_peak_rmse_top5)`. A point dominates another if neither
coordinate is worse and at least one is better. Identical points keep the
earliest epoch. A dominator is eligible whenever the dominated point is
eligible, so pruning cannot remove the final constrained winner.

Only the requested constrained frontier is maintained unless auxiliary saving
requests both. The final global best overall defines the limit.
`eligible_candidate_count` counts feasible points on the final frontier;
`frontier_candidate_count` counts all points on that frontier. Files are
retained for necessary frontier candidates and requested simple-best roles.
There is no arbitrary cap: exact Pareto storage can grow with the epoch count.

## Canonical and auxiliary artifacts

Every completed run has one canonical `best_<stem>.pth`. With `overall` and
auxiliary saving disabled, the original online strict-`<` canonical save is
unchanged; no candidate store, auxiliary directory, or manifest is created.
Scalar best-epoch metadata for all three simple roles is still tracked.

Simple peak modes update the canonical file online. Constrained modes publish
it after all epochs and final resolution. The final layout is:

```text
<output_dir>/
  best_<stem>.pth
  metrics_<stem>.jsonl
  summary_<stem>.json
  config_<stem>.json
  config_used.sh
  test_preds_<stem>.npz
  checkpoint_selection_<stem>.json          # non-default mode or auxiliary saving
  aux_checkpoints/<stem>/epoch_<NNNN>.pth    # distinct auxiliary winners
```

During training, required candidates are stored at
`checkpoint_candidates/<stem>/epoch_<NNNN>.pth`. Candidate and auxiliary names
do not match `best_*.pth`. Shared directories are scoped by run stem.

With auxiliary saving enabled, all five roles resolve from the same VAL
history and tolerance. Roles selecting the same epoch share one physical
file. The primary winner uses the canonical file; other distinct winners use
auxiliary files. The manifest lists every shared role explicitly.

A replacement candidate is atomically saved before any unreferenced predecessor
is deleted. Dominance on one frontier cannot delete a checkpoint still needed
by another frontier or a simple-best role. Finalization verifies stored epochs
and VAL dictionaries, atomically publishes the artifacts and manifest, then
removes candidates. Failed writes preserve sources for retry; other runs and
unexpected recovery files remain untouched.

## Summary, manifest and checkpoint metadata

`best_epoch` and `best_val_rmse` describe the primary selected checkpoint.
`val` and `test` contain fresh final evaluations of that checkpoint.
`summary["checkpoint_selection"]` contains:

```text
checkpoint_selection_mode, checkpoint_selection_metric_key
checkpoint_overall_tol, save_aux_checkpoints
selected_epoch
selected_val_rmse_all, selected_val_rmse_peak5
selected_val_true_peak_rmse_top5
best_overall_epoch, best_overall_val_rmse_all
best_peak5_epoch, best_peak5_val_rmse_peak5
best_true_peak_epoch, best_true_peak_val_true_peak_rmse_top5
```

Constrained primaries also include `selected_overall_limit`,
`selected_eligible_candidate_count`, and `selected_frontier_candidate_count`.
The manifest path is included whenever a manifest is produced.

The manifest records the configured rule, tolerance, auxiliary setting, and a
`primary` entry with role, path, epoch, full VAL metrics, and source key. Its
`roles` object has exactly the five supported names; unavailable roles are
`null`. Available entries include the three selection metrics,
`selection_metric_key`, and `shared_roles`. Constrained entries include the
limit and candidate counts.

Checkpoint metadata includes `checkpoint_role`, `checkpoint_roles`,
`checkpoint_candidate`, `selection_metric_key`, `selection_metric_value`,
`checkpoint_selection_mode`, `checkpoint_overall_tol`, and `selection_by_role`.
For shared artifacts the representative role is first, with the primary role
first when applicable. Temporary files are marked as candidates.

## Scheduler, DDP and TEST isolation

Checkpoint selection does not alter losses, optimizer updates, learning rates,
random sampling, augmentation, dropout, gradient accumulation, or model
architecture. ReduceLROnPlateau still uses `rmse_all` for
`rop_metric=val_rmse_phys`, or `rmse_peak5` for `rop_metric=val_rmse_peak`.

Only rank 0 selects and writes artifacts, using globally aggregated VAL
metrics and the existing DDP padding conventions. After the final epoch,
selection and canonical publication precede model reload, full VAL
reevaluation, and one TEST evaluation. The TEST loader is constructed after
selection. Auxiliary saving never triggers extra TEST evaluations. Later
inference uses `infer.py --ckpt <manifest role path>`.

The mode and tolerance should be defined using TRAIN/VAL reasoning. Apply the
same selection rule to comparable models; comparisons using different rules
are checkpoint-selection ablations. Old runs and archived checkpoint/config
interfaces are not reconstructed or adapted.

## Regression coverage

`tests/test_checkpoints.py` covers exact modes and source keys, five distinct
winners, overall and peak ties, constraints, randomized brute-force Pareto
comparisons, deduplicated storage, retention ordering, and failed-write
recovery. `tests/test_checkpoint_pipeline.py` covers all modes and auxiliary
settings, final table values, fixed default predictions, unchanged training
trajectories across selection modes, independent scheduler choices, explicit
auxiliary inference, and two-rank rank-0-only selection and TEST evaluation.
