# Configurable validation-only checkpoint selection

Instruction #7 uses the current post-#6 revision
`d8734bcc66f5bb4cb0b75b908becd7d0c988c1d4` as its base, including the follow-up
that made loss configuration resolution uniform. Implementation and validation
were performed directly in `/home/exouser/media/volume/PACT-Data/StormSurge`.
The existing untracked P1 files and unrelated moves of two documentation files
were preserved. No experiment configurations were generated, no real training
experiments were launched, and Instruction #8 is outside this change.

## Controls and metric mapping

| Shell setting | CLI option | Default |
| --- | --- | --- |
| `CHECKPOINT_SELECTION` | `--checkpoint_selection` | `overall` |
| `CHECKPOINT_OVERALL_TOL` | `--checkpoint_overall_tol` | `0.01` |
| `SAVE_AUX_CHECKPOINTS` | `--save_aux_checkpoints` | `0` |

Tolerance is a finite, nonnegative **relative fraction**, not an absolute RMSE
increment. Auxiliary saving uses the project's boolean parser, including
`0`/`1`, `false`/`true` and `no`/`yes`. Invalid modes, tolerances and booleans fail
at argument parsing. These fields appear in resolved config JSON,
`config_used.sh`, and checkpoint `training_config`, and are included in the
normal config hash. Adding the resolved default fields changes new-run hashes;
historical hashes are not imitated by hiding settings.

| Mode | Exact VAL source key | Feasible checkpoints |
| --- | --- | --- |
| `overall` | `rmse_all` | All completed epochs |
| `peak5` | `rmse_peak5` | All completed epochs |
| `peak_magnitude` | `peak_magnitude_rmse_top5` | All completed epochs |
| `constrained_peak5` | `rmse_peak5` | Within the final overall limit |
| `constrained_peak_magnitude` | `peak_magnitude_rmse_top5` | Within the final overall limit |

`rmse_peak5` measures trajectory error across **all horizons** of the true
split-relative top-5% windows. `peak_magnitude_rmse_top5` measures the error
between the predicted and true **physical window maxima** in those same
windows. The existing post-#6 metric dictionary is used directly; no aliases,
new metric population, or extra forward passes are introduced. In particular,
the fixed-TRAIN event metric is not used for these selection modes.

The selector receives only aggregated VAL metrics and epoch numbers. It does
not inspect loss settings, model architecture, TRAIN predictions, or TEST.
Positive peak loss is not required for peak-based selection, and enabling
peak loss does not change the selection mode. There are no config-age checks
or implicit overrides of explicit amplitude, shape, peak or tail settings.

## Exact constrained selection and ties

After all epochs, let `r_best = min_e VAL_e["rmse_all"]`. The admissible set is

```text
VAL_e["rmse_all"] <= (1 + CHECKPOINT_OVERALL_TOL) * r_best
```

The configured peak metric is minimized over that set. For `0.01`, the limit
is `1.01 * r_best`; for `0`, only exact overall minima qualify. No numerical
slack is added. A finite tolerance so large that the product overflows admits
all finite RMSE candidates; JSON then records the limit as `"unbounded"`.

`overall` retains the **first** epoch reaching the minimum overall RMSE, exactly
as the original strict `<` save rule. Every peak-based role uses the tuple
`(target_peak_metric, rmse_all, epoch)` in ascending order. Filesystem ordering
never participates in selection. Missing or nonfinite selection metrics fail
explicitly; the current evaluation pipeline supplies all three required keys.

`CheckpointSelector` in `emulator/training/checkpoints.py` tracks independent
two-dimensional Pareto frontiers for `(rmse_all, rmse_peak5)` and
`(rmse_all, peak_magnitude_rmse_top5)`. A point dominates another if neither
coordinate is worse and at least one is better. Identical points keep the
earliest epoch. A dominator is always eligible whenever the dominated point
is eligible, and is preferred by the selection tuple, so pruning cannot
remove the final constrained winner. The full-history best overall scalar is
always tracked separately, including its earliest-epoch tie rule.

Only a requested constrained target's frontier is maintained; auxiliary
saving requests both frontiers. At completion the **final global** best overall
defines the limit, and the eligible frontier is filtered and minimized. No
moving best-so-far limit is used to make a final constrained decision.
`eligible_candidate_count` counts feasible points on the **final frontier**;
`frontier_candidate_count` counts all points on that frontier.

All epochs are not saved as a blanket policy. Files are retained only for the
union of necessary frontier candidates and requested simple-best roles. There
is no arbitrary cap or approximate eviction: in the worst case every epoch
is nondominated, so exact Pareto storage can still grow with the epoch count.

## Canonical and auxiliary artifacts

Every completed run has exactly one canonical `best_<stem>.pth`, with the
same stem structure and existing payload fields. `overall` with auxiliary
saving disabled still executes the original online strict-`<` canonical
`torch.save`; no candidate store, candidate directory, auxiliary directory,
or selection manifest is needed. Scalar best-epoch metadata for all three
simple rules is tracked even on that lightweight default path.

Simple peak modes update the canonical file online with their comparison rule.
Constrained primary modes publish it only after all epochs and exact final
resolution. The final layout is:

```text
<output_dir>/
  best_<stem>.pth
  metrics_<stem>.jsonl
  summary_<stem>.json
  config_<stem>.json
  config_used.sh
  test_preds_<stem>.npz
  checkpoint_selection_<stem>.json          # non-default mode or auxiliary saving
  aux_checkpoints/<stem>/epoch_<NNNN>.pth    # distinct non-primary winners, if requested
```

During training, required candidates are stored at
`checkpoint_candidates/<stem>/epoch_<NNNN>.pth`. Candidate and auxiliary names
do not match `best_*.pth`, even in recursive searches. Shared directories are
scoped by the run stem so one run cannot clean another run's files.

With auxiliary saving enabled, all five roles resolve using the same training
history and configured tolerance. Roles selecting the same epoch share one
physical file. The primary epoch uses the canonical file; only other distinct
winner epochs need auxiliary files. If all five choose one epoch, only the
canonical model file is retained and the manifest records all five aliases.

## Safe retention and finalization

`CandidateCheckpointStore` computes the union of files referenced by both
required frontiers and the retained simple-best roles. A replacement candidate
is serialized to a temporary file and atomically renamed **before** any
unreferenced predecessor is deleted. Becoming dominated on one frontier never
deletes a checkpoint that is still required on the other frontier or by a
simple-best role. This also protects the earliest overall winner when later
epochs have equal overall RMSE and better peaks.

Finalization verifies each candidate's stored epoch and complete VAL dictionary,
publishes every required canonical/auxiliary artifact atomically, then publishes
the manifest atomically. Only after all these steps succeed are temporary
candidates removed. A failed save or manifest write leaves source candidates
available for recovery; final materialization can be retried. Unexpected files
and other run directories are preserved. There is no automatic training-resume
feature, and an interrupted write does not imply a completed selection.

## Summary, manifest and checkpoint metadata

Historical `best_epoch`, `best_val_rmse`, `val`, and `test` remain. The first
three describe the **primary selected** checkpoint, whose overall RMSE need
not be the smallest for a peak-based mode. Default `overall` semantics are
unchanged. `summary["checkpoint_selection"]` contains:

```text
checkpoint_selection_mode, checkpoint_selection_metric_key
checkpoint_overall_tol, save_aux_checkpoints
selected_epoch
selected_val_rmse_all, selected_val_rmse_peak5
selected_val_peak_magnitude_rmse_top5
best_overall_epoch, best_overall_val_rmse_all
best_peak5_epoch, best_peak5_val_rmse_peak5
best_peak_magnitude_epoch, best_peak_magnitude_val_peak_magnitude_rmse_top5
```

Constrained primaries also include `selected_overall_limit`,
`selected_eligible_candidate_count`, and `selected_frontier_candidate_count`.
The manifest path is included whenever a manifest is produced.

The manifest records the configured rule/tolerance/auxiliary setting and a
`primary` entry with role, path, epoch, full VAL metrics and exact source key.
Its `roles` object has all five names; unavailable weight roles are `null`.
Each available role has its path, epoch, full `val`, the three selection
metrics, `selection_metric_key`, and explicit `shared_roles`. Constrained
entries also record the limit and candidate counts. Scalar metadata in the
summary does not falsely advertise unavailable model weights as artifacts.

Checkpoints retain their existing model-state semantics and gain
`checkpoint_role`, `checkpoint_roles`, `checkpoint_candidate`,
`selection_metric_key`, `selection_metric_value`, `checkpoint_selection_mode`,
`checkpoint_overall_tol`, and `selection_by_role`. A shared artifact lists every
role; `checkpoint_role` is the representative first role, with the primary
first when applicable. Temporary files are marked as candidates, and their
metadata describes why they were retained at the time they were saved.
Additional fields are ignored safely by existing readers.

## Scheduler, DDP and TEST isolation

The model, losses, optimizer, gradient accumulation, random sampling,
augmentation, dropout, AMP, TF32 and TRAIN/VAL populations are unchanged.
Cosine/warmup configuration is unchanged. ReduceLROnPlateau still steps on
`rmse_all` for `rop_metric=val_rmse_phys`, or `rmse_peak5` for
`rop_metric=val_rmse_peak`, independently of the checkpoint rule.

Only rank 0 constructs the selector, writes checkpoints or publishes the
manifest. It consumes the existing globally aggregated validation metrics,
including the established DDP padding conventions. Other ranks do not select
from their own local shards.

The lifecycle remains `TRAIN -> VAL` each epoch, then final selection,
canonical publication, canonical model reload, and **one TEST evaluation**.
TEST loader construction follows final selection as well. Auxiliary saving
never triggers TEST. Explicit later inference uses the unchanged interface
`infer.py --ckpt <manifest role path>`; no inference selector is added.

## Validation and changed files

The [machine-readable validation report](audit/evidence/checkpoint_selection_validation.json)
and [full test output](audit/evidence/checkpoint_selection_tests.txt) record
the complete regression run. The new coverage includes:

- Seventeen selector/storage tests: five distinct winners; overall/peak/epoch
  ties; one epoch; monotonic sequences; identical points; zero and very large
  tolerances; late overall or peak improvements; invalid inputs; frontier
  activation; exact scalar summaries; duplicate-role storage; cross-frontier
  and overall-role references; save-before-delete ordering; failed candidate
  writes; failed materialization/manifest with retry; corrupt-source rejection;
  and preservation of other runs and recovery files.
- 9,000 exact brute-force comparisons over 250 randomly generated histories,
  two targets, six tolerances and three insertion orders (forward, reverse,
  shuffled), with independent checks of frontier and feasible counts.
- Thirty pipeline combinations of five modes, two auxiliary settings, and
  single/direct/severity-shape heads. Each checks actual epoch weight tensors,
  complete stored VAL metrics, role metadata, unique canonical glob, manifests,
  config snapshots/hash, and exactly one TEST after final selection.
- A frozen default reference captured before edits from `d8734bc`, confirming
  the earliest minimum, historical summary fields, canonical name structure,
  selected model state, exact TEST metrics/predictions and absence of extra
  candidate/auxiliary artifacts.
- Three frozen synthetic training histories from the exact `d8734bc` trainer,
  exercising real CPU optimizer/backprop, dropout, augmentation, accumulation,
  and positive #2/#3/#6 losses where applicable. Across all five modes, six
  epochs per variant match the frozen losses, forward-output hashes, weight
  hashes, metrics, RNG states and learning rates exactly. An additional ten
  runs compare both existing ROP metric choices across all five modes.
- A real two-rank Gloo run verifies equal globally aggregated VAL dictionaries,
  rank-0-only selection/checkpoint/manifest writes, selected artifact contents,
  one canonical file, and one rank-0 TEST after finalization.
- CLI validation, boolean parsing, shell forwarding from an existing source
  config, all ten loss modes with zero/positive peak loss, and explicit auxiliary
  inference through the current `--ckpt` interface.
- All 64 existing P0/P1 configs dry-run unchanged. Source bytes, old command
  arguments and old parsed values match the saved pre-edit reference exactly;
  only the three new resolved fields are added. Twenty-nine existing
  model/loss/data/engine/metric/inference files are byte-identical to the base.

Production changes are limited to `emulator/training/checkpoints.py`,
`emulator/training/arguments.py`, `train.py`, and `train.sh`. Tests add
`tests/test_checkpoints.py`, `tests/test_checkpoint_pipeline.py`, and two
post-#6 JSON references in `tests/fixtures/`. Existing mocked epoch dictionaries
in `tests/test_excess_amplitude.py` and `tests/test_severity_shape.py` now include
the direct peak metric already emitted by the real post-#6 pipeline.
Documentation changes are this report, README, and the audit evidence files.

## Prospective comparison protocol

The selection mode and relative tolerance must be predefined using TRAIN/VAL
reasoning. TEST cannot choose the mode or tolerance. This infrastructure does
not select a scientific protocol for a future study.

For a fair model comparison, apply the same selection rule to every comparable
baseline/model with these validation metrics, including single, dual direct,
and dual severity-shape. Comparing an overall-selected baseline with a
peak-selected proposed model must instead be labeled a checkpoint-selection
ablation. This is protocol guidance; code does not enforce paper-specific
model/loss combinations.

Historical P0/P1 runs retained overall-best weights only. A peak-best epoch
reported in an old log does not create its missing weights. The selector does
not reconstruct old checkpoints or retroactively reinterpret old runs; the
new artifact choices apply to future runs.
