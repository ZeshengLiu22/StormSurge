# Episode GT-aligned peak implementation report

## Outcome

Single now supports the physical objective
`GlobalMSE + EXCEEDANCE_LOSS_WEIGHT * TailMSE + EPISODE_GT_ALIGNED_PEAK_WEIGHT * EpisodeGTAlignedPeakMSE`.
The new weight defaults to **0.0**. Omitted settings preserve the former objective.
Positive weight requires Single with global MSE, Tail MSE and the fixed TRAIN Q95.
No formal training experiment was launched. Tests use small temporary fixtures;
the saved station data were read only. Historical result files were untouched.

## Construction and canonical targets

`train.py` builds metadata once from TRAIN GT, TRAIN target timestamps and the
fixed TRAIN physical Q95. `build_episode_peak_targets` calls the final evaluator's
`gt_event_episodes`, then its shared `gt_episode_peak_indices` helper. These sort
chronologically, use strict `GT > tau`, join only exact 3600-second steps and keep
the earliest GT maximum on ties. Normal/missing hours and split boundaries end
an episode. Single-hour episodes remain valid; forecast boundaries do not split
an episode. There is no event merging, padding, or predicted argmax in the loss.

Every episode has one boolean `is_episode_gt_peak=True`. `episode_id` is also
attached to TRAIN batches, with -1 outside episodes. Metadata is saved separately,
in new checkpoints, and in run summaries.

The saved representation has six targets `[t,...,t+5]` at six-hour forecast
centers. Audits found **zero repeated TRAIN target timestamps** at every station.
Input histories overlap, but target blocks do not. The unique timestamp is the
natural canonical occurrence; a duplicate tie rule is unnecessary. Existing
loader/threshold validation continues to reject overlapping target blocks before
training. GlobalMSE and TailMSE behavior is unchanged.

## Exact reduction

Let N be all eligible target occurrences in TRAIN, J the number of episodes
(equal to the canonical peak count), and p = J/N. For a batch with M target points:

```
EpisodeGTAlignedPeakMSE = sum(mask_peak * squared_error) / (M * p)
```

Over the complete representation, this equals
`sum_j (pred(t_j*) - GT(t_j*))**2 / J`. Thus the intended weight is one per
episode, independent of episode duration or number of forecast blocks crossed.
p is fixed from TRAIN. The numerator includes zeros at every unselected point.
An empty mask produces exact differentiable zero with finite zero peak gradients.
Positive weight rejects a nonpositive or nonfinite TRAIN prevalence.

The existing shuffle visits each sample without replacement. Uniform minibatch
means estimate this same objective; gradient accumulation keeps the existing
average of microbatch means. For unequal microbatches, realised per-point
coefficients vary with shuffled position, but the expectation is equal. Epoch
loss components are instead weighted by target count, recovering the exact mean
of the predictions encountered during the traversal. They are not a frozen
checkpoint evaluation because parameters change during training.

DistributedSampler can pad by repeating samples when TRAIN window count is not
divisible by world size. Positive episode loss **rejects padded/truncated or
replacement sampling** explicitly. It supports single-process traversal and
unpadded distributed traversal. This prevents silent duplicate supervision and
preserves the requested p = J/N without changing sampling. For the audited data,
there are 13,224 TRAIN windows per station.

## Saved TRAIN audit

Source: `Data/Grid4_New/NCEP/graphs`, chronological 60/20/20 splits, seed 42.
The full provenance and numeric checks are in [audit.json](audit.json).

| Station | TRAIN Q95 (m) | TRAIN points | Episodes = canonical targets | Fixed p | Cross-window episodes | Single-hour episodes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| CBBT | 0.191533852369 | 79344 | 185 = 185 | 0.002331619278 | 125 | 20 |
| Lewes | 0.251904280484 | 79344 | 198 = 198 | 0.002495462795 | 155 | 15 |
| Battery | 0.383809146285 | 79344 | 437 = 437 | 0.005507662835 | 211 | 69 |
| Boston | 0.314852280915 | 79344 | 542 = 542 | 0.006831014317 | 197 | 248 |

All 1,362 physical episodes have one peak target, including 688 crossing forecast
blocks and 352 lasting one hour. An independent earliest-timestamp oracle agrees
with each mask. Position-dependent synthetic prediction errors on the saved GT
verify both the equal-episode mean and squared final EpisodeGTAlignedPeakRMSE.
Maximum loss/oracle discrepancy is 2.17e-19 m². Saved data contain no tied episode
maxima; deterministic synthetic tests cover ties and timestamp reordering.

## Config, logs and selection

`EPISODE_GT_ALIGNED_PEAK_WEIGHT` is forwarded by `train.sh` to
`--episode_gt_aligned_peak_weight` and stored in `LossConfig`/resolved snapshots.
Generated configs explicitly use 0.0. No old ambiguous peak-weight option exists.

Single epoch logs include raw GlobalMSE, raw TailMSE, weighted Tail, raw
EpisodeGTAlignedPeakMSE, weighted episode peak, encountered canonical peak count,
fixed TRAIN prevalence, and total loss. JSONL loss records use the same components.

Only `overall` (minimum VAL AllRMSE) and `exceedance` (minimum VAL ExceedanceRMSE)
remain. Exceedance is the default primary research selector; overall is the
conventional sensitivity selector. New checkpoints have no BEA score/settings or
window peak metrics. Removed role names are rejected. Both roles retain earliest
strict minima and receive the same timestamp-aware final VAL and TEST passes.

## Metric cleanup

The formal registry, epoch logs, summaries, inference reports, and checkpoint
comparison tables no longer expose forecast-window peak amplitude/alignment or
window timing metrics. Dead implementations and their tests were removed.
Offline forecast-window peak comparison tables were also removed. Internal GT
population counts still support unchanged Tail/Dual thresholds. Existing Dual
raw-excess amplitude supervision and its separate branch diagnostics remain.

Final metric dictionaries contain exactly eleven entries: AllRMSE/MAE,
ExceedanceRMSE/MAE, EpisodePeakRMSE/MAE/Bias,
EpisodeGTAlignedPeakRMSE/MAE/Bias, and EpisodePeakTimingMAEHours.

## Validation

- Complete pytest suite: **245 passed**, plus **649 passing subtests**, in 66.80 seconds. No failures or skips; 24 dependency/runtime warnings.
- The training environment lacks preprocessing-only `netCDF4` and `xarray`. The full run appended the existing local `cge` environment for `netCDF4` and `/tmp/stormsurge-test-deps` for a temporary xarray install. No project dependencies or training environment packages were changed.
- Saved-data audit: all four stations passed reconstruction, uniqueness, one-target-per-episode, and final-metric parity checks.
- Documentation interface audit: passed, with exactly 11 canonical keys and 25 supported loss fields.
- Shell syntax (`train.sh`, `infer.sh`, `infer_multi.sh`) and `git diff --check`: passed.

Tests used `/home/exouser/.conda/envs/torchpyg-cu12x/bin/python`. The complete
run log and JUnit results are also available for this workspace session at
`/tmp/stormsurge-final-tests.log` and `/tmp/stormsurge-final-tests.xml`.

The requested A–J checks cover:

- A: saved TRAIN reconstruction and final-evaluator parity at all four stations.
- B: saved and synthetic episodes crossing forecast boundaries stay whole.
- C: zero duplicate saved TRAIN timestamps; synthetic overlap is rejected before supervision.
- D: single-hour episodes each contribute a peak.
- E: large predictions away from the GT maximum do not enter the peak loss.
- F: episode count equals canonical supervision target count, with one mask entry per episode.
- G: zero weight matches the previous GlobalMSE + Tail objective **bitwise** in values and gradients, for FP32/FP64 and three Tail weights.
- H: finite, nonzero gradients reach model parameters; empty masks have finite zero peak gradients.
- I: only two selectors can create checkpoints; removed roles fail validation.
- J: each role's saved final VAL/TEST metrics equals direct timestamp-aware reevaluation and contains exactly eleven keys.

Additional checks cover fixed-prevalence batch partitioning, strict threshold
precision, timestamp formats/order, split and missing-hour boundaries, sampler
padding rejection, generated configs, actual fixture training/inference,
checkpoint durability, earliest ties, and selector-independent training trajectories.

## Remaining semantic differences

There is no population, GT peak timestamp or amplitude-target discrepancy between
training and final EpisodeGTAlignedPeak evaluation on the audited representation.
The loss is MSE and the final metric is its square root for fixed predictions on
the same data. Training uses a fixed TRAIN prevalence and stochastic updates;
final VAL/TEST evaluation directly averages their own GT episodes. Unsupported
padded distributed traversal is rejected instead of changing the scientific
normalization or duplicating peak targets.

## Files changed

- [README.md](../../README.md)
- [configs/README.md](../../configs/README.md)
- [configs/s0_refresh/README.md](../../configs/s0_refresh/README.md)
- [configs/wqe_factorial_multickpt/README.md](../../configs/wqe_factorial_multickpt/README.md)
- [docs/CHECKPOINT_SELECTION.md](../../docs/CHECKPOINT_SELECTION.md)
- [docs/FORMULATION.md](../../docs/FORMULATION.md)
- [docs/LOSSES.md](../../docs/LOSSES.md)
- [docs/METRICS.md](../../docs/METRICS.md)
- [emulator/data/graph_store.py](../../emulator/data/graph_store.py)
- [emulator/data/targets.py](../../emulator/data/targets.py)
- [emulator/inference/dual_diagnostics.py](../../emulator/inference/dual_diagnostics.py)
- [emulator/training/arguments.py](../../emulator/training/arguments.py)
- [emulator/training/checkpoint_selection.py](../../emulator/training/checkpoint_selection.py)
- [emulator/training/engine.py](../../emulator/training/engine.py)
- [emulator/training/episode_peaks.py](../../emulator/training/episode_peaks.py)
- [emulator/training/losses.py](../../emulator/training/losses.py)
- [emulator/training/metrics.py](../../emulator/training/metrics.py)
- [emulator/training/reporting.py](../../emulator/training/reporting.py)
- [infer.py](../../infer.py)
- [reports/episode_peak/REPORT.md](../../reports/episode_peak/REPORT.md)
- [reports/episode_peak/audit.json](../../reports/episode_peak/audit.json)
- [reports/episode_peak/validation.json](../../reports/episode_peak/validation.json)
- [tests/test_checkpoint_pipeline.py](../../tests/test_checkpoint_pipeline.py)
- [tests/test_checkpoint_selection.py](../../tests/test_checkpoint_selection.py)
- [tests/test_config_interfaces.py](../../tests/test_config_interfaces.py)
- [tests/test_dual_experiments.py](../../tests/test_dual_experiments.py)
- [tests/test_episode_peak_loss.py](../../tests/test_episode_peak_loss.py)
- [tests/test_fixed_event_diagnosis.py](../../tests/test_fixed_event_diagnosis.py)
- [tests/test_hourly_metrics.py](../../tests/test_hourly_metrics.py)
- [tests/test_inference_reporting.py](../../tests/test_inference_reporting.py)
- [tests/test_model_reporting.py](../../tests/test_model_reporting.py)
- [tests/test_pipeline.py](../../tests/test_pipeline.py)
- [tests/test_severity_shape.py](../../tests/test_severity_shape.py)
- [tests/test_training_reporting.py](../../tests/test_training_reporting.py)
- [tests/test_wqe_factorial_configs.py](../../tests/test_wqe_factorial_configs.py)
- [tools/audit_episode_peak_targets.py](../../tools/audit_episode_peak_targets.py)
- [tools/audit_event_excess_targets.py](../../tools/audit_event_excess_targets.py)
- [tools/event_diagnostics.py](../../tools/event_diagnostics.py)
- [tools/generate_configs.py](../../tools/generate_configs.py)
- [train.py](../../train.py)
- [train.sh](../../train.sh)

The 104 generated training shell configs under `configs/baseline_ablation`,
`configs/wqe`, `configs/s0_refresh`, and `configs/wqe_factorial_multickpt` now
set the zero default peak weight and select primary `exceedance`. Their model,
optimizer, Tail, sampling, and schedule settings are unchanged.
