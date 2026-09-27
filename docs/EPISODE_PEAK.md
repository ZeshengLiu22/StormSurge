# Episode-level GT-aligned peak supervision

## Scientific motivation

The active Single objective supervises three populations:

1. **GlobalMSE:** all supervised hours.
2. **Tail-MSE:** every strict exceedance hour above the fixed TRAIN Q95.
3. **EpisodeGTAlignedPeak-MSE:** one GT peak amplitude per physical extreme
   episode, so long episodes do not receive extra peak targets.

The current [Single Tail × EpisodePeak 4×4](../configs/single_tail_episodepeak_4x4/README.md)
varies the two additional weights while keeping the global loss as MSE.

## Episode definition

Fit $\tau$ as Q95 of unique physical TRAIN target hours, as specified in
[FORMULATION](FORMULATION.md). Within each split, sort unique timestamps
chronologically and form maximal sequences with strict $GT>\tau$ and exactly
3600-second continuity.

- A non-exceedance hour (including $GT=\tau$), missing hour, or split boundary
  breaks an episode. Gaps are never bridged.
- Single-hour episodes are valid. Episodes may cross forecast-block boundaries.
- For episode $E_j$, choose $t_j^*$ as the earliest timestamp attaining its GT
  maximum. Prediction does not define the episode or this GT peak timestamp.

[episode_peaks.py](../emulator/training/episode_peaks.py) builds TRAIN targets
once using the evaluator's `gt_event_episodes` and `gt_episode_peak_indices`.
The boolean `is_episode_gt_peak` marks exactly one target per episode;
`episode_id` is -1 outside episodes. These fields follow samples through
shuffling and batching.

## Training objective

The physical objective is exactly:

```text
GlobalMSE + lambda_T * TailMSE + lambda_P * EpisodeGTAlignedPeakMSE
```

For a minibatch with $M$ target points, let $d=\hat y-y$,
$H=\mathbf1[y>\tau]$, and $P=\mathbf1[\mathrm{is\_episode\_gt\_peak}]$.
Let $N$ be the number of unique TRAIN target points, $J$ the number of physical
TRAIN episodes, $q_H$ the strict TRAIN exceedance-hour fraction, and
$p=J/N$ the fixed canonical TRAIN peak prevalence. Then

$$
\mathrm{GlobalMSE}=\frac1M\sum d^2,\qquad
\mathrm{TailMSE}=\frac1{Mq_H}\sum H d^2,\qquad
\mathrm{EpisodeGTAlignedPeakMSE}=\frac1{Mp}\sum P d^2.
$$

The means include zeros at unselected points. Both prevalences are fixed from
TRAIN; the denominators never use a minibatch's exceedance or peak count.
Over the full TRAIN representation, the episode term becomes

$$
\frac1{Np}\sum_{j=1}^J(\hat y_{t_j^*}-y_{t_j^*})^2
=\frac1J\sum_{j=1}^J(\hat y_{t_j^*}-y_{t_j^*})^2.
$$

Thus each physical episode has equal intended weight, independent of duration
or the number of forecast blocks it crosses. All three terms have units m².
A batch without canonical peaks has an exact differentiable zero peak loss.
Positive episode weight requires a finite, positive TRAIN prevalence.

Uniform shuffling without replacement gives equal expected weight per episode.
Gradient accumulation retains the average of microbatch means; unequal
microbatches can have different realized per-point coefficients. Epoch loss
components instead use target-count weighting to recover the exact traversal
mean. Those logs use predictions encountered during training, while final
evaluation uses a fixed checkpoint. See [LOSSES](LOSSES.md) for all reductions.

No forecast-window peak objective or timing objective is active in this Single
family.

## Relationship to evaluation

All episode metrics use the same GT episodes. Let $\hat t_j^*$ be the earliest
timestamp of the prediction maximum within GT episode $E_j$.

| Quantity | Per-episode comparison |
| --- | --- |
| EpisodePeak | Predicted maximum $\hat y_{\hat t_j^*}$ versus GT maximum $y_{t_j^*}$; each may occur at a different time. |
| EpisodeGTAlignedPeak | Prediction $\hat y_{t_j^*}$ versus GT $y_{t_j^*}$ at the canonical GT peak time. |
| EpisodePeakTiming | Absolute difference between $\hat t_j^*$ and $t_j^*$ in hours. |

The loss directly supervises the **EpisodeGTAlignedPeak amplitude** quantity.
It does not supervise timing. For fixed predictions on the same population,
its full-dataset value equals EpisodeGTAlignedPeakRMSE squared. Final VAL/TEST
metrics average the episodes in their own split, using the saved TRAIN threshold.

[METRICS](METRICS.md) lists exactly 11 canonical final metrics. The two
[checkpoint roles](CHECKPOINT_SELECTION.md) are `exceedance` (primary, minimum
VAL ExceedanceRMSE) and `overall` (secondary conventional sensitivity, minimum
VAL AllRMSE). Episode metrics are computed during final evaluation.

## Saved TRAIN audit

The saved NCEP Grid4_New audit uses chronological 60/20/20 splits and seed 42.
Source provenance and numeric checks are in [audit.json](audits/episode_peak/audit.json).

| Station | TRAIN Q95 (m) | TRAIN points | Episodes = canonical targets | Fixed $p=J/N$ | Cross-window episodes | Single-hour episodes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| CBBT | 0.191533852369 | 79344 | 185 = 185 | 0.002331619278 | 125 | 20 |
| Lewes | 0.251904280484 | 79344 | 198 = 198 | 0.002495462795 | 155 | 15 |
| Battery | 0.383809146285 | 79344 | 437 = 437 | 0.005507662835 | 211 | 69 |
| Boston | 0.314852280915 | 79344 | 542 = 542 | 0.006831014317 | 197 | 248 |

All 1,362 episodes have one peak target, including 688 crossing forecast blocks
and 352 lasting one hour. Each station has 13,224 TRAIN windows and zero
duplicate TRAIN target timestamps. The saved data have no tied GT maxima;
synthetic tests cover the earliest-maximum tie rule.

## Sampling constraints

Every TRAIN target timestamp must occur exactly once. The saved representation
has disjoint six-hour target blocks; overlapping forcing histories are allowed.
Overlapping target blocks are rejected before supervision.

Positive episode weight requires traversal without replacement or dropped
samples. Padded or truncated distributed sampling is rejected because it can
repeat or omit canonical peaks. Use one process or an unpadded distributed
traversal with a world size dividing the TRAIN window count. The sampler is
validated without changing its behavior.

## Configuration

`EPISODE_GT_ALIGNED_PEAK_WEIGHT` defaults to **0.0**. `train.sh` forwards it as
`--episode_gt_aligned_peak_weight`. A positive value requires Single, global
MSE, Tail MSE, and TRAIN Q95. The Tail coefficient remains
`EXCEEDANCE_LOSS_WEIGHT`. Zero episode weight skips the additional term.

TRAIN count, prevalence, threshold, and construction rules are saved in
`episode_peak_metadata.json`, checkpoints, and run summaries. The
[active family README](../configs/single_tail_episodepeak_4x4/README.md) defines
the exact weight grid, anchor cells, and manifests.

## Verification

The [saved validation record](audits/episode_peak/validation.json) is the
implementation verification snapshot:

- **Saved-data parity:** all four stations match an independent earliest-peak
  oracle and squared final EpisodeGTAlignedPeakRMSE. Maximum discrepancy:
  $2.17\times10^{-19}$ m².
- **Zero-weight regression:** loss values and gradients match the previous
  GlobalMSE + Tail objective bitwise in FP32/FP64 at three Tail weights.
- **Episode-count equality:** every physical TRAIN episode has exactly one
  canonical peak target.
- **Cross-window episodes:** saved-data and synthetic checks preserve episodes
  spanning forecast blocks; separate tests cover missing hours, split boundaries,
  single-hour episodes, ties, and duplicate rejection.
- **Test-suite result:** the recorded implementation run passed 245 tests and
  649 subtests, with no failures or skips.

Current numerical checks are in [test_episode_peak_loss.py](../tests/test_episode_peak_loss.py).
The [saved-data audit tool](../tools/audit_episode_peak_targets.py) reconstructs
targets and checks parity without training.
