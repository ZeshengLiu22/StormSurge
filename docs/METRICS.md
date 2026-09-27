# Canonical forecast metrics

The public evaluator is [metrics.py](../emulator/training/metrics.py).
It consumes physical predictions and GT with shape `[windows, horizons]`.
Final VAL/TEST evaluation supplies matching target timestamps and returns exactly
**11 metrics**. Epoch evaluation returns the four hourly metrics used below.
Errors and biases are stored in meters; the compact console displays millimeters.
Timing is stored and displayed in hours. Undefined populations serialize as `null`
and display as `NA`.

## Populations and timestamps

All metrics reuse the fixed physical TRAIN hourly Q95 threshold from
[Formulation](FORMULATION.md), with strict `GT > tau`. AllRMSE/MAE use all target
hours; ExceedanceRMSE/MAE use only those strict exceedances.

`gt_event_episodes` sorts the target timestamps within each split. Consecutive
exceedances join only when timestamps differ by exactly 3600 seconds. A normal
hour, missing hour, or split boundary terminates the episode. Single-hour episodes
are valid. Episodes can cross forecast blocks. There is no merging or padding.
Duplicate timestamps within a split fail explicitly; identical timestamps in
separate splits belong to separate physical series.

For episode $E_j$, define $t_j^*$ as the **earliest** timestamp attaining its GT
maximum, and $\hat t_j^*$ as the earliest timestamp attaining the prediction
maximum within the same GT episode. The shared `gt_episode_peak_indices` helper
also supplies the TRAIN loss targets. Predictions never define the episode.

Define $R(d)=\sqrt{\operatorname{mean}(d^2)}$,
$M(d)=\operatorname{mean}|d|$, and $B(d)=\operatorname{mean}(d)$.
Every episode has equal weight in an episode reduction.

| Metric | Error / formula | Population | Unit | Key |
| --- | --- | --- | --- | --- |
| AllRMSE | $R(\hat y-y)$ | All hours | m | `all_rmse` |
| AllMAE | $M(\hat y-y)$ | All hours | m | `all_mae` |
| ExceedanceRMSE | $R(\hat y-y)$ | Strict GT exceedance hours | m | `exceedance_rmse` |
| ExceedanceMAE | $M(\hat y-y)$ | Strict GT exceedance hours | m | `exceedance_mae` |
| EpisodePeakRMSE | $R(\hat y_{\hat t_j^*}-y_{t_j^*})$ | GT episodes | m | `episode_peak_rmse` |
| EpisodePeakMAE | $M(\hat y_{\hat t_j^*}-y_{t_j^*})$ | GT episodes | m | `episode_peak_mae` |
| EpisodePeakBias | $B(\hat y_{\hat t_j^*}-y_{t_j^*})$ | GT episodes | m | `episode_peak_bias` |
| EpisodeGTAlignedPeakRMSE | $R(\hat y_{t_j^*}-y_{t_j^*})$ | GT episodes | m | `episode_gt_aligned_peak_rmse` |
| EpisodeGTAlignedPeakMAE | $M(\hat y_{t_j^*}-y_{t_j^*})$ | GT episodes | m | `episode_gt_aligned_peak_mae` |
| EpisodeGTAlignedPeakBias | $B(\hat y_{t_j^*}-y_{t_j^*})$ | GT episodes | m | `episode_gt_aligned_peak_bias` |
| EpisodePeakTimingMAEHours | $\operatorname{mean}_j|\hat t_j^*-t_j^*|/3600$ | GT episodes | h | `episode_peak_timing_mae_hours` |

Bias is prediction minus GT. Smaller RMSE/MAE is better; bias closer to zero is
better. The aligned metric samples the prediction at the GT peak time. The
[Single episode loss](LOSSES.md#single-episode-gt-aligned-peak-amplitude-loss)
uses that same amplitude error, squared. It adds no timing supervision.

Both retained [checkpoints](CHECKPOINT_SELECTION.md) receive the same fresh final
VAL and TEST evaluation, including timestamps. Canonical JSON dictionaries,
Markdown tables, and console summaries contain these same eleven metrics.
Counts and prevalences belong to TRAIN metadata; offline Dual branch and episode
severity diagnostics have separate tables.

## Numerical example

GT `[[0,3,5],[4,0,0],[0,0,6]]`, predictions
`[[2,6,2],[3,2,1],[0,2,4]]`, hourly times 0 through 8, and tau 2 produce
GT episodes at hours `[1,2,3]` and `[8]`. The GT peaks are hours 2 and 8.
Their aligned errors are -3 and -2, so EpisodeGTAlignedPeakRMSE is
$\sqrt{13/2}$ m. Predicted maxima within those episodes give errors +1 and -2,
so EpisodePeakRMSE is $\sqrt{5/2}$ m. EpisodePeakTimingMAEHours is 0.5 h.

The implementation and numerical tests also cover timestamp order, ties,
missing hours, split boundaries, strict threshold precision, and empty episodes.
