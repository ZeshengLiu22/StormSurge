# Evaluation metrics

The implementation is [`evaluate_metrics`](../emulator/training/metrics.py). Training, validation, final checkpoint evaluation, inference, and event diagnostics use this metric dictionary. Inputs are physical predictions and targets shaped `[N, K]`, plus the fixed TRAIN threshold `tau_physical`. Computation converts these arrays to NumPy float64. Evaluation never refits the extreme threshold or selects extremes by a split-dependent rank.

Every final metric report identifies the configured TRAIN hourly target percentile, physical threshold in meters, and strict comparison `y > tau`. Prediction exports carry the threshold and schema metadata. A target-domain evaluation reuses the source TRAIN threshold unchanged.

Let $y_{ih}$ and $\hat y_{ih}$ denote truth and prediction in meters; $d_{ih}=\hat y_{ih}-y_{ih}$. Define all target positions $A$, extreme positions $X=\{(i,h):y_{ih}>\tau\}$, predicted extreme positions $P=\{(i,h):\hat y_{ih}>\tau\}$, and GT Event Windows $W=\{i:\exists h,(i,h)\in X\}$. Let $C=\{(i,h):i\in W\}$ contain **every horizon** of a qualifying window. See [Formulation](FORMULATION.md) for the threshold and timestamp contract.

For any nonempty finite error vector $v$, define $R(v)=\sqrt{\operatorname{mean}(v^2)}$, $M(v)=\operatorname{mean}(\lvert v\rvert)$, and $B(v)=\operatorname{mean}(v)$. Underprediction percentage for a paired population is $U_\delta=100\operatorname{mean}(\mathbf1[\hat y<y-\delta])$. Comparisons are strict, including at tolerances 0, 0.005, 0.010, and 0.020 meters. Positive signed error means overprediction; negative means underprediction.

In the tables, **E/F** means available during TRAIN and VAL epochs, final VAL/TEST evaluation, and inference; **F** means final evaluation/inference/scientific diagnostics with target timestamps. Epoch metrics describe the predictions generated during that epoch; training parameters can change between batches. Final evaluation uses one frozen checkpoint. Down arrows mean smaller is better, up arrows larger is better, and “zero” means distance from zero is preferable. Count/rate diagnostics have no optimization direction. A smaller underprediction percentage describes less underprediction but does not penalize overprediction by itself.

Empty regression/timing populations and undefined fractions are Python `None`, written as JSON `null` and displayed as `NA`. Counts are zero for empty populations; prevalence rates are `null` when the total denominator is zero. Precision is `null` with no predicted extreme hours; recall is `null` with no GT extreme hours. F1 is zero if either set is nonempty and there are no true positives, and `null` if both sets are empty. Episode keys are omitted when timestamps are not supplied. Nonfinite predictions, targets, or thresholds fail explicitly.

## 1. Overall

| Metric | Formula | Population | Unit | Sign | Better | Available | Internal key |
| --- | --- | --- | --- | --- | --- | --- | --- |
| AllRMSE | $R(d_A)$ | All $NK$ target hours | m | Nonnegative | ↓ | E/F | `all_rmse` |
| AllMAE | $M(d_A)$ | All $NK$ target hours | m | Nonnegative | ↓ | E/F | `all_mae` |

Every supervised target timestamp counts once within a split. Distributed evaluation removes only padded copies of the same sampler sample ID. Dataset timestamp duplication is an error, not a deduplication rule.

## 2. Extreme-hour metrics

Regression and underprediction metrics use individual **GT target hours with $y>\tau$**. They do not include other horizons merely because their window contains an extreme. Let $TP=\lvert X\cap P\rvert$.

| Metric | Formula | Population | Unit | Sign | Better | Available | Internal key |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ExtremeHourN | $\lvert X\rvert$ | GT extreme hours | hours/count | Nonnegative | Diagnostic | E/F | `extreme_hour_n` |
| ExtremeHourRate | $\lvert X\rvert/(NK)$ | All target hours | Fraction [0,1] | Nonnegative | Diagnostic | E/F | `extreme_hour_rate` |
| ExceedanceRMSE | $R(d_X)$ | GT extreme hours $X$ | m | Nonnegative | ↓ | E/F | `exceedance_rmse` |
| ExceedanceMAE | $M(d_X)$ | GT extreme hours $X$ | m | Nonnegative | ↓ | E/F | `exceedance_mae` |
| ExceedanceBias | $B(d_X)$ | GT extreme hours $X$ | m | Prediction − truth | Zero | E/F | `exceedance_bias` |
| ExceedanceUnder% | $U_0(X)$ | GT extreme hours $X$ | % [0,100] | Counts negative errors | ↓ underprediction | E/F | `exceedance_under_pct` |
| ExceedanceUnder5mm% | $U_{0.005}(X)$ | GT extreme hours $X$ | % [0,100] | Error below −5 mm | ↓ underprediction | E/F | `exceedance_under_5mm_pct` |
| ExceedanceUnder10mm% | $U_{0.010}(X)$ | GT extreme hours $X$ | % [0,100] | Error below −10 mm | ↓ underprediction | E/F | `exceedance_under_10mm_pct` |
| ExceedanceUnder20mm% | $U_{0.020}(X)$ | GT extreme hours $X$ | % [0,100] | Error below −20 mm | ↓ underprediction | E/F | `exceedance_under_20mm_pct` |
| ExceedancePrecision | $TP/\lvert P\rvert$ | Predicted extreme hours $P$ | Fraction [0,1] | Nonnegative | ↑ | E/F | `exceedance_precision` |
| ExceedanceRecall | $TP/\lvert X\rvert$ | GT extreme hours $X$ | Fraction [0,1] | Nonnegative | ↑ | E/F | `exceedance_recall` |
| ExceedanceF1 | $2TP/(\lvert X\rvert+\lvert P\rvert)$ | GT/predicted extreme-hour classification | Fraction [0,1] | Nonnegative | ↑ | E/F | `exceedance_f1` |

### Lead-wise diagnostics

For every lead $h=0,\ldots,K-1$, use $X_h=\{i:y_{ih}>\tau\}$. Lead zero is the target at the forecast reference time $t$; later targets are $t+h$ hours. These use the same fixed threshold. `include_leadwise=True` is the canonical API default.

| Metric | Formula | Population | Unit | Sign | Better | Available | Internal key for lead zero |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ExtremeHourN@Lead0 | $\lvert X_0\rvert$ | GT extremes at lead 0 | hours/count | Nonnegative | Diagnostic | E/F | `extreme_hour_n_lead_0` |
| ExceedanceRMSE@Lead0 | $R((d_{i0})_{i\in X_0})$ | GT extremes at lead 0 | m | Nonnegative | ↓ | E/F | `exceedance_rmse_lead_0` |

Substitute the integer lead in the key suffix for each subsequent horizon. A lead with no extreme hours has count zero and RMSE `null`.

## 3. Event-window metrics

A window qualifies when any GT horizon exceeds $\tau$. The trajectory metrics then score **all K horizons** of that window, including normal hours.

| Metric | Formula | Population | Unit | Sign | Better | Available | Internal key |
| --- | --- | --- | --- | --- | --- | --- | --- |
| EventWindowN | $\lvert W\rvert$ | GT Event Windows | Windows/count | Nonnegative | Diagnostic | E/F | `event_window_n` |
| EventWindowRate | $\lvert W\rvert/N$ | All forecast windows | Fraction [0,1] | Nonnegative | Diagnostic | E/F | `event_window_rate` |
| EventWindowRMSE | $R(d_C)$ | All horizons of GT Event Windows | m | Nonnegative | ↓ | E/F | `event_window_rmse` |
| EventWindowMAE | $M(d_C)$ | All horizons of GT Event Windows | m | Nonnegative | ↓ | E/F | `event_window_mae` |

## 4. Window-peak metrics

For each $i\in W$, define $p_i=\max_h\hat y_{ih}-\max_h y_{ih}$. The predicted maximum can occur at a different horizon from the GT maximum. Each GT Event Window contributes one error.

| Metric | Formula | Population | Unit | Sign | Better | Available | Internal key |
| --- | --- | --- | --- | --- | --- | --- | --- |
| WindowPeakRMSE | $R(p)$ | One maximum-amplitude error per GT Event Window | m | Nonnegative | ↓ | E/F | `window_peak_rmse` |
| WindowPeakMAE | $M(p)$ | One maximum-amplitude error per GT Event Window | m | Nonnegative | ↓ | E/F | `window_peak_mae` |
| WindowPeakBias | $B(p)$ | One maximum-amplitude error per GT Event Window | m | Predicted maximum − GT maximum | Zero | E/F | `window_peak_bias` |

## 5. GT-aligned peak metrics

For each $i\in W$, let $h_i^*=\operatorname{first\ argmax}_h y_{ih}$ and $a_i=\hat y_{i,h_i^*}-y_{i,h_i^*}$. The prediction is sampled at the **GT maximum's horizon**, regardless of where its own maximum occurs. Underprediction percentages use these same aligned pairs.

| Metric | Formula | Population | Unit | Sign | Better | Available | Internal key |
| --- | --- | --- | --- | --- | --- | --- | --- |
| GTAlignedPeakRMSE | $R(a)$ | GT-peak horizon in every GT Event Window | m | Nonnegative | ↓ | E/F | `gt_aligned_peak_rmse` |
| GTAlignedPeakMAE | $M(a)$ | GT-peak horizon in every GT Event Window | m | Nonnegative | ↓ | E/F | `gt_aligned_peak_mae` |
| GTAlignedPeakBias | $B(a)$ | GT-peak horizon in every GT Event Window | m | Prediction at GT peak − GT peak | Zero | E/F | `gt_aligned_peak_bias` |
| GTAlignedPeakUnder% | $U_0$ on aligned pairs | GT-peak horizon in every GT Event Window | % [0,100] | Counts negative aligned errors | ↓ underprediction | E/F | `gt_aligned_peak_under_pct` |
| GTAlignedPeakUnder5mm% | $U_{0.005}$ on aligned pairs | GT-peak horizon in every GT Event Window | % [0,100] | Aligned error below −5 mm | ↓ underprediction | E/F | `gt_aligned_peak_under_5mm_pct` |
| GTAlignedPeakUnder10mm% | $U_{0.010}$ on aligned pairs | GT-peak horizon in every GT Event Window | % [0,100] | Aligned error below −10 mm | ↓ underprediction | E/F | `gt_aligned_peak_under_10mm_pct` |
| GTAlignedPeakUnder20mm% | $U_{0.020}$ on aligned pairs | GT-peak horizon in every GT Event Window | % [0,100] | Aligned error below −20 mm | ↓ underprediction | E/F | `gt_aligned_peak_under_20mm_pct` |

## 6. Peak timing

Let $\hat h_i^*=\operatorname{first\ argmax}_h\hat y_{ih}$ for each GT Event Window. Both GT and predicted ties use the first horizon. Production targets have interval $\Delta t=1$ hour; the metric API accepts `target_interval_hours` explicitly.

| Metric | Formula | Population | Unit | Sign | Better | Available | Internal key |
| --- | --- | --- | --- | --- | --- | --- | --- |
| PeakTimingMAESteps | $\operatorname{mean}_{i\in W}(\lvert\hat h_i^*-h_i^*\rvert)$ | GT Event Windows | Horizon steps | Nonnegative | ↓ | E/F | `peak_timing_mae_steps` |
| PeakTimingMAEHours | $\Delta t\operatorname{mean}_{i\in W}(\lvert\hat h_i^*-h_i^*\rvert)$ | GT Event Windows | Hours | Nonnegative | ↓ | E/F | `peak_timing_mae_hours` |

## 7. Event-episode metrics

`gt_event_episodes` reconstructs chronological GT target hours, checks timestamp uniqueness **within each split**, and partitions strict exceedances into maximal contiguous runs. Adjacent hours must differ by exactly 3600 seconds. A normal hour, missing hour, or split boundary breaks a run. Single-hour episodes are retained. Episodes may cross forecast-window boundaries. Their membership is determined entirely by GT and the saved threshold, so every compared model uses the same intervals.

Let $G_j$ be episode $j$, $J$ its total count, $t_j^*$ the earliest GT peak timestamp, and $\hat t_j^*$ the earliest predicted peak timestamp **within the same $G_j$**. Define

$$
p_j=\max_{t\in G_j}\hat y_t-\max_{t\in G_j}y_t,
\qquad a_j=\hat y_{t_j^*}-y_{t_j^*},
$$

$$
q_j=\sum_{t\in G_j}\left[\max(\hat y_t-\tau,0)-\max(y_t-\tau,0)\right]\times1\text{ hour}.
$$

The excess area is a sum of hourly values; it is not trapezoidal integration. All episode reductions weight episodes equally, independent of duration. No prediction-defined episode matching occurs.

| Metric | Formula | Population | Unit | Sign | Better | Available | Internal key |
| --- | --- | --- | --- | --- | --- | --- | --- |
| EpisodeN | $J$ | Maximal contiguous GT extreme-hour runs | Episodes/count | Nonnegative | Diagnostic | F | `episode_n` |
| EpisodePeakRMSE | $R(p)$ | One peak error over the exact timestamps of each GT episode | m | Nonnegative | ↓ | F | `episode_peak_rmse` |
| EpisodePeakMAE | $M(p)$ | One peak error over the exact timestamps of each GT episode | m | Nonnegative | ↓ | F | `episode_peak_mae` |
| EpisodePeakBias | $B(p)$ | One peak error over the exact timestamps of each GT episode | m | Predicted maximum − GT maximum | Zero | F | `episode_peak_bias` |
| EpisodeGTAlignedPeakRMSE | $R(a)$ | Earliest GT peak timestamp of each GT episode | m | Nonnegative | ↓ | F | `episode_gt_aligned_peak_rmse` |
| EpisodeGTAlignedPeakMAE | $M(a)$ | Earliest GT peak timestamp of each GT episode | m | Nonnegative | ↓ | F | `episode_gt_aligned_peak_mae` |
| EpisodeGTAlignedPeakBias | $B(a)$ | Earliest GT peak timestamp of each GT episode | m | Prediction at GT peak − GT peak | Zero | F | `episode_gt_aligned_peak_bias` |
| EpisodeGTAlignedPeakUnder% | $100\operatorname{mean}_j\mathbf1[a_j<0]$ | Earliest GT peak timestamp of each GT episode | % [0,100] | Counts negative aligned errors | ↓ underprediction | F | `episode_gt_aligned_peak_under_pct` |
| EpisodePeakTimingMAEHours | $\operatorname{mean}_j\lvert\hat t_j^*-t_j^*\rvert$ in hours | Exact timestamps of each GT episode | Hours | Nonnegative | ↓ | F | `episode_peak_timing_mae_hours` |
| EpisodeDetectionRecall | $\operatorname{mean}_j\mathbf1[\exists t\in G_j:\hat y_t>\tau]$ | GT episodes | Fraction [0,1] | Nonnegative | ↑ | F | `episode_detection_recall` |
| EpisodeExcessAreaMAE | $M(q)$ | Excess integrals over the exact timestamps of each GT episode | m·h | Nonnegative | ↓ | F | `episode_excess_area_mae` |
| EpisodeExcessAreaBias | $B(q)$ | Excess integrals over the exact timestamps of each GT episode | m·h | Predicted area − GT area | Zero | F | `episode_excess_area_bias` |

### Worked comparison: five different RMSE populations

Use the same nine consecutive hours for all metrics below, divided into three illustrative three-horizon forecast windows. Set $\tau=2$ m. This shortened example makes the reductions visible; the production horizon count comes from the dataset.

| Hour | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Window | A | A | A | B | B | B | C | C | C |
| Truth (m) | 0 | 3 | 5 | 4 | 0 | 0 | 0 | 0 | 6 |
| Prediction (m) | 2 | 6 | 2 | 3 | 2 | 1 | 0 | 2 | 4 |
| Signed error (m) | 2 | 3 | −3 | −1 | 2 | 1 | 0 | 2 | −2 |

- GT extreme hours are **1, 2, 3, 8**. Their errors are $[3,-3,-1,-2]$, giving ExceedanceRMSE $=\sqrt{23/4}\approx2.397916$ m.
- All three windows contain an extreme hour. EventWindowRMSE includes all nine errors: $\sqrt{36/9}=2$ m.
- GT window maxima are $[5,4,6]$ and predicted window maxima are $[6,3,4]$. WindowPeakRMSE $=\sqrt{(1^2+(-1)^2+(-2)^2)/3}=\sqrt2\approx1.414214$ m.
- GT maxima occur at hours **2, 3, 8**. Predictions at those hours give errors $[-3,-1,-2]$. GTAlignedPeakRMSE $=\sqrt{14/3}\approx2.160247$ m.
- GT episodes are **hours 1–3** and **hour 8**. The first episode crosses the A/B window boundary. GT episode maxima are $[5,6]$; predictions over the same episode timestamps have maxima $[6,4]$. EpisodePeakRMSE $=\sqrt{(1^2+(-2)^2)/2}=\sqrt{5/2}\approx1.581139$ m.

Thus all five quantities differ, although they use identical predictions, truth, and threshold. The episode excess areas are GT $[6,4]$ m·h and predicted $[5,2]$ m·h, so EpisodeExcessAreaMAE is 1.5 m·h and EpisodeExcessAreaBias is −1.5 m·h. These values are checked in [the numerical metric tests](../tests/test_hourly_metrics.py).

[Event diagnostics and Figures 6–8](../tools/event_diagnostics.py) use this same GT-defined episode population and exact episode timestamps. Saved split labels keep episodes isolated when an export contains multiple source splits. [Dual branch diagnostics](../emulator/inference/dual_diagnostics.py) additionally report gate calibration and body/raw-excess errors; those mechanism diagnostics do not replace the forecast metrics above or enter checkpoint selection.

Severity-bin diagnostics apply the same reductions to episodes in each shared GT severity bin. They additionally expose `episode_gt_aligned_peak_under_5mm_pct`, `episode_gt_aligned_peak_under_10mm_pct`, and `episode_gt_aligned_peak_under_20mm_pct`: respectively $100\operatorname{mean}_j\mathbf1[\hat y_{t_j^*}<y_{t_j^*}-\delta]$ for $\delta=0.005,0.010,0.020$ m over that bin's GT episodes. These are final scientific diagnostics, in percent; smaller means less underprediction beyond the stated tolerance, with the same caveat about overprediction. Unpopulated quantile bins are omitted from the diagnostic tables. They supplement the 42 canonical forecast metrics and are not epoch or checkpoint-selection terms.
