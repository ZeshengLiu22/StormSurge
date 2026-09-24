# Forecasting task and hourly extreme formulation

This document defines the task, units and extreme populations used by all output
heads, objectives, checkpoint roles and evaluation tools.

## 1. Physical target and actual temporal structure

The supervised target is storm-surge residual water level in meters. The graph
preprocessor constructs each target vector as `float32(nc) - float32(nc_tide)`;
these physical `graph.y` values are the labels consumed by training. Input
normalization does not alter stored labels.

For forecast origin $t_i$, configured forcing history $H$ hours and current
preprocessed horizon count $K=6$, the task is

$$
(X_{t_i-H},X_{t_i-H+6},\ldots,X_{t_i})
\longrightarrow (y_{t_i},y_{t_i+1},\ldots,y_{t_i+5}). \tag{1}
$$

- Forcing history is sampled every **six hours**, including current forcing.
- `history_hours` must be a nonnegative multiple of six; sequence length is
  $T=H/6+1$. The stored source graphs contain nine forcing slices spanning
  48 hours. `ForcingGraphView` keeps the requested final $T$ slices.
- Targets are sampled every **one hour**, and the first target is **at $t_i$**.
  Lead indices 0–5 therefore mean 0–5 hours from the origin.
- The model obtains `out_channels` from the TRAIN graph target width. Current
  preprocessing and audited station datasets use six horizons; tensor utilities
  also support other explicitly timestamped hourly widths.
- History contains forcing fields and coordinates; it does not add historical
  target water levels to the supervised target population.

`preprocessing/time_align_unified.py` places origins at six-hour intervals from
November 1 00:00 UTC within each available winter season. For sample index $i$,
`hour_start = 6 * time_index` and the source target slice is `[6*i:6*i+6]`.
Thus adjacent target index sets are

$$
\{6i,\ldots,6i+5\},\qquad\{6i+6,\ldots,6i+11\}, \tag{2}
$$

whose intersection is empty. This construction permits overlapping **forcing
histories** while keeping supervised **target blocks disjoint**. Seasonal or
missing-data gaps are not filled during evaluation.

The production timestamp contract uses integer UTC seconds. A graph must contain
explicit `target_timestamps` or a recorded `center_time`; the latter expands to
`center_time + [0,1,...,K-1] hours`. When both are provided, lead zero must equal
`center_time`. Every block must have consecutive hourly horizons. Missing temporal
metadata or repeated supervised target timestamps raises an informative error.
There is no filename-based guess and no silent target deduplication.

The data-only audit checks every actual graph's `center_time`, `hour_start`,
`time_index`, target width and `nc - nc_tide` reconstruction. It additionally
checks uniqueness across the entire station dataset, including split boundaries.

## 2. Split construction and normalization

`ForcingGraphStore.split` assigns complete season/year groups to TRAIN, VAL and
TEST. Defaults are `train_ratio=0.6`, `val_ratio=0.2`, `shuffle_years=0` and
`seed=42`. Available year-group names are sorted; optional seeded shuffling occurs
before assignment. Split counts use Python `round` with small-dataset guards. At least one group
per split is retained when at least three groups exist; training requires
nonempty TRAIN and VAL groups. Within each
split, graph indices are sorted. Splits and seeds are saved with the checkpoint.

The station data audit records the exact resulting group names. For the current
36-season NCEP datasets, TRAIN has 22 seasons (1979–1980 through 2000–2001), VAL
has seven (2001–2002 through 2007–2008), and TEST has seven (2008–2009 through
2014–2015). These memberships are determined by the available files and split
configuration, not hard-coded year lists in the trainer.

Target means $\mu_h$ and scales $\sigma_h$ are fitted on TRAIN targets per
horizon. `fit_statistics` accumulates target sums and squares in FP64, casts
moments to FP32 and uses $\sigma_h=\sqrt{v_h+10^{-6}}$. Normalized targets and
physical predictions are

$$
z_{ih}=(y_{ih}-\mu_h)/\sigma_h,\qquad
\widehat y_{ih}=\mu_h+\sigma_h\widehat z_{ih}. \tag{3}
$$

Feature normalization is described in [BACKBONE](BACKBONE.md). Neither feature
normalization nor target normalization determines the extreme population.

## 3. The one TRAIN hourly threshold

Let $\mathcal T_{\mathrm{train}}$ contain every unique supervised TRAIN target
timestamp exactly once. With `exceedance_percentile` / `EXCEEDANCE_PERCENTILE`
(default 95), set $p=\text{exceedance\_percentile}/100$, where $0<p<1$:

$$
\tau=Q_p\bigl(\{y_t:t\in\mathcal T_{\mathrm{train}}\}\bigr). \tag{4}
$$

`fit_loss_thresholds` calls `supervised_targets`, orders physical labels by their
verified timestamps and computes exactly:

```python
tau = float(np.quantile(hourly_values, exceedance_percentile / 100., method="linear"))
```

The labels are stored physical values promoted to FP64 for fitting. For sorted
zero-indexed values $x_0,\ldots,x_{n-1}$, let $u=(n-1)p$, $j=\lfloor u\rfloor$
and $a=u-j$. Linear interpolation gives
$Q_p=(1-a)x_j+a x_{j+1}$, with the usual endpoint when $a=0$.

Only `graph.y` from TRAIN indices enters this calculation. History/input tensors,
VAL, TEST and OOD labels are excluded. Duplicates stop execution; values are not
silently deduplicated. Q95 describes approximately the highest five percent of
TRAIN hourly target values; interpolation and tied labels determine the exact
strict exceedance count.

One physical scalar $\tau$, stored as `tau_physical`, applies to every lead.
Its normalized representation is the length-$K$ vector

$$
\tau_h^{(z)}=(\tau-\mu_h)/\sigma_h, \tag{5}
$$

stored as `tau_normalized`. This vector represents the same physical water level
under per-horizon normalization. Metric and training masks compare physical
labels in FP64 to preserve strictness near floating-point threshold boundaries.

## 4. Three distinct ground-truth populations

**Extreme Hour:** an individual target hour satisfying $y_t>\tau$. Equality
is not extreme. Its true physical excess is $e_t^*=\max(y_t-\tau,0)$.

**Event Window:** a supervised forecast block with at least one Extreme Hour:
$E_i=\mathbf1[\exists h:y_{ih}>\tau]$. A window may contain both extreme and
ordinary hours. The gate target is $E_i$; Event Window trajectory metrics score
all its horizons.

**Event Episode:** a maximal chronological contiguous sequence of GT Extreme
Hours. Adjacency is exactly one hour, gap tolerance is zero, and minimum duration
is one extreme hour. A sub-threshold hour, missing timestamp or split boundary
breaks the episode. An episode may cross a forecast-window boundary. Predictions
never determine episode membership, and independently detected predicted episodes
are not matched to GT episodes.

For example, take $\tau=0.20$ m and two adjacent six-hour blocks:

| Hour | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| GT (m) | .10 | .20 | .30 | .35 | .10 | .25 | .40 | .10 | .20 | .10 | .10 | .10 |
| Extreme | no | no | yes | yes | no | yes | yes | no | no | no | no | no |

There are four Extreme Hours, two Event Windows and two Event Episodes:
`[2,3]` and `[5,6]`. The second episode crosses the block boundary. If hour 6
belongs to another split, `[5]` and `[6]` are separate episodes. Removing hour 3
would also interrupt continuity; missing timestamps are never bridged.

## 5. Transfer, metadata and diagnostics

The NCEP source TRAIN threshold is reused unchanged for that station/model's VAL,
TEST, CMIP6/OOD and future inference. `infer.py` requires saved threshold metadata
and never calls a threshold fitter. In-source `--scope all` evaluation preserves
saved TRAIN/VAL/TEST labels for episode isolation; an external domain is evaluated
as its own series using the same source threshold.

Every run and checkpoint saves `metric_schema="hourly_q95_v1"`,
`threshold_schema="train_hourly_q95_v1"`, `exceedance_percentile`, `tau_physical`,
`tau_normalized`, `train_target_hour_count`, `train_extreme_hour_count`,
`train_extreme_hour_rate`, `train_event_window_count`, `train_event_window_rate`
and `train_episode_count`. Threshold metadata also includes the quantile method,
TRAIN Event Window prevalence (`event_prior`) and fit provenance.

[METRICS](METRICS.md) defines reductions on each population. The offline diagnostic
tool freezes GT episode timestamps and IDs once for all compared models. Shared
GT severity quantile bins describe that fixed population; binning does not change
the extreme threshold or membership.

Implementation sources: [target preprocessing](../preprocessing/time_align_unified.py),
[timestamp validation](../emulator/data/targets.py),
[store and view](../emulator/data/graph_store.py),
[statistics and threshold fitting](../emulator/data/stats.py),
[trainer](../train.py), [inference](../infer.py), and
[data audit](../tools/audit_hourly_threshold.py).
