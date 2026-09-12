# Optional final-output extreme peak supervision and exact peak metrics

Instruction #6 is implemented on post-#3 revision
`50ec23fc23866cb225c594bb403e1505704cda79`, directly in the existing
`/home/exouser/media/volume/PACT-Data/StormSurge` repository. The tracked tree
was clean at entry; the pre-existing untracked `experiment_config/P1_WeightProbe/`
was preserved. All 120 existing tests passed before source edits. The completed
#2 amplitude and #3 severity/shape implementations remain available.

## Independent controls and exact objective

```bash
PEAK_LOSS_WEIGHT="0"
PEAK_POOL="max"
PEAK_POOL_BETA="20.0"
```

The corresponding Python/CLI options are `peak_loss_weight` /
`--peak_loss_weight`, `peak_pool` / `--peak_pool`, and `peak_pool_beta` /
`--peak_pool_beta`. The weight must be finite and nonnegative; the pool must
be `max` or `smoothmax`. Beta must be finite and positive when a positive
weight activates smoothmax, and has inverse-meter units. It is unused for
hard max or zero weight. All three resolved controls are included in shell
snapshots, Python config JSON and checkpoint `training_config`.

Let `y_i,h` and `yhat_i,h` be the final **physical** truth and prediction in
meters, directly supplied to `ForecastLoss.forward(output, prediction, target)`:

```text
m_true_i = peak_pool(y_i,:)
m_pred_i = peak_pool(yhat_i,:)
E_i      = 1[max_h(y_i,h) > tau_train]
q_E      = TRAIN event_count / TRAIN train_windows
L_peak   = (1 / (B * q_E)) * sum_i E_i * (m_pred_i - m_true_i)^2
L_total  = L_existing_prediction_tail_slope
           + PEAK_LOSS_WEIGHT * L_peak
           + L_existing_dual_branches_amplitude_shape
```

`L_peak` has units of square meters. Masking precedes squaring, so an event-free
batch/rank has exactly zero loss and zero finite gradients even with large
non-event errors. The denominator is fixed from TRAIN, including when the
current minibatch event fraction differs. It is neither `tail_frac`, the
nominal percentile fraction, nor the gate's clipped initialization prior.

`train.py` already fits `fit_loss_thresholds` for single and dual heads. It now
passes `fitted["tau_phys"]` as the new trailing optional `event_threshold`
argument, separately from the existing positional `peak_threshold`, which
still means the **tail** threshold. `fitted["event_prior"]` remains the exact
#2 empirical TRAIN prior. `ForecastLoss` validates these only when required
by the active objective. Existing positional callers remain valid.

The strict event comparison uses the hard physical truth peak in FP64 against
the original fitted threshold, matching `fit_loss_thresholds`' TRAIN event
count even when a percentile lies between neighboring FP32 values. It never
round-trips physical truth through normalized excess targets. Equality at
`tau_train` is excluded; the established tail loss retains its own `>=`
threshold convention unchanged. No VAL/TEST threshold or prior is fitted.

`emulator/training/final_peak.py::final_peak_terms` is a pure reusable helper
returning `(event, m_pred, m_true, loss)`. It imports the existing
`peak_pool` and `validate_event_prior` from `excess_amplitude.py`; there is no
second pooling or prior-validation implementation. The shared prior error
message now refers to event-only supervision generally; its validation and
the #2/#3 numerical operations are unchanged.

Hard `max` is the canonical default. The optional sensitivity mode reuses
#2's **bounded softmax-weighted mean**, centered before multiplying by beta:

```text
smoothmax(v) = sum_h softmax(beta * (v_h - max(v)))_h * v_h
```

The same pool is applied to truth and prediction, so equal trajectories have
exactly zero peak loss in either mode. Increasing beta approaches hard max.
This training surrogate never changes the event population or the hard-max
evaluation metrics below.

## Three distinct scientific quantities

| Objective | Supervised quantity | Population/normalization |
|---|---|---|
| Existing `L_tail` | Entire final physical trajectory | Existing TRAIN tail threshold, fixed `tail_frac` |
| #2 `L_excess_amp` | Peak of the internal ungated physical excess branch | Strict TRAIN events, exact `q_E` |
| #6 `L_peak` | Peak of the reconstructed final physical prediction | Strict TRAIN events, exact `q_E` |

The new helper never uses normalized `output.prediction`, body, excess,
severity or shape separately. It works with baseline/single, dual/direct and
dual/severity_shape. Its gradient follows the final prediction into the
single head/backbone, or body/gate/excess/shared context, or
body/gate/severity/shape/shared context where locally differentiable. Hard-max
tests use a unique predicted peak; tied-max gradient allocation is not assumed.

The optional term is added **before** the dual-supervision block and its
`no_branch_supervision` early return. All four named branch ablations
(`no_excess_loss`, `no_gate_bce`, `fixed_gate`, `no_branch_supervision`) retain
the requested positive peak weight. Existing amplitude/shape ablation behavior
is unchanged. All six amplitude/shape/peak combinations are tested over the
ten existing loss modes. No combinatorial `loss_mode`, asymmetric loss,
checkpoint-selection control or model parameter is added.

With weight zero, `final_peak_terms` is not called at all. The legacy loss
arithmetic, model architecture/state, outputs, gradients, RNG and strict old
checkpoint reconstruction remain unchanged. Metrics are still produced.

For optional training diagnostics, call `terms.diagnostics()` on a result
of `final_peak_terms`. It returns tensor scalars `peak_loss`, `pred_peak_mean`,
`true_peak_mean`, `peak_bias`, `event_count`, `event_fraction`. These are batch
diagnostics: the means/bias cover the whole batch using the training pool;
the loss uses masked strict events. For diagnostic-only use, call the helper
on detached physical tensors under `torch.no_grad()`. There is no mutable
criterion cache or additional automatic forward pass.

## Hard physical evaluation metrics

For each unique sample, define hard `t = max(y)`, `p = max(yhat)`, signed peak
error `e = p - t`, true peak index `j = argmax(y)`, predicted peak index
`k = argmax(yhat)`, point error `d = yhat[j] - y[j]` and timing error
`a = abs(k - j)`. Argmax ties use the standard **first occurrence**. Timing
is in forecast steps, without conversion to hours.

The following eleven stems are each suffixed with `_all`, `_top5` and `_event`,
adding **33 flat metric keys**:

| Stem | Definition over the selected population |
|---|---|
| `peak_magnitude_rmse` | `sqrt(mean(e**2))` |
| `peak_magnitude_mae` | `mean(abs(e))` |
| `peak_bias` | `mean(e)`; negative means underprediction |
| `peak_underprediction_fraction` | `mean(p < t)` |
| `peak_underprediction_mean` | `mean(t - p \| p < t)` |
| `peak_overprediction_fraction` | `mean(p > t)` |
| `peak_overprediction_mean` | `mean(p - t \| p > t)` |
| `true_peak_point_rmse` | `sqrt(mean(d**2))` |
| `true_peak_point_mae` | `mean(abs(d))` |
| `true_peak_point_bias` | `mean(d)` |
| `peak_timing_mae_steps` | `mean(a)` |

For example, `peak_magnitude_rmse_top5` and `peak_timing_mae_steps_event` are
available during each validation epoch. Empty populations and absent
underprediction/overprediction conditional means are `None` / JSON `null`.
An absent TRAIN threshold makes all `_event` metrics unavailable.

The three populations are all unique windows, the legacy split-relative top5
windows, and fixed TRAIN events `t > tau_train`. The shared
`unique_rows_and_top5_indices` first retains the first record for each sample
ID, restoring increasing ID order. It then returns the exact pre-existing
`ceil(0.05*N)` top5 indices. Evaluation preserves FP32 truth sorting with
NumPy quicksort; training and overall inference preserve their prior input
precision/stable ordering. Legacy trajectory top5 and all new top5 metrics
consume these same indices. This also preserves the historical distinction
between evaluation and overall inference tie conventions.

All **new** populations count each unique sample ID once after DDP gathering.
Legacy validation `rmse_all`/`mae_all` deliberately still include sampler
padding, FP32 batch means and FP64 weighted accumulation. Legacy
`rmse_peak5`/`mae_peak5` retain their original FP32 cast/reduction. All four
keys remain unchanged. Old four-column `summarize_windows` callers still
receive their four metrics; normal engine/inference now supply seven columns.

`physical_peak_columns` supplies shared hard-peak features to training,
validation, test and inference. The engine retains only seven scalar values
per window: ID, truth peak, window MSE/MAE, signed peak error, signed true-peak
point error and timing. Three extra scalars suffice; no `[N,K]` VAL archive
or extra inference pass is added. New peak differences are computed in FP64
from physical values, independently of legacy trajectory reduction precision.

## Logs, checkpoints, summary and inference

`format_metrics` defaults to the historical four metrics plus
`peak_magnitude_rmse_top5`. `extended=True` displays the full dictionary when
explicitly requested. Ordinary epoch and best-checkpoint stdout stay concise.
Full Train/Val dictionaries are stored in each JSONL epoch record, checkpoint
`val`, and best-checkpoint `summary.json` Val/Test fields. The final Test uses
the same engine and peak metric implementation.

Ordinary `infer.py` reports all direct peak metrics without
`--dual_diagnostics`, for all three head formulations. Overall metrics are
in `metrics.json` and the established per-year JSON report; each year's entry
also includes the expanded dictionary. Prediction exports keep their original
`y_true`, `y_pred`, `tags` fields. The saved threshold is taken from
`dual_metadata["tau_phys"]` when present, otherwise
`loss_thresholds["tau_phys"]` (including old single checkpoints). Reports
include `event_threshold_phys` and `event_threshold_source`. When neither
exists, event metrics are null; all/top5 metrics remain available.
Legacy unrelated empty past/future group/timing NaN conventions remain intact.

Production checkpoint selection **still minimizes validation `rmse_all`**.
A regression test deliberately gives the peak metric and `rmse_all` conflicting
epoch rankings and verifies that the latter determines the saved checkpoint.
TEST is evaluated only after reloading that checkpoint and has no selection role.

## Changed files and validation

Runtime changes are in `emulator/training/final_peak.py`, `losses.py`,
`arguments.py`, `excess_amplitude.py` (shared error wording only), `metrics.py`,
`engine.py`, plus `train.py`, `train.sh` and `infer.py`. Documentation is in
this file and the root README. New tests are `test_final_peak.py`,
`test_peak_metrics.py`, `test_peak_reporting.py` and the small saved-compatible
`tests/fixtures/post3_peak_metrics.json`. Existing amplitude/severity metadata
assertions now include the separate event threshold; pipeline/training tests
accept and verify the expanded metric dictionary and null populations.

The [machine-readable validation report](audit/evidence/final_peak_validation.json)
records **156 passed, zero failures/errors/skips**, including all 120 existing
tests and 36 new tests (22 loss, 12 metric, 2 reporting), and lists changed-file
hashes and all 64 unchanged P0/P1 config hashes. The
[full regression log](audit/evidence/final_peak_tests.txt) records the tests
and runtime (42.388 seconds). CPU BF16, CUDA FP16/BF16 and two-rank Gloo
passed. Pyflakes, `git diff --check` and `bash -n train.sh` passed. Pre-change references include ten model/optional-loss
combinations across zero/positive history, all ten prediction modes (100
loss/gradient references), and four legacy engine metrics at batch sizes
1, 7 and 41. Each matches exactly at zero peak weight. A frozen post-#3
NPZ-compatible fixture and independent NumPy calculations verify legacy and
new metrics, ties, strict thresholds and duplicate handling.

All 32 `P0_QuickRun` and 32 `P1_WeightProbe` configs were compared under
`DRY_RUN=1`. Old CLI arguments, parsed values, model formulation and run tags
match exactly after removing the three newly defaulted controls; all config
file hashes remain unchanged. Default peak weight is zero. Representative
profiles include P0 Battery dual MSE, P0 Lewes single MSE, P0 Boston dual
tail+slope, and P1 CBBT excess-10/no-tail and Battery excess-5/tail-0.10.

No scientific peak weight was chosen, no experiment config was generated,
and no real-data training, experiment or VAL/TEST tuning was started. Existing
regression tests use temporary synthetic fixtures; new checkpoint/reporting
tests mock training epochs and never run an optimizer step. Instructions
#7 and #8 remain unimplemented. No additional project checkout was created.
