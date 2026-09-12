# Optional physical excess peak-amplitude supervision

The later [severity-shape upgrade](SEVERITY_SHAPE.md) reuses this exact amplitude
objective through reconstructed `output.excess`. Its shape targets share the
physical target helper; it adds no separate severity-loss weight. Direct remains
the default. Both excess-removing ablations now also disable optional shape loss.

This training-only objective directly supervises the maximum physical excess
inside a true event window. It adds no parameters and leaves `ForecastOutput`,
checkpoint model reconstruction, and `prediction = body + p * excess` unchanged.
It is independent of the existing trajectory-wise `EXCESS_LOSS_WEIGHT` and all
prediction/tail/slope loss modes. No new loss-mode strings are required.

## Definition and TRAIN metadata

For normalized targets `z = (y - y_mean) / y_std`, use the existing head's
TRAIN-derived horizon thresholds `t_h = (tau_phys - y_mean_h) / y_std_h`:

```text
E_i             = any_h(z_i,h > t_h)          # strict dual event; ties excluded
r_target_norm   = max(z - t, 0)
r_target_phys   = r_target_norm * y_std       # meters, horizon by horizon
r_pred_phys     = output.excess * y_std       # meters, horizon by horizon
a_target_i      = pool_h(r_target_phys_i,h)
a_pred_i        = pool_h(r_pred_phys_i,h)
L_excess_amp    = mean_i[E_i * (a_pred_i - a_target_i)^2] / q_E
L_total_new     = L_total_existing + excess_amp_loss_weight * L_excess_amp
```

`L_excess_amp` has units of square meters. Convert to physical units **before**
pooling: differing horizon scales can change which horizon has the maximum.
The physical target agrees, up to floating-point roundoff, with
`max(y_phys - tau_phys, 0)` using the same TRAIN threshold.

`q_E` comes only from `fit_loss_thresholds(...)["event_prior"]`, saved as the
exact `event_count / train_windows` for strict TRAIN events. `train.py` passes it
to `ForecastLoss(..., event_prior=fitted["event_prior"])`. It is neither
`tail_frac`, the nominal percentile fraction, the gate's clipped initialization
prior, nor the event fraction in a minibatch. Missing, nonfinite, or nonpositive
priors fail clearly when the objective is active.

Every event contributes with denominator `batch_size * q_E`; no division by
event count or current batch prevalence occurs. Masking before squaring is
algebraically equivalent to the formula above and gives an event-free batch
an exactly zero, differentiable loss. The fixed prior also preserves normal
DDP averaging across equal-sized local batches, including ranks with no events.
The existing trajectory-wise excess loss retains its original whole-batch,
whole-horizon denominator and does not acquire prior normalization.

## Configuration

The defaults, accepted by self-contained shell configs and saved in the
existing resolved configuration/checkpoint snapshots, are:

```bash
EXCESS_AMP_LOSS_WEIGHT="0"
EXCESS_AMP_POOL="max"
EXCESS_AMP_BETA="20.0"
```

CLI equivalents are `--excess_amp_loss_weight`, `--excess_amp_pool`, and
`--excess_amp_beta`. Weight must be finite and nonnegative. At zero weight the
new tensor computation is skipped entirely; old four-argument `ForecastLoss`
callers and old checkpoints remain supported. Single/baseline training remains
unchanged at zero weight, and a positive weight requires the supervised dual
exceedance head (`--model perceiver3 --head_type dual`).

`max` uses the exact physical horizon maximum and does not read beta.
`smoothmax` is an optional sensitivity mode: `sum(softmax(beta * r) * r)`.
The implementation centers values at their maximum before forming the softmax
logits for numerical stability. The result is bounded by the input range, with
no additive log-sum-exp horizon bias. Both prediction and target use the same
operator, so identical trajectories have exactly zero amplitude error.
Beta has **inverse-meter** units and must be finite and positive only when
smoothmax supervision is active. Physical products, pooling and squared errors
promote FP16/BF16 inputs to FP32; FP64 callers retain their precision.

To enable the objective in a future config, supply a scientifically chosen
positive weight via `AMPLITUDE_WEIGHT`; this change selects no such weight:

```bash
EXCESS_AMP_LOSS_WEIGHT="${AMPLITUDE_WEIGHT:?Set the chosen positive amplitude weight}"
EXCESS_AMP_POOL="max"
EXCESS_AMP_BETA="20.0"
```

## Ablations and diagnostics

`no_excess_loss` sets both `excess_loss_weight` and `excess_amp_loss_weight` to
zero, removing all explicit excess-branch supervision. `no_branch_supervision`
also zeros the body and gate weights and disables `dual_loss`. `none`,
`no_gate_bce` and `fixed_gate` retain a requested positive amplitude weight.
The new optional weight is never forced from zero to one by branch-supervision
enforcement.

`emulator.training.excess_amplitude.excess_amplitude_terms` is a pure reusable
helper exposing the strict event mask, physical predicted/target trajectories,
pooled amplitudes, and the scalar auxiliary loss. These values can support
future severity targets or component diagnostics without criterion state or
changing `ForecastLoss.forward`'s scalar return. Normal epoch logging is
unchanged; the three exact config values are always recorded.

The existing `infer.py --dual_diagnostics` export now adds overall/per-year:

- `excess_amp_rmse`, `excess_amp_mae`, `excess_amp_bias`;
- `pred_excess_amp_mean`, `target_excess_amp_mean`.

These metrics reuse exported `excess_phys`, `y_true` and the saved TRAIN
`tau_phys`, select only true strict-threshold events, and measure **hard maximum**
amplitudes in meters even when the training pool was smoothmax. They use the
ungated excess branch. Bias is prediction minus target. Undefined event metrics
are JSON `null`. The pure `summarize_excess_amplitude` helper is also available
for aligned physical excess arrays and event masks.

No inference pass or prediction field is added. Production checkpoint selection
continues to minimize validation `rmse_all`; these diagnostic metrics never
select checkpoints or tune any settings. This upgrade includes no experiments,
P2 configs, severity model, or real-data training.

## Upgrade validation (2026-09-12)

The [validation report](audit/evidence/excess_amplitude_validation.json) lists
every modified file, exact defaults/prior/formula, per-module test counts and
all 64 historical config comparisons. The [full test log](audit/evidence/excess_amplitude_tests.txt)
records **95 passed, zero failures/errors/skips**, including all five requested
regression modules and 23 new amplitude tests. CPU BF16, H100 CUDA FP16/BF16 and
two-rank Gloo DDP passed; DDP gradients match the global batch even with an
event-free rank. Pyflakes, `git diff --check` and `bash -n train.sh` also passed.

All 32 existing P0 and 32 existing P1 configs passed `DRY_RUN=1` comparisons
against commit `9e615c4`: all previous resolved values are identical, with only
the three new inactive defaults added. Source config hashes are unchanged.
Representative cases include P0 Battery dual MSE, P0 Boston dual tail+slope,
P0 Lewes single MSE, P1 Battery excess-5/tail-0.10, and P1 CBBT
excess-10/no-tail. The untracked P1 directory is preserved and excluded from
this code commit.

The broader regression run also identified an existing stale pipeline assertion
that omitted the baseline's `[Best]` log messages. Its failure was reproduced
on the unchanged baseline; the test now checks each message against the actual
validation improvement. Production logging was not changed. Existing local
preprocessing test dependencies were reused without modifying the training
environment. All training exercised by the tests uses temporary synthetic
fixtures; no experimental or real-data training was launched.
