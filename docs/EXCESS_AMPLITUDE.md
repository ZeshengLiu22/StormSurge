# True-peak-time excess amplitude supervision

This optional objective supervises the ungated physical excess branch at the observed target peak horizon. It adds no parameters and leaves `prediction = body + p * excess`, the soft body cap, gate, and trajectory supervision unchanged. The default direct-dual model is unchanged. The optional [severity-shape formulation](SEVERITY_SHAPE.md) uses the same objective through reconstructed `output.excess`.

## Definition

For normalized target `z = (y_phys - y_mean) / y_std`, the saved TRAIN scalar threshold `tau_phys` gives per-horizon `t_h = (tau_phys - y_mean_h) / y_std_h`:

```text
h*_i            = argmax_h(y_phys[i,h])          # first occurrence on ties
E_i             = 1[max_h(y_phys[i,h]) > tau_phys]
e_target_norm   = clamp_min(z - t, 0)
e_target_phys   = e_target_norm * y_std          # meters, horizon by horizon
e_pred_phys     = output.excess * y_std          # ungated excess, meters
a_target_i      = e_target_phys[i,h*_i]
a_pred_i        = e_pred_phys[i,h*_i]
q_E             = TRAIN_event_count / TRAIN_window_count
L_amp_true      = mean_i[E_i * (a_pred_i - a_target_i)**2] / q_E
```

The original physical target determines both the strict event mask and peak index. Horizon-specific normalization can change an argmax, so normalized target peaks must not choose the horizon. The physical excess target agrees up to floating-point roundoff with `(y_phys - tau_phys)_+`.

Because `tau_phys` is the same scalar at every horizon, strict event windows satisfy `argmax(e_target_phys) = argmax(y_phys)` with the same first-occurrence tie rule, and `max(e_target_phys) = e_target_phys[h*]`. The target amplitude therefore stays the same. The prediction changes from its own maximum excess to its excess at the true target peak time. Predicting the right amplitude at the wrong horizon can no longer satisfy this objective.

For example, target excess `[.02, .04, .15, .08]` and prediction `[.02, .15, .10, .08]` have equal maxima. Their aligned amplitude squared error is nevertheless `(.10 - .15)**2` at the third horizon, before the fixed event-prior reduction.

## Reduction and gradients

`q_E` is exactly `fit_loss_thresholds(...)["event_prior"]`, the empirical strict TRAIN event fraction. It is neither nominal `0.05`, batch prevalence, a clipped gate initialization prior, nor VAL/TEST prevalence. `train.py` passes the fitted prior and scalar physical event threshold to `ForecastLoss`. An active amplitude objective requires a finite positive prior and finite saved threshold.

Each event has denominator `batch_size * q_E`. Masking before squaring gives event-free batches differentiable exact zero and non-event windows zero gradients. Equal-sized partitions preserve sample additivity, existing gradient accumulation, and ordinary DDP gradient averaging, including an event-free rank.

The amplitude term directly differentiates only `output.excess[i,h*_i]`. The wrong predicted-max horizon, body, gate and final reconstructed output receive no direct gradient from this term. Shared context can receive gradients through the excess branch. In the optional severity-shape architecture, the gathered excess depends on both severity and shape through the existing reconstruction; that architecture is unchanged. Physical products and squares promote FP16/BF16 inputs to FP32 and preserve FP64 inputs.

The ordinary direct-dual trajectory loss remains responsible for the rest of the excess trajectory. Body, excess and gate losses retain their weights and reductions. Tail keeps its existing threshold, `>=` event rule, `tail_frac` denominator and slope combinations.

## Configuration and objective

The only amplitude control is `EXCESS_AMP_LOSS_WEIGHT`, corresponding to `--excess_amp_loss_weight` and `LossConfig.excess_amp_loss_weight`. It defaults to `0`, must be finite and nonnegative, and skips amplitude computation at zero. A positive weight requires a supervised PACT dual exceedance head. There are no amplitude pooling or temperature settings.

```bash
EXCESS_AMP_LOSS_WEIGHT="0"
```

For the production direct-dual model:

```text
L_total = L_base + tail_lambda * L_tail
          + body_loss_weight * L_body
          + excess_loss_weight * L_excess
          + gate_loss_weight * L_gate
          + excess_amp_loss_weight * L_amp_true
```

Optional terms apply only when enabled. Existing slope variants remain available. There is no separate final-output peak objective, timing objective, new threshold or new event weighting. No compatibility adapter reconstructs obsolete objective configurations.

`no_excess_loss` disables trajectory excess and amplitude supervision. `no_branch_supervision` also disables body/gate supervision. `none`, `no_gate_bce` and `fixed_gate` retain a requested positive amplitude weight; no ablation forces this optional weight to one.

## Diagnostics

`infer.py --dual_diagnostics` reports overall/per-year strict-event diagnostics using the original `y_true` peak index, ungated exported `excess_phys`, and saved TRAIN threshold:

- `true_peak_excess_rmse`, `true_peak_excess_mae`, `true_peak_excess_bias`;
- `pred_true_peak_excess_mean`, `target_true_peak_excess_mean`.

All compare `e_pred_phys[h*]` with `e_target_phys[h*]`; bias is prediction minus target. Undefined event diagnostics are JSON `null`. They reuse the existing inference pass and never select checkpoints or fit thresholds. Ordinary forecast peak metrics and checkpoint selection are documented in [the README](../README.md#metrics-and-artifacts) and [checkpoint selection](CHECKPOINT_SELECTION.md).

Older validation logs under `docs/audit/evidence/` record the objective used at their historical revisions. They do not define this current interface.
