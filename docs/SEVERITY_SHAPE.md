# Optional severity-aware excess decomposition

This upgrade starts from the completed Instruction #2 revision
`e2846dd9e6a83397640cb6f39401e03d2bc9d84a`. The tracked production working tree
was clean, and its untracked `experiment_config/P1_WeightProbe/` directory was
preserved. All 95 post-#2 tests passed before any source changes. The existing
physical excess-amplitude objective remains the canonical amplitude control.

## Defaults and compatibility

```bash
EXCESS_FORMULATION="direct"
SHAPE_LOSS_WEIGHT="0"
SEVERITY_SHAPE_EPS="1e-6"
EXCESS_AMP_LOSS_WEIGHT="0"
EXCESS_AMP_POOL="max"
EXCESS_AMP_BETA="20.0"
```

The three new CLI options are `--excess_formulation`, `--shape_loss_weight` and
`--severity_shape_eps`. All #2 CLI options remain available. Shape weight must
be finite and nonnegative; epsilon must be finite and positive. A positive
shape weight requires `severity_shape`, which requires `--model perceiver3
--head_type dual`. Shape supervision is skipped completely at weight zero;
neither optional amplitude nor shape supervision is ever forced on.

`direct` still builds the original `ExceedanceHead`, with the same `body`,
`excess`, `gate`, `threshold` and optional fixed-gate buffer, construction order,
parameter sizes and initialization. It creates no severity/shape modules or
target-scale buffer. All existing `ForecastOutput` fields retain their meaning;
two trailing optional diagnostic fields, `severity_phys` and `excess_shape`,
default to `None`. Existing positional construction remains supported.

Legacy `model_config` dictionaries need no migration: the new dataclass fields
default to `direct`, `target_y_std=None` and `severity_shape_eps=1e-6`. Production
reconstruction remains `ModelConfig(**saved_config)`, `build_model(config)`, then
`load_state_dict(saved_state, strict=True)`. The direct prediction/loss paths
remain unchanged, including direct models with #2 amplitude supervision.

## Physical factorization and units

The optional `SeverityShapeHead` consumes the unchanged PACT forecast context
`C` of shape `[B,K,hidden]`, after the existing horizon readout and LayerNorm.
Let `C_bar = mean_h(C)` and `eps = SEVERITY_SHAPE_EPS`:

```text
body_norm       = threshold - softplus(threshold - body(C))
p_event         = sigmoid(gate(C_bar))           # same event branch as direct
severity_phys   = softplus(severity(C_bar))       # [B], meters
shape_raw       = softplus(shape(C)) + eps        # [B,K], unitless
shape           = shape_raw / max_h(shape_raw)
excess_phys     = severity_phys[:,None] * shape   # [B,K], meters
output.excess   = excess_phys / target_y_std      # [B,K], normalized excess
prediction_norm = body_norm + p_event * output.excess
prediction_phys = body_phys + p_event * severity_phys[:,None] * shape
```

The gate remains window-level and unitless; `fixed_gate` still uses the exact
empirical TRAIN prevalence. Epsilon is added to the FP32 softplus output
before dividing by the per-window maximum. Shape has maximum exactly one,
including when every softplus value is smaller than the default epsilon or
underflows to zero. In the all-underflow case shape is uniformly one, so
severity still equals the physical maximum excess. Adding epsilon retains
below-epsilon gradients until softplus itself underflows; there is no clamp
on the predicted shape. No normalization crosses the batch axis, and no label
enters any prediction branch.

This corrects the normalization edge case reviewed at `f673555a`. Previously
only the denominator was floored, allowing maxima below one (or zero) and
breaking severity's amplitude interpretation. Severity-shape outputs and
gradients intentionally change with this correction, including small rounding
changes away from the edge. Existing weight shapes and checkpoint loading
remain valid. Single/direct arithmetic is unchanged. The old frozen severity
fixture is preserved as historical evidence; corrected severity-shape training
is checked for exact equivalence across all five checkpoint-selection modes.

`train.py` supplies `stats_cpu["y_std"].tolist()` only in `severity_shape` mode.
`ModelConfig.target_y_std` saves this TRAIN vector, and the new head registers
it as an FP32 `[1,K]` `target_y_std` buffer. Its length must equal `out_channels`,
and every entry must be finite and strictly positive. Each entry is a physical
target standard deviation in meters. The head divides physical excess by
each horizon's own scale; no scalar or averaged scale is substituted. Both
the configuration vector and model-state buffer survive checkpoint loading.

Branch logits are converted to FP32 before softplus, normalization and physical
reconstruction, retaining the direct head's AMP policy. The body and gate use
the existing small MLP architecture; severity and shape use the same MLP
width/dropout convention (`head_hidden` or twice the context width).

## Initialization

Severity and shape final-layer weights are zero. Their biases are respectively
`log(expm1(0.1))` and `log(expm1(1.0))`. Thus initial severity is approximately
**0.1 meters** and softplus shape is one, so raw shape is `1 + eps`.
Normalized shape is exactly one and initial physical event contribution is `p_event * 0.1 m` at
each horizon. Earlier MLP layers retain the existing initialization. The gate
keeps zero final weights and the existing clipped-prior logit bias; fixed gate
keeps the unclipped empirical prior. No new severity statistic is fitted.

The severity-shape head has one more regression MLP than direct, since one
trajectory branch is replaced by two branches. This is an explicit architecture
comparison, with unchanged backbone and existing trajectory supervision.
[Parameter reporting](MODEL_PARAMETERS.md) now records total/trainable/head
counts for every model and individual head branches in JSON. At context width
128 and head width 256 the extra MLP contains exactly 33,281 parameters; the
report counts actual modules without constructing an extra comparison model.
Zero final weights initially block that branch's gradient to shared context;
this is expected initialization behavior. Gradient-routing tests use nonzero
synthetic final weights to verify every intended connection.

## Shared targets and independent losses

Instruction #2's strict event/physical target operations are extracted without
arithmetic changes into `physical_excess_target` in
`emulator/training/excess_amplitude.py`. Both amplitude and shape helpers reuse
it, and both reuse the same `validate_event_prior`:

```text
z            = (y_phys - y_mean) / y_std
E_i          = any_h(z_i,h > threshold_h)         # strict TRAIN event
r_target     = clamp_min(z - threshold, 0) * y_std
a_target     = max_h(r_target)                   # meters; always hard max
shape_target = r_target / clamp_min(a_target[:,None], eps)
L_shape      = mean_i[E_i * mean_h((shape_i,h - shape_target_i,h)^2)] / q_E
```

The physical target is the same as `clamp_min(y_phys - tau_train, 0)` within
floating-point tolerance. It never uses the predicted body. A valid event has
positive amplitude; amplitudes below epsilon remain safe and their target
shape maximum may be below one. Target construction retains its existing
amplitude denominator floor in meters. The predicted dimensionless shape uses
additive epsilon before max normalization; this correction does not change
the target, strict event definition, or any loss weight/reduction.

`q_E` remains exactly `fit_loss_thresholds(...)["event_prior"]`, the strict
TRAIN `event_count / train_windows`, passed by `train.py` to `ForecastLoss`.
Neither `tail_frac`, the nominal percentile fraction, current batch prevalence,
nor gate-initialization clipping changes it. A missing/nonfinite/nonpositive
prior fails clearly if amplitude or shape supervision is active. `L_shape` is
dimensionless and is not amplitude-weighted. It uses the whole batch denominator
and has exactly zero loss/gradients on event-free batches, including DDP ranks.

The total objective adds independently controlled terms:

```text
L_existing_prediction_and_tail_slope
  + body_loss_weight       * L_body
  + excess_loss_weight     * L_excess
  + gate_loss_weight       * L_gate
  + excess_amp_loss_weight * L_excess_amp
  + shape_loss_weight      * L_shape
```

`L_excess` is the unchanged physical trajectory loss on `output.excess`, with
its existing whole-batch/horizon reduction. `L_excess_amp` is #2's unchanged
physical amplitude objective in square meters. It receives reconstructed
normalized excess in either formulation; there is no second severity loss or
`SEVERITY_LOSS_WEIGHT` control. Under hard-max pooling,
it supervises `severity_phys` directly. Canonical severity-shape comparisons
should use `EXCESS_AMP_POOL="max"` for this interpretation. `smoothmax` still
pools the reconstructed physical trajectory on both sides and remains an
optional sensitivity mode; it does not redefine the scalar severity target.

Prediction loss connects body, gate, severity, shape and shared context.
Trajectory excess loss connects severity and shape. Hard-max amplitude loss
connects severity, with the unit-peak shape factor canceling. Shape loss directly connects the shape branch. Amplitude/shape objectives
do not directly supervise body or gate; shared-context coupling is expected.

## Ablations and future configuration capability

`no_excess_loss` now disables trajectory excess, amplitude and shape weights.
`no_branch_supervision` disables those three plus body and gate supervision.
`no_gate_bce` and `fixed_gate` may retain both optional terms. Historical
body/excess/gate enforcement is unchanged; optional weights can always stay zero.

The interface supports all six requested comparisons without source edits:

| Formulation | Amplitude weight | Shape weight | Interpretation |
|---|---|---|---|
| direct | 0 | 0 | Existing post-#2 model |
| direct | positive | 0 | Existing model with #2 amplitude supervision |
| severity_shape | 0 | 0 | Factorization with original trajectory supervision |
| severity_shape | positive | 0 | Factorization plus amplitude supervision |
| severity_shape | 0 | positive | Factorization plus shape supervision |
| severity_shape | positive | positive | Both optional objectives |

Here `positive` is a placeholder for a later scientific choice. This upgrade
chooses neither loss weight, creates no experiment configs and starts no
experiments. No VAL/TEST tuning or future peak-output, checkpoint-selection or
asymmetric-underprediction objectives are implemented.

## Checkpoints, run identity and diagnostics

Training snapshots preserve `excess_formulation`, `shape_loss_weight`, all #2
amplitude options and `severity_shape_eps`. Severity-shape model config also
stores `target_y_std`. Dual metadata identifies the formulation and, for the new
head, epsilon. These fields survive the existing checkpoint path; no external
architectural guessing is needed. Automatic shell run tags append
`_efseverity_shape` only for the new formulation, preserving old direct names.

Normal inference prediction exports still contain only `y_true`, `y_pred` and
`tags`. With `--dual_diagnostics`, the existing pass exports `gate_probability`,
`body_phys`, `excess_phys`, final prediction and existing target/event arrays;
severity-shape additionally exports `severity_phys` (`[N]`, meters) and
`excess_shape` (`[N,K]`, unitless). Direct checkpoints need neither array.
Diagnostic JSON/NPZ metadata records `excess_formulation` and includes
`severity_shape_eps` when relevant. Existing #2 amplitude diagnostics continue
to use the reconstructed physical excess and the saved strict TRAIN threshold.
No additional forward pass or inference-time fit is introduced. Checkpoint
selection continues to minimize validation `rmse_all`.

## Validation results

The [implementation/validation report](audit/evidence/severity_shape_validation.json)
lists all modified files and the complete checks. The [full test log](audit/evidence/severity_shape_tests.txt)
records **120 passed, zero failures/errors/skips**, including the six requested
regression modules, all 23 Instruction #2 amplitude tests and 25 new
severity-shape tests. CPU BF16, H100 CUDA FP16/BF16 and two-rank Gloo DDP passed.
Pyflakes on changed Python files, `git diff --check` and `bash -n train.sh` passed.

Before editing, six PACT references were saved from the post-#2 implementation:
single, learned dual and fixed-gate dual, each with zero and positive history.
The updated default matches their initialization, state keys/values, strict
reconstruction, original outputs and constructor/forward RNG bit for bit.
An independent frozen direct-head reference also verifies exact gradients.

All **32 P0_QuickRun + 32 P1_WeightProbe** configs were compared under
`DRY_RUN=1` with the pre-change commands. Old parsed values and run tags match
exactly, the three new defaults are inactive/direct, #2 amplitude weights stay
zero, and all source hashes are unchanged. No new experiment config was created.
Representative checks include P0 Battery dual MSE, P0 Boston dual tail+slope,
P0 Lewes single MSE, P1 Battery excess-5/tail-0.10 and P1 CBBT excess-10/no-tail.
Training exercised by the pre-existing regression suite uses temporary synthetic
fixtures; new checkpoint tests mock epochs and do not run an optimizer. No
real-data training, tuning or experiment was launched.
