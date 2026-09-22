# Training objectives

The objective is implemented by `ForecastLoss` in [losses.py](../emulator/training/losses.py), with [amplitude](../emulator/training/excess_amplitude.py) and [shape](../emulator/training/excess_shape.py) helpers. Every extreme target or mask uses the fixed TRAIN threshold defined in [FORMULATION.md](FORMULATION.md). Architecture constraints are specified in [DUAL_EXCEEDANCE.md](DUAL_EXCEEDANCE.md) and [SEVERITY_SHAPE.md](SEVERITY_SHAPE.md).

<!-- choices loss_mode: mse,wqe,wmse,mse_slope,wqe_slope,wmse_slope -->

## Notation and reductions

For a minibatch of B windows and K hourly horizons, let y and prediction ŷ have shape [B,K] and units meters. Define error dᵢₕ = ŷᵢₕ − yᵢₕ, strict hourly mask Mᵢₕ = 1[yᵢₕ > τ], and Event Window indicator Eᵢ = maxₕ Mᵢₕ. The normalized target is zᵢₕ = (yᵢₕ − μₕ)/σₕ, using TRAIN means μ and positive scales σ. A physical body prediction is b̂ᵢₕ = μₕ + σₕb̂ⁿᵢₕ; raw physical excess is êᵢₕ = σₕêⁿᵢₕ. Thus ŷᵢₕ = b̂ᵢₕ + gᵢêᵢₕ for Dual.

The fixed TRAIN prevalences are:

- q_H = number of strict TRAIN Extreme Hours / number of TRAIN target hours; stored as `train_extreme_hour_rate` and passed as `extreme_hour_prior`.
- q_E = number of TRAIN Event Windows / number of TRAIN windows; stored as `event_prior` and `train_event_window_rate`.

Neither prevalence is recomputed from a minibatch. A mean over B×K always includes masked zeros. Dividing by q_H or q_E gives a fixed weight per qualifying hour or window and preserves sample-weighted batch partitioning. Enabled hourly supervision requires q_H > 0; enabled amplitude or shape supervision requires q_E > 0. The constructor rejects invalid required priors. Batches containing no qualifying samples produce differentiable zero for these masked objectives.

Production loss calls construct strict events and physical excess from physical targets and scalar τ. Comparisons and subtraction use FP64 before excess is returned in the target's arithmetic precision; this preserves the strict boundary when normalized or FP32 threshold representations round. Physical amplitude products are computed in FP32 or FP64.

## Prediction loss

`LOSS_MODE_LIST` selects one or more modes in the launcher; each command passes `loss_mode` through `--loss_mode`. The default is `mse`.

The global prediction objective `L_pred ∈ {MSE, WQE}` and the Dual conditional raw-excess objective `L_excess ∈ {MSE, WQE}` are independently selectable. Legacy WMSE and slope modes remain supported. `EXCESS_LOSS_MODE` / `--excess_loss_mode` defaults to `mse` and affects only the Dual raw-excess trajectory term. Existing configs that omit it retain the original MSE/MSE objective. For Single, the excess choice is inactive, which is stated in the startup log.

**MSE**, used by `mse` and `mse_slope`:

    L_pred = (1 / BK) Σᵢₕ dᵢₕ².                                      (1)

**Smooth weighted MSE**, used by `wmse` and `wmse_slope`:

    wᵢₕ = 1 + α sigmoid((yᵢₕ − τ) / max(s, 10⁻⁶ m))
    L_pred = (1 / BK) Σᵢₕ wᵢₕ dᵢₕ².                                (2)

`WMSE_ALPHA` / `wmse_alpha` defaults to α = 4, and `WMSE_S` / `wmse_s` defaults to s = 0.1 m. The denominator is BK, without dividing by the sum of weights. Weighting uses signed physical target values and the same τ as every other objective.

**Weighted quantile–expectile (WQE)**, used by `wqe` and `wqe_slope`, follows [Longo et al., Earth's Future (2026), equations 5–8](https://doi.org/10.1029/2025EF007072). For residual ε = target − prediction, positive ε means underprediction. Using the fixed per-horizon TRAIN scale σ = `y_std`:

    u = ε / σ
    Q(u) = τ_q u                 if u ≥ 0; (τ_q − 1) u otherwise
    E(u) = τ_e u²                if u ≥ 0; (1 − τ_e) u² otherwise
    ρ_scaled(ε; σ) = σ² [w_q Q(u) + w_e E(u)]
    L_pred = mean(ρ_scaled(y − ŷ; σ)).

The reusable `wqe_penalty` returns the pointwise tensor. Global WQE averages **all target hours**, with no Q95 or Event Window mask. The shared parameters for global and excess WQE are:

| CLI field | Launcher variable | Published default |
| --- | --- | ---: |
| `--wqe_quantile_tau` | `WQE_QUANTILE_TAU` | 0.25 |
| `--wqe_expectile_tau` | `WQE_EXPECTILE_TAU` | 0.82 |
| `--wqe_quantile_weight` | `WQE_QUANTILE_WEIGHT` | 1/6 = 0.16666666666666667 |
| `--wqe_expectile_weight` | `WQE_EXPECTILE_WEIGHT` | 5/6 = 0.83333333333333333 |

Both taus must be finite and strictly between zero and one. Weights must be finite, nonnegative, and sum to one within floating-point tolerance; scales must be finite and positive. TRAIN statistics receive no gradients, while the prediction remains differentiable.

The [released implementation](https://github.com/Emi-Lo/storm-surge-surrogate-model/blob/v1.3/custom_for_model.py#L289-L305) standardizes the output target before applying [WQE](https://github.com/Emi-Lo/storm-surge-surrogate-model/blob/v1.3/custom_losses.py). Here, evaluating WQE on TRAIN-standardized residual magnitudes preserves that geometry across physical unit scales, including centimeter-valued data. Rescaling by `y_std²` keeps the result on the physical squared-loss scale used by the existing objective and branch coefficients. It does not make WQE numerically equal to MSE or equalize their gradients.

All prediction objectives have physical squared units (m² for meter-valued data), apply to every target hour, and support Single, Direct Dual, and Severity–Shape Dual. Their direct paths reach the Single regression head or the Dual body, excess, and learned gate. For Severity–Shape, the excess path reaches severity and shape.

## Dual branch supervision

Targets at every horizon are b*ᵢₕ = min(yᵢₕ,τ) and e*ᵢₕ = max(yᵢₕ−τ,0). The following losses are supported by both Dual excess parameterizations.

**Body loss:**

    L_body = (1 / BK) Σᵢₕ (b̂ᵢₕ − b*ᵢₕ)².                         (3)

All hours contribute. Units are m². `BODY_LOSS_WEIGHT` / `body_loss_weight` defaults to 1. This term directly supervises only the body branch.

**Conditional raw excess trajectory loss:**

`EXCESS_LOSS_MODE="mse"` retains the existing expression exactly:

    L_excess = (1 / BK) Σᵢₕ Eᵢ (êᵢₕ − e*ᵢₕ)².                    (4)

`EXCESS_LOSS_MODE="wqe"` replaces only its pointwise penalty, using the same helper and shared WQE parameters as the global prediction objective:

    L_excess = (1 / BK) Σᵢₕ Eᵢ ρ_scaled(e*ᵢₕ − êᵢₕ; σₕ).

Every horizon of a GT Event Window contributes, including horizons whose target excess is zero. The denominator includes all windows; this term has no q_E division. Units are m². `EXCESS_LOSS_WEIGHT` / `excess_loss_weight` defaults to 1. The supervised quantity is raw excess before multiplication by the gate. This term directly supervises the Direct excess branch or both reconstructed Severity–Shape factors.

**Window gate BCE:**

    v = (1 / K) Σₕ σₕ²
    L_gate = (v / B) Σᵢ BCEWithLogits(ℓᵢ, Eᵢ).                     (5)

The learned gate has one logit ℓᵢ per window, with gᵢ = sigmoid(ℓᵢ). BCE is averaged over all windows; multiplying by v gives units m². `GATE_LOSS_WEIGHT` / `gate_loss_weight` defaults to 1. Only the gate branch receives a direct gradient from this term. The fixed-gate ablation has no learned logit and returns zero for this term.

## Hourly exceedance loss

The independent extreme-aware coefficient is `EXCEEDANCE_LOSS_WEIGHT` / `exceedance_loss_weight`, default 0:

    L_exceedance = (1 / (BK q_H)) Σᵢₕ Mᵢₕ dᵢₕ².                  (6)

This is additional physical MSE only on strict GT Extreme Hours. Its units are m². A normal horizon within an Event Window has zero contribution. The fixed q_H is the observed TRAIN hourly prevalence, including the effect of ties at τ. The term remains ordinary squared error when the primary prediction objective uses WMSE or WQE, or the excess objective uses WQE. `EXCEEDANCE_LOSS_WEIGHT` is independent of `EXCESS_LOSS_MODE`; switching either WQE placement also leaves body, gate, amplitude, shape, and slope definitions unchanged.

All three head choices support this objective. Its gradient follows the final prediction through the same branches as L_pred. The threshold, target mask, and prevalence are constants.

## GT-aligned raw excess amplitude loss

For each window, select the first maximum of the original physical target:

    h*ᵢ = first argmaxₕ yᵢₕ
    âᵢ = êᵢ,h*ᵢ
    a*ᵢ = max(yᵢ,h*ᵢ − τ, 0)
    L_amp = (1 / (B q_E)) Σᵢ Eᵢ (âᵢ − a*ᵢ)².                    (7)

`EXCESS_AMP_LOSS_WEIGHT` / `excess_amp_loss_weight` defaults to 0. The loss is measured in m². Each GT Event Window contributes one squared error. The selected horizon is GT-aligned: a larger predicted excess at another horizon cannot substitute for excess at h*ᵢ. A strict Event Window has a*ᵢ > 0; the scalar threshold ensures h*ᵢ also selects a maximum of the target excess.

The predicted quantity is raw excess; the gate and body do not enter (7). For Direct Dual, only the selected output excess coordinate receives the direct output gradient. For Severity–Shape, âᵢ = severityᵢ × shapeᵢ,h*ᵢ; both factors receive gradients, and shape's maximum normalization can propagate a gradient to its normalizing maximum. Shared branch parameters and the backbone can affect multiple horizons. Single does not support amplitude supervision.

## Optional temporal shape loss

Severity–Shape supports `SHAPE_LOSS_WEIGHT` / `shape_loss_weight`, default 0. Define the target amplitude and dimensionless shape:

    a*ᵢ = maxₕ e*ᵢₕ
    s*ᵢₕ = e*ᵢₕ / a*ᵢ   if Eᵢ = 1; otherwise 0
    L_shape = (1 / (BK q_E)) Σᵢₕ Eᵢ (ŝᵢₕ − s*ᵢₕ)².             (8)

Every horizon of a GT Event Window contributes. The loss is dimensionless. Any positive target amplitude is used exactly, even when it is smaller than `severity_shape_eps`. Non-event windows use denominator 1 internally and retain zero targets. This term directly supervises the shape branch; it has no direct path to severity, body, or gate. The shared backbone receives its gradient.

`EXCESS_FORMULATION` / `excess_formulation` defaults to `direct`; a positive shape weight requires `severity_shape`. `SEVERITY_SHAPE_EPS` / `severity_shape_eps` defaults to 10⁻⁶ and must be finite and positive. Its architectural role is specified in [SEVERITY_SHAPE.md](SEVERITY_SHAPE.md). There is no separate scalar severity objective: severity is trained through reconstructed excess and final prediction, with optional GT-aligned amplitude supervision.

## Slope loss

The suffix `_slope` enables slope matching when K > 1 and `slope_lambda` is nonzero:

    rᵢₕ = (ŷᵢ,ₕ₊₁ − ŷᵢₕ) − (yᵢ,ₕ₊₁ − yᵢₕ)
    uᵢ = sigmoid((τ − maxₕ yᵢₕ) / max(s_mask, 10⁻⁶ m))
    L_slope = (1 / (B(K−1))) Σᵢ Σₕ₌₀ᴷ⁻² uᵢ ρ(rᵢₕ).            (9)

This is a soft weight over all windows and adjacent target pairs. The GT window maximum enters a smooth weight around the same τ; there is no separately fitted cutoff. With one-hour targets, differences are hourly increments in meters; no explicit time division is performed.

`SLOPE_ROBUST` / `slope_robust` selects:

    charb: ρ(r) = sqrt(r² + ε²)
    huber: ρ(r) = r² / 2                       if |r| ≤ δ
                   δ(|r| − δ/2)               otherwise.

Charbonnier has units m and includes its positive ε baseline at zero error. Huber has units m². The numeric settings are:

| CLI field | Launcher variable | Default | Meaning |
| --- | --- | ---: | --- |
| `slope_lambda` | `SLOPE_LAMBDA_LIST` | 0.01 | Coefficient, active only in slope modes |
| `slope_mask_s` | `SLOPE_MASK_S_LIST` | 0.1 | Smooth window weight scale, m |
| `slope_robust` | `SLOPE_ROBUST` | `charb` | Penalty choice |
| `slope_charb_eps` | `SLOPE_CHARB_EPS` | 0.001 | Charbonnier ε, m |
| `slope_huber_delta` | `SLOPE_HUBER_DELTA` | 0.05 | Huber δ, m |

The code floors ε and δ at 10⁻¹² m, and the sigmoid scale at 10⁻⁶ m. All head choices support slope matching; its prediction path reaches their active regression branches and learned gate.

## Complete objective and effective settings

With absent terms assigned coefficient zero, the training objective is:

    L = L_pred + λ_H L_exceedance + λ_slope L_slope
        + λ_body L_body + λ_excess L_excess + λ_gate L_gate
        + λ_amp L_amp + λ_shape L_shape.                        (10)

The code adds numerical terms with their configured scales. A dimensionally uniform m² interpretation assigns λ_shape units m² and the Charbonnier λ_slope units m; the other coefficients, including Huber λ_slope, are dimensionless. No additional rescaling is performed beyond the equations above.

`DUAL_LOSS` / `dual_loss` defaults to 1 for Dual and is resolved to 0 for Single. `DUAL_ABLATION` / `dual_ablation` defaults to `none`. In ordinary Dual runs, requesting `dual_loss=0` or a zero body/excess/gate coefficient restores required supervision and emits a warning. Positive coefficients are preserved. Named ablations explicitly control removals:

| `dual_ablation` | Effective branch change | Prediction/exceedance/slope objectives |
| --- | --- | --- |
| `none` | Body, excess, gate supervision required | Retained |
| `no_gate_bce` | Gate BCE weight = 0 | Retained; learned gate still receives prediction gradients |
| `no_excess_loss` | Excess trajectory, amplitude, shape weights = 0 | Retained; excess still receives prediction gradients |
| `no_branch_supervision` | All five branch auxiliary weights = 0; `dual_loss=0` | Retained |
| `fixed_gate` | Gate fixed at TRAIN q_E; gate BCE weight = 0 | Retained; no learned gate parameters |

All enabled terms can update the shared backbone. Targets, normalization statistics, masks, selected GT peak indices, and fitted priors have no learned gradient. The following table records direct head paths and compatibility; shared backbone gradients are additional.

| Term | Directly supervised head paths | Single | Direct Dual | Severity–Shape |
| --- | --- | --- | --- | --- |
| Prediction, exceedance, slope | Single regression; Dual body, gate, excess; reconstructed severity and shape | Yes | Yes | Yes |
| Body | Body | No | Yes | Yes |
| Excess trajectory | Raw excess; severity and shape through their product | No | Yes | Yes |
| Gate BCE | Learned gate | No | Yes | Yes |
| Amplitude | GT-aligned raw excess; severity and shape through their product | No | Yes | Yes |
| Shape | Shape | No | No | Yes |

## Canonical generated experiments

[tools/generate_configs.py](../tools/generate_configs.py) generates the matched S0/D0/D1/D2/D3 variants. Their primary objective is MSE and slope weight is inactive. Generator branch weights override parser defaults: λ_body = 1, λ_excess = 2, λ_gate = 0.5 in every Dual variant. The parser defaults for these three fields are all 1.

    L_S0 = L_pred
    L_D0 = L_pred + L_body + 2 L_excess + 0.5 L_gate
    L_D1 = L_D0 + 0.025 L_exceedance
    L_D2 = L_D0 + 0.003 L_amp
    L_D3 = L_D0 + 0.025 L_exceedance + 0.003 L_amp.                (11)

| Variant | Pred | Body | Excess | Gate | Exceedance | Amp | Shape | Slope |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| S0 Single | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| D0 DualBase | 1 | 1 | 2 | 0.5 | 0 | 0 | 0 | 0 |
| D1 Exceedance | 1 | 1 | 2 | 0.5 | 0.025 | 0 | 0 | 0 |
| D2 Amp | 1 | 1 | 2 | 0.5 | 0 | 0.003 | 0 | 0 |
| D3 ExceedanceAmp | 1 | 1 | 2 | 0.5 | 0.025 | 0.003 | 0 | 0 |

The optional Severity–Shape D0–D3 configs use the same matrix with the alternative excess parameterization. Their generated shape weight is 0; explicitly enabling shape supervision adds λ_shape L_shape to the corresponding objective. S0 has no Severity–Shape counterpart.

## WQE placement experiments

Generate the separate [WQE family](../configs/wqe) with:

```bash
python tools/generate_configs.py --family wqe
DRY_RUN=1 USE_TMUX=0 bash train.sh configs/wqe/train_config_NCEP_CBBT_W1_GlobalWQE.sh
```

The existing S0/D0/D1/D2/D3 files are preserved. D0 supplies the MSE/MSE control (W0); W1/W2/W3 add 12 configs across CBBT, Lewes, Battery, and Boston:

WQE configs save results under `/home/exouser/media/share/PACT/WQE_Results`, with a separate run-name/timestamp directory for each experiment. Override this during generation with `--results-root` when needed; the current S0/D0/D1/D2/D3 generator default remains `./All_Results`.

| Variant | `LOSS_MODE_LIST` | `EXCESS_LOSS_MODE` |
| --- | --- | --- |
| Existing D0 / W0 control | `("mse")` | `"mse"` (default) |
| W1_GlobalWQE | `("wqe")` | `"mse"` |
| W2_ExcessWQE | `("mse")` | `"wqe"` |
| W3_BothWQE | `("wqe")` | `"wqe"` |

Every new run uses the published WQE parameters and Direct D0 settings: body weight 1, excess weight 2, gate weight 0.5, with exceedance, amplitude, and shape weights zero and no active slope term. Startup logs report the effective global loss, Dual excess loss (or Single inactivity), and all four shared WQE parameters. Checkpoint selection, metrics, thresholds, splits, and architecture retain their existing definitions.

## Optimization and validation boundaries

[train.py](../train.py) uses Adam with fixed `weight_decay=1e-5`, which contributes its optimizer L2 regularization to trainable parameters. It is separate from the scalar returned by `ForecastLoss`. Dropout belongs to the architecture described in [BACKBONE.md](BACKBONE.md). There is no additional independent peak objective or explicit scalar severity loss.

[engine.py](../emulator/training/engine.py) supplies physical predictions and physical targets to the objective. Gradient accumulation divides each microbatch mean loss by the number of microbatches in its accumulation group, including the final partial group. Thus equally sized microbatches reproduce the corresponding combined-batch loss; unequal microbatches retain equal microbatch weights. All evaluation and checkpoint selection use the independent canonical physical metrics described in [METRICS.md](METRICS.md) and [CHECKPOINT_SELECTION.md](CHECKPOINT_SELECTION.md).
