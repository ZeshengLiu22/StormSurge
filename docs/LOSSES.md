# Training objectives

The objective is implemented by `ForecastLoss` in [losses.py](../emulator/training/losses.py), with [amplitude](../emulator/training/excess_amplitude.py) and [shape](../emulator/training/excess_shape.py) helpers. Every extreme target or mask uses the fixed TRAIN threshold defined in [FORMULATION.md](FORMULATION.md). Architecture constraints are specified in [DUAL_EXCEEDANCE.md](DUAL_EXCEEDANCE.md) and [SEVERITY_SHAPE.md](SEVERITY_SHAPE.md).

<!-- choices loss_mode: mse,wqe,wmse,mse_slope,wqe_slope,wmse_slope -->
<!-- choices excess_loss_mode: mse,wqe -->
<!-- choices exceedance_loss_mode: mse,wqe -->

## Notation and reductions

For a minibatch of $B$ windows and $K$ hourly horizons, let $y$ and prediction $\hat y$ have shape `[B,K]` and units meters. Define error $d_{ih}=\hat y_{ih}-y_{ih}$, strict hourly mask $M_{ih}=\mathbf1[y_{ih}>\tau]$, and Event Window indicator $E_i=\max_h M_{ih}$. The normalized target is $z_{ih}=(y_{ih}-\mu_h)/\sigma_h$, using TRAIN means $\mu$ and positive scales $\sigma$. A physical body prediction is $\hat b_{ih}=\mu_h+\sigma_h\hat b^{\mathrm{norm}}_{ih}$; raw physical excess is $\hat e_{ih}=\sigma_h\hat e^{\mathrm{norm}}_{ih}$. Thus $\hat y_{ih}=\hat b_{ih}+g_i\hat e_{ih}$ for Dual.

The fixed TRAIN prevalences are:

- $q_H$ = number of strict TRAIN Extreme Hours / number of TRAIN target hours; stored as `train_extreme_hour_rate` and passed as `extreme_hour_prior`.
- $q_E$ = number of TRAIN Event Windows / number of TRAIN windows; stored as `event_prior` and `train_event_window_rate`.

Neither prevalence is recomputed from a minibatch. A mean over $B\times K$ always includes masked zeros. Dividing by $q_H$ or $q_E$ gives a fixed weight per qualifying hour or window and preserves sample-weighted batch partitioning. Enabled hourly supervision requires $q_H>0$; enabled amplitude or shape supervision requires $q_E>0$. The constructor rejects invalid required priors. Batches containing no qualifying samples produce differentiable zero for these masked objectives.

Production loss calls construct strict events and physical excess from physical targets and scalar $\tau$. Comparisons and subtraction use FP64 before excess is returned in the target's arithmetic precision; this preserves the strict boundary when normalized or FP32 threshold representations round. Physical amplitude products are computed in FP32 or FP64.

## Prediction loss

`LOSS_MODE_LIST` selects one or more modes in the launcher; each command passes `loss_mode` through `--loss_mode`. The default is `mse`.

The global prediction objective $L_{\mathrm{pred}}\in\{\mathrm{MSE},\mathrm{WQE}\}$ and the Dual conditional raw-excess objective $L_{\mathrm{excess}}\in\{\mathrm{MSE},\mathrm{WQE}\}$ are independently selectable. Legacy WMSE and slope modes remain supported. `EXCESS_LOSS_MODE` / `--excess_loss_mode` defaults to `mse` and affects only the Dual raw-excess trajectory term. Existing configs that omit it retain the original MSE/MSE objective. The additional hourly Tail penalty is independently selected by `EXCEEDANCE_LOSS_MODE` / `--exceedance_loss_mode`, also defaulting to `mse`. For Single, the excess choice is inactive, which is stated in the startup log.

**MSE**, used by `mse` and `mse_slope`:

$$
L_{\mathrm{pred}}=\frac{1}{BK}\sum_{i,h}d_{ih}^2. \tag{1}
$$

**Smooth weighted MSE**, used by `wmse` and `wmse_slope`:

$$
\begin{aligned}
w_{ih}&=1+\alpha\,\operatorname{sigmoid}\!\left(\frac{y_{ih}-\tau}{\max(s,10^{-6}\,\mathrm{m})}\right),\\
L_{\mathrm{pred}}&=\frac{1}{BK}\sum_{i,h}w_{ih}d_{ih}^2.
\end{aligned} \tag{2}
$$

`WMSE_ALPHA` / `wmse_alpha` defaults to $\alpha=4$, and `WMSE_S` / `wmse_s` defaults to $s=0.1\,\mathrm{m}$. The denominator is $BK$, without dividing by the sum of weights. Weighting uses signed physical target values and the same $\tau$ as every other objective.

**Weighted quantile–expectile (WQE)**, used by `wqe` and `wqe_slope`, follows [Longo et al., Earth's Future (2026), equations 5–8](https://doi.org/10.1029/2025EF007072). For residual $\varepsilon=\mathrm{target}-\mathrm{prediction}$, positive $\varepsilon$ means underprediction. Using the fixed per-horizon TRAIN scale $\sigma=$ `y_std`:

$$
\begin{aligned}
u&=\varepsilon/\sigma,\\
Q(u)&=\begin{cases}
\tau_q u,&u\ge0,\\
(\tau_q-1)u,&u<0,
\end{cases}\\
E(u)&=\begin{cases}
\tau_e u^2,&u\ge0,\\
(1-\tau_e)u^2,&u<0,
\end{cases}\\
\rho_{\mathrm{scaled}}(\varepsilon;\sigma)&=\sigma^2\left[w_q Q(u)+w_e E(u)\right],\\
L_{\mathrm{pred}}&=\operatorname{mean}\!\left(\rho_{\mathrm{scaled}}(y-\hat y;\sigma)\right).
\end{aligned}
$$

The reusable `wqe_penalty` returns the pointwise tensor. Global WQE averages **all target hours**, with no Q95 or Event Window mask. The shared parameters for global, raw-excess, and Tail WQE are:

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

Targets at every horizon are $b_{ih}^*=\min(y_{ih},\tau)$ and $e_{ih}^*=\max(y_{ih}-\tau,0)$. The following losses are supported by both Dual excess parameterizations.

**Body loss:**

$$
L_{\mathrm{body}}=\frac{1}{BK}\sum_{i,h}(\hat b_{ih}-b_{ih}^*)^2. \tag{3}
$$

All hours contribute. Units are m². `BODY_LOSS_WEIGHT` / `body_loss_weight` defaults to 1. This term directly supervises only the body branch.

**Conditional raw excess trajectory loss:**

`EXCESS_LOSS_MODE="mse"` retains the existing expression exactly:

$$
L_{\mathrm{excess}}=\frac{1}{BK}\sum_{i,h}E_i(\hat e_{ih}-e_{ih}^*)^2. \tag{4}
$$

`EXCESS_LOSS_MODE="wqe"` replaces only its pointwise penalty, using the same helper and shared WQE parameters as the global prediction objective:

$$
L_{\mathrm{excess}}=\frac{1}{BK}\sum_{i,h}E_i\,\rho_{\mathrm{scaled}}(e_{ih}^*-\hat e_{ih};\sigma_h).
$$

Every horizon of a GT Event Window contributes, including horizons whose target excess is zero. The denominator includes all windows; this term has no $q_E$ division. Units are m². `EXCESS_LOSS_WEIGHT` / `excess_loss_weight` defaults to 1. The supervised quantity is raw excess before multiplication by the gate. This term directly supervises the Direct excess branch or both reconstructed Severity–Shape factors.

**Window gate BCE:**

$$
\begin{aligned}
v&=\frac{1}{K}\sum_h\sigma_h^2,\\
L_{\mathrm{gate}}&=\frac{v}{B}\sum_i\operatorname{BCEWithLogits}(\ell_i,E_i).
\end{aligned} \tag{5}
$$

The learned gate has one logit $\ell_i$ per window, with $g_i=\operatorname{sigmoid}(\ell_i)$. BCE is averaged over all windows; multiplying by $v$ gives units m². `GATE_LOSS_WEIGHT` / `gate_loss_weight` defaults to 1. Only the gate branch receives a direct gradient from this term. The fixed-gate ablation has no learned logit and returns zero for this term.

## Hourly exceedance (Tail) loss

The independent coefficient is `EXCEEDANCE_LOSS_WEIGHT` / `exceedance_loss_weight`,
default 0. `EXCEEDANCE_LOSS_MODE` / `--exceedance_loss_mode` selects its pointwise
penalty, defaulting to `mse` for existing configs:

$$
\begin{aligned}
L_{\mathrm{exceedance}}&=\frac{1}{BKq_H}\sum_{i,h}M_{ih}\,p_{ih},\\
p_{ih}&=\begin{cases}
d_{ih}^2,&\text{Tail-MSE},\\
\rho_{\mathrm{scaled}}(y_{ih}-\hat y_{ih};\sigma_h),&\text{Tail-WQE}.
\end{cases}
\end{aligned} \tag{6}
$$

Tail-WQE simply uses `wqe_penalty(prediction, target, y_std)` in place of the
pointwise squared error. It shares all four WQE parameters and the detached TRAIN
`y_std` with global and raw-excess WQE; scales must be finite and positive when
any WQE mode is selected, including Tail-WQE with global/excess MSE.

Both modes have physical squared units (m²) and select only strict GT Extreme
Hours, `y > tau`. Ties and normal horizons within Event Windows contribute zero.
The mean includes all $BK$ target positions with masked zeros. The denominator
$q_H$ is the fixed observed TRAIN hourly prevalence, including the effect of
ties at $\tau$. It is never replaced by the batch's extreme count or event rate.
An extreme-free batch has a differentiable zero Tail contribution.

Global, raw-excess, and Tail modes are independent; switching one leaves the
other penalties, body, gate, amplitude, shape, and slope definitions unchanged.
Setting the Tail weight to zero disables either Tail mode. All three head
choices support it; its gradient follows the final prediction through the same
branches as $L_{\mathrm{pred}}$. The threshold, target mask, and prevalence are
constants.

## GT-aligned raw excess amplitude loss

For each window, select the first maximum of the original physical target:

$$
\begin{aligned}
h_i^*&=\operatorname{first\,argmax}_h y_{ih},\\
\hat a_i&=\hat e_{i,h_i^*},\\
a_i^*&=\max(y_{i,h_i^*}-\tau,0),\\
L_{\mathrm{amp}}&=\frac{1}{Bq_E}\sum_i E_i(\hat a_i-a_i^*)^2.
\end{aligned} \tag{7}
$$

`EXCESS_AMP_LOSS_WEIGHT` / `excess_amp_loss_weight` defaults to 0. The loss is measured in m². Each GT Event Window contributes one squared error. The selected horizon is GT-aligned: a larger predicted excess at another horizon cannot substitute for excess at $h_i^*$. A strict Event Window has $a_i^*>0$; the scalar threshold ensures $h_i^*$ also selects a maximum of the target excess.

The predicted quantity is raw excess; the gate and body do not enter (7). For Direct Dual, only the selected output excess coordinate receives the direct output gradient. For Severity–Shape, $\hat a_i=\mathrm{severity}_i\times\mathrm{shape}_{i,h_i^*}$; both factors receive gradients, and shape's maximum normalization can propagate a gradient to its normalizing maximum. Shared branch parameters and the backbone can affect multiple horizons. Single does not support amplitude supervision.

## Optional temporal shape loss

Severity–Shape supports `SHAPE_LOSS_WEIGHT` / `shape_loss_weight`, default 0. Define the target amplitude and dimensionless shape:

$$
\begin{aligned}
a_i^*&=\max_h e_{ih}^*,\\
s_{ih}^*&=\begin{cases}
e_{ih}^*/a_i^*,&E_i=1,\\
0,&E_i=0,
\end{cases}\\
L_{\mathrm{shape}}&=\frac{1}{BKq_E}\sum_{i,h}E_i(\hat s_{ih}-s_{ih}^*)^2.
\end{aligned} \tag{8}
$$

Every horizon of a GT Event Window contributes. The loss is dimensionless. Any positive target amplitude is used exactly, even when it is smaller than `severity_shape_eps`. Non-event windows use denominator 1 internally and retain zero targets. This term directly supervises the shape branch; it has no direct path to severity, body, or gate. The shared backbone receives its gradient.

`EXCESS_FORMULATION` / `excess_formulation` defaults to `direct`; a positive shape weight requires `severity_shape`. `SEVERITY_SHAPE_EPS` / `severity_shape_eps` defaults to $10^{-6}$ and must be finite and positive. Its architectural role is specified in [SEVERITY_SHAPE.md](SEVERITY_SHAPE.md). There is no separate scalar severity objective: severity is trained through reconstructed excess and final prediction, with optional GT-aligned amplitude supervision.

## Slope loss

The suffix `_slope` enables slope matching when $K>1$ and `slope_lambda` is nonzero:

$$
\begin{aligned}
r_{ih}&=(\hat y_{i,h+1}-\hat y_{ih})-(y_{i,h+1}-y_{ih}),\\
u_i&=\operatorname{sigmoid}\!\left(\frac{\tau-\max_h y_{ih}}{\max(s_{\mathrm{mask}},10^{-6}\,\mathrm{m})}\right),\\
L_{\mathrm{slope}}&=\frac{1}{B(K-1)}\sum_i\sum_{h=0}^{K-2}u_i\,\rho(r_{ih}).
\end{aligned} \tag{9}
$$

This is a soft weight over all windows and adjacent target pairs. The GT window maximum enters a smooth weight around the same $\tau$; there is no separately fitted cutoff. With one-hour targets, differences are hourly increments in meters; no explicit time division is performed.

`SLOPE_ROBUST` / `slope_robust` selects:

$$
\begin{aligned}
\rho_{\mathrm{charb}}(r)&=\sqrt{r^2+\varepsilon^2},\\
\rho_{\mathrm{huber}}(r)&=\begin{cases}
r^2/2,&\lvert r\rvert\le\delta,\\
\delta(\lvert r\rvert-\delta/2),&\lvert r\rvert>\delta.
\end{cases}
\end{aligned}
$$

Charbonnier has units m and includes its positive $\varepsilon$ baseline at zero error. Huber has units m². The numeric settings are:

| CLI field | Launcher variable | Default | Meaning |
| --- | --- | ---: | --- |
| `slope_lambda` | `SLOPE_LAMBDA_LIST` | 0.01 | Coefficient, active only in slope modes |
| `slope_mask_s` | `SLOPE_MASK_S_LIST` | 0.1 | Smooth window weight scale, m |
| `slope_robust` | `SLOPE_ROBUST` | `charb` | Penalty choice |
| `slope_charb_eps` | `SLOPE_CHARB_EPS` | 0.001 | Charbonnier $\varepsilon$, m |
| `slope_huber_delta` | `SLOPE_HUBER_DELTA` | 0.05 | Huber $\delta$, m |

The code floors $\varepsilon$ and $\delta$ at $10^{-12}$ m, and the sigmoid scale at $10^{-6}$ m. All head choices support slope matching; its prediction path reaches their active regression branches and learned gate.

## Complete objective and effective settings

With absent terms assigned coefficient zero, the training objective is:

$$
\begin{aligned}
L={}&L_{\mathrm{pred}}+\lambda_H L_{\mathrm{exceedance}}+\lambda_{\mathrm{slope}}L_{\mathrm{slope}}\\
&+\lambda_{\mathrm{body}}L_{\mathrm{body}}+\lambda_{\mathrm{excess}}L_{\mathrm{excess}}+\lambda_{\mathrm{gate}}L_{\mathrm{gate}}\\
&+\lambda_{\mathrm{amp}}L_{\mathrm{amp}}+\lambda_{\mathrm{shape}}L_{\mathrm{shape}}.
\end{aligned} \tag{10}
$$

The code adds numerical terms with their configured scales. A dimensionally uniform m² interpretation assigns $\lambda_{\mathrm{shape}}$ units m² and the Charbonnier $\lambda_{\mathrm{slope}}$ units m; the other coefficients, including Huber $\lambda_{\mathrm{slope}}$, are dimensionless. No additional rescaling is performed beyond the equations above.

`DUAL_LOSS` / `dual_loss` defaults to 1 for Dual and is resolved to 0 for Single. `DUAL_ABLATION` / `dual_ablation` defaults to `none`. In ordinary Dual runs, requesting `dual_loss=0` or a zero body/excess/gate coefficient restores required supervision and emits a warning. Positive coefficients are preserved. Named ablations explicitly control removals:

| `dual_ablation` | Effective branch change | Prediction/exceedance/slope objectives |
| --- | --- | --- |
| `none` | Body, excess, gate supervision required | Retained |
| `no_gate_bce` | Gate BCE weight = 0 | Retained; learned gate still receives prediction gradients |
| `no_excess_loss` | Excess trajectory, amplitude, shape weights = 0 | Retained; excess still receives prediction gradients |
| `no_branch_supervision` | All five branch auxiliary weights = 0; `dual_loss=0` | Retained |
| `fixed_gate` | Gate fixed at TRAIN $q_E$; gate BCE weight = 0 | Retained; no learned gate parameters |

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

[tools/generate_configs.py](../tools/generate_configs.py) generates the matched S0/D0/D1/D2/D3 variants. Their primary objective is MSE and slope weight is inactive. Generator branch weights override parser defaults: $\lambda_{\mathrm{body}}=1$, $\lambda_{\mathrm{excess}}=2$, $\lambda_{\mathrm{gate}}=0.5$ in every Dual variant. The parser defaults for these three fields are all 1.

$$
\begin{aligned}
L_{\mathrm{S0}}&=L_{\mathrm{pred}},\\
L_{\mathrm{D0}}&=L_{\mathrm{pred}}+L_{\mathrm{body}}+2L_{\mathrm{excess}}+0.5L_{\mathrm{gate}},\\
L_{\mathrm{D1}}&=L_{\mathrm{D0}}+0.025L_{\mathrm{exceedance}},\\
L_{\mathrm{D2}}&=L_{\mathrm{D0}}+0.003L_{\mathrm{amp}},\\
L_{\mathrm{D3}}&=L_{\mathrm{D0}}+0.025L_{\mathrm{exceedance}}+0.003L_{\mathrm{amp}}.
\end{aligned} \tag{11}
$$

| Variant | Pred | Body | Excess | Gate | Exceedance | Amp | Shape | Slope |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| S0 Single | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| D0 DualBase | 1 | 1 | 2 | 0.5 | 0 | 0 | 0 | 0 |
| D1 Exceedance | 1 | 1 | 2 | 0.5 | 0.025 | 0 | 0 | 0 |
| D2 Amp | 1 | 1 | 2 | 0.5 | 0 | 0.003 | 0 | 0 |
| D3 ExceedanceAmp | 1 | 1 | 2 | 0.5 | 0.025 | 0.003 | 0 | 0 |

The optional Severity–Shape D0–D3 configs use the same matrix with the alternative excess parameterization. Their generated shape weight is 0; explicitly enabling shape supervision adds $\lambda_{\mathrm{shape}}L_{\mathrm{shape}}$ to the corresponding objective. S0 has no Severity–Shape counterpart.

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

Every new run uses the published WQE parameters and Direct D0 settings: body weight 1, excess weight 2, gate weight 0.5, with exceedance, amplitude, and shape weights zero and no active slope term. Startup logs report the effective global loss, Dual excess loss (or Single inactivity), Tail mode and weight, and all four shared WQE parameters. Checkpoint selection, metrics, thresholds, splits, and architecture retain their existing definitions.

## Full factorial experiments

The [Single family](../configs/s0_refresh/README.md) contains 24 configs
(4 stations × 2 global modes × 3 Tail levels). The
[Dual family](../configs/wqe_factorial_multickpt/README.md) contains 48
(4 stations × 2 global modes × 2 raw-excess modes × 3 Tail levels).
Generate them with `--family s0_refresh` and `--family wqe_factorial_multickpt`.

| Label | Meaning | Control |
| --- | --- | --- |
| G0 / G1 | Global MSE / WQE | `LOSS_MODE_LIST` |
| E0 / E1 | Raw-excess MSE / WQE, Dual only | `EXCESS_LOSS_MODE` |
| T0 | Tail off | `EXCEEDANCE_LOSS_WEIGHT=0`, inactive mode `mse` |
| T1 | Tail-MSE | `EXCEEDANCE_LOSS_MODE=mse`, weight 0.025 |
| T2 | Tail-WQE | `EXCEEDANCE_LOSS_MODE=wqe`, weight 0.025 |

The historical Single suffix S0_Single_Tail_WQE maps to G1_T1: global WQE plus
Tail-MSE. G0_T2 and G1_T2 add Tail-WQE. Both families preserve their model/training
settings, result roots, shared WQE parameters, and four VAL checkpoint roles.
Amplitude, shape, and slope are inactive. Single has no branch objectives;
Dual retains body/excess/gate weights 1/2/0.5.

## Optimization and validation boundaries

[train.py](../train.py) uses Adam with fixed `weight_decay=1e-5`, which contributes its optimizer L2 regularization to trainable parameters. It is separate from the scalar returned by `ForecastLoss`. Dropout belongs to the architecture described in [BACKBONE.md](BACKBONE.md). There is no additional independent peak objective or explicit scalar severity loss.

[engine.py](../emulator/training/engine.py) supplies physical predictions and physical targets to the objective. Gradient accumulation divides each microbatch mean loss by the number of microbatches in its accumulation group, including the final partial group. Thus equally sized microbatches reproduce the corresponding combined-batch loss; unequal microbatches retain equal microbatch weights. All evaluation and checkpoint selection use the independent canonical physical metrics described in [METRICS.md](METRICS.md) and [CHECKPOINT_SELECTION.md](CHECKPOINT_SELECTION.md).
