# Severity–Shape excess parameterization

`SeverityShapeHead` in [heads.py](../emulator/models/heads.py) is a supported experimental Dual formulation. It replaces the excess branch with a physical severity factor and a dimensionless horizon shape. The upstream [backbone](BACKBONE.md), capped body, and Event Window gate retain the interfaces defined in [DUAL_EXCEEDANCE.md](DUAL_EXCEEDANCE.md).

<!-- choices excess_formulation: direct,severity_shape -->

The default is `EXCESS_FORMULATION=direct`. Select `EXCESS_FORMULATION=severity_shape` / `--excess_formulation severity_shape` with `MODEL=perceiver3` and `HEAD_TYPE=dual`. This parameterization supports both spatial encoders and every supported PACT temporal block.

## Context and branch networks

The head receives horizon contexts $C\in\mathbb R^{B\times K\times d}$. It forms a shared window context $\bar c_i=(1/K)\sum_h C_{ih}\in\mathbb R^d$. Severity and the learned gate use $\bar c_i$; shape and body use each $C_{ih}$. The production head uses arithmetic mean pooling.

Each branch uses `head_mlp`: Linear(d,2d) → LeakyReLU(0.1) → Dropout(`head_dropout`) → Linear(2d,1). A supplied internal `head_hidden` changes the intermediate width. Severity and shape outputs are cast to FP32 before their positive activations and normalization.

## Severity

For raw scalar severity score $a_i$:

$$
\hat A_i=\operatorname{softplus}(a_i). \tag{1}
$$

The output has shape `[B]` and units m. The factor is nonnegative and represents the maximum physical raw excess over the forecast horizons because predicted shape has unit maximum. The corresponding GT interpretation is $A_i^*=\max_h\max(y_{ih}-\tau,0)$, using the canonical hourly TRAIN $\tau$ from [FORMULATION.md](FORMULATION.md).

The severity branch's final linear weights are initialized to zero and its bias to $\log(\exp(0.1)-1)$. Initial severity is therefore 0.1 m for every window. No label-derived severity scale is fitted. There is no standalone loss comparing $\hat A$ directly with $A^*$.

## Temporal shape

For raw shape scores $s_{ih}$ and positive `SEVERITY_SHAPE_EPS` / `severity_shape_eps`:

$$
\begin{aligned}
u_{ih}&=\operatorname{softplus}(s_{ih})+\varepsilon,\\
\hat s_{ih}&=\frac{u_{ih}}{\max_j u_{ij}}.
\end{aligned} \tag{2}
$$

The output has shape `[B,K]` and is dimensionless. The default $\varepsilon$ is $10^{-6}$. It is added before maximum normalization, ensuring a positive denominator even when softplus underflows. For finite branch scores, $\hat s_{ih}>0$ and $\max_h\hat s_{ih}=1$. There is no sum-to-one constraint: $\sum_h\hat s_{ih}$ may range above 1 up to $K$. The maximum operation is a hard maximum, with PyTorch's first-index gradient behavior at tied maxima.

Shape final linear weights initialize to zero and the bias to $\log(\exp(1)-1)$. Thus $u$ is initially constant at approximately $1+\varepsilon$ and every normalized horizon shape initially equals 1.

GT shape uses $e_{ih}^*=\max(y_{ih}-\tau,0)$. In a GT Event Window, $s_{ih}^*=e_{ih}^*/\max_j e_{ij}^*$, giving maximum 1 and exact zeros at normal hours. Any positive GT excess amplitude is used exactly, including amplitudes below $\varepsilon$. Non-event target shape is zero. The positive architectural $\varepsilon$ does not floor target amplitudes.

## Reconstructed excess and final prediction

The registered `target_y_std` buffer contains TRAIN physical target scales $\sigma$ in shape `[1,K]`. It must be finite, positive, and match the number of horizons. Reconstruction is:

$$
\begin{aligned}
\hat e^{\mathrm{phys}}_{ih}&=\hat A_i\hat s_{ih},\\
\hat e^{\mathrm{norm}}_{ih}&=\hat A_i\hat s_{ih}/\sigma_h,\\
\hat y^{\mathrm{norm}}_{ih}&=\hat b^{\mathrm{norm}}_{ih}+g_i\hat e^{\mathrm{norm}}_{ih},\\
\hat y^{\mathrm{phys}}_{ih}&=\mu_h+\sigma_h\hat b^{\mathrm{norm}}_{ih}+g_i\hat A_i\hat s_{ih}.
\end{aligned} \tag{3}
$$

Each reconstructed tensor has shape `[B,K]`; physical excess and prediction have units m, while normalized quantities are dimensionless. Target means $\mu$ enter physical body reconstruction; excess is a difference and uses only $\sigma$. The raw physical excess has maximum $\hat A_i$. The window scalar gate multiplies every reconstructed excess horizon, making the gated contribution's maximum $g_i\hat A_i$. This need not equal the maximum final prediction because body also varies across horizons.

The learned gate is $g_i=\operatorname{sigmoid}(\operatorname{gate}(\bar c_i))$, shape `[B,1]`. Its final weights initialize to zero, and its bias is the logit of TRAIN Event Window prevalence clipped to $[10^{-6},1-10^{-6}]$. With `DUAL_ABLATION=fixed_gate`, $g$ is the unmodified empirical TRAIN prevalence and there is no gate network.

`ForecastOutput` stores normalized `prediction`, `body`, `excess`, and `threshold`; window `gate_logits` and `gate_probability`; physical `severity_phys` `[B]`; and dimensionless `excess_shape` `[B,K]`. `gate_logits` is `None` for a fixed gate. Inference performs the same reconstruction with fitted buffers and learned network outputs. GT labels are used only for supervision and evaluation.

## Direct and Severity–Shape comparison

| Feature | Direct | Severity–Shape |
| --- | --- | --- |
| Production class | `ExceedanceHead` | `SeverityShapeHead` |
| Excess branch input | Context at each horizon | Window mean for severity; each horizon for shape |
| Learned excess outputs | $K$ positive normalized excess values | One physical severity and $K$ shape scores |
| Physical excess | $\sigma_h\operatorname{softplus}(\mathrm{raw}_{ih})$ | $\hat A_i\hat s_{ih}$ |
| Cross-horizon normalization | None in excess output | Shape divided by its hard maximum |
| Shape constraint | No explicit factor | Nonnegative with maximum 1 |
| Initial excess | 0.1 normalized units at each horizon | 0.1 m at each horizon |
| Extra fixed buffer | None beyond $\tau$ | TRAIN target standard deviations |
| Body and gate | Capped body and scalar Event Window gate | Same production definitions |
| Backbone | PACT contexts | Same PACT contexts |
| Default formulation | Yes | No; experimental option |

Direct excess-capacity experiments are separate options described in [DUAL_EXCEEDANCE.md](DUAL_EXCEEDANCE.md). Their configuration is restricted to Direct.

## Loss interaction

The complete equations, populations, reductions, units, and coefficients are defined in [LOSSES.md](LOSSES.md). Their effects on the factors are:

| Objective | Severity direct path | Shape direct path | Supervised quantity |
| --- | --- | --- | --- |
| Prediction MSE/WMSE | Yes | Yes | Final physical prediction |
| Hourly exceedance | Yes | Yes | Final prediction at GT Extreme Hours |
| Slope | Yes | Yes | Adjacent final prediction increments |
| Excess trajectory | Yes | Yes | Reconstructed raw excess over GT Event Windows |
| GT-aligned amplitude | Yes | Yes | $\hat A_i\hat s_{i,h_i^*}$ at the first GT target maximum |
| Shape | No | Yes | Dimensionless normalized shape over GT Event Windows |
| Body / gate BCE | No | No | Their respective body / gate outputs |

All enabled objectives also reach the shared backbone through their active branch paths. Amplitude supervision gathers shape at the GT peak, and the hard maximum in (2) may propagate gradients through the shape denominator. It is therefore not a standalone severity-target comparison.

`SHAPE_LOSS_WEIGHT` defaults to 0; a positive value enables dimensionless shape supervision. Generated optional Severity–Shape D0–D3 configs retain the same prediction/branch/exceedance/amplitude coefficients as their Direct counterparts and keep shape weight 0. Named Dual ablations apply the same effective loss removals to either parameterization.

## Source and configuration checks

This description follows `SeverityShapeHead`, `ForecastOutput`, and `head_mlp` in [heads.py](../emulator/models/heads.py), the formulation checks in [dual.py](../emulator/common/dual.py), the physical loss targets in [excess_shape.py](../emulator/training/excess_shape.py), the parser in [arguments.py](../emulator/training/arguments.py), and model construction/checkpoint buffers in [train.py](../train.py). [test_severity_shape.py](../tests/test_severity_shape.py) verifies reconstruction, initial values, constraints, gradient paths, low-precision arithmetic, and checkpoint/inference round trips.
