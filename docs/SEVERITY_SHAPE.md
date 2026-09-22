# Severity–Shape excess parameterization

`SeverityShapeHead` in [heads.py](../emulator/models/heads.py) is a supported experimental Dual formulation. It replaces the excess branch with a physical severity factor and a dimensionless horizon shape. The upstream [backbone](BACKBONE.md), capped body, and Event Window gate retain the interfaces defined in [DUAL_EXCEEDANCE.md](DUAL_EXCEEDANCE.md).

<!-- choices excess_formulation: direct,severity_shape -->

The default is `EXCESS_FORMULATION=direct`. Select `EXCESS_FORMULATION=severity_shape` / `--excess_formulation severity_shape` with `MODEL=perceiver3` and `HEAD_TYPE=dual`. This parameterization supports both spatial encoders and every supported PACT temporal block.

## Context and branch networks

The head receives horizon contexts C ∈ ℝᴮˣᴷˣᵈ. It forms a shared window context c̄ᵢ = (1/K)Σₕ Cᵢₕ ∈ ℝᵈ. Severity and the learned gate use c̄ᵢ; shape and body use each Cᵢₕ. The production head uses arithmetic mean pooling.

Each branch uses `head_mlp`: Linear(d,2d) → LeakyReLU(0.1) → Dropout(`head_dropout`) → Linear(2d,1). A supplied internal `head_hidden` changes the intermediate width. Severity and shape outputs are cast to FP32 before their positive activations and normalization.

## Severity

For raw scalar severity score aᵢ:

    Âᵢ = softplus(aᵢ),        shape [B], units m.                 (1)

The factor is nonnegative and represents the maximum physical raw excess over the forecast horizons because predicted shape has unit maximum. The corresponding GT interpretation is A*ᵢ = maxₕ max(yᵢₕ−τ,0), using the canonical hourly TRAIN τ from [FORMULATION.md](FORMULATION.md).

The severity branch's final linear weights are initialized to zero and its bias to log(exp(0.1)−1). Initial severity is therefore 0.1 m for every window. No label-derived severity scale is fitted. There is no standalone loss comparing Â directly with A*.

## Temporal shape

For raw shape scores sᵢₕ and positive `SEVERITY_SHAPE_EPS` / `severity_shape_eps`:

    uᵢₕ = softplus(sᵢₕ) + ε
    ŝᵢₕ = uᵢₕ / maxⱼ uᵢⱼ,    shape [B,K], dimensionless.         (2)

The default ε is 10⁻⁶. It is added before maximum normalization, ensuring a positive denominator even when softplus underflows. For finite branch scores, ŝᵢₕ > 0 and maxₕ ŝᵢₕ = 1. There is no sum-to-one constraint: Σₕ ŝᵢₕ may range above 1 up to K. The maximum operation is a hard maximum, with PyTorch's first-index gradient behavior at tied maxima.

Shape final linear weights initialize to zero and the bias to log(exp(1)−1). Thus u is initially constant at approximately 1+ε and every normalized horizon shape initially equals 1.

GT shape uses e*ᵢₕ = max(yᵢₕ−τ,0). In a GT Event Window, s*ᵢₕ = e*ᵢₕ / maxⱼ e*ᵢⱼ, giving maximum 1 and exact zeros at normal hours. Any positive GT excess amplitude is used exactly, including amplitudes below ε. Non-event target shape is zero. The positive architectural ε does not floor target amplitudes.

## Reconstructed excess and final prediction

The registered `target_y_std` buffer contains TRAIN physical target scales σ in shape [1,K]. It must be finite, positive, and match the number of horizons. Reconstruction is:

    êᵖʰʸˢᵢₕ = Âᵢ ŝᵢₕ                   [B,K], m
    êⁿᵢₕ = Âᵢ ŝᵢₕ / σₕ                  [B,K], normalized
    ŷⁿᵢₕ = b̂ⁿᵢₕ + gᵢ êⁿᵢₕ               [B,K]
    ŷᵖʰʸˢᵢₕ = μₕ + σₕ b̂ⁿᵢₕ + gᵢ Âᵢ ŝᵢₕ.                  (3)

Target means μ enter physical body reconstruction; excess is a difference and uses only σ. The raw physical excess has maximum Âᵢ. The window scalar gate multiplies every reconstructed excess horizon, making the gated contribution's maximum gᵢÂᵢ. This need not equal the maximum final prediction because body also varies across horizons.

The learned gate is gᵢ = sigmoid(gate(c̄ᵢ)), shape [B,1]. Its final weights initialize to zero, and its bias is the logit of TRAIN Event Window prevalence clipped to [10⁻⁶,1−10⁻⁶]. With `DUAL_ABLATION=fixed_gate`, g is the unmodified empirical TRAIN prevalence and there is no gate network.

`ForecastOutput` stores normalized `prediction`, `body`, `excess`, and `threshold`; window `gate_logits` and `gate_probability`; physical `severity_phys` [B]; and dimensionless `excess_shape` [B,K]. `gate_logits` is `None` for a fixed gate. Inference performs the same reconstruction with fitted buffers and learned network outputs. GT labels are used only for supervision and evaluation.

## Direct and Severity–Shape comparison

| Feature | Direct | Severity–Shape |
| --- | --- | --- |
| Production class | `ExceedanceHead` | `SeverityShapeHead` |
| Excess branch input | Context at each horizon | Window mean for severity; each horizon for shape |
| Learned excess outputs | K positive normalized excess values | One physical severity and K shape scores |
| Physical excess | σₕ softplus(rawᵢₕ) | Âᵢ ŝᵢₕ |
| Cross-horizon normalization | None in excess output | Shape divided by its hard maximum |
| Shape constraint | No explicit factor | Nonnegative with maximum 1 |
| Initial excess | 0.1 normalized units at each horizon | 0.1 m at each horizon |
| Extra fixed buffer | None beyond τ | TRAIN target standard deviations |
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
| GT-aligned amplitude | Yes | Yes | Âᵢ ŝᵢ,h*ᵢ at the first GT target maximum |
| Shape | No | Yes | Dimensionless normalized shape over GT Event Windows |
| Body / gate BCE | No | No | Their respective body / gate outputs |

All enabled objectives also reach the shared backbone through their active branch paths. Amplitude supervision gathers shape at the GT peak, and the hard maximum in (2) may propagate gradients through the shape denominator. It is therefore not a standalone severity-target comparison.

`SHAPE_LOSS_WEIGHT` defaults to 0; a positive value enables dimensionless shape supervision. Generated optional Severity–Shape D0–D3 configs retain the same prediction/branch/exceedance/amplitude coefficients as their Direct counterparts and keep shape weight 0. Named Dual ablations apply the same effective loss removals to either parameterization.

## Source and configuration checks

This description follows `SeverityShapeHead`, `ForecastOutput`, and `head_mlp` in [heads.py](../emulator/models/heads.py), the formulation checks in [dual.py](../emulator/common/dual.py), the physical loss targets in [excess_shape.py](../emulator/training/excess_shape.py), the parser in [arguments.py](../emulator/training/arguments.py), and model construction/checkpoint buffers in [train.py](../train.py). [test_severity_shape.py](../tests/test_severity_shape.py) verifies reconstruction, initial values, constraints, gradient paths, low-precision arithmetic, and checkpoint/inference round trips.
