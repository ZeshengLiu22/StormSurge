# Direct Dual exceedance head

The production `ExceedanceHead` in [`heads.py`](../emulator/models/heads.py) receives the shared PACT contexts from [BACKBONE](BACKBONE.md) and predicts a capped body, nonnegative excess and one Event Window gate. It uses the single physical threshold from [FORMULATION](FORMULATION.md):

\[
 \tau=Q_p\bigl(\{y_t:t\text{ is a TRAIN supervised target hour}\}\bigr),
 \qquad p=\mathrm{exceedance\_percentile}/100.
\]

The default is TRAIN hourly Q95. All splits and corresponding OOD evaluations retain the same source TRAIN threshold. The model's normalized threshold vector represents this one physical value under per-horizon normalization.

## Architecture

Let $C_{ih}\in\mathbb R^d$ be the context of horizon $h$ in window $i$, with $C\in\mathbb R^{B\times K\times d}$. Let $\mu_h\in\mathbb R$ and $s_h>0$ denote TRAIN target mean and scale, and define

\[
 z_{ih}=\frac{y_{ih}-\mu_h}{s_h},\qquad
 \theta_h=\frac{\tau-\mu_h}{s_h}.
\]

`ModelConfig.peak_threshold_norm` carries $(\theta_0,\ldots,\theta_{K-1})$; `train.py` fills it from `tau_normalized`. `peak_prior` carries the observed TRAIN Event Window rate. These are construction fields, not additional fitted extreme definitions. The head registers `threshold` as a non-trainable FP32 buffer `[1,K]`.

Production body, excess and gate each use an independent MLP:

\[
 f(v)=W_2\operatorname{Dropout}\!\left(\operatorname{LeakyReLU}_{0.1}(W_1v+b_1)\right)+b_2,
\]

with widths $d\to w\to1$, normally $w=2d$. `head_dropout` controls their dropout, and the Python/checkpoint `head_hidden` field can override $w$. Parameters are not shared among branches, but each branch shares its MLP across horizons.

### Normalized implementation

With independent scalar raw outputs $r_{ih}=f_b(C_{ih})$ and $a_{ih}=f_e(C_{ih})$, the implementation computes

\[
 \hat b^{\rm norm}_{ih}=\theta_h-\operatorname{softplus}(\theta_h-r_{ih}),\qquad
 \hat e^{\rm norm}_{ih}=\operatorname{softplus}(a_{ih}).
\]

It mean-pools the horizon contexts for a scalar window gate:

\[
 \bar C_i=\frac1K\sum_hC_{ih},\qquad
 \ell_i=f_g(\bar C_i),\qquad g_i=\sigma(\ell_i),
\]
\[
 \hat z_{ih}=\hat b^{\rm norm}_{ih}+g_i\hat e^{\rm norm}_{ih}.
\]

Raw branch outputs are converted to FP32 before the body cap, softplus and sigmoid. The body is upper-bounded by $\theta_h$, with no lower bound. Excess is nonnegative. A learned sigmoid gate is shared across all horizons, rather than predicting separate hourly probabilities. No hard gate threshold is applied during inference.

### Physical interpretation

Define physical body and raw excess:

\[
 \hat b_{ih}=\mu_h+s_h\hat b^{\rm norm}_{ih}
     =\tau-s_h\operatorname{softplus}(\theta_h-r_{ih}),\qquad
 \hat e_{ih}=s_h\hat e^{\rm norm}_{ih}.
\]

Then

\[
 \boxed{\hat y_{ih}=\hat b_{ih}+g_i\hat e_{ih}},\qquad
 \hat b_{ih}\le\tau,\quad \hat e_{ih}\ge0.
\]

Body and excess have units of meters; $g_i$ is dimensionless. The body cap is strict in exact finite arithmetic, but floating-point softplus underflow can reach equality. The final prediction can exceed $\tau$. Because body may lie below $\tau$, the branch quantity $g_i\hat e_{ih}$ is not generally identical to $\max(\hat y_{ih}-\tau,0)$.

### Initialization and gate meaning

Let $q_E=N_{\rm TRAIN}^{-1}\sum_iE_i$ be the observed TRAIN Event Window rate. [`initial_gate_prior`](../emulator/common/dual.py) clips this rate to $\tilde q_E=\min(\max(q_E,10^{-6}),1-10^{-6})$ only for finite learned-logit initialization. The gate's final linear weights are zero and its bias is $\operatorname{logit}(\tilde q_E)$, so its initial prediction is the same probability in every window. The saved empirical rate remains $q_E$.

The excess MLP's last weights are zero and its bias is $\log(\operatorname{expm1}(0.1))$. Initial normalized raw excess is therefore 0.1 at each horizon, corresponding to physical excess $0.1s_h$ meters. Body parameters retain the ordinary MLP initialization. This initialization uses the Event Window rate, which generally differs from the extreme-hour rate.

The learned gate is supervised as the probability that at least one target hour exceeds $\tau$. It is a soft reconstruction factor; it is not an event detector based on predicted percentiles, and its value need not equal the frequency of individual extreme hours.

### Output contract and inference

`ForecastOutput` stores:

| Field | Direct Dual shape | Space / meaning |
|---|---|---|
| `prediction` | `[B,K]` | Reconstructed normalized forecast $\hat z$ |
| `body` | `[B,K]` | Normalized capped body |
| `excess` | `[B,K]` | Normalized raw nonnegative excess before gating |
| `gate_logits` | `[B,1]` | One learned logit per window; `None` for fixed gate |
| `threshold` | `[1,K]` | Broadcast normalized TRAIN threshold buffer |
| `gate_probability` | `[B,1]` | One soft gate per window |
| `severity_phys`, `excess_shape` | `None` | Used only by [Severity–Shape](SEVERITY_SHAPE.md) |

The engine converts `prediction` to meters using the saved $\mu_h,s_h$. Optional exported `body_phys`, `excess_phys`, `gate_probability` and `gate_logits` preserve the physical branch decomposition for [event diagnostics](../tools/event_diagnostics.py), which verifies reconstruction numerically. The forward method receives only contexts and fixed fitted quantities. Ground truth is needed for training supervision and evaluation populations, not for inference gating or reconstruction.

## Supervision

Physical targets at every horizon are

\[
 b^*_{ih}=\min(y_{ih},\tau),\qquad e^*_{ih}=\max(y_{ih}-\tau,0),\qquad
 E_i=\mathbf1\{\exists h:y_{ih}>\tau\}.
\]

Thus $y_{ih}=b^*_{ih}+e^*_{ih}$, and $y=\tau$ is not extreme. Normalized branch targets are $(b^*_{ih}-\mu_h)/s_h$ and $e^*_{ih}/s_h$. The production loss receives physical targets and the full-precision `tau_physical` when constructing the strict event mask and excess; its mask is not reconstructed by comparing rounded normalized buffers.

| Branch | Learned output | Constraint | Target | Direct auxiliary supervision |
|---|---|---|---|---|
| Body | $\hat b^{\rm norm}_{ih}$ | Physical $\hat b_{ih}\le\tau$ | $b^*_{ih}$ | Every horizon of every TRAIN window |
| Raw excess | $\hat e^{\rm norm}_{ih}$ | Physical $\hat e_{ih}\ge0$ | $e^*_{ih}$ | Every horizon of GT Event Windows, including zero-excess horizons |
| Gate | $\ell_i,g_i$ | $0\le g_i\le1$ | $E_i$ | All TRAIN windows through BCE, unless disabled by an explicit ablation |
| Final forecast | $\hat z_{ih}$ | Body plus gated excess | $y_{ih}$ | All TRAIN horizons through the primary prediction objective |

The final prediction objective propagates through all active branches. Optional exceedance supervision uses strict GT extreme hours of the final forecast; optional amplitude supervision acts on raw excess at the GT-aligned Event Window peak. See [LOSSES](LOSSES.md) for the exact masks, denominators, units, coefficients and gradient paths. Architecture and reconstruction do not change between generated D0/D1/D2/D3 conditions.

## Supported configuration and ablations

Production direct Dual uses `--model perceiver3 --head_type dual --excess_formulation direct`. The only gate/method choices are `gate_mode=window` and `dual_mode=exceedance`. The default `dual_ablation=none` enforces body, excess and gate supervision: setting a required coefficient to zero is resolved back to a positive value by `enforce_dual_loss`. An intentional removal uses a named ablation.

| `dual_ablation` | Architectural effect | Supervision disabled |
|---|---|---|
| `none` | Learned body, excess and gate | None |
| `no_gate_bce` | Same head | Gate BCE |
| `no_excess_loss` | Same head | Excess trajectory, amplitude and shape terms |
| `no_branch_supervision` | Same head | Body, excess trajectory, gate BCE, amplitude and shape terms |
| `fixed_gate` | Gate network removed; $g_i=q_E$ is a buffer | Gate BCE |

The fixed gate retains the exact empirical $q_E$, including possible 0 or 1, and has no learned logit. In all ablations the primary prediction objective remains; optional final-forecast exceedance and slope objectives also remain when enabled. The detailed objective mapping is in [LOSSES](LOSSES.md).

## Optional direct excess and gate-pooling presets

`exceedance_head_experiment` is omitted by default. When explicitly selected, it constructs `ExceedanceHead_Experiment` with the same body cap, positive excess activation, gate semantics and reconstruction. It requires PACT, a Dual head and `excess_formulation=direct`.

Let $F$ denote the sequence `Linear(d,2d) → LeakyReLU(0.1) → Dropout(head_dropout) → Linear(2d,d)`, applied to each horizon context. The currently supported values are:

| Preset | Excess computation before softplus |
|---|---|
| `legacy` | MLP $d\to w\to1$, with production $w=2d$ or `head_hidden` |
| `c1` | MLP $d\to4d\to1$ |
| `c2` | $\operatorname{Linear}_{d\to1}(F(C))$ |
| `c2r` | $\operatorname{Linear}_{d\to1}(C+F(C))$ |
| `c3` | $\operatorname{Linear}_{d\to1}(C+F(\operatorname{LN}_{\rm affine}(C)))$ |

The `legacy` value is a supported preset name for the production MLP layout. Every preset retains the same initial normalized excess 0.1 by zero-initializing the final excess projection's weights and using the same softplus-inverse bias. Body and gate retain their production MLP widths. These options change head capacity only; they do not modify the backbone or threshold.

For these optional presets, `exceedance_gate_pooling=mean` averages horizon contexts. `learned` instead computes

\[
 a_{ih}=w^\mathsf T\operatorname{LN}_{\rm affine}(C_{ih})+b,\quad
 \alpha_{ih}=\operatorname{softmax}_h(a_{ih}),\quad
 \bar C_i=\sum_h\alpha_{ih}C_{ih}.
\]

The gate MLP receives this pooled vector. Its affine normalization and scalar scoring layer use their standard initializers; the gate output itself still starts at $\tilde q_E$ because its final projection is initialized as above. `fixed_gate` omits learned pooling parameters as well as the gate network. The pooling setting has an effect only when a direct head preset is selected; the production head and Severity–Shape head use mean pooling.
