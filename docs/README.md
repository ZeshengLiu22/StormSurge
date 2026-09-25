# PACT documentation

This documentation describes the supported implementation. Read it in this order:
forecasting task → hourly extreme definition → backbone representation → output
formulation → training objectives → validation checkpoint selection → evaluation.

```text
forcing/history + graph + optional station metadata
                        ↓
                  spatial encoder
                        ↓
                station/node readout
                        ↓
                  temporal module
                        ↓
                  horizon contexts
                        ↓
       ┌───────────────────────────────┐
       │ Single Head                   │
       └───────────────────────────────┘
                      OR
       ┌───────────────────────────────┐
       │ Dual Exceedance Head          │
       │ body + gate × excess          │
       └───────────────────────────────┘
                        ↓
                 physical prediction
```

| Document | Role |
| --- | --- |
| [FORMULATION](FORMULATION.md) | Target timestamps, physical units, the one TRAIN threshold, Extreme Hours, Event Windows and Event Episodes. |
| [BACKBONE](BACKBONE.md) | Inputs, supported encoders and temporal modules, readouts, stability and parameter accounting. |
| [DUAL_EXCEEDANCE](DUAL_EXCEEDANCE.md) | Direct Dual architecture, physical reconstruction, branch targets and supported ablations. |
| [SEVERITY_SHAPE](SEVERITY_SHAPE.md) | Optional severity × shape excess parameterization sharing the same backbone. |
| [LOSSES](LOSSES.md) | Complete objectives, masks, reductions, units, gradients, controls and experiment variants. |
| [CHECKPOINT_SELECTION](CHECKPOINT_SELECTION.md) | Four VAL-selected checkpoints from each training trajectory and final comparison. |
| [METRICS](METRICS.md) | Authoritative metric dictionary, worked example, lead-wise and fixed-episode diagnostics. |

The [repository README](../README.md) contains commands. Implementation references
inside each document identify the source of its equations and configuration.
