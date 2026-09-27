# PACT documentation

This documentation describes the supported implementation. The current Single
development experiment is the [Tail × EpisodePeak 4×4](../configs/single_tail_episodepeak_4x4/README.md).
Read in this order: [FORMULATION](FORMULATION.md) → [BACKBONE](BACKBONE.md) →
output formulation (Single in BACKBONE, or the Dual documents below) →
[LOSSES](LOSSES.md) / [EPISODE_PEAK](EPISODE_PEAK.md) →
[CHECKPOINT_SELECTION](CHECKPOINT_SELECTION.md) → [METRICS](METRICS.md).

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
| [EPISODE_PEAK](EPISODE_PEAK.md) | True episode construction, canonical GT peak supervision, normalization, and implementation audit. |
| [CHECKPOINT_SELECTION](CHECKPOINT_SELECTION.md) | Two VAL-selected checkpoint roles: exceedance and overall, plus timestamp-aware final evaluation. |
| [METRICS](METRICS.md) | Authoritative metric dictionary, worked example, lead-wise and fixed-episode diagnostics. |

The [repository README](../README.md) contains commands. Implementation references
inside each document identify the source of its equations and configuration.
