# PACT documentation

The **ACTIVE** scientific development path is the
[Single Tail × EpisodePeak 4×4](../configs/single_tail_episodepeak_4x4/README.md).
The [configuration overview](../configs/README.md#status-vocabulary) defines
experiment-family and implementation statuses.

## Current Single documentation

This sequence describes the current active Single development pipeline:
[FORMULATION](FORMULATION.md) → [BACKBONE](BACKBONE.md) →
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
                        ↓
                 physical prediction
```

| Document | Role |
| --- | --- |
| [FORMULATION](FORMULATION.md) | Target timestamps, physical units, the one TRAIN threshold, Extreme Hours, Event Windows and Event Episodes. |
| [BACKBONE](BACKBONE.md) | Inputs, supported encoders and temporal modules, readouts, stability and parameter accounting. |
| [LOSSES](LOSSES.md) | Current Single objective, feature status table, and complete supported loss formulas and controls. |
| [EPISODE_PEAK](EPISODE_PEAK.md) | True episode construction, canonical GT peak supervision, normalization, and implementation audit. |
| [CHECKPOINT_SELECTION](CHECKPOINT_SELECTION.md) | Two VAL-selected checkpoint roles: exceedance and overall, plus timestamp-aware final evaluation. |
| [METRICS](METRICS.md) | Authoritative metric dictionary, worked example, lead-wise and fixed-episode diagnostics. |

## Supported legacy / optional formulations

These Dual implementations remain supported and tested for historical reproduction
and optional extensions. They are outside the current Single research path.

| Document | Implementation status and role |
| --- | --- |
| [DUAL_EXCEEDANCE](DUAL_EXCEEDANCE.md) | SUPPORTED LEGACY / historical Dual formulation: Direct Dual architecture, physical reconstruction, branch targets and supported ablations. |
| [SEVERITY_SHAPE](SEVERITY_SHAPE.md) | SUPPORTED LEGACY / optional experimental Dual formulation: severity × shape excess parameterization sharing the same backbone. |

The [repository README](../README.md) contains commands. Implementation references
inside each document identify the source of its equations and configuration.
