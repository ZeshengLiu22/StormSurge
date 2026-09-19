# A0 HeadCapacity validation — PASS

Validated UTC: 2026-09-16T23:33:08.196006+00:00  
Repository commit: `823a195b3be8e9778b272c64a61022dc4de8e2a0`

Exactly 12 standalone configs, three per station and four per formulation. All 12 passed `bash -n`,
`DRY_RUN=1 bash train.sh <config>`, and the production argument parser; each emitted exactly one run.
All non-experimental controls match the validated production configs. No runtime config inheritance.

| Station | Single base | Direct dual | Severity-shape dual | Total |
| --- | ---: | ---: | ---: | ---: |
| CBBT | 1 | 1 | 1 | 3 |
| Lewes | 1 | 1 | 1 | 3 |
| Battery | 1 | 1 | 1 | 3 |
| Boston | 1 | 1 | 1 | 3 |

Tail, excess amplitude, explicit shape, final peak, and slope supervision are OFF in all 12 configs.
Both duals use BODY=1, trajectory EXCESS=2, GATE=0.5, exceedance/window gating, dual loss 1 and ablation none.
Single uses dual loss 0 and only a regression head. Every run selects `overall` with auxiliary checkpoints 0.

The unchanged launcher omits tail/slope arguments under `mse`; the parser retains inactive defaults
`tail_lambda=0.1` and `slope_lambda=0.01`. The config coefficients are explicitly zero and neither term
enters the MSE objective. Similarly, single-head branch weights are explicitly zero in the config;
the launcher omits these arguments and the parser disables dual supervision (`dual_loss=0`).

## Parameter counts

Counts use emulator.models.count_model_parameters on CPU models built from parsed A0 settings, actual graph tensor dimensions, and the six existing station-coordinate features. Only one TRAIN-year file per station is memory-mapped onto the meta device for shapes. Placeholder thresholds=0, target scales=1 and prior=0.05 are used only for counting; they are non-parameter buffers/initialization and do not change registered capacity. Training still fits actual TRAIN statistics. No model forward, optimizer, training or queue submission occurs.

| Station | Model | Total / trainable | Head | Backbone |
| --- | --- | ---: | ---: | ---: |
| CBBT | single_base | 618,885 | 33,281 | 585,604 |
| CBBT | dual_direct_base | 685,447 | 99,843 | 585,604 |
| CBBT | dual_severity_base | 718,728 | 133,124 | 585,604 |
| Lewes | single_base | 618,885 | 33,281 | 585,604 |
| Lewes | dual_direct_base | 685,447 | 99,843 | 585,604 |
| Lewes | dual_severity_base | 718,728 | 133,124 | 585,604 |
| Battery | single_base | 618,885 | 33,281 | 585,604 |
| Battery | dual_direct_base | 685,447 | 99,843 | 585,604 |
| Battery | dual_severity_base | 718,728 | 133,124 | 585,604 |
| Boston | single_base | 618,885 | 33,281 | 585,604 |
| Boston | dual_direct_base | 685,447 | 99,843 | 585,604 |
| Boston | dual_severity_base | 718,728 | 133,124 | 585,604 |

Severity-shape adds **33,281** trainable parameters over direct: one 128→256→1 branch MLP,
**4.855%** more total capacity and **33.333%** more head capacity.
Direct excess capacity is 33,281; severity plus shape capacity is 66,562. Body/gate/backbone are unchanged.
This is the requested three-formulation study; no fourth capacity-matched model was added.

See [manifest.csv](manifest.csv), [commands_all.txt](commands_all.txt),
[full verification](validation_report.json), and [branch counts](validation/parameter_counts.json).
The `validation/` directory contains all 12 dry-run transcripts.

Model/loss/launcher hashes and result-tree contents were unchanged by validation.
**No training started, no jobs submitted, and no model forward passes or optimizer steps executed.**
