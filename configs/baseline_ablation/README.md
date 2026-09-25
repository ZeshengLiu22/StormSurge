# Baseline ablation

**Status: Temporarily archived（暂时归档）.**

Formerly `current`. This family contains 20 matched NCEP configurations:
CBBT, Lewes, Battery, and Boston × five Single/Dual loss conditions.

| Variant | Head | Objective |
| --- | --- | --- |
| S0 Single | Single | Global prediction MSE |
| D0 DualBase | Dual | Global MSE + body MSE + 2 × excess MSE + 0.5 × gate BCE |
| D1 Exceedance | Dual | D0 + 0.025 × Tail-MSE |
| D2 Amp | Dual | D0 + 0.003 × amplitude loss |
| D3 ExceedanceAmp | Dual | D0 + 0.025 × Tail-MSE + 0.003 × amplitude loss |

Tail-MSE uses strict exceedances above the fixed TRAIN Q95 threshold.
Amplitude supervision acts on raw excess at the ground-truth-aligned event peak.
See the [loss definitions](../../docs/LOSSES.md) and [manifest](manifest.csv).

Results use `./All_Results`. The generator's existing `--family current` option
(also the default) writes to this directory. To inspect a configuration from
the repository root:

```bash
DRY_RUN=1 USE_TMUX=0 bash train.sh configs/baseline_ablation/train_config_NCEP_CBBT_D0_DualBase.sh
```

D0 supplies the MSE/MSE control for the archived [WQE placement family](../wqe/README.md).
The [Single refresh](../s0_refresh/README.md) and
[Dual factorial](../wqe_factorial_multickpt/README.md) provide the matched
global-loss/Tail comparisons. See the [configuration overview](../README.md).
