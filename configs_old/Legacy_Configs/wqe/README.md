# WQE placement

**Status: ACHIEVED / ARCHIVED — completed WQE placement study.**

This study compared WQE on the global prediction, raw-excess branch, or both.
All 12 manifest cells have saved final VAL/TEST summaries and completed logs
(verified 2026-09-27). WQE remains supported for historical/diagnostic use,
outside the current [Single Tail × EpisodePeak 4×4](../single_tail_episodepeak_4x4/README.md).

This completed family contains 12 NCEP configurations: CBBT, Lewes, Battery,
and Boston × three WQE placements on the direct Dual head.
The [baseline D0](../baseline_ablation/README.md) supplies the MSE/MSE control.

| Variant | Global prediction loss | Raw-excess branch loss |
| --- | --- | --- |
| D0 / W0 control | MSE | MSE |
| W1 GlobalWQE | WQE | MSE |
| W2 ExcessWQE | MSE | WQE |
| W3 BothWQE | WQE | WQE |

All WQE configs retain D0's body/excess/gate weights of 1/2/0.5. Tail-MSE,
amplitude, and shape losses are disabled. WQE uses quantile tau 0.25,
expectile tau 0.82, and weights 1/6 and 5/6. See the
[loss definitions](../../docs/LOSSES.md) and [manifest](manifest.csv).

Results use `/home/exouser/media/share/PACT/WQE_Results`.
The generator retains `--family wqe` for reproduction. To inspect a config
from the repository root:

```bash
DRY_RUN=1 USE_TMUX=0 bash train.sh configs/wqe/train_config_NCEP_CBBT_W1_GlobalWQE.sh
```

The [Dual factorial](../wqe_factorial_multickpt/README.md) expands these
placements with MSE/MSE controls and Tail off/MSE/WQE comparisons.
See the [configuration overview](../README.md) for archive status and related families.
