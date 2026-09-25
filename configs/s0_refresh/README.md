# Single G × T factorial

Status: not archived. See the [configuration overview](../README.md).

This family contains 24 Single-head PACT configs: CBBT, Boston, Battery, and
Lewes × global MSE/WQE × Tail off/MSE/WQE. Each station has six cells.
All use `HEAD_TYPE=single` and `DUAL_LOSS=0`.

**G = Global**: the final prediction loss over all target hours. G0 uses MSE
(mean squared error); G1 uses WQE (weighted quantile–expectile loss).
**T = Tail**: an additional final-prediction loss only at strict TRAIN Q95
extreme hours (`y > tau`). T0 disables it; T1 uses MSE; T2 uses WQE.
Both enabled Tail modes have weight 0.025 and fixed TRAIN `q_H` normalization.
G0 still enables MSE; only T0 means off. There is no G2.

**E = Excess** selects the raw-excess branch loss in Dual runs (E0 = MSE,
E1 = WQE; no E2). Single has no such branch, so its filenames use G/T only.
For example, G1_T2 means global WQE + Tail-WQE at weight 0.025.
See the [shared naming guide](../README.md#g--e--t-和-0--1--2-的含义).

| Cell | Global prediction loss | Tail penalty | Tail weight | Previous config/run suffix |
| --- | --- | --- | ---: | --- |
| G0_T0 | MSE | Off | 0 | `S0_Single` |
| G0_T1 | MSE | MSE | 0.025 | `S0_Single_Tail` |
| G0_T2 | MSE | WQE | 0.025 | New |
| G1_T0 | WQE | Off | 0 | `S0_Single_WQE` |
| G1_T1 | WQE | MSE | 0.025 | `S0_Single_Tail_WQE` |
| G1_T2 | WQE | WQE | 0.025 | New |

The historical `S0_Single_Tail_WQE` used global WQE plus Tail-MSE and maps to
G1_T1. T2 introduces the independently selected Tail-WQE penalty.
The generator replaces the 16 old filenames with the complete factorial.
Files are named `train_config_NCEP_<station>_G<g>_T<t>.sh`; the
[manifest](manifest.csv) records all loss modes, weights, and run names.

Results retain `/home/exouser/media/share/PACT/0924_s0_refresh/`, with distinct
`NCEP_<station>_G<g>_T<t>__<timestamp>/` run directories. Existing result
directories keep their historical names.

## Loss definitions

G0/G1 select `LOSS_MODE_LIST=("mse")` / `("wqe")` over all target hours.
T0 disables Tail with `EXCEEDANCE_LOSS_WEIGHT=0`. T1/T2 select
`EXCEEDANCE_LOSS_MODE="mse"` / `"wqe"`, both at weight 0.025.
T0 records the inactive mode as `mse`.

```text
Tail_MSE = mean(1[y > tau] * (prediction - y)^2) / q_H
Tail_WQE = mean(1[y > tau] * wqe_penalty(prediction, y, y_std)) / q_H
Single loss = global_loss + EXCEEDANCE_LOSS_WEIGHT * Tail_loss
```

`tau` is the fixed Q95 of unique TRAIN target hours. The mask is strictly
`y > tau`, including within Event Windows; ties contribute zero. The mean
includes all target positions, and `q_H` is the fixed observed TRAIN
extreme-hour fraction. Tail-WQE uses the same detached TRAIN `y_std` and shared
parameters as global WQE:

```bash
WQE_QUANTILE_TAU=0.25
WQE_EXPECTILE_TAU=0.82
WQE_QUANTILE_WEIGHT=0.16666666666666667
WQE_EXPECTILE_WEIGHT=0.83333333333333333
```

`EXCESS_LOSS_MODE="mse"` is inactive for Single. Body/excess/gate weights
remain the inactive Single defaults 1/1/1. Amplitude and shape weights remain
zero. See the [loss definitions](../../docs/LOSSES.md).

## Matched training and evaluation

All model and training settings are preserved: GraphSAGE + Transformer, width
128, 24-hour history, batch size 256 with four accumulation steps, learning rate
`5e-3`, 300 epochs, cosine schedule with five warmup epochs, BF16 AMP, and TF32.
Station selection, NCEP Grid4_New data, chronological 60/20/20 splits, seed 42,
normalization, station metadata, and loader settings remain the same.
The [Dual factorial](../wqe_factorial_multickpt/README.md) provides corresponding
G/E/T cells, with active branch weights 1/2/0.5.

All four VAL-selected roles are retained: `overall`, `exceedance`,
`aligned_peak`, and `bea` (Balanced Event-Aware). BEA minimizes the fixed score
`0.50*AllRMSE + 0.25*ExceedanceRMSE + 0.25*GTAlignedPeakRMSE`.
Each role is reevaluated on VAL and TEST; `overall` supplies the primary alias.
See [checkpoint selection](../../docs/CHECKPOINT_SELECTION.md).

The interpreter remains `/home/exouser/.conda/envs/torchpyg-cu12x/bin/python`.
Export `S0_PYTHON_BIN` to deliberately override it.

## Generation and launchers

From the repository root, regenerate and inspect commands without training:

```bash
python tools/generate_configs.py --family s0_refresh
DRY_RUN=1 USE_TMUX=0 bash train.sh configs/s0_refresh/train_config_NCEP_CBBT_G0_T2.sh
DRY_RUN=1 bash configs/s0_refresh/launch_all.sh
```

[launch_all.sh](launch_all.sh) covers all 24 jobs; each `launch_GgTt.sh`
covers the four stations for one cell (for example,
[launch_G1T2.sh](launch_G1T2.sh)). Order is G, then T, then CBBT, Boston,
Battery, Lewes. With `DRY_RUN=1`, launchers print resolved commands, bypass
CUDA preflight and the queue, and create no result directories.

Invoking a launcher without `DRY_RUN=1` submits its jobs through `qsub_local`
after [preflight.sh](preflight.sh) verifies training imports and CUDA.
Generation alone never submits jobs. Tests reproduce all configs and the
manifest, parse dry-run arguments, compare unchanged settings to Single
controls, check all four checkpoint roles, and exercise launch order using a
mock queue.
