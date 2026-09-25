# Dual G × E × T factorial with multiple checkpoints

Status: not archived. See the [configuration overview](../README.md).

This family contains 48 direct Dual-head NCEP configs: CBBT, Boston, Battery,
and Lewes × global MSE/WQE × raw-excess MSE/WQE × Tail off/MSE/WQE.
Each station has 12 cells. Files are named
`train_config_NCEP_<station>_G<g>_E<e>_T<t>.sh`.

| Factor | 0 | 1 | 2 |
| --- | --- | --- | --- |
| G = Global: global prediction loss | MSE | WQE | — |
| E = Excess: raw-excess branch loss | MSE | WQE | — |
| T = Tail: additional Q95 extreme-hour loss | Off | MSE, weight 0.025 | WQE, weight 0.025 |

MSE means mean squared error; WQE means weighted quantile–expectile loss.
**G0/E0 enable MSE; only T0 means off.** G/E have two choices (0/1); T has
three (0/1/2). There is no G2 or E2.

G compares final predictions with targets over all target hours. E supervises
the raw excess **before gate multiplication**, across all horizons in GT Event
Windows. T adds loss on final predictions only at strict extreme hours
(`y > tau`). For example, G1_E0_T2 means global WQE + raw-excess MSE + Tail-WQE
at weight 0.025; body MSE and gate BCE keep their existing definitions.
Single has no E branch and uses G/T names only. See the
[shared naming guide](../README.md#g--e--t-和-0--1--2-的含义).

G maps to `LOSS_MODE_LIST`; E maps to `EXCESS_LOSS_MODE`.
T1/T2 use `EXCEEDANCE_LOSS_MODE="mse"` / `"wqe"`, both with
`EXCEEDANCE_LOSS_WEIGHT=0.025`. T0 uses weight 0 and inactive mode `mse`.
Existing T0/T1 cells preserve their meanings; T2 adds four cells per station.

Tail uses the strict `y > tau` mask at fixed TRAIN Q95, averaging over all
positions with masked zeros and dividing by fixed TRAIN `q_H`. Tail-WQE
replaces only the pointwise squared error with `wqe_penalty`. All three WQE
placements share detached TRAIN `y_std`, quantile tau 0.25, expectile tau 0.82,
and weights 1/6 and 5/6. Global, raw-excess, and Tail modes are independent.

Body/excess/gate weights remain 1/2/0.5. Body remains MSE; amplitude and shape
losses are disabled. All model and training hyperparameters are preserved:
GraphSAGE + Transformer, width 128, 24-hour history, batch size 256, four
accumulation steps, learning rate `5e-3`, 300 epochs, five warmup epochs,
cosine schedule, BF16 AMP, TF32, chronological 60/20/20 splits, and seed 42.
See the [manifest](manifest.csv) and [loss definitions](../../docs/LOSSES.md).

`multickpt` refers to four VAL-selected checkpoints from one training trajectory:
`overall`, `exceedance`, `aligned_peak`, and `bea` (Balanced Event-Aware).
BEA minimizes `0.50*AllRMSE + 0.25*ExceedanceRMSE + 0.25*GTAlignedPeakRMSE`
with fixed weights. Every role is reevaluated on VAL and TEST; `overall`
supplies the primary result. See [checkpoint selection](../../docs/CHECKPOINT_SELECTION.md).

The configured Python is `/home/exouser/.conda/envs/torchpyg-cu12x/bin/python`;
export `WQEF_PYTHON_BIN` to override it. Results retain the historical root
`/home/exouser/media/share/PACT/All_results_0922_wqe_factorial_multickpt`, using
`NCEP_<station>_G<g>_E<e>_T<t>__<timestamp>/` run directories.

From the repository root, regenerate and inspect commands without training:

```bash
python tools/generate_configs.py --family wqe_factorial_multickpt
DRY_RUN=1 USE_TMUX=0 bash train.sh configs/wqe_factorial_multickpt/train_config_NCEP_CBBT_G0_E0_T2.sh
DRY_RUN=1 bash configs/wqe_factorial_multickpt/launch_all.sh
```

[launch_all.sh](launch_all.sh) covers all 48 jobs; each of the 12
`launch_GgEeTt.sh` subsets covers four stations (for example,
[launch_G0E0T2.sh](launch_G0E0T2.sh)). Order is G, then E, then T, then CBBT,
Boston, Battery, Lewes. With `DRY_RUN=1`, launchers print resolved commands,
bypass CUDA preflight and the queue, and create no result directories.
Invoking a launcher without that flag submits its jobs through `qsub_local`
after [preflight.sh](preflight.sh) verifies imports and CUDA availability.
Generation alone never submits jobs.

Tests reproduce all configs and the manifest, parse dry-run arguments, compare
unchanged settings to existing Dual controls, check all four checkpoint roles,
and exercise launch order using a mock queue.
The [Single factorial](../s0_refresh/README.md) supplies matching G × T controls.
