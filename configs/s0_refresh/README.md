# S0 refresh

Status: not archived. See the [configuration overview](../README.md) for the
other experiment families and archive status.

Sixteen matched Single-head PACT runs: CBBT, Lewes, Battery, and Boston, each
with four prediction-loss/Tail conditions. All use `HEAD_TYPE=single` and
`DUAL_LOSS=0`.

| Condition | Global prediction loss | Tail-MSE weight | Config/run suffix |
| --- | --- | ---: | --- |
| S0 | MSE | 0 | `S0_Single` |
| S0_WQE | WQE | 0 | `S0_Single_WQE` |
| S0_Tail | MSE | 0.025 | `S0_Single_Tail` |
| S0_Tail_WQE | WQE | 0.025 | `S0_Single_Tail_WQE` |

Files are named `train_config_NCEP_<station>_<suffix>.sh` in this directory.
All results use `/home/exouser/media/share/PACT/0924_s0_refresh/`, with distinct
`NCEP_<station>_<suffix>__<timestamp>/` run directories.

The original MSE configs were based on `main` `49bb29f`; the WQE comparisons
were added from `34ff3f0`. The Tail variants were added against verified
`main` `9072c89`. Each station's corresponding `G0_E0_T0`, `G1_E0_T0`,
`G0_E0_T1`, and `G1_E0_T1` configs in `configs/wqe_factorial_multickpt`
supply the matched protocol and loss settings.

The common backbone and training settings match the latest factorial configs:
GraphSAGE + Transformer, width 128, 24-hour history, batch size 256 with four
accumulation steps, learning rate `5e-3`, 300 epochs, cosine schedule with five
warmup epochs, BF16 AMP, and TF32. Station selection, NCEP Grid4_New data,
chronological 60/20/20 splits, seed 42, normalization, station metadata settings,
and loader settings are preserved.

The interpreter remains explicitly pinned to
`/home/exouser/.conda/envs/torchpyg-cu12x/bin/python`, matching the latest factorial
configs. Export `S0_PYTHON_BIN` to deliberately override it.

## Loss definitions

`S0` uses global prediction MSE. `S0_WQE` uses the current TRAIN-scale-aware WQE
prediction objective across all target hours, with the published parameters
used by the latest experiments:

```bash
WQE_QUANTILE_TAU=0.25
WQE_EXPECTILE_TAU=0.82
WQE_QUANTILE_WEIGHT=0.16666666666666667
WQE_EXPECTILE_WEIGHT=0.83333333333333333
```

Both Tail variants add `EXCEEDANCE_LOSS_WEIGHT=0.025` times the existing strict
hourly exceedance MSE:

```text
Tail_MSE = mean(1[y > tau] * (prediction - y)^2) / q_H
S0_Tail loss = global_MSE + 0.025 * Tail_MSE
S0_Tail_WQE loss = global_WQE + 0.025 * Tail_MSE
```

`tau` is the fixed Q95 of unique TRAIN target hours. `q_H` is the observed strict
TRAIN extreme-hour fraction. The mean includes all target positions, with zero
contribution outside `y > tau`; normalization uses the fixed TRAIN fraction.
The Tail term remains ordinary physical squared error when global prediction
loss is WQE. These settings match each station's `G0_E0_T1` and `G1_E0_T1`
configs.

Amplitude and shape weights remain zero, the experimental head selector remains
empty, and no dual branch is constructed. `EXCESS_LOSS_MODE="mse"` is inactive
for every Single-head run; it does not add a branch objective. Tail is controlled
independently by `EXCEEDANCE_LOSS_WEIGHT`.

## Shared evaluation and logging

Metrics use the current shared implementation and the same fixed TRAIN Q95
threshold with strict `y > tau`. All six VAL-selected roles are retained:
overall, exceedance, aligned_peak, equal, peak_priority, and eventaware.
Each is reevaluated on VAL and TEST; `overall` is the primary alias. Legacy
checkpoint weights remain `0.65/0.20/0.15`.

The current launcher/trainer provide the same config snapshots, launcher and
training logs, epoch JSONL, summaries, checkpoint comparisons, and per-role
prediction exports. Single has no dual-branch diagnostics.

## Launch commands

Run the desired commands from a shell where `qsub_local` is available.

```bash
cd /home/exouser/StormSurge
```

S0: global MSE.

```bash
qsub_local train.sh S0_CBBT_0924 configs/s0_refresh/train_config_NCEP_CBBT_S0_Single.sh
qsub_local train.sh S0_Lewes_0924 configs/s0_refresh/train_config_NCEP_Lewes_S0_Single.sh
qsub_local train.sh S0_Battery_0924 configs/s0_refresh/train_config_NCEP_Battery_S0_Single.sh
qsub_local train.sh S0_Boston_0924 configs/s0_refresh/train_config_NCEP_Boston_S0_Single.sh
```

S0_WQE: global WQE.

```bash
qsub_local train.sh S0WQE_CBBT_0924 configs/s0_refresh/train_config_NCEP_CBBT_S0_Single_WQE.sh
qsub_local train.sh S0WQE_Lewes_0924 configs/s0_refresh/train_config_NCEP_Lewes_S0_Single_WQE.sh
qsub_local train.sh S0WQE_Battery_0924 configs/s0_refresh/train_config_NCEP_Battery_S0_Single_WQE.sh
qsub_local train.sh S0WQE_Boston_0924 configs/s0_refresh/train_config_NCEP_Boston_S0_Single_WQE.sh
```

S0_Tail: global MSE plus Tail-MSE.

```bash
qsub_local train.sh S0Tail_CBBT_0924 configs/s0_refresh/train_config_NCEP_CBBT_S0_Single_Tail.sh
qsub_local train.sh S0Tail_Lewes_0924 configs/s0_refresh/train_config_NCEP_Lewes_S0_Single_Tail.sh
qsub_local train.sh S0Tail_Battery_0924 configs/s0_refresh/train_config_NCEP_Battery_S0_Single_Tail.sh
qsub_local train.sh S0Tail_Boston_0924 configs/s0_refresh/train_config_NCEP_Boston_S0_Single_Tail.sh
```

S0_Tail_WQE: global WQE plus Tail-MSE.

```bash
qsub_local train.sh S0TailWQE_CBBT_0924 configs/s0_refresh/train_config_NCEP_CBBT_S0_Single_Tail_WQE.sh
qsub_local train.sh S0TailWQE_Lewes_0924 configs/s0_refresh/train_config_NCEP_Lewes_S0_Single_Tail_WQE.sh
qsub_local train.sh S0TailWQE_Battery_0924 configs/s0_refresh/train_config_NCEP_Battery_S0_Single_Tail_WQE.sh
qsub_local train.sh S0TailWQE_Boston_0924 configs/s0_refresh/train_config_NCEP_Boston_S0_Single_Tail_WQE.sh
```

## Configuration comparison

Relative to their paired runs without Tail, the new Tail configs change only
`exceedance_loss_weight` (`0` to `0.025`) and run identity (`run_tag` and the
run-specific `output_dir`) in the resolved training arguments. WQE versus MSE
changes only `loss_mode` and run identity. WQE parameters are explicit in WQE
configs and equal the inactive defaults resolved by the MSE configs.

Compared with the corresponding latest factorial cells, differences are the
Single head, disabled dual supervision, inactive S0 branch defaults (excess/gate
weights `1/1` instead of `2/0.5`), and output/run identity. The runtime is the same,
with the S0-specific override variable. The existing eight configs are preserved.

## Validation

All 16 configs passed Bash syntax checks, launcher dry runs, argument parsing,
and comparisons with their corresponding latest factorial cells. The eight Tail
additions were also compared against their paired configs without Tail; only
Tail weight and run identity differ. All 16 documented launch commands resolve
to distinct result directories under the shared root.

All 27 existing tests in `test_exceedance_loss`, `test_wqe_loss`, and
`test_wqe_configs` passed. The original eight configs remain byte-for-byte
unchanged. No station jobs were submitted.
