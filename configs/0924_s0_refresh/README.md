# 0924 S0 refresh

Eight Single-head PACT runs for CBBT, Lewes, Battery, and Boston: four MSE
baselines and four matched global-WQE comparisons. The MSE configs are based on
`main` commit `49bb29ff783a07a1761a7533a60147898a0f9132` (verified against remote
`main`). References are each station's `configs/current/*_S0_Single.sh` and
`configs/0922_wqe_factorial_multickpt/*_G0_E0_T0.sh`. The WQE variants were added
from `main` `34ff3f0` using each station's `*_G1_E0_T0.sh` WQE parameters.

The shared backbone and training settings match all 32 latest factorial configs:
GraphSAGE + Transformer, width 128, 24-hour history, batch size 256 with four
accumulation steps, learning rate `5e-3`, 300 epochs, cosine schedule with five
warmup epochs, BF16 AMP, and TF32. Station selection, NCEP Grid4_New data,
chronological 60/20/20 splits, seed 42, normalization, station metadata settings,
and loader settings are preserved.

`HEAD_TYPE=single` and `DUAL_LOSS=0` apply to all eight runs. The original S0
configs use plain prediction MSE; the `_WQE` variants use global prediction WQE.
Tail, amplitude, and shape weights are zero; the experimental head selector is
empty. The retained dual settings are inactive for Single.

Metrics use the current shared implementation and a fixed Q95 threshold fitted
on unique TRAIN target hours, with strict `y > tau`. All six VAL-selected roles
are retained: overall, exceedance, aligned_peak, equal, peak_priority, and
eventaware. Each is reevaluated on VAL and TEST; `overall` is the primary alias.
The current launcher/trainer provide the same config snapshots, launcher and
training logs, epoch JSONL, summaries, checkpoint comparisons, and per-role
prediction exports. Single has no dual-branch diagnostics.

Results go to
`/home/exouser/media/share/PACT/0924_s0_refresh/`, with separate run directories
`NCEP_<station>_S0_Single__<timestamp>/` for MSE and
`NCEP_<station>_S0_Single_WQE__<timestamp>/` for WQE.
The runtime is explicitly pinned to
`/home/exouser/.conda/envs/torchpyg-cu12x/bin/python`, matching the latest factorial
configs. Export `S0_PYTHON_BIN` to deliberately override it.

Launch the four MSE baselines from a shell where `qsub_local` is available:

```bash
cd /home/exouser/StormSurge
qsub_local train.sh S0_CBBT_0924 configs/0924_s0_refresh/train_config_NCEP_CBBT_S0_Single.sh
qsub_local train.sh S0_Lewes_0924 configs/0924_s0_refresh/train_config_NCEP_Lewes_S0_Single.sh
qsub_local train.sh S0_Battery_0924 configs/0924_s0_refresh/train_config_NCEP_Battery_S0_Single.sh
qsub_local train.sh S0_Boston_0924 configs/0924_s0_refresh/train_config_NCEP_Boston_S0_Single.sh
```

Compared with each latest `G0_E0_T0` control, the MSE variants differ in the
head (`dual` to `single`), dual supervision (`1` to `0`), inactive excess
and gate weights (`2`/`0.5` to the canonical S0 defaults `1`/`1`), and output/run
identity. Inactive WQE parameter assignments are omitted; both loss-mode
selectors remain `mse`. The interpreter is unchanged, with an S0-specific
override variable. All other resolved arguments match.

The MSE variants retain the current S0 training/evaluation arguments. This
refresh adds the explicit runtime, explicit inactive excess-MSE selector, and
requested result root. The six-role protocol comes from current `train.py`.

Original MSE validation: all four configs passed Bash syntax checks, launcher
dry runs, argument parsing, and comparison with station-matched S0 and factorial
controls.
Shared resolved settings were also checked against all 32 factorial variants.
Training imports and CUDA availability passed using the pinned interpreter.
All 33 existing tests in `test_hourly_threshold`, `test_hourly_metrics`,
`test_eventaware_checkpoints`, and `test_checkpoint_pipeline` passed.
No station jobs were submitted.

## Matched WQE runs

The four `train_config_NCEP_<station>_S0_Single_WQE.sh` configs retain the paired
S0 backbone, training settings, datasets, metric definitions, all six checkpoint
roles, evaluation, logging, runtime, and result root. Their only resolved
argument changes from the paired MSE config are `loss_mode=mse` to `wqe`,
`run_tag`, and the run-specific `output_dir`.

WQE parameters are explicitly pinned to the latest `G1_E0_T0` settings:

```bash
LOSS_MODE_LIST=("wqe")
WQE_QUANTILE_TAU=0.25
WQE_EXPECTILE_TAU=0.82
WQE_QUANTILE_WEIGHT=0.16666666666666667
WQE_EXPECTILE_WEIGHT=0.83333333333333333
```

The global WQE prediction loss uses the current TRAIN-scale-aware implementation
across all target hours. `EXCESS_LOSS_MODE="mse"` remains an inactive Single-head
setting; it does not add an MSE term or a dual branch. The WQE parameter values
also equal the inactive defaults resolved by the paired MSE configurations.

```bash
cd /home/exouser/StormSurge
qsub_local train.sh S0WQE_CBBT_0924 configs/0924_s0_refresh/train_config_NCEP_CBBT_S0_Single_WQE.sh
qsub_local train.sh S0WQE_Lewes_0924 configs/0924_s0_refresh/train_config_NCEP_Lewes_S0_Single_WQE.sh
qsub_local train.sh S0WQE_Battery_0924 configs/0924_s0_refresh/train_config_NCEP_Battery_S0_Single_WQE.sh
qsub_local train.sh S0WQE_Boston_0924 configs/0924_s0_refresh/train_config_NCEP_Boston_S0_Single_WQE.sh
```

WQE validation: all four configs passed Bash syntax checks, launcher dry runs,
argument parsing, and comparisons with their paired S0 MSE and latest
`G1_E0_T0` configs. All 22 existing tests in `test_wqe_loss` and
`test_wqe_configs` passed. The four MSE configs remain byte-for-byte
unchanged. No station jobs were submitted.
