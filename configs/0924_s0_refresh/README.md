# 0924 S0 refresh

Four Single-head PACT baselines for CBBT, Lewes, Battery, and Boston, based on
`main` commit `49bb29ff783a07a1761a7533a60147898a0f9132` (verified against remote
`main`). References are each station's `configs/current/*_S0_Single.sh` and
`configs/0922_wqe_factorial_multickpt/*_G0_E0_T0.sh`.

The shared backbone and training settings match all 32 latest factorial configs:
GraphSAGE + Transformer, width 128, 24-hour history, batch size 256 with four
accumulation steps, learning rate `5e-3`, 300 epochs, cosine schedule with five
warmup epochs, BF16 AMP, and TF32. Station selection, NCEP Grid4_New data,
chronological 60/20/20 splits, seed 42, normalization, station metadata settings,
and loader settings are preserved.

`HEAD_TYPE=single`, `DUAL_LOSS=0`, and plain prediction MSE define S0. Tail,
amplitude, and shape weights are zero; the experimental head selector is empty.
The retained dual settings are inactive for Single.

Metrics use the current shared implementation and a fixed Q95 threshold fitted
on unique TRAIN target hours, with strict `y > tau`. All six VAL-selected roles
are retained: overall, exceedance, aligned_peak, equal, peak_priority, and
eventaware. Each is reevaluated on VAL and TEST; `overall` is the primary alias.
The current launcher/trainer provide the same config snapshots, launcher and
training logs, epoch JSONL, summaries, checkpoint comparisons, and per-role
prediction exports. Single has no dual-branch diagnostics.

Results go to
`/home/exouser/media/share/PACT/0924_s0_refresh/NCEP_<station>_S0_Single__<timestamp>/`.
The runtime is explicitly pinned to
`/home/exouser/.conda/envs/torchpyg-cu12x/bin/python`, matching the latest factorial
configs. Export `S0_PYTHON_BIN` to deliberately override it.

From a shell where `qsub_local` is available:

```bash
cd /home/exouser/StormSurge
qsub_local train.sh S0_CBBT_0924 configs/0924_s0_refresh/train_config_NCEP_CBBT_S0_Single.sh
qsub_local train.sh S0_Lewes_0924 configs/0924_s0_refresh/train_config_NCEP_Lewes_S0_Single.sh
qsub_local train.sh S0_Battery_0924 configs/0924_s0_refresh/train_config_NCEP_Battery_S0_Single.sh
qsub_local train.sh S0_Boston_0924 configs/0924_s0_refresh/train_config_NCEP_Boston_S0_Single.sh
```

Compared with each latest `G0_E0_T0` control, the resolved argument differences
are the head (`dual` to `single`), dual supervision (`1` to `0`), inactive excess
and gate weights (`2`/`0.5` to the canonical S0 defaults `1`/`1`), and output/run
identity. Inactive WQE parameter assignments are omitted; both loss-mode
selectors remain `mse`. The interpreter is unchanged, with an S0-specific
override variable. All other resolved arguments match.

Compared with current S0, the training/evaluation arguments are unchanged; this
refresh adds the explicit runtime, explicit inactive excess-MSE selector, and
requested result root. The six-role protocol comes from current `train.py`.

Validation: all four configs passed Bash syntax checks, launcher dry runs,
argument parsing, and comparison with station-matched S0 and factorial controls.
Shared resolved settings were also checked against all 32 factorial variants.
Training imports and CUDA availability passed using the pinned interpreter.
All 33 existing tests in `test_hourly_threshold`, `test_hourly_metrics`,
`test_eventaware_checkpoints`, and `test_checkpoint_pipeline` passed.
No station jobs were submitted.
