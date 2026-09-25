# WQE/Tail factorial with multiple checkpoints

Status: not archived.

This family contains 32 direct Dual-head NCEP configurations: CBBT, Boston,
Battery, and Lewes × global MSE/WQE × raw-excess MSE/WQE × Tail-MSE off/on.
Files are named `train_config_NCEP_<station>_G<g>_E<e>_T<t>.sh`.

| Factor | 0 | 1 |
| --- | --- | --- |
| G: global prediction loss | MSE | WQE |
| E: raw-excess branch loss | MSE | WQE |
| T: additional Tail-MSE | Disabled | Weight 0.025 |

Body/excess/gate weights are 1/2/0.5. Body remains MSE; Tail remains MSE on
strict TRAIN Q95 extreme hours. Amplitude and shape losses are disabled.
See the [manifest](manifest.csv) and [loss definitions](../../docs/LOSSES.md).

`multickpt` refers to six VAL-selected checkpoints from one training trajectory:
overall, exceedance, aligned_peak, equal, peak_priority, and eventaware.
Every role is reevaluated on VAL and TEST; overall supplies the primary result.
See [Checkpoint selection](../../docs/CHECKPOINT_SELECTION.md).

The configured Python is `/home/exouser/.conda/envs/torchpyg-cu12x/bin/python`;
export `WQEF_PYTHON_BIN` to override it. Results retain the historical root
`/home/exouser/media/share/PACT/All_results_0922_wqe_factorial_multickpt`.
Generate this family with `python tools/generate_configs.py --family wqe_factorial_multickpt`.

From the repository root, inspect one config without starting training:

```bash
DRY_RUN=1 USE_TMUX=0 bash train.sh configs/wqe_factorial_multickpt/train_config_NCEP_CBBT_G0_E0_T0.sh
```

[launch_all.sh](launch_all.sh) submits all 32 jobs; each `launch_GgEeTt.sh`
submits the four stations for one cell. Launchers run [preflight.sh](preflight.sh)
to check imports and CUDA availability before submitting jobs through `qsub_local`.

The [Single refresh](../s0_refresh/README.md) supplies matching Single-head
global-loss/Tail controls. See the [configuration overview](../README.md).
