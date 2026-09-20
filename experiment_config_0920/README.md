# 0920 direct-dual formulation study

**4 stations × 4 formulations = 16 configs: 12 training + 4 post-processing.**
Stations are CBBT, Lewes, Battery and Boston. Every station has exactly these entries:

| Formulation | Mode | Model reconstruction | Excess supervision | Checkpoint |
| --- | --- | --- | --- | --- |
| F0_Soft | train | `b + q e` | event | Selected by overall VAL RMSE |
| F1_Hard | postprocess | `b_phys + 1[q >= 0.5] e_phys` | Reuses F0 | The corresponding already-selected F0 checkpoint |
| F2_AdditiveEvent | train | `b + e` | event | Selected by overall VAL RMSE |
| F3_AdditiveAll | train | `b + e` | all | Selected by overall VAL RMSE |

Config folder:
`/home/exouser/media/volume/PACT-Data/StormSurge/experiment_config_0920/configs/`

Result root:
`/home/exouser/media/volume/PACT-Data/StormSurge/All_results_0920/`

`/home/exouser/media` is the existing symlink to `/media`. The result root is ignored
by the repository rule `/All_results_0920/`. Training output folders are
`0920_<Station>_<Formulation>__<TIMESTAMP>/`; F1 has its own similarly named folder
under the same result root. No 0919 configs or outputs are modified.

The [manifest](manifest.csv) contains exactly 16 rows. F1 has `mode=postprocess`,
`source_checkpoint=0920_<Station>_F0_Soft`, a source config and a checkpoint path
template. Its model/loss fields describe the source F0; blank epochs/accumulation
fields mean that F1 performs no training. The template is documentation and a
dry-run placeholder, not a glob that selects a checkpoint at execution time.

F0/F2/F3 copy each station's frozen 0919 `Legacy_Mean` settings into standalone
configs. They never source the historical config at runtime. The capacity-study
`EXCEEDANCE_HEAD_EXPERIMENT` setting is removed completely, so F0 constructs the
production `ExceedanceHead` and F2/F3 the matching additive head. Gate pooling is
explicitly `mean`. No learned pooling, C1/C2/C2R/C3 or severity-shape is selected.
Apart from reconstruction, supervision scope and run identity, their resolved
training arguments are identical within each station.

The inherited protocol is seed 42, deterministic kernels off, LR 5e-3, 300 epochs,
GraphSAGE + Transformer, history 24 h, hidden width 128, graph/Transformer depth 2,
read heads 8/8, FF multiplier 4, graph/head dropout 0.05 and Transformer dropout 0.
Batch size is 256 with accumulation 4, Adam weight decay remains 1e-5, cosine
scheduling has five warmup epochs from 0.1 and minimum LR 1e-6. TRAIN z-score
statistics, chronological 0.6/0.2/remainder year splits, TRAIN 95th-percentile event
threshold, six station coordinate features, no elevation/bathymetry, no input
augmentation/clipping, and the private DataLoader RNG stream are preserved.

All three trained formulations use MSE plus body/excess/gate weights **1/2/0.5**,
`DUAL_LOSS=1`, `DUAL_ABLATION=none`, overall checkpoint selection and no auxiliary
checkpoints. Tail/slope, excess-amplitude, shape and peak loss coefficients are
explicitly zero. The existing shell interface uses `LOSS_MODE_LIST`, `LR_LIST`,
`HISTORY_HOURS_LIST`, `TAIL_LAMBDA_LIST` and `SLOPE_LAMBDA_LIST`, each with one value.
Under MSE the unchanged launcher omits inactive tail/slope flags, leaving their
unused Python defaults at 0.1/0.01; neither term enters the objective.

Execution settings match 0919: one GPU (`CUDA_VISIBLE_DEVICES=0`), BF16 AMP, TF32,
one Torch thread, no loader workers/pinning/persistent workers/prefetching, and
`fork`. The interpreter is
`/media/volume/PACT-Data/conda_envs/torchpyg-cu124/bin/python`, without conda activation.
Input graphs and station metadata retain the 0919 paths. No model/training code
is changed by this suite.

Validate and dry-run without launching jobs:

```bash
cd /home/exouser/media/volume/PACT-Data/StormSurge
/media/volume/PACT-Data/conda_envs/torchpyg-cu124/bin/python -B experiment_config_0920/generate_configs.py --check
/media/volume/PACT-Data/conda_envs/torchpyg-cu124/bin/python -B experiment_config_0920/validate_configs.py --check-only
bash experiment_config_0920/run_all.sh --dry-run
# One station: three training dry-runs plus one F1 dry-run
bash experiment_config_0920/run_CBBT.sh --dry-run
```

Validation checks the exact matrix and filenames, every inherited 0919 setting,
all 16 commands through the production argument parsers, the result ignore rule,
source/reference hashes, and F1 rejection of incompatible checkpoints. It creates
no result directories and launches no queue, training or inference jobs. Recorded
commands are in [dry_run_commands.txt](dry_run_commands.txt); resolved settings and
results are in [validation_report.json](validation_report.json).

When training is separately authorized, `bash experiment_config_0920/run_all.sh`
queues only the **12 training runs**, using the existing one-slot qsub_local/tmux
convention. The four `run_<Station>.sh` wrappers queue three each. F1 is deliberately
not queued before F0 is available. GPU 0 and BF16 are required; the launcher has
no CPU/precision fallback. [commands_all.txt](commands_all.txt) lists the 16 entries
with their modes and is a command reference, not a batch submission script.

After F0 finishes, use its exact overall-selected `best_*.pth` for F1, for example:

```bash
cd /home/exouser/media/volume/PACT-Data/StormSurge
F0_CHECKPOINT_CBBT='/absolute/path/to/0920_CBBT_F0_Soft__TIMESTAMP/best_perceiver3_....pth'
/media/volume/PACT-Data/conda_envs/torchpyg-cu124/bin/python -B experiment_config_0920/run_f1.py \
  experiment_config_0920/configs/postprocess_config_0920_CBBT_F1_Hard.sh \
  --checkpoint "$F0_CHECKPOINT_CBBT"
```

`F0_CHECKPOINT` in the environment is an alternative to `--checkpoint`. Add
`--dry-run` to validate an existing checkpoint and print its inference command.
The runner verifies the source station, 0920 F0 run tag, production architecture,
event supervision and canonical training/selection settings. It never finds a
"latest" file, optimizes a threshold or selects a second checkpoint. The threshold
is fixed at **0.5** with an inclusive comparison.

F1 uses the saved held-out TEST tags and `infer.py --dual_diagnostics`. Its
`f1_metrics.json` copies the existing `hard_gate_0p5` metrics, with the source
checkpoint path and SHA-256 for traceability. All ten standard trajectory/peak
metrics are included. Normal `y_pred` and normal `metrics.json` still describe F0's
soft-gated output; F1 metrics also remain available in `dual_diagnostics.json`.
No metric formulas, predictions or checkpoints are replaced.

Suite generation and validation did not launch training or evaluation jobs.
