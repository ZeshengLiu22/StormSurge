# A0 HeadCapacity — NCEP 24 h

Compare prediction-head parameterization and excess-branch capacity before adding
peak-aware auxiliary losses. Exactly **12 runs**: CBBT, Lewes, Battery, and Boston,
each with these three formulations:

| Tag | Head | Excess formulation | Dual loss | BODY | Trajectory EXCESS | GATE |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| `single_base` | single | direct (unused) | 0 | 0 | 0 | 0 |
| `dual_direct_base` | dual | direct | 1 | 1 | 2 | 0.5 |
| `dual_severity_base` | dual | severity_shape | 1 | 1 | 2 | 0.5 |

Every config is self-contained with explicit assignments and no config inheritance.
All use `LOSS_MODE_LIST=("mse")`; tail, excess-amplitude, shape, final-peak, and
slope loss coefficients are explicitly zero. The severity-shape formulation keeps
the existing trajectory-wise excess supervision at weight 2; its shape branch
learns through ordinary prediction and dual supervision, with no explicit shape loss.
The single-head objective contains prediction MSE only.

Both duals use `DUAL_MODE="exceedance"`, `DUAL_ABLATION="none"`, window gating,
and the TRAIN 95th-percentile event threshold. All runs select the minimum overall
validation RMSE checkpoint (`CHECKPOINT_SELECTION="overall"`, `SAVE_AUX_CHECKPOINTS=0`).

The common protocol preserves the validated production settings:

- NCEP / 24 h / perceiver3 / GraphSAGE / Transformer; hidden 128, graph layers 2,
  node/time read heads 8, Transformer layers 2, FF multiplier 4.0, max time steps 32.
  Graph/head dropout 0.05; Transformer dropout 0.0. Existing branch MLP width is 256.
- Batch 256, accumulation 4, 300 epochs, Adam LR 5e-3 and weight decay 1e-5;
  cosine scheduler, warmup 5 epochs from factor 0.1, min LR 1e-6; seed 42.
- Chronological year-group split 0.6/0.2/remainder, shuffle/future-only disabled.
  TRAIN-fitted z-score normalization; OOD, clipping, and feature augmentation disabled.
- Existing six coordinate-derived station features retained; elevation/bathymetry off.
- One GPU (`CUDA_VISIBLE_DEVICES=0`), BF16 AMP and TF32; one Torch thread, zero
  loader workers/prefetch, pinning/persistent workers off, multiprocessing `fork`.
- Existing NCEP graph and station JSON paths, direct production Python interpreter,
  no conda activation, and existing `qsub_local`/tmux session and exit-status behavior.

## Files and commands

[manifest.csv](manifest.csv) lists all 12 configs, identities, loss settings,
result patterns, and usable `qsub_local` commands. [commands_all.txt](commands_all.txt)
contains those 12 commands in station/formulation order. They have **not been submitted**.
Use them from the current repository in an interactive shell where `qsub_local` is available:

```bash
cd /media/volume/PACT-Data/StormSurge
qsub_local train.sh A0_CBBT_single_base experiment_config_0916/A0_HeadCapacity/train_config_NCEP_CBBT_24h_single_base.sh
```

The existing local queue uses one slot by default; its worker waits for the tmux
session and propagates its exit status. Each config honors the queue's `SESSION_NAME`.
Semantic run names and results follow:

```text
All_results_0916/A0_HeadCapacity/<RUN_NAME>__<TIMESTAMP>/
All_results_0916/A0_HeadCapacity/NCEP_CBBT_24h_dual_severity_base__20260916_123456/
```

The second path is an example, not an executed run. The result root is prepared
with a local `.gitignore`; no training artifacts are created during validation.

## Validation without training

```bash
cd /media/volume/PACT-Data/StormSurge
/media/volume/PACT-Data/conda_envs/torchpyg-cu124/bin/python -B experiment_config_0916/A0_HeadCapacity/validate_configs.py
```

The validator checks the complete 4×3 inventory, explicit settings, production
protocol, manifest commands, `bash -n`, all 12 `DRY_RUN=1` commands, resolved Python
arguments, output naming, and identical direct/severity supervision. It also uses
`emulator.models.count_model_parameters` on CPU models and verifies unchanged
model/loss/launcher files and result contents. It never invokes `train.py`, executes
a model forward pass, creates an optimizer, or submits a job.

The unchanged launcher omits tail/slope CLI flags under `mse`, leaving inactive
Python defaults of 0.1/0.01; both objectives remain OFF and both config coefficients
are explicitly zero. Single-head branch flags are also omitted; the parser sets
`dual_loss=0` and the model has only a regression head.

[validation_report.md](validation_report.md) gives the findings and parameter comparison;
[validation_report.json](validation_report.json) records every resolved config and command.
[validation/parameter_counts.json](validation/parameter_counts.json) includes per-branch
counts, construction details, actual data dimensions, and split year groups.
The same directory contains a dry-run transcript for each of the 12 configs.
[provenance.json](provenance.json) identifies the production references used for verification;
the configs never source these references at runtime.

The capacity comparison uses identical backbones and existing branch widths.
Severity-shape adds one branch MLP (33,281 parameters) over direct dual, doubling
excess-branch capacity from 33,281 to 66,562. No fourth model or width sweep is included.
