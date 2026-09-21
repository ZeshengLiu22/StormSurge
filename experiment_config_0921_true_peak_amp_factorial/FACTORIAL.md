# Final matched Tail × true-peak Amp factorial

Production `main` / local `origin/main` / HEAD:
`af53de756860765a9809b0c5c44bc828c2a08359`.
The starting tree had only the existing untracked diagnosis directory.
[initial_audit.json](initial_audit.json) records the initial status and hashes.
Production model, training, loss, metric, and launcher source files are unchanged.

The formal experiment is **20 runs: five each for CBBT, Lewes, Battery, Boston**.
Final standalone configs are in [configs/](configs/); every sweep axis has one
entry. [manifest.csv](manifest.csv) records each config, run name and command.
Results will be written outside Git to:

`/media/share/PACT/Results/All_results_0921_true_peak_amp_factorial`

Run directories use `<run_name>__<timestamp>`. The launcher sends checkpoints,
metrics and logs to that directory. Config generation and validation do not
create the results root or submit jobs.

| Condition | Head | Base/loss mode | Tail | True-peak Amp |
|---|---|---|---:|---:|
| S0 — Single Base | single | mse | OFF | OFF |
| D0 — Dual Base | production direct dual | mse | OFF | OFF |
| D1 — Dual + Tail | production direct dual | mse_tail | 0.025 | OFF |
| D2 — Dual + Amp | production direct dual | mse | OFF | 0.003 |
| D3 — Dual + Tail + Amp | production direct dual | mse_tail | 0.025 | 0.003 |

All Dual runs use Q95, body/excess/gate weights **1 / 2 / 0.5**, full branch
supervision, no ablation, a window gate with mean pooling, and the production
soft body cap. Amp is the unchanged physical ungated excess loss at
`argmax_h y[h]`, normalized by the empirical strict TRAIN event prior. There
are no capacity variants or pooling controls for the Amp objective.

Tail uses `TAIL_FRAC=0.05`. Tail-enabled configs set
`TAIL_LAMBDA_LIST=("0.025")`. Plain MSE disables Tail; those configs retain the
unused production Tail default `0.10`, which the launcher does not pass as an
argument. Manifest `tail_lambda=0` means an inactive term;
`tail_lambda_config` separately records the unused shell default. Inactive
slope/WMSE fields retain defaults, but those objectives are never selected.
Single has `DUAL_LOSS=0` and Amp weight zero; its dual-only manifest fields
are `N/A`. No additional coefficient, seed, LR, or checkpoint-selection study
is part of this factorial.

The scientific contrasts are S0→D0 for the supervised direct-dual formulation,
D0→D1 for Tail, D0→D2 for Amp, D1→D3 for Amp with Tail, and D2→D3 for Tail with
Amp. D0/D1/D2/D3 provide the Tail × Amp interaction.

## Frozen protocol

- NCEP at `./Data/Grid4_New/NCEP/graphs`; chronological year groups;
  train/val ratios 0.6/0.2; `SHUFFLE_YEARS=0`, `FUTURE_ONLY=0`, seed 42.
- `perceiver3`, GraphSAGE, Transformer, history 24 h, hidden 128, spatial
  layers 2, dropout/head dropout 0.05, node/time read heads 8/8, Transformer
  layers 2, FF multiplier 4.0, Transformer dropout 0.0, max time steps 32.
- `USE_SITE_ELEVATION=0`, `USE_BATHYMETRY=0`. Production coordinate metadata
  remains enabled, giving six station features.
- Z-score; OOD disabled; clipping, feature augmentation, augmentation
  probability, scale and bias all zero. Unused percentile fields remain explicit.
- Batch 256 × accumulation 4 × one GPU = effective batch 1024. LR 5e-3,
  300 epochs, production Adam weight decay 1e-5, cosine schedule, warmup
  5 epochs from factor 0.1, minimum LR 1e-6. No gradient clipping;
  `DETERMINISTIC=0`.
- BF16 AMP and TF32 enabled; workers 0, Torch threads 1, pin/persistent
  workers disabled, prefetch factor 0, multiprocessing context `fork`.
  Explicit existing local Python:
  `/media/volume/PACT-Data/conda_envs/torchpyg-cu124/bin/python`.
- All runs use `CHECKPOINT_SELECTION="overall"`, tolerance 0.01 and
  `SAVE_AUX_CHECKPOINTS=0`: selection uses VAL overall RMSE. TEST is used
  only by the existing final reporting path after selection.
- Reporting stays canonical: AllRMSE, AllMAE, Top5RMSE, Top5MAE,
  TruePeakRMSE, TruePeakMAE, TruePeakBias, TruePeakUnder%, TimingSteps.
  The last five use the production `_top5` true-peak-time definitions.

## Validation and initialization

[validation/validation_report.md](validation/validation_report.md) and
[validation/validation_report.json](validation/validation_report.json) contain
the completed checks. All 20 configs pass `bash -n`, the actual `train.sh`
dry-run branch, resolved shell snapshots, and production argparse; each emits
one training command. Config text, parsed arguments and manifest fields are
checked against the prescribed values. After run identity is excluded,
D0–D3 differ only by Tail/Amp factors. The no-Tail modes retain their normal
inactive defaults. No obsolete controls appear in the final configs.

Actual production model construction gives **618,885 Single parameters** and
**685,447 Dual parameters** at every station (all trainable). The common
backbone has **585,604** parameters; Single/Dual heads have **33,281 / 99,843**.
D0–D3 have identical architecture, parameter counts and every initialized
state tensor within each station, using one call to the production seed
routine per independent construction. S0 also has the identical initialized
backbone. No RNG behavior is changed in production.

The CPU initialization check uses the preserved TRAIN-fitted normalization
and thresholds, after checking the current filename split and every TRAIN
source file's size and modification time. It opens one TRAIN file per station
to verify five input features, six output horizons and sufficient history.
It does not refit statistics, load VAL/TEST graph files, run a forward/backward
pass, or invoke an optimizer. Hashes of initialized model state are in the
JSON report; no model checkpoint is written.

The existing diagnosis used `USE_SITE_ELEVATION=1` (seven station features).
The formal configs follow the requested `USE_SITE_ELEVATION=0`; this accounts
for 128 fewer parameters than the diagnostic Dual model. The coefficients
remain fixed at 0.025 / 0.003. The diagnosis, its scripts, original README,
data and artifacts are preserved byte-for-byte; all 43 initial files pass
SHA-256 verification. [DIAGNOSIS.md](DIAGNOSIS.md) and [README.md](README.md)
describe the earlier diagnostic stage; this file describes the final factorial.

To repeat validation from the repository root without training:

```bash
/media/volume/PACT-Data/conda_envs/torchpyg-cu124/bin/python -B \
  experiment_config_0921_true_peak_amp_factorial/validate_configs.py
```

## Launch files — prepared, not executed

Run in the repository's interactive shell, where the existing `qsub_local`
function is available:

```bash
cd /home/exouser/media/volume/PACT-Data/StormSurge
source experiment_config_0921_true_peak_amp_factorial/commands_all.txt
```

Alternatively, submit one station by sourcing its corresponding command file:

- [commands_CBBT.txt](commands_CBBT.txt)
- [commands_Lewes.txt](commands_Lewes.txt)
- [commands_Battery.txt](commands_Battery.txt)
- [commands_Boston.txt](commands_Boston.txt)

Choose the complete list or the station lists once; sourcing both would submit
duplicates. [commands_all.txt](commands_all.txt) contains the exact 20 commands,
each in this established form:

```bash
qsub_local train.sh 0921_CBBT_S0_Single experiment_config_0921_true_peak_amp_factorial/configs/train_config_0921_CBBT_S0_Single.sh
```

No jobs were launched during preparation or validation.
