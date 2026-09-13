# P2 Peak-Aware Factorial validation — PASS

Validated UTC: 2026-09-13T05:37:44.099133+00:00  
Repository: `/media/volume/PACT-Data/StormSurge`  
Baseline commit: `3bfe08c0978a4b9278e5536debf1852e9fceb54f`  
Repository commit at validation: `3bfe08c0978a4b9278e5536debf1852e9fceb54f`  
Current `train.sh` SHA-256: `b096e7d03de2e8ed028f99c3c41e467f59dbf04089fb741472a898a2db0bd22d`

The validated source includes the authorized opt-in launcher enhancement.
The baseline commit identifies the unchanged production Python/model/loss code
and P0/P1 protocol; the launcher hash identifies the precise additional change.

## Experiment inventory and full factorial

| Station | Direct | Severity-shape | Total | Factorial coverage |
| --- | ---: | ---: | ---: | --- |
| CBBT | 8 | 16 | 24 | PASS |
| Lewes | 8 | 16 | 24 | PASS |
| Battery | 8 | 16 | 24 | PASS |
| Boston | 8 | 16 | 24 | PASS |
| Total | 32 | 64 | 96 | PASS |

All direct `(Tail, Amp, Peak)` and severity-shape `(Tail, Amp, Shape, Peak)`
combinations occur exactly once per station. Duplicate/missing/extra combinations:
**0/0/0**. Direct shape weights are always zero. All 96 configs are standalone,
contain singleton sweep arrays, pass `bash -n`, and have unique experiment IDs
1–96, semantic names, Python run tags, output directories and queue labels.

## Exact weights and active semantics

| Control | Direct OFF / ON | Severity-shape OFF / ON |
| --- | --- | --- |
| Active tail coefficient | 0 / 0.025 | 0 / 0.025 |
| Excess amplitude | 0 / 0.0007 | 0 / 0.0025 |
| Shape supervision | always 0 | 0 / 0.00003 |
| Final physical peak | 0 / 0.0035 | 0 / 0.0035 |

Every config declares `TAIL_LAMBDA_LIST=("0.025")`, preserving P0. Tail OFF
uses `mse`; no `--tail_lambda` is passed and the tail branch is inactive. The
inactive parser default is recorded but is not an experimental level. Tail ON
uses `mse_tail` and resolves the active coefficient to exactly 0.025. CPU-only
loss-forward checks certified that MSE never reads the tail threshold, even at
nonzero tail coefficients; no backward pass, optimizer or training loop ran.
No P2 mode enables slope supervision; P0's inactive slope controls are retained.

Frozen weights: body **1**, excess **2**, gate **0.5**; tail fraction **0.05**;
TRAIN exceedance percentile **95**. Amplitude and final-peak pools are hard
`max`; both inactive smoothmax betas are **20**; severity-shape epsilon **1e-6**.
Checkpoint selection is **overall**, primary minimum **VAL rmse_all**,
auxiliary checkpoints **0**, overall tolerance **0.01** for every config.

## Frozen architecture, data and training controls

- `perceiver3` / GraphSAGE / Transformer; history 24 h; hidden 128; graph layers 2;
  dropout/head dropout 0.05; node/time read heads 8; Transformer layers 2,
  FF multiplier 4.0, dropout 0; max time steps 32.
- Dual exceedance/window head; dual loss 1; ablation none; event definition unchanged.
- Chronological train/validation ratios 0.6/0.2; shuffle 0; future-only 0;
  inactive future threshold 2030; seed 42.
- Batch 256, accumulation 4; Adam, LR 0.005, weight decay 1e-5; 300 epochs;
  cosine; warmup 5, **start factor 0.1**; min LR 1e-6; max gradient norm 0;
  deterministic kernels 0. P0's inactive ROP/WMSE/slope settings are unchanged.
- One GPU/CUDA device 0; AMP bf16 and TF32 on; threads 1; workers, pin memory,
  persistent workers and prefetch all 0; multiprocessing fork.
- OOD disabled; zscore; clipping and augmentation/probability/scale/bias all 0.
- Site elevation and bathymetry off; existing six lat/lon-derived features
  validated for all four station JSON files; station metadata remains enabled.
- P0 data roots, station JSON path, verified Python interpreter and activation-off
  settings are preserved. No training data were loaded.

All configured fields and all resolved non-factor CLI fields were compared with
the corresponding P0 dual-MSE reference for **every config: 96 comparisons across
four stations**, with **zero unexpected differences**. Intentional changes are
the P2 factors, run identity and result root; amplitude/shape/peak/formulation and
checkpoint settings that were implicit P0 production defaults are now explicit.
The exact complete controls, including inactive values and production defaults
`device=auto` and `use_station_meta=1`, are in
[`validation/frozen_controls.json`](validation/frozen_controls.json).

## Dry runs, backward compatibility and command files

**96/96 current-repository P2 DRY_RUNs succeeded**, each resolving exactly one
command that passed the production argument parser. Semantic `T/A/S/P` names
appear in both the output directory and Python `run_tag`. The launcher preserves
timestamp naming and refuses to overwrite existing run directories.

All **32 P0 + 32 P1** full dry-run `CMD:` lines are **byte-for-byte unchanged**
against the baseline launcher when `PYTHON_RUN_TAG_BASE` is absent. Explicit
empty overrides were also checked. P2's nonempty override and resolved launcher
snapshot were verified before generation. Tail-launcher behavior was not changed.

| Command file | Commands |
| --- | ---: |
| commands_all.txt | 96 |
| commands_CBBT.txt | 24 |
| commands_Lewes.txt | 24 |
| commands_Battery.txt | 24 |
| commands_Boston.txt | 24 |

All command lines match the manifest and locally supported
`qsub_local train.sh <label> <config_path>` syntax, inspected from `.bashrc`.
The combined list uses CBBT, Lewes, Battery, Boston order. Command lists were
created as text and were never executed.

## Repository protection and evidence

Of **369 baseline tracked files**, only the authorized
`train.sh` enhancement differs from task start; all **368 others** retain identical SHA-256
hashes. Production Python/model/loss/training mathematics, P0 and P1 are unchanged.
All **643** existing result-tree entries retain their metadata;
no result/checkpoint was written and no P2 result directory was created.
**No training, qsub_local call, job submission or tmux execution occurred.**

- [`validation_report.json`](validation_report.json): all 96 resolved namespaces,
  CLI token lists, exact weights, config hashes and per-run P0 differences.
- [`validation/dry_runs.txt`](validation/dry_runs.txt): all 96 launcher outputs.
- [`validation/p0_reference_dry_runs.txt`](validation/p0_reference_dry_runs.txt):
  the four station references used for every non-factor comparison.
- [`validation/launcher_regression.json`](validation/launcher_regression.json):
  baseline/current P0/P1 command strings and semantic-tail/tag regression checks.
- [`validation/original_state.json`](validation/original_state.json): original
  tracked-file hashes and existing-results metadata for the integrity checks.

This certifies configuration and launcher behavior; training performance and
scientific outcomes have not been measured.
