# P3 — Learning-rate stability

This controlled experiment asks whether the P2 severity-shape degradation
persists when the initial LR falls from 0.005 to 0.002 or 0.001. It contains
**31 independent, standalone treatment configs** copied from 12 exact P2
semantic sources. Generation and validation do not start training.

## Matrix

Counts below are audited from [manifest.csv](manifest.csv). Every treatment,
its reason and its exact P2 source path/hash are in the manifest. The complete
31-row audit is printed in [matrix.txt](matrix.txt) and the
[validation report](validation_report.md).

| Station | LR5e-3 | LR2e-3 | LR1e-3 | Total |
| --- | ---: | ---: | ---: | ---: |
| CBBT | 0 | 0 | 0 | 0 |
| Lewes | 1 | 1 | 1 | 3 |
| Battery | 3 | 6 | 6 | 15 |
| Boston | 3 | 5 | 5 | 13 |
| Total | 7 | 12 | 12 | **31** |

Part A: four configs × Battery/Boston × two low LRs = **16**.
Part B: seven alert cells × three LRs = **21** nominal treatments.
Six treatments belong to both parts; they are present once and use
`reason=core_probe+p2_degradation_alert`. The union is **31**.

| Core config | Scientific role |
| --- | --- |
| T1A1S0P1 | Strongest current cross-station severity-shape candidate |
| T1A1S0P0 | Tests whether Peak was rescuing Tail+Amp optimization failures |
| T0A1S0P1 | Tests the strong station sensitivity of Amp+Peak |
| T0A0S0P0 | Severity baseline to separate formulation instability from auxiliary-loss interaction |

Every LR5e-3 treatment is a **new P3 rerun**, including a fresh timestamped
output directory. No P2 result directory is used as a P3 run.

## Exact P2 protocol

**Interface correction:** the existing P2 implementation uses
`HEAD_TYPE="dual"`, `DUAL_MODE="exceedance"` and
`EXCESS_FORMULATION="severity_shape"`. The request's phrase
`dual_mode=severity_shape` describes the intended formulation, but is not a
valid value of the current CLI. P3 preserves the exact P2 interface and
severity-shape mathematics. See `emulator/training/arguments.py` in the
repository; `dual_mode` only accepts `exceedance`.

The generator replaces only `LR_LIST`, `SESSION_NAME`, `PACT_RUN_NAME` and
`ALL_RESULTS_ROOT`, plus the experiment header comment. Every other source
line is retained, including the existing
`PYTHON_RUN_TAG_BASE="${PACT_RUN_NAME}"` mechanism. The copied comments that
mention P0/P2 document inherited controls.

| Control | Verified P2/P3 setting |
| --- | --- |
| Data | `/media/share/PACT/Data/Grid4_New/NCEP/graphs`; original station JSON paths; no external TEST root |
| Split | Chronological year groups, train 0.6 / VAL 0.2; shuffle 0; future-only 0 |
| Normalization | Existing TRAIN-derived normalization; `zscore`; clipping 0 |
| Architecture | perceiver3 / GraphSAGE / Transformer, history 24 h, hidden 128, graph layers 2 |
| Attention | Node/time heads 8/8; Transformer layers 2, FF multiplier 4, dropout 0, max time steps 32 |
| Head | Dual supervised exceedance, severity-shape excess, window gate, ablation none |
| Dropout | Model/head 0.05/0.05 |
| Branch weights | Body 1, excess 2, gate 0.5 |
| Events | TRAIN window-maximum exceedance percentile 95; tail fraction 0.05 |
| Batch | 256; gradient accumulation 4; one GPU, CUDA device 0 |
| Optimization | Adam, weight decay 1e-5, 300 epochs; LR is the only optimization treatment |
| Schedule | Cosine, warmup 5 epochs, warmup start factor 0.1, min LR 1e-6 |
| Stability | Seed 42; deterministic execution 0; gradient clipping 0 |
| Precision | AMP enabled, BF16, TF32 on |
| Loader | Torch threads 1; workers/pin memory/persistent workers/prefetch all 0; context fork |
| Checkpoint | Overall minimum VAL RMSE; auxiliary checkpoints 0; overall tolerance 0.01 |
| OOD/augmentation | Disabled; augmentation probability/scale/bias all 0 |
| Station features | Site elevation/bathymetry off; existing coordinate metadata retained |

Warmup retains P2's factor 0.1 relative to each selected base LR. The cosine
schedule, warmup length, number of epochs and minimum LR are unchanged.

| Factor | OFF | ON |
| --- | --- | --- |
| Tail | `loss_mode=mse` | `loss_mode=mse_tail`, active lambda 0.025 |
| Amp | 0 | 0.0025 |
| Shape | 0 | 0.00003 |
| Peak | 0 | 0.0035 |

Every config retains `TAIL_LAMBDA_LIST=("0.025")`. Tail-OFF runs omit
`--tail_lambda`, exactly as P2 does; the parsed default 0.1 is inactive under
MSE. It is not an additional treatment. Severity epsilon remains 1e-6;
amplitude and final-peak pooling remain hard `max`; inactive pool betas are
20. No loss definition, coefficient or normalization is changed.

## Original P2 degradation references

These user-supplied historical values are documentation, **not success
thresholds**. They also appear in the manifest's `p2_reference_*` columns.
Approximation marks are retained below; only the two stated early epochs
were supplied as exact values.

| Station | Config | Selected epoch | VAL rmse_all (mm) | Additional observation |
| --- | --- | ---: | ---: | --- |
| Lewes | T0A0S0P1 | ≈283 | ≈31.933 | |
| Battery | T0A0S1P1 | ≈271 | ≈43.668 | |
| Battery | T1A0S1P0 | ≈255 | ≈43.769 | |
| Battery | T1A1S0P0 | 3 | ≈95.713 | Late VAL ≈193 mm |
| Boston | T0A1S0P1 | ≈260 | ≈38.393 | |
| Boston | T1A0S0P0 | ≈286 | ≈30.366 | |
| Boston | T1A1S0P0 | 27 | ≈40.035 | |

New LR5e-3 reruns help distinguish systematic LR degradation from run-to-run
variation with seed 42 and nondeterministic execution. This experiment does
not select station-specific LRs using TEST, and all completed and failed run
artifacts must remain available for later analysis.

## Start runs

Use the usual local interactive Bash shell where `qsub_local` is defined.
The wrapper is **sourced** so it can call that existing shell function. It
executes in a subshell and preserves the caller's directory and shell options.
The P2 interpreter is used directly:
`/media/volume/PACT-Data/conda_envs/torchpyg-cu124/bin/python`, with `DO_CONDA=0`.

First enter the repository:

```bash
cd /media/volume/PACT-Data/StormSurge
```

Choose one of these commands for the desired scope:

```bash
# All 31 runs
source experiment_config/P3_LRStability/launch.sh --submit --all

# Battery only: 15
source experiment_config/P3_LRStability/launch.sh --submit --station Battery

# Boston only: 13
source experiment_config/P3_LRStability/launch.sh --submit --station Boston

# Lewes only: 3
source experiment_config/P3_LRStability/launch.sh --submit --station Lewes

# LR2e-3 only: 12
source experiment_config/P3_LRStability/launch.sh --submit --lr 2e-3

# All degradation-alert treatments: 21 (includes the six shared low-LR treatments)
source experiment_config/P3_LRStability/launch.sh --submit --reason p2_degradation_alert
```

To preview, omit `--submit` or use `--dry-run`. Preview performs the same
full 31-run validation and prints deferred commands; it does not submit jobs.

```bash
bash experiment_config/P3_LRStability/launch.sh --dry-run --station Battery
bash experiment_config/P3_LRStability/launch.sh --lr LR2e-3
bash experiment_config/P3_LRStability/launch.sh --semantic-config T1A1S0P0
bash experiment_config/P3_LRStability/launch.sh --station Boston --lr 0.001 --reason p2_degradation_alert
```

Filters combine by intersection; repeated values within one filter combine
by union. `--reason core_probe` selects 16 treatments;
`--reason p2_degradation_alert` selects 21; the exact combined reason selects
the six overlaps. An empty selection or any invalid manifest/config fails
before the first submission. Selection preserves the order in `manifest.csv`.
Separate invocations are separate submissions: submitting `--all` and then a
subset intentionally queues new reruns of that subset.

Submission uses the working P2 syntax:

```bash
qsub_local train.sh P3_Battery_SS_T1_A1_S0_P0_LR2e-3 experiment_config/P3_LRStability/train_config_P3_Battery_SS_T1_A1_S0_P0_LR2e-3.sh
```

Each config is also independently usable with `bash train.sh <config>`,
after running the validator below. The existing local queue defaults to one
slot, passes the run label as `SESSION_NAME`, and waits for the existing tmux
launcher. P3 does not change the queue helper, worker, shared `train.sh`, or
P0/P1/P2 naming behavior. All experiment outputs go to:

```text
All_Results/P3_LRStability/<P3 semantic run name>__<execution timestamp>/
```

Python `run_tag` is `<P3 semantic run name>_<execution timestamp>`. The
existing launcher's refusal to overwrite a same-name, same-second result
directory is preserved. The results group starts empty; no training was run
as part of preparation.

## Revalidate and reproduce

```bash
# Read-only validation, full matrix and per-run P2 diff summary
/media/volume/PACT-Data/conda_envs/torchpyg-cu124/bin/python -B \
  experiment_config/P3_LRStability/validate_configs.py

# Optionally save fresh reports to a separate location
/media/volume/PACT-Data/conda_envs/torchpyg-cu124/bin/python -B \
  experiment_config/P3_LRStability/validate_configs.py --report-dir /tmp/p3-validation

# Reproduce generated files; refuses to overwrite edited generated content
python3 -B experiment_config/P3_LRStability/generate_configs.py

# Exercise filters and rejection paths with a mock queue; never trains/submits
/media/volume/PACT-Data/conda_envs/torchpyg-cu124/bin/python -B \
  experiment_config/P3_LRStability/test_workflow.py
```

Validation checks all 108 configured shell fields, all 99 resolved Python
fields and every raw argv token per treatment. Allowed differences are only
identity/output and LR; the seven LR5e-3 reruns must have identical LR to P2.
The expected matrix is checked before any dry run. All configs must match
their source copies exactly after the documented replacements; additional
settings, missing settings, duplicate treatments and nested sweeps fail.

`provenance.json` records source config hashes, the source repository commit,
and hashes of the shared launcher, training code, model/loss/normalization
implementation and station metadata. Validation rejects later source drift.
It also checks the existing Adam call and verifies that dry runs changed no
result artifacts. Data are not loaded; training and convergence remain to be
measured when the experiment is run.

Evidence: [validation_report.md](validation_report.md),
[validation_report.json](validation_report.json), [dry_runs.txt](dry_runs.txt),
[workflow_checks.json](workflow_checks.json), and [integrity_report.json](integrity_report.json).
