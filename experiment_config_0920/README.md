# 0920 direct-dual formulation study

## Study Outcome — Concluded / Archived

**This branch is a completed exploratory study with a negative/mixed scientific
outcome.**

> Engineering objective: achieved.
>
> Scientific formulation hypothesis: not supported strongly enough to continue.

### Achieved

- Implemented the direct-dual formulation study on `study/direct-dual-formulation`
  without modifying `main`, keeping the experimental changes isolated and reversible.
- Added and tested the intended F0/F1/F2/F3 formulations, with reconstruction,
  supervision-scope, checkpoint/inference tests and formulation diagnostics.
- Added configurations for CBBT, Lewes, Battery and Boston, with a
  [manifest](manifest.csv), [provenance](provenance.json) and
  [validation evidence](validation_report.json).
- Successfully evaluated the principal trained formulation changes (F0, F2 and F3)
  on CBBT. This screening was sufficient to decide whether the alternative
  reconstruction and supervision formulations were promising enough to continue.

| Formulation | Mode | Approximate prediction | Excess supervision |
| --- | --- | --- | --- |
| **F0** (`F0_Soft`) | Original soft-gated model | `body + q * excess` | Event windows |
| **F1** (`F1_Hard`) | Post-processing diagnostic derived from F0 | `body_phys + 1[q >= 0.5] * excess_phys` | Reuses F0; no retraining |
| **F2** (`F2_AdditiveEvent`) | Additive model | `body + excess` | Event windows |
| **F3** (`F3_AdditiveAll`) | Additive model | `body + excess` | All samples/windows, including zero excess targets on non-events |

Here `q` is the learned window event probability. F2/F3 retain the gate as an
auxiliary event predictor, but it does not multiply the excess in reconstruction.
These definitions follow the [heads](../emulator/models/heads.py),
[supervision scopes](../emulator/training/losses.py) and
[F1 diagnostic](../emulator/inference/dual_diagnostics.py).
F0/F2/F3 checkpoints were selected by overall VAL RMSE; F1 reuses the corresponding
already-selected F0 checkpoint with a fixed, inclusive threshold of 0.5.
F1 is a diagnostic/post-processing formulation, not evidence of a successful new
model. No completed F1 scientific result is claimed here.

### Failed / Not Supported

**The scientific objective of improving peak-aware prediction through these
formulation changes was not achieved.** The main evidence is the completed CBBT
screening, with the following final TEST metrics:

| Variant | AllRMSE | AllMAE | Top5RMSE | Top5MAE | PeakRMSE | PeakMAE | PeakBias | Under% | TruePeakRMSE | TimingSteps |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| F0 | 0.022430 | 0.016945 | 0.036866 | 0.026565 | 0.042608 | 0.032122 | -0.027928 | 86.01% | 0.044448 | 1.373057 |
| F2 | 0.022605 | 0.017174 | 0.035999 | 0.026750 | 0.042237 | 0.032391 | -0.028187 | 86.01% | 0.044240 | 1.388601 |
| F3 | 0.022938 | 0.017343 | 0.037823 | 0.027683 | 0.044286 | 0.034154 | -0.030487 | 86.01% | 0.046638 | 1.290155 |

Error and bias values are in metres. Top5 metrics use the 5% of TEST windows with
the largest true peaks. PeakRMSE, PeakMAE, PeakBias, Under%, TruePeakRMSE and
TimingSteps refer to that same top-5% population. TruePeakRMSE evaluates the
prediction at the true peak time; TimingSteps is mean absolute peak timing error
in forecast steps.

The values match the final `TEST` log rows and `test` fields in `summary_*.json`
for these original run directories under the ignored `All_results_0920/` root:

- F0: `0920_CBBT_F0_Soft__20260920_194155/`
- F2: `0920_CBBT_F2_AdditiveEvent__20260920_200719/`
- F3: `0920_CBBT_F3_AdditiveAll__20260920_203247/`

F2 produced only small improvements in Top5RMSE and PeakRMSE. These did not
translate into better overall error, peak bias or underprediction rate. PeakBias
remained approximately **-0.028 m**, and the reported underprediction rate remained
exactly **86.01%** for both F0 and F2. Removing the soft gate factor `q` therefore
did not resolve the primary peak-amplitude underprediction problem in this
screening.

F3 degraded most relevant metrics relative to F0 and F2. Its improved timing did
not offset the worse overall and peak-amplitude errors or more negative peak bias.
The results do not support extending excess supervision to all samples and do
not justify continuing the F2/F3 formulation sweep.

### Evaluation scope and stopping decision

Other-station runs were intentionally stopped before the planned sweep completed,
after the CBBT screening showed insufficient evidence to justify further compute
expenditure. Lewes F0 reached final evaluation, while Lewes F2 was interrupted;
the other-station formulation comparisons were not completed. **This branch does
not establish a four-station scientific comparison.** Configuration coverage and
individual or partial run artifacts must not be interpreted as a completed
comparison across CBBT, Lewes, Battery and Boston. Missing results are not inferred.

## Branch Disposition

- **Status:** concluded / archived exploratory study.
- **Merge into `main`:** no.
- **Reason:** implementation was successful, but the tested direct-dual formulation
  changes did not provide a sufficiently strong or consistent improvement in
  peak-aware prediction.
- **Preserve branch:** yes, for provenance, reproducibility and possible future
  reference.
- **Future work:** start future peak-aware work from clean `main` in a new branch
  rather than extending this branch.

This is a scientific stopping decision, not a software failure.

## Archived experiment setup and commands

The original setup and commands below are retained for provenance and
reproducibility. The study is concluded; the configuration matrix describes
implemented coverage, not completed scientific results.

**4 stations × 4 formulations = 16 configs: 12 training + 4 post-processing.**
Stations are CBBT, Lewes, Battery and Boston. Each station has configurations for
the four formulations defined above.

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
