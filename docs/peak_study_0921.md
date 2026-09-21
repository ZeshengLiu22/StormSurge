# 0921 fixed-event diagnosis, LR comparison, and optional checkpoint score

No experiment training has been launched. The completed 5e-3 baselines remain frozen.

## Task 1: completed, analysis only

The report is `/home/exouser/peak_branch_diagnosis_0921/DIAGNOSIS.md`.
It includes 202 fixed GT TEST windows at CBBT and 208 at Lewes, exactly the same IDs
for S0/D0/D1/D2/D3. Both stations have 662 extreme TRAIN windows out of 13,224.
Per-event CSVs, four requested summary tables plus consistency/attribution tables,
and ten plots per station in PNG and PDF are saved separately from training results.
The incomplete Lewes D3 attempt at 22:48:15 was excluded; the completed 22:55:43 run is used.

The mechanism supported by these selected checkpoints is raw-excess underestimation.
In the highest GT quartile, Amp increases the raw-excess contribution by 3.054 mm
at CBBT and decreases it by 2.078 mm at Lewes relative to D0. The gate is almost one
there; body deficits are about 2 mm. Lewes has larger physical excess targets and a
larger observed TEST/TRAIN tail increase, with the same number of extreme TRAIN windows.
This is descriptive evidence, not proof of the optimizer-level cause.

Archived TEST outputs confirm the opposite Amp UnderAny directions: CBBT
87.624% → 84.158%; Lewes 87.981% → 88.942%. The Lewes RMSE change is approximately
0.003 mm and should not be interpreted as meaningful degradation. Replay differences
from the nondeterministic BF16/TF32 baselines are measured explicitly. Branch
components and final predictions always come from the same forward pass. No model,
head, loss, metric definition, normalization, split, or threshold implementation was changed.

## Task 2: ten prepared configs, no training

| Method | CBBT run name | Lewes run name |
| --- | --- | --- |
| S0 | 0921_CBBT_S0_Single_LR3e3 | 0921_Lewes_S0_Single_LR3e3 |
| D0 | 0921_CBBT_D0_DualBase_LR3e3 | 0921_Lewes_D0_DualBase_LR3e3 |
| D1 | 0921_CBBT_D1_Tail_LR3e3 | 0921_Lewes_D1_Tail_LR3e3 |
| D2 | 0921_CBBT_D2_Amp_LR3e3 | 0921_Lewes_D2_Amp_LR3e3 |
| D3 | 0921_CBBT_D3_TailAmp_LR3e3 | 0921_Lewes_D3_TailAmp_LR3e3 |

Each file is `experiment_config_0921_lr3e3/configs/train_config_<run name>.sh`.
It is copied from the corresponding original 0921 shell config. Only `LR_LIST`,
run identifiers, output root, and descriptive comments differ. The validator
checks non-comment shell content and runs the existing launcher in DRY_RUN mode,
then parses its command using the real training parser and compares every resolved
argument with the completed parent's saved JSON. Only `lr`, `run_tag`, and
`output_dir` may differ; LR must be 0.005 → 0.003. Parent SHA256s are recorded.

`experiment_config_0921_lr3e3/validation/config_diff.json` reports PASS for all ten.
All runs retain 300 epochs, min_lr=1e-6, existing precision/determinism and batching,
and `CHECKPOINT_SELECTION=overall`, `SAVE_AUX_CHECKPOINTS=0`. The peak-aware tracker
is off. The result root is `/media/share/PACT/Results/All_results_0921_lr3e3`.

`/home/exouser/lr_comparison_0921/paired_metrics.csv` contains the existing baseline
metrics and explicitly pending 3e-3 columns. `learning_curves.csv` records selected
epoch, minimum-Val-AllRMSE epoch, and inclusive epochs 250–300 mean/std for VAL All
and TruePeak RMSE. No comparison conclusions are generated before the new runs exist.

## Task 3: isolated optional tracker

Default behavior remains exactly the existing strict-`<` best-Val-AllRMSE selection
and `best_<stem>.pth` output. No reference file is loaded and no new tracker is
created by default. The new CLI arguments use suppressed defaults, so historical
resolved configurations and filename hashes also remain unchanged.

Enable either `--track_peakaware 1` with selection `overall`, or explicitly select
`--checkpoint_selection peakaware`. The shell equivalents are `TRACK_PEAKAWARE=1`
and `CHECKPOINT_SELECTION=overall|peakaware`. When enabled, both `best_overall.pt`
and `best_peakaware.pt` are retained throughout training. The primary existing
`best_<stem>.pth` reflects the requested selection at completion; `best_overall.pt`
always retains the original strict-minimum-overall winner. If both roles select one
epoch, final evaluation is shared rather than injecting replay noise into the delta.

The exact implementation in `emulator/training/peakaware_checkpoints.py` is:

```python
score = (
    0.65 * (val['rmse_all'] / ref_all)
    + 0.20 * (val['rmse_peak5'] / ref_top5)
    + 0.15 * (val['true_peak_rmse_top5'] / ref_truepeak)
)
```

The TRUEPEAK term preserves the current trainer's displayed **top5 window**
population. It does not silently switch to the fixed-threshold event population
used by Task 1. Weights are configurable by `CKPT_SCORE_W_ALL`,
`CKPT_SCORE_W_TOP5`, `CKPT_SCORE_W_TRUEPEAK` (corresponding lowercase CLI flags);
they must be finite, nonnegative, and sum to one. References must be positive and
finite. Earliest epoch wins exact score ties. Only VAL metrics enter this score;
Under%, PeakBias, and TEST never enter selection. There is no automatic fallback.

Frozen references in `checkpoint_score_refs.json` are the S0 **saved checkpoint
epoch** VAL metrics, before the final replay. These are the same metric definitions
and evaluation path used each validation epoch; their tiny differences from a final
BF16 re-evaluation are not rounded away. Each source checkpoint/epoch and SHA256 is recorded.

| Station | ref_all | ref_top5 | ref_truepeak |
| --- | --- | --- | --- |
| CBBT | 0.023442892410193077 | 0.02942943386733532 | 0.03310695422448863 |
| Lewes | 0.023992552450784275 | 0.03638144209980965 | 0.04215794976369823 |

On improvement, the optional tracker prints epoch, all three VAL RMSEs and score.
The score is also stored per epoch in JSONL. At completion both policies are evaluated
on full VAL and TEST without distributed padding; results and deltas are printed
and written to `checkpoint_policy_comparison.csv` and `.json`. Delta is
peakaware minus overall for AllRMSE, Top5RMSE, TruePeakRMSE, PeakBias and UnderPercent
(the latter in percentage points). A **strictly greater than 1%** VAL AllRMSE
degradation is flagged, without fallback. The saved selection-epoch and final
re-evaluation flags are both reported. This opt-in tracker uses its own artifacts;
it is intentionally not combined with the older `save_aux_checkpoints=1` Pareto feature.

## Commands

Run from the production repository using the existing training environment:

```bash
cd /media/volume/PACT-Data/StormSurge
PACT_PYTHON=/media/volume/PACT-Data/conda_envs/torchpyg-cu124/bin/python

# Analysis only; verified replay caches are reused.
"$PACT_PYTHON" tools/peak_branch_diagnosis.py --stage all
# CSV/report/plot regeneration only, no forward passes:
"$PACT_PYTHON" tools/peak_branch_diagnosis.py --stage analyze

# Verify all ten LR-only differences without starting training:
"$PACT_PYTHON" tools/prepare_lr3e3.py

# Explicitly launch the ten NEW LR runs sequentially, only when requested:
bash experiment_config_0921_lr3e3/launch.sh

# Populate clean paired tables after the LR sweep; --replay is inference only:
"$PACT_PYTHON" tools/compare_lr0921.py --replay
```

For a **future separate** checkpoint-policy experiment, choose a parent after
reviewing the LR results. This example copies one parent while keeping its LR,
losses, and all other scientific settings; it does not modify the ten sweep files:

```bash
"$PACT_PYTHON" tools/make_peakaware_config.py \
  --parent experiment_config_0921_lr3e3/configs/train_config_0921_CBBT_D2_Amp_LR3e3.sh \
  --output /home/exouser/peakaware_future/CBBT_D2_peakaware.sh \
  --selection peakaware
DRY_RUN=1 USE_TMUX=0 bash train.sh /home/exouser/peakaware_future/CBBT_D2_peakaware.sh
# Future explicit launch only:
USE_TMUX=0 bash train.sh /home/exouser/peakaware_future/CBBT_D2_peakaware.sh
```

Use `--selection overall` in that preparation command to retain the old primary
selection while tracking and reporting both policies. No future policy run was
created or launched as part of the LR sweep.

## Files and validation

Modified production files: `train.py`, `train.sh`, `emulator/training/arguments.py`.
Added feature files: `emulator/training/peakaware_checkpoints.py`,
`checkpoint_score_refs.json`, this document, and tests
`test_peakaware_checkpoints.py`, `test_fixed_event_diagnosis.py`.
Added isolated tools: `pact_diagnostic_io.py`, `peak_branch_diagnosis.py`,
`prepare_lr3e3.py`, `compare_lr0921.py`, `freeze_checkpoint_refs.py`, and
`make_peakaware_config.py`. Added config folder: `experiment_config_0921_lr3e3/`
(ten configs, launch script, parent manifest, dry commands/resolved arguments/diff report).

Checks include exact event membership, missing-ID rejection, shared bins, physical
branch reconstruction (maximum under 3e-7 m), untouched baseline hashes, ten resolved
config comparisons, score normalization/invalid references/ties, distinct checkpoint
winners, post-training VAL/TEST reports, and the >1% boundary. Existing checkpoint,
CLI, metric-reporting, and two-process CPU regression tests pass. A small real CPU
fixture verifies identical training trajectory with and without tracking; comparison
against the original trainer also verifies bitwise-equivalent default losses,
forwards, weights, LR, RNG states and validation metrics for Single and direct Dual.
These are synthetic tests, not PACT experiment training.

Task 1 does not require any production training edits. Task 3 is removable by
reverting only the three modified production files and removing its helper/reference
files; all diagnosis tools and LR-only configurations remain independent.
