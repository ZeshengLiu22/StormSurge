# Four-station excess-risk 2×2 study

This study changes only aggregation of the production direct-dual horizon-wise
excess MSE. Stage-0 diagnostics motivate testing excess deficit with two
independent factors: normalization by the exact TRAIN event prior and relative
physical excess-magnitude weighting within an event.

The experiment covers **CBBT, Lewes, Battery, and Boston**, with E0/E1/E2/E3
at each station: **16 standalone configs**. Each station fits its own TRAIN
threshold and event prior; the four modes at that station share the fitted
values. Priors and thresholds are never pooled or copied across stations.

Branch: `study/excess-risk-weighting`, based on upstream-verified clean tracked
`main` at `d665a4b0f54700a2564e54187f6c067c470d1df0`. The original main checkout
contained untracked `All_results_0920/` artifacts, so implementation uses the clean
worktree `/home/exouser/StormSurge_excess_risk`. Main and those artifacts were
preserved. No archived commit was merged or cherry-picked.

## Objectives

Using the unchanged `dual_excess_target` event and target:

\[
e^*_{ih}=\max(y^{norm}_{ih}-\tau_h^{norm},0),\quad
E_i=\mathbf{1}[\exists h:e^*_{ih}>0],\quad
\ell_{ih}=[(\hat e_{ih}-e^*_{ih})\sigma_h]^2.
\]

The denominator is the fixed value returned by `fit_loss_thresholds` on TRAIN:

\[
q_E=N_{\mathrm{TRAIN\ event}}/N_{\mathrm{TRAIN\ windows}}.
\]

It is never replaced by nominal 0.05 or current-batch prevalence. Requested
`train_prior` normalization rejects missing, nonfinite or nonpositive priors;
`none` retains the existing prior requirements.

For magnitude weighting, first convert each horizon's target excess to meters:

\[
m_{ih}=e^*_{ih}\sigma_h,\quad
r_{ih}=\frac{m_{ih}}{\max_j m_{ij}+\epsilon},\quad
w^{raw}_{ih}=1+\alpha r_{ih},\quad
\widetilde w_{ih}=\frac{w^{raw}_{ih}}{H^{-1}\sum_j w^{raw}_{ij}}.
\]

The implementation uses `torch.finfo(physical.dtype).tiny` as the zero-division
guard, with FP16/BF16 magnitudes promoted to FP32. Alpha is finite and
nonnegative; all 16 configs use **alpha = 1**. Raw weights at zero, half-max,
and max physical excess are 1, 1.5, and 2 before mean-one normalization. Every
window has mean horizon weight one, including event-free windows. Zero-excess
horizons inside events retain supervision. Mean-one weighting redistributes
supervision; correlation with squared error can still change the scalar loss.

| Run | Event normalization | Horizon weighting | Exact excess objective |
|---|---|---|---|
| E0_Uniform | none | uniform | \(L_{E0}=\operatorname{mean}_{i,h}[E_i\ell_{ih}]\) |
| E1_PriorNorm | train_prior | uniform | \(L_{E1}=\operatorname{mean}_{i,h}[E_i\ell_{ih}]/q_E\) |
| E2_Magnitude | none | relative_magnitude | \(L_{E2}=\operatorname{mean}_{i,h}[E_i\widetilde w_{ih}\ell_{ih}]\) |
| E3_PriorNorm_Magnitude | train_prior | relative_magnitude | \(L_{E3}=\operatorname{mean}_{i,h}[E_i\widetilde w_{ih}\ell_{ih}]/q_E\) |

The total training objective in every run is
\(L=\mathrm{MSE}_{phys}+L_{body}+2L_{ex}+0.5L_{gate}\).
`EXCESS_LOSS_WEIGHT=2` is fixed. Prior normalization intentionally increases the
effective strength of excess supervision; there is no compensating change.

E0 executes the original production expression directly:

```python
excess = (((output.excess - excess_target) * y_std).square() * event).mean()
```

It does not construct weights, apply epsilon, or divide by a prior. Focused
tests verify exact scalar and gradient agreement with production arithmetic
and unchanged RNG state. The new settings do not enter `ModelConfig`.

## Configuration and provenance

| CLI / LossConfig field | Shell variable | Default |
|---|---|---|
| `excess_event_normalization` | `EXCESS_EVENT_NORMALIZATION` | `none` |
| `excess_horizon_weighting` | `EXCESS_HORIZON_WEIGHTING` | `uniform` |
| `excess_magnitude_alpha` | `EXCESS_MAGNITUDE_ALPHA` | `1.0` |

The first two accept `none,train_prior` and `uniform,relative_magnitude`,
respectively. Active aggregation requires direct dual supervision. Settings
are saved in resolved shell/JSON configs, checkpoint `training_config`,
checkpoint `dual_metadata`, and summary `dual_metadata`. Omitted fields retain
the defaults when constructing `LossConfig`; old checkpoint inference remains
unchanged, including checkpoints without these metadata fields.

Each config is standalone. [manifest.csv](manifest.csv) lists the matrix,
paths, output directories, and commands. Apart from run identity, the only
differences within a station are the two factor switches. Between stations,
only station selection and run identity differ; the shared training protocol
is identical. Existing CBBT configs remain byte-for-byte unchanged.

| Station | Configs | Filename prefix |
|---|---|---|
| CBBT | E0, E1, E2, E3 | `configs/train_config_0921_CBBT_` |
| Lewes | E0, E1, E2, E3 | `configs/train_config_0921_Lewes_` |
| Battery | E0, E1, E2, E3 | `configs/train_config_0921_Battery_` |
| Boston | E0, E1, E2, E3 | `configs/train_config_0921_Boston_` |

## Matched protocol and historical reference

The protocol reference is read using `git show` only:

`study/direct-dual-formulation:experiment_config_0920/configs/train_config_0920_CBBT_F0_Soft.sh`

Generation pins that file to archived commit
`7516f9c5148d3068ad86b984028d417fd226f782`, with file SHA256
`68aeaae4d6128aa1129ab4c3b61cdb83edbb378bfc4bb40d3c247998808d96e3`.
The archived `DIRECT_DUAL_RECONSTRUCTION=soft_gate` and
`EXCESS_SUPERVISION_SCOPE=event` selectors are omitted because main's production
direct-dual implementation already fixes those semantics. Main supplies all
implementation code. `EXCEEDANCE_HEAD_EXPERIMENT` is explicitly empty.
The same pinned CBBT F0 protocol is used for all four stations, changing only
station selection and run identity when extending it to Lewes, Battery, and
Boston. Each run loads the selected station's existing metadata JSON.

The matched protocol is NCEP at all four stations; PACT (`perceiver3`), GraphSAGE, Transformer;
24-hour history, hidden width 128, two spatial layers; production direct dual
head and mean gate pooling; exceedance percentile 95; MSE with body/excess/gate
weights 1/2/0.5. Training uses LR `5e-3`, 300 epochs, batch size 256, accumulation
4, Adam with unchanged `weight_decay=1e-5`, cosine scheduling, five warmup epochs,
seed 42, `DETERMINISTIC=0`, BF16 AMP, and TF32. Site elevation and bathymetry
features are both off; the existing station metadata feature vector remains
enabled. OOD augmentation, input clipping, and gradient clipping are off.

Tail and slope shell coefficients are zero, and MSE leaves both objectives
inactive. The unchanged launcher omits tail/slope CLI coefficients under MSE;
their parsed values remain the historical **inactive** defaults 0.1/0.01.
Excess-amplitude, shape, and final-peak coefficients resolve to zero.
No optional auxiliary loss, model/head architecture, soft body cap, gate BCE,
body parameterization, reconstruction `body + q * excess`, threshold fitting,
optimizer, data ordering, initialization, or checkpoint criterion changes.

The existing **0920 F0 result is an external historical reference**. Its run
directory is `All_results_0920/0920_CBBT_F0_Soft__20260920_194155` in the original
checkout. **New E0 is the matched implementation-control run**; E1/E2/E3 are
new experimental runs. The historical result is not substituted for E0, and
archived source equivalence beyond the protocol has not been asserted.
`DETERMINISTIC=0` means identical config/seed or initial CPU parameters do not
guarantee identical training trajectories. No new training results exist yet.

## Validation and execution

Run from this study worktree, whose `train.py` contains the new switches:

```bash
cd /home/exouser/StormSurge_excess_risk
STUDY_PY=/media/volume/PACT-Data/conda_envs/torchpyg-cu124/bin/python
PYTHONDONTWRITEBYTECODE=1 "$STUDY_PY" experiment_config_0921_excess_risk/generate_configs.py --check
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' "$STUDY_PY" experiment_config_0921_excess_risk/validate_configs.py
```

The validator runs all 16 launcher dry runs, parses the actual commands,
compares every non-factor setting to that station's E0 and the archived protocol,
checks that stations differ only in station selection and run identity, and
checks optional losses, overall checkpoint selection, and unique output paths.
For each station it fits actual TRAIN thresholds four times on the same selected
population and constructs all four models on CPU. Within each station it verifies
equal thresholds, priors, `ModelConfig`, parameter counts, and initial
parameter/buffer state. It never calls a training epoch or optimizer.
`--config-only` skips data loading and CPU model construction and writes separate
evidence. [validation/validation.json](validation/validation.json) records the
completed four-station validation; [validation/tests.json](validation/tests.json)
records the unchanged implementation's unit-suite outcome and environment.
That suite ran 222 tests:
217 passed, 5 CUDA-only tests skipped, and no failures or errors. Focused
checks passed 22 tests after normalization and 37 after magnitude weighting.

All 16 dry runs and actual TRAIN validations passed. The fitted thresholds are
station-specific and identical across E0/E1/E2/E3 within each station:

| Station | TRAIN windows | Strict events | `tau_phys` (meters) |
|---|---:|---:|---:|
| CBBT | 13,224 | 662 | 0.21263161301612854 |
| Lewes | 13,224 | 662 | 0.281014084815979 |
| Battery | 13,224 | 662 | 0.43208199739456177 |
| Boston | 13,224 | 662 | 0.3491159677505493 |

Each station independently fitted `event_prior = 662 / 13224 = 0.050060496067755596`.
The equal counts are a result of these data, not a shared or nominal prior.
All models have 685,447 parameters, including 99,843 head parameters; initial
CPU model state is identical across the four modes within each station.

Thus E1/E3 apply `1/q_E = 19.97583081570997`; with the fixed coefficient 2, the
coefficient on the original weighted/unweighted mean is `39.95166163141994`.

Focused tests can be rerun without GPU access:

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' PYTHONPATH=.:tests \
  "$STUDY_PY" -m unittest test_excess_risk test_dual_loss test_dual_experiments test_config_interfaces -v
```

Full discovery uses `-m unittest discover -s tests -v`. Distributed CPU tests
need local sockets; preprocessing tests need the dependencies pinned in
`environment_dataprep.yml`. The recorded full run uses temporary dependencies
without changing the training environment; CUDA-only tests are skipped.

Training commands are prepared in [commands_all.txt](commands_all.txt). Execute
them only after an explicit decision to start GPU training. The file lists all
16 individual commands; examples for the four E0 controls are:

```bash
bash train.sh experiment_config_0921_excess_risk/configs/train_config_0921_CBBT_E0_Uniform.sh
bash train.sh experiment_config_0921_excess_risk/configs/train_config_0921_Lewes_E0_Uniform.sh
bash train.sh experiment_config_0921_excess_risk/configs/train_config_0921_Battery_E0_Uniform.sh
bash train.sh experiment_config_0921_excess_risk/configs/train_config_0921_Boston_E0_Uniform.sh
```

Each uses the existing tmux launcher on GPU 0; run one at a time, waiting for
its session to finish before the next command. To preview a command without
launching, prefix it with `DRY_RUN=1`.

All future training artifacts go to
`/media/share/PACT/Results/All_results_0921_excess_risk/<RUN_NAME>__<TIMESTAMP>`.
Validation has created no experiment results directory. Checkpoints and
training/evaluation results must remain outside Git.

## Evaluation

Primary checkpoint selection remains **overall validation RMSE**, with no
auxiliary checkpoints. TEST metrics never select a checkpoint. Final evaluation
retains AllRMSE, AllMAE, Top5RMSE, Top5MAE, PeakRMSE, PeakMAE, PeakBias, Under%,
TruePeakRMSE, and TimingSteps. Mechanism diagnostics emphasize **PeakBias,
Under%, TruePeakRMSE, and PeakRMSE**; AllRMSE/AllMAE remain safeguards for general
performance. No metric becomes a new training objective or selection rule.

Implementation is split into `Add opt-in TRAIN-prior excess normalization`,
`Add opt-in within-event excess-magnitude weighting`, and
`Add matched excess-risk 2x2 experiment configs` commits for separate review
and rollback.
