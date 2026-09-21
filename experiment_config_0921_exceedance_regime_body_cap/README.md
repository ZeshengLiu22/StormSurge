# TRAIN exceedance regime and exact body cap

This study asks whether a broader TRAIN high-surge regime improves the unchanged
TEST top 5% extremes. It starts from clean production `main` at
`d665a4b0f54700a2564e54187f6c067c470d1df0`, on branch
`study/exceedance-regime-body-cap` in
`/home/exouser/StormSurge_exceedance_regime_body_cap`.
The implementation commit is `06dd17ecbc527842a70edf6520a123c9edc6975b`.
No changes from `study/excess-risk-weighting` were merged or cherry-picked.

Exactly these six conditions are generated for CBBT, Lewes, Battery and Boston:

| ID | TRAIN percentile | Excess weight | Body cap |
| --- | ---: | ---: | --- |
| T0 | 95 | 2 | soft |
| T1 | 95 | 2 | exact |
| T2 | 90 | 1 | soft |
| T3 | 90 | 1 | exact |
| T4 | 90 | 2 | soft |
| T5 | 90 | 2 | exact |

T0/T1 isolates the Q95 body-cap effect; T0/T4 isolates the TRAIN regime;
T2/T4 isolates Q90 excess weight; T2/T3 and T4/T5 isolate the Q90 body cap.
Q90/weight 1 approximately compensates for doubled event prevalence; Q90/weight 2
retains the current weight over the broader regime. There is no Q95/weight 1 run.

`DUAL_BODY_CAP="soft"` / `--dual_body_cap soft` retains the original
`b = tau - softplus(tau - r)`. Opt-in `exact` uses `b = min(r, tau)`.
Reconstruction remains `y_hat = b + sigmoid(gate_logits) * excess`.
Excess stays in normalized target units. Capacity, pooling, gate, initialization,
loss definitions, optimizer and evaluation code are unchanged. Missing checkpoint
or config fields default to soft. The mode is saved in model/training configs,
resolved shell snapshots, dual checkpoint metadata and summaries.

All runs use the production `ExceedanceHead`: no capacity variant or learned gate
pooling. TRAIN-prior normalization, relative-magnitude weighting, tail, amplitude,
final/window peak, shape, asymmetric and physical-excess objectives are not enabled.
No severity decomposition or hard gate is used. Existing opt-in features on main
remain inactive. Exact cap is restricted to the production direct dual head.

The standalone configs inherit values at generation time from each station's
0916 direct-dual reference, which also underlies the recent 0919 CBBT protocol.
They do not source old configs at runtime. Within each station, their resolved
settings differ only in percentile, excess weight, cap and run/output identity.

Protocol: NCEP; perceiver3; GraphSAGE + Transformer; 24 h history; hidden 128;
two spatial layers; two Transformer layers; read heads 8/8; FF multiplier 4;
graph/head dropout .05 and Transformer dropout 0; six station features;
MSE + 1×body + {1,2}×excess + .5×gate; Adam LR .005 and weight decay 1e-5;
300 epochs; batch 256; accumulation 4; cosine scheduler; five warmup epochs from
factor .1; minimum LR 1e-6; seed 42; deterministic 0; BF16 AMP; TF32 on;
overall VAL RMSE checkpoint selection; no auxiliary checkpoints, OOD augmentation,
feature clipping or gradient clipping. All optional auxiliary losses are off.
Under MSE, the existing launcher omits tail/slope flags: their inactive parser
defaults .1/.01 never enter the objective, despite explicit zero shell coefficients.

Dataset: `/media/share/PACT/Data/Grid4_New/NCEP/graphs`.
Station metadata: `/media/volume/PACT-Data/StormSurge/station_json` (read only).
Each station has 13,224 TRAIN, 4,208 VAL and 3,848 TEST windows. The chronological
year-group split is TRAIN 1979_1980 through 2000_2001, VAL 2001_2002 through
2007_2008, and TEST 2008_2009 through 2014_2015. Thresholds use strict
`max_h(Y_h) > tau_phys` on TRAIN only:

| Station | Q95 threshold (m) | Q95 events | Q90 threshold (m) | Q90 events |
| --- | ---: | ---: | ---: | ---: |
| CBBT | 0.212631613 | 662 | 0.133888617 | 1,323 |
| Lewes | 0.281014085 | 662 | 0.185592353 | 1,323 |
| Battery | 0.432081997 | 662 | 0.339308053 | 1,323 |
| Boston | 0.349115968 | 662 | 0.290335149 | 1,323 |

All six models have **685,447 parameters**, including 99,843 head parameters.
All random weights and RNG consumption match. Full initialized states match
within each percentile. Moving Q95 to Q90 changes the threshold buffer and the
gate bias initialized from the empirical TRAIN prevalence, following the existing
rule. First- and second-epoch sample permutations match across all six conditions
on the real TRAIN window IDs. Deterministic kernels remain disabled as requested.

Evaluation retains **TEST top 5%**, exactly `ceil(.05 * 3848) = 193` windows per
station, selected by true window maximum using the existing tie handling.
AllRMSE/AllMAE still use the full TEST split. Top5RMSE, Top5MAE, PeakRMSE, PeakMAE,
PeakBias, Under%, TruePeakRMSE and TimingSteps use the existing top-5% subset.
`tail_frac=.05` stays fixed. Additional `_event` diagnostics follow the TRAIN
threshold, and are not the primary extreme metrics. No Q90 evaluation replaces
the top-5% results.

Validation passed: seven focused tests, all 217 full-suite tests, 24 launcher dry
runs, matched reference settings, real-data threshold/split/initialization checks,
and all 24 synthetic GPU BF16/TF32 forward/loss/backward checks. GPU smoke checks
take no optimizer steps. Primary main remained clean at the base SHA. No study
jobs were submitted, and no training outputs or checkpoints were added to Git.
See [validation_report.md](validation_report.md),
[full numerical evidence](validation_report.json),
[test transcript](validation/full_tests.txt),
[manifest](manifest.csv), and [changed files](changed_files.txt).

Regenerate/check or validate without submitting jobs:

```bash
cd /home/exouser/StormSurge_exceedance_regime_body_cap
/media/volume/PACT-Data/conda_envs/torchpyg-cu124/bin/python -B experiment_config_0921_exceedance_regime_body_cap/generate_configs.py --check
/media/volume/PACT-Data/conda_envs/torchpyg-cu124/bin/python -B experiment_config_0921_exceedance_regime_body_cap/validate_configs.py --check-only --gpu-smoke
bash experiment_config_0921_exceedance_regime_body_cap/run_all.sh --dry-run
```

Launch all 24 runs through the existing `qsub_local`/tmux queue, one GPU job at a time:

```bash
cd /home/exouser/StormSurge_exceedance_regime_body_cap
bash experiment_config_0921_exceedance_regime_body_cap/run_all.sh
```

Alternatively, submit six runs for one station (do not also submit all 24):

```bash
bash experiment_config_0921_exceedance_regime_body_cap/run_all.sh CBBT
bash experiment_config_0921_exceedance_regime_body_cap/run_all.sh Lewes
bash experiment_config_0921_exceedance_regime_body_cap/run_all.sh Battery
bash experiment_config_0921_exceedance_regime_body_cap/run_all.sh Boston
```

The launcher validates first and requires GPU 0 with BF16 support, with no CPU or
precision fallback. It captures this experimental worktree as the working directory.
[commands_all.txt](commands_all.txt) contains the exact 24 individual queue commands
for an interactive shell with `qsub_local`, after the same validation/GPU checks.
[dry_run_commands.txt](dry_run_commands.txt) records all resolved Python commands;
its fixed timestamp is an inspection placeholder, not a launch timestamp.

All formal artifacts go outside Git under
`/media/share/PACT/Results/All_results_0921_exceedance_regime_body_cap/`, in
`0921_<Station>_<ID>_Q<percentile>_W<weight>_<cap>__<TIMESTAMP>/` directories.
