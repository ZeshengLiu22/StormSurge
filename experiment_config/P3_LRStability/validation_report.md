# P3 LR stability validation — PASS

```text
P3_LRStability

Unique runs: 31

By LR:
    5e-3: 7
    2e-3: 12
    1e-3: 12

By station:
    CBBT: 0
    Lewes: 3
    Battery: 15
    Boston: 13

Core probe: 16
Degradation-alert treatments: 21 nominal
Overlap removed: 6
Final unique: 31

Station | T A S P | LR | reason
Lewes | 0 0 0 1 | 5e-3 | p2_degradation_alert
Lewes | 0 0 0 1 | 2e-3 | p2_degradation_alert
Lewes | 0 0 0 1 | 1e-3 | p2_degradation_alert
Battery | 0 0 0 0 | 2e-3 | core_probe
Battery | 0 0 0 0 | 1e-3 | core_probe
Battery | 0 0 1 1 | 5e-3 | p2_degradation_alert
Battery | 0 0 1 1 | 2e-3 | p2_degradation_alert
Battery | 0 0 1 1 | 1e-3 | p2_degradation_alert
Battery | 0 1 0 1 | 2e-3 | core_probe
Battery | 0 1 0 1 | 1e-3 | core_probe
Battery | 1 0 1 0 | 5e-3 | p2_degradation_alert
Battery | 1 0 1 0 | 2e-3 | p2_degradation_alert
Battery | 1 0 1 0 | 1e-3 | p2_degradation_alert
Battery | 1 1 0 0 | 5e-3 | p2_degradation_alert
Battery | 1 1 0 0 | 2e-3 | core_probe+p2_degradation_alert
Battery | 1 1 0 0 | 1e-3 | core_probe+p2_degradation_alert
Battery | 1 1 0 1 | 2e-3 | core_probe
Battery | 1 1 0 1 | 1e-3 | core_probe
Boston | 0 0 0 0 | 2e-3 | core_probe
Boston | 0 0 0 0 | 1e-3 | core_probe
Boston | 0 1 0 1 | 5e-3 | p2_degradation_alert
Boston | 0 1 0 1 | 2e-3 | core_probe+p2_degradation_alert
Boston | 0 1 0 1 | 1e-3 | core_probe+p2_degradation_alert
Boston | 1 0 0 0 | 5e-3 | p2_degradation_alert
Boston | 1 0 0 0 | 2e-3 | p2_degradation_alert
Boston | 1 0 0 0 | 1e-3 | p2_degradation_alert
Boston | 1 1 0 0 | 5e-3 | p2_degradation_alert
Boston | 1 1 0 0 | 2e-3 | core_probe+p2_degradation_alert
Boston | 1 1 0 0 | 1e-3 | core_probe+p2_degradation_alert
Boston | 1 1 0 1 | 2e-3 | core_probe
Boston | 1 1 0 1 | 1e-3 | core_probe

Config-vs-P2 validation: PASS
31 P3 configs vs 12 exact P2 semantic sources.
Per run: 108 shell fields and 99 parsed Python fields compared.
Unexpected differences: 0. Shared train.sh and Python implementation unchanged.
LR5e-3: identity/output only. Low LR: identity/output plus LR.
Mode: DUAL_MODE=exceedance; EXCESS_FORMULATION=severity_shape (exact P2 interface).

Per-run diff summary (all PASS):
P3_Lewes_SS_T0_A0_S0_P1_LR5e-3: lr unchanged at 0.005; run_tag, output_dir
P3_Lewes_SS_T0_A0_S0_P1_LR2e-3: lr 0.005 -> 0.002; run_tag, output_dir
P3_Lewes_SS_T0_A0_S0_P1_LR1e-3: lr 0.005 -> 0.001; run_tag, output_dir
P3_Battery_SS_T0_A0_S0_P0_LR2e-3: lr 0.005 -> 0.002; run_tag, output_dir
P3_Battery_SS_T0_A0_S0_P0_LR1e-3: lr 0.005 -> 0.001; run_tag, output_dir
P3_Battery_SS_T0_A0_S1_P1_LR5e-3: lr unchanged at 0.005; run_tag, output_dir
P3_Battery_SS_T0_A0_S1_P1_LR2e-3: lr 0.005 -> 0.002; run_tag, output_dir
P3_Battery_SS_T0_A0_S1_P1_LR1e-3: lr 0.005 -> 0.001; run_tag, output_dir
P3_Battery_SS_T0_A1_S0_P1_LR2e-3: lr 0.005 -> 0.002; run_tag, output_dir
P3_Battery_SS_T0_A1_S0_P1_LR1e-3: lr 0.005 -> 0.001; run_tag, output_dir
P3_Battery_SS_T1_A0_S1_P0_LR5e-3: lr unchanged at 0.005; run_tag, output_dir
P3_Battery_SS_T1_A0_S1_P0_LR2e-3: lr 0.005 -> 0.002; run_tag, output_dir
P3_Battery_SS_T1_A0_S1_P0_LR1e-3: lr 0.005 -> 0.001; run_tag, output_dir
P3_Battery_SS_T1_A1_S0_P0_LR5e-3: lr unchanged at 0.005; run_tag, output_dir
P3_Battery_SS_T1_A1_S0_P0_LR2e-3: lr 0.005 -> 0.002; run_tag, output_dir
P3_Battery_SS_T1_A1_S0_P0_LR1e-3: lr 0.005 -> 0.001; run_tag, output_dir
P3_Battery_SS_T1_A1_S0_P1_LR2e-3: lr 0.005 -> 0.002; run_tag, output_dir
P3_Battery_SS_T1_A1_S0_P1_LR1e-3: lr 0.005 -> 0.001; run_tag, output_dir
P3_Boston_SS_T0_A0_S0_P0_LR2e-3: lr 0.005 -> 0.002; run_tag, output_dir
P3_Boston_SS_T0_A0_S0_P0_LR1e-3: lr 0.005 -> 0.001; run_tag, output_dir
P3_Boston_SS_T0_A1_S0_P1_LR5e-3: lr unchanged at 0.005; run_tag, output_dir
P3_Boston_SS_T0_A1_S0_P1_LR2e-3: lr 0.005 -> 0.002; run_tag, output_dir
P3_Boston_SS_T0_A1_S0_P1_LR1e-3: lr 0.005 -> 0.001; run_tag, output_dir
P3_Boston_SS_T1_A0_S0_P0_LR5e-3: lr unchanged at 0.005; run_tag, output_dir
P3_Boston_SS_T1_A0_S0_P0_LR2e-3: lr 0.005 -> 0.002; run_tag, output_dir
P3_Boston_SS_T1_A0_S0_P0_LR1e-3: lr 0.005 -> 0.001; run_tag, output_dir
P3_Boston_SS_T1_A1_S0_P0_LR5e-3: lr unchanged at 0.005; run_tag, output_dir
P3_Boston_SS_T1_A1_S0_P0_LR2e-3: lr 0.005 -> 0.002; run_tag, output_dir
P3_Boston_SS_T1_A1_S0_P0_LR1e-3: lr 0.005 -> 0.001; run_tag, output_dir
P3_Boston_SS_T1_A1_S0_P1_LR2e-3: lr 0.005 -> 0.002; run_tag, output_dir
P3_Boston_SS_T1_A1_S0_P1_LR1e-3: lr 0.005 -> 0.001; run_tag, output_dir

No training, queue submission or tmux session was started.
```

## Fixed controls read from P2

| Config variable | Value |
| --- | --- |
| AMP_DTYPE | `bf16` |
| BATCH_SIZE | `256` |
| BODY_LOSS_WEIGHT | `1` |
| CHECKPOINT_OVERALL_TOL | `0.01` |
| CHECKPOINT_SELECTION | `overall` |
| CNN_INTERMEDIATE_CHANNEL | `29` |
| CONDA_ENV | `/media/volume/PACT-Data/conda_envs/torchpyg-cu124` |
| CUDA_VISIBLE_DEVICES | `0` |
| DETERMINISTIC | `0` |
| DISABLE_OOD | `1` |
| DO_CONDA | `0` |
| DROPOUT | `0.05` |
| DUAL_ABLATION | `none` |
| DUAL_LOSS | `1` |
| DUAL_MODE | `exceedance` |
| ENCODER_TYPE | `GraphSAGE` |
| EPOCHS | `300` |
| EXCEEDANCE_PERCENTILE | `95` |
| EXCESS_AMP_BETA | `20` |
| EXCESS_AMP_POOL | `max` |
| EXCESS_FORMULATION | `severity_shape` |
| EXCESS_LOSS_WEIGHT | `2` |
| FUTURE_ONLY | `0` |
| FUTURE_YEAR_THRESHOLD | `2030` |
| GATE_LOSS_WEIGHT | `0.5` |
| GATE_MODE | `window` |
| GRAD_ACCUM_STEPS | `4` |
| HEAD_DROPOUT | `0.05` |
| HEAD_TYPE | `dual` |
| HIDDEN_CHANNELS | `128` |
| HISTORY_HOURS_LIST | `24` |
| MAX_GRAD_NORM | `0` |
| MAX_TIME_STEPS | `32` |
| MIN_LR | `1e-6` |
| MODEL | `perceiver3` |
| MP_CONTEXT | `fork` |
| NODE_READ_HEADS | `8` |
| NUM_LAYERS | `2` |
| NUM_WORKERS | `0` |
| PEAK_POOL | `max` |
| PEAK_POOL_BETA | `20` |
| PERSISTENT_WORKERS | `0` |
| PIN_MEMORY | `0` |
| PREFETCH_FACTOR | `0` |
| PYTHONDONTWRITEBYTECODE | `1` |
| PYTHON_BIN | `/media/volume/PACT-Data/conda_envs/torchpyg-cu124/bin/python` |
| ROOT_DIR | `/media/share/PACT/Data/Grid4_New/NCEP/graphs` |
| ROP_COOLDOWN | `0` |
| ROP_FACTOR | `0.5` |
| ROP_METRIC | `val_rmse_phys` |
| ROP_MIN_LR | `1e-6` |
| ROP_PATIENCE | `20` |
| ROP_THRESHOLD | `1e-4` |
| RUN_DIR_NAME_STYLE | `runname_timestamp` |
| SAVE_AUX_CHECKPOINTS | `0` |
| SCHEDULER | `cosine` |
| SEED | `42` |
| SEVERITY_SHAPE_EPS | `1e-6` |
| SHUFFLE_YEARS | `0` |
| SLOPE_CHARB_EPS | `1e-3` |
| SLOPE_HUBER_DELTA | `0.05` |
| SLOPE_LAMBDA_LIST | `0.01` |
| SLOPE_MASK_S_LIST | `0.10` |
| SLOPE_ROBUST | `charb` |
| STATION_JSON_DIR | `/media/volume/PACT-Data/StormSurge/station_json` |
| TAIL_FRAC | `0.05` |
| TAIL_LAMBDA_LIST | `0.025` |
| TEMPORAL_BLOCK | `Transformer` |
| TEST_ROOT_DIR | `` |
| TIME_READ_HEADS | `8` |
| TORCH_THREADS | `1` |
| TRAIN_PY | `train.py` |
| TRAIN_RATIO | `0.6` |
| TRANSFORMER_DROPOUT | `0.0` |
| TRANSFORMER_FF_MULT | `4.0` |
| TRANSFORMER_LAYERS | `2` |
| USE_AMP | `1` |
| USE_BATHYMETRY | `0` |
| USE_SITE_ELEVATION | `0` |
| USE_TF32 | `1` |
| USE_TMUX | `1` |
| VAL_RATIO | `0.2` |
| WARMUP_EPOCHS | `5` |
| WARMUP_START_FACTOR | `0.1` |
| WMSE_ALPHA | `4.0` |
| WMSE_Q_LIST | `95` |
| WMSE_S | `0.10` |
| WMSE_USE_ABS | `1` |
| X_AUG | `0` |
| X_AUG_BIAS | `0` |
| X_AUG_PROB | `0` |
| X_AUG_SCALE | `0` |
| X_CLIP | `0` |
| X_NODES_PER_GRAPH | `256` |
| X_NORM | `zscore` |
| X_P_HI | `99.0` |
| X_P_LO | `1.0` |
| num_gpus | `1` |

Adam with weight decay 1e-5 is verified in `train.py`. All shared implementation files and station JSON files are checked against `provenance.json`. All actual argv tokens match P2 after replacing only LR, run tag and output directory. The saved dry-run timestamp is synthetic and is never used for real launches.

These checks establish configuration equivalence and launcher command validity. No data loading, model execution, optimizer step, training or convergence check was performed.
