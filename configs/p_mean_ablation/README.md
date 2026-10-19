# NCEP pressure-mode ablation

Twelve assignment-only training configs compare `anomaly`, `raw`, and
`anomaly_plus_mean` at Battery, Boston, CBBT, and Lewes. Every run uses the
`CNN_GRU_36H` PACT candidate (`MODEL=perceiver3`) and seed 42.

Each station's reference is its seed-42 config in
`configs/architecture_avgrank_multiseed/CNN_GRU_36H/`, which retains the
architecture-ablation training settings. Only the pressure representation,
experiment names, and output destinations differ. Settings include:

- CNN intermediate width 29, hidden width 128, two spatial layers, GRU,
  and a 36-hour forcing history.
- Chronological 60/20/20 TRAIN/VAL/TEST split and TRAIN-only z-score statistics.
- Batch size 256 per rank, four GPUs, 300 epochs, learning rate `5e-3`, cosine
  schedule, and five warmup epochs.
- Global physical MSE with tail and episode-peak weights both zero; primary
  checkpoint selection `overall`, retaining the `exceedance` checkpoint too.
- BF16 AMP and TF32 enabled; feature clipping, augmentation, site elevation,
  and bathymetry disabled, matching the reference configs.

Configs are stored as:

```text
configs/p_mean_ablation/<pressure_mode>/<station>/
  train_config_NCEP_<station>_CNN_GRU_36H_p<pressure_mode>_seed42_G0_T0000_EP0000.sh
```

Outputs go to:

```text
/glade/u/home/zel/scratch/PACT_StormSurge_FormalRuns/pressure_mode_ablation/
  <pressure_mode>/<station>/
    NCEP_<station>_CNN_GRU_36H_p<pressure_mode>_seed42_G0_T0000_EP0000__<timestamp>/
```

The data root remains `./dataset/NCEP/graphs`. Non-default modes require
`p_mean_hist` for every stored history timestep, with a consistent `p_mean_curr`
when present. Missing metadata raises an error. Existing graph files are used
without preprocessing. Raw pressure and the appended mean feature use the stable
FP64 variance calculation during z-score fitting; inference reuses saved statistics.

From the StormSurge repository root, inspect all twelve commands without starting
training:

```bash
for config in configs/p_mean_ablation/*/*/train_config_*.sh; do
  DRY_RUN=1 USE_TMUX=0 bash train.sh "$config" || break
done
```

Run one config inside an allocated four-GPU Derecho job:

```bash
PYTHON_BIN=/glade/work/zel/conda-envs/derecho-ml/bin/python USE_TMUX=0 \
  bash train.sh configs/p_mean_ablation/raw/Battery/train_config_NCEP_Battery_CNN_GRU_36H_praw_seed42_G0_T0000_EP0000.sh
```

The configs retain the launcher's usual optional tmux behavior; `USE_TMUX=0`
keeps a scheduler or sequential runner attached until training finishes.
