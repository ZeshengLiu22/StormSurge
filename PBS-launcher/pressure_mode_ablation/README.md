# Station-grouped pressure-mode jobs

Each of the four PBS scripts runs one station's `CNN_GRU_36H`, seed-42 configs
sequentially in this order: `anomaly`, `raw`, `anomaly_plus_mean`. Each run uses
all four GPUs, and a failure stops the station job. The four stations are
Battery, Boston, CBBT, and Lewes: four PBS jobs cover twelve training runs.

Resources match the existing `architecture_avgrank_multiseed/CNN_GRU_36H`
launchers: project `ULHI0005`, queue `main@desched1`, economy priority, one node
with 64 CPUs, 487 GB RAM, and four GPUs. The two-hour walltime covers all three
runs in a station job. The scripts use `derecho-ml`, `cuda/12.9.0`, and
`USE_TMUX=0`, retaining the original training settings from
`configs/p_mean_ablation/`.

From the repository root, preview all four jobs without submitting or training:

```bash
cd /glade/u/home/zel/scratch/StormSurge
for station in Battery Boston CBBT Lewes; do
  DRY_RUN=1 bash "PBS-launcher/pressure_mode_ablation/run_derecho_${station}.pbs" || break
done
```

Submit all four station jobs:

```bash
for station in Battery Boston CBBT Lewes; do
  qsub "PBS-launcher/pressure_mode_ablation/run_derecho_${station}.pbs" || break
done
```

PBS stdout/stderr logs are combined under this launcher directory. Training
results retain each mode's and station's separate output path:

```text
/glade/u/home/zel/scratch/PACT_StormSurge_FormalRuns/pressure_mode_ablation/
  <pressure_mode>/<station>/<run_name>__<timestamp>/
```

Every config is checked before the first run. Real jobs also check that four
CUDA GPUs and NCCL are available. As specified by the configs, the two
non-default modes require existing `p_mean_hist` metadata in the graph files.
