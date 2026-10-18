# StormSurge PBS launchers

## Architecture ablation on Derecho

`architecture_ablation/` contains nine history directories (`0H`, `6H`, `12H`,
`18H`, `24H`, `30H`, `36H`, `42H`, `48H`). Each contains exactly four PBS scripts:
`run_derecho_Boston.pbs`, `run_derecho_CBBT.pbs`, `run_derecho_Lewes.pbs`, and
`run_derecho_Battery.pbs`.

Each job runs its station's eight configurations sequentially:
`GNN_MLP`, `GNN_LSTM`, `GNN_GRU`, `GNN_Attn`, `CNN_MLP`, `CNN_LSTM`, `CNN_GRU`,
`CNN_Attn`. Every run uses all four GPUs through
`python -m torch.distributed.run --nproc_per_node=4`, as constructed by `train.sh`.
This gives 36 jobs and 288 training runs. A failed run stops its station job.

| Setting | Value |
| --- | --- |
| System / queue | Derecho / `main@desched1` |
| Project | `ULHI0005` |
| Priority | `economy` |
| Resources per job | 1 GPU node, 64 CPUs, 487 GB RAM, 4 A100 GPUs |
| Walltime per job | `12:00:00` for all eight runs together |
| Python | `/glade/work/zel/conda-envs/derecho-ml/bin/python` |
| CUDA module | `cuda/12.9.0` |
| Execution | `USE_TMUX=0`, four DDP ranks, one GPU per rank |

The shared `architecture_ablation/run_station.sh` sets both `PYTHON_BIN` and
`S0_PYTHON_BIN` so the configurations use the Derecho environment. It checks all
config paths and, inside the batch job, checks that four CUDA GPUs and NCCL are
available before training. The existing BF16/TF32, batch size, optimizer,
model, loss, and data settings come from the original configuration files.

From the repository root, preview a job without loading data, creating training
outputs, or submitting to PBS:

```bash
cd /glade/u/home/zel/scratch/StormSurge
DRY_RUN=1 bash PBS-launcher/architecture_ablation/0H/run_derecho_Boston.pbs
```

Submit one station/history job:

```bash
qsub PBS-launcher/architecture_ablation/0H/run_derecho_Boston.pbs
```

Submit all 36 jobs:

```bash
for history in 0H 6H 12H 18H 24H 30H 36H 42H 48H; do
  for station in Boston CBBT Lewes Battery; do
    qsub "PBS-launcher/architecture_ablation/${history}/run_derecho_${station}.pbs"
  done
done
```

PBS output and errors are combined in each history directory under a unique
job-name/job-ID filename. Each training run retains its own `launcher.log`,
training log, resolved config, checkpoints, and `exit_status` beneath the
configured `../PACT_StormSurge_FormalRuns` output root. Job walltime covers all
eight runs; PBS terminates a job if it reaches that limit.

Resource and submission references:
[Derecho GPU job examples](https://ncar-hpc-docs.readthedocs.io/en/latest/pbs/job-scripts/derecho-job-script-examples/),
[queue limits and priority](https://ncar-hpc-docs.readthedocs.io/en/latest/pbs/charging/),
[server-qualified queue names](https://ncar-hpc-docs.readthedocs.io/en/latest/pbs/peer-scheduling/).
