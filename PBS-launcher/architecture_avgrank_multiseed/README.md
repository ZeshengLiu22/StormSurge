# Architecture AvgRank three-seed study

This isolated study contains 96 new training configs and eight independent Derecho
PBS jobs. Every job runs Battery, Boston, CBBT and Lewes in that order, with seeds
42, 123 and 2026 consecutively within each station: 12 sequential runs, each using
all four A100 GPUs for 300 epochs. Seed 42 is a fresh run in the new result tree.

| Group | Architecture | Four-station minutes/seed | Three-seed minutes | Walltime | PBS job name |
| --- | --- | ---: | ---: | --- | --- |
| A | CNN_GRU_48H | 46.557 | 139.671 | 03:30:00 | arA_GRU48 |
| A | CNN_LSTM_42H | 42.606 | 127.818 | 03:15:00 | arA_LSTM42 |
| A | CNN_GRU_30H | 35.194 | 105.582 | 02:45:00 | arA_GRU30 |
| C | CNN_Attn_48H | 46.727 | 140.181 | 03:30:00 | arC_Attn48 |
| C | CNN_MLP_36H | 37.910 | 113.730 | 03:00:00 | arC_MLP36 |
| C | CNN_LSTM_30H | 35.193 | 105.579 | 02:45:00 | arC_LSTM30 |
| C | CNN_Attn_24H | 33.087 | 99.261 | 02:45:00 | arC_Attn24 |
| C | CNN_LSTM_24H | 32.251 | 96.753 | 02:45:00 | arC_LSTM24 |

Group A uses LR=3e-3 with no clipping; Group C uses LR=5e-3 with clipping=0.25.
Configs copy their exact station/history/architecture references from
`configs/architecture_ablation_LR3e-3` and
`configs/architecture_ablation_LR5e-3_Clip0.25`. Only `SEED`, `ALL_RESULTS_ROOT`,
`PACT_RUN_NAME`, `PYTHON_RUN_TAG_BASE` and documentation comments differ.
Attention retains `TEMPORAL_BLOCK="Transformer"`. All configs retain the CNN
encoder, NCEP data, raw pressure, Global MSE, math attention backend, module weight
decay policy, overall checkpoint selection, and the reference precision and
scheduler settings. The 0.6/0.2 chronological year splits, `SHUFFLE_YEARS=0`,
`FUTURE_ONLY=0` and `FUTURE_YEAR_THRESHOLD=2030` are identical across all seeds and
stations. The existing split implementation uses the seed only when shuffling.

## Files and results

Paths below are relative to `/glade/u/home/zel/scratch/StormSurge`:

- 36 configs: `configs/architecture_avgrank_multiseed_LR3e-3/`.
- 60 configs: `configs/architecture_avgrank_multiseed_LR5e-3_Clip0.25/`.
- Three PBS scripts and one group wrapper: `PBS-launcher/architecture_avgrank_multiseed_LR3e-3/`.
- Five PBS scripts and one group wrapper: `PBS-launcher/architecture_avgrank_multiseed_LR5e-3_Clip0.25/`.
- Shared runner, generator/validator, CPU-only behavior tests and this guide:
  `PBS-launcher/architecture_avgrank_multiseed/`.

Config hierarchy: `<Architecture>/<Station>/seed<SEED>/<config.sh>`.
PBS hierarchy: `<Architecture>/run_derecho.pbs`.
Each PBS output destination `<Architecture>/pbs_logs/` exists before submission;
tracked `.gitkeep` files preserve these directories in future checkouts.

Results are created by the unchanged `train.sh` only during actual execution:

```text
/glade/u/home/zel/scratch/PACT_StormSurge_FormalRuns_1021_LR3e-3/architecture_avgrank_multiseed/<Architecture>/<Station>/seed<SEED>/
/glade/u/home/zel/scratch/PACT_StormSurge_FormalRuns_1021_LR5e-3_Clip0.25/architecture_avgrank_multiseed/<Architecture>/<Station>/seed<SEED>/
```

`RUN_DIR_NAME_STYLE="runname_timestamp"` preserves a fresh, uniquely named run
directory inside each seed folder. Both run-name fields include `_seed<SEED>`.
Existing results are never checked for reuse, skipped, retried, or overwritten.

## Validate without training

From the repository root:

```bash
python3 PBS-launcher/architecture_avgrank_multiseed/experiments.py validate --dry-run
PYTHONDONTWRITEBYTECODE=1 python3 PBS-launcher/architecture_avgrank_multiseed/test_launchers.py
DRY_RUN=1 bash PBS-launcher/architecture_avgrank_multiseed_LR3e-3/CNN_GRU_48H/run_derecho.pbs
```

Validation checks the exact 96-config matrix, source equivalence, result roots,
fixed splits, Bash syntax, eight PBS definitions, pre-existing PBS log directories,
12 ordered configs per PBS, and coverage of all 96 configs exactly once. Preflight
validates all 12 configs before any training or CUDA setup. Dry-run prints the
ordered paths and exits without importing PyTorch, loading modules, checking
CUDA, creating experiment outputs, or invoking `train.sh`.

The default interpreter is
`/glade/work/zel/conda-envs/derecho-ml/bin/python`; the reference `PYTHON_BIN`
override is retained and also exported as `S0_PYTHON_BIN`. It must be executable,
including for dry-run. Behavior tests use temporary fixture repositories, a
harmless shell stub in place of `train.sh`, and mock module/CUDA probes; they do
not import PyTorch or run a model. They cover sequential success, individual
failure and continuation, preflight rejection, no-output dry-run and SIGTERM
cleanup. Their temporary scratch directories are removed afterward.

To regenerate missing config/PBS files after reviewing the matrix:

```bash
python3 PBS-launcher/architecture_avgrank_multiseed/experiments.py generate
```

Generation accepts identical files but refuses to replace differing files.

## Execution and failure records

Actual execution requires a PBS allocation. The shared runner retains
`cuda/12.9.0`, the Derecho Python environment, four-visible-GPU/NCCL checks,
single-thread limits, `USE_TMUX=0`, and the `train.sh` DDP launcher with
`--nproc_per_node=4`. It clears the two Slurm variables read by `train.sh`, and
clears inherited `PACT_RUNSTAMP` so each invocation gets a new timestamp.
`train.sh` continues to choose localhost and a random master port.

Every start and end prints group, architecture, station, seed, config path and
UTC timestamps; every end also prints the exit code. Per-run console logs are at
`<PBS_GROUP>/<Architecture>/execution_logs/<Station>/seed<SEED>/<launch_id>.log`.
The existing per-training logs/checkpoints remain inside the result directory.

Each invocation creates a unique append-only TSV event manifest at
`<PBS_GROUP>/<Architecture>/manifests/<launch_id>.tsv`. All 12 runs begin as
`PENDING`, then acquire `RUNNING` and a terminal `SUCCEEDED`, `FAILED` or
`INTERRUPTED` record. The last event for a config is its current state.
Runs left after an aborted sequence receive `NOT_RUN`; an incomplete manifest
must not be treated as a successful three-seed experiment. A failed run is never
retried; after its process group exits, the runner continues and ultimately exits
nonzero if any run failed. Existing successful outputs remain intact.

Each training invocation has its own process group using `setsid`. On SIGTERM or
SIGINT, the runner signals the entire active group, waits up to ten seconds,
force-stops survivors, reaps its child, records interruption, and exits 143 or 130.
Residual processes after a normal child exit also trigger cleanup and failure;
the sequence stops if cleanup cannot be confirmed. The PBS scripts use `exec`
so the batch shell does not conceal the runner's status or signals.

Compared with `architecture_ablation/run_station.sh`, this runner groups jobs by
architecture and iterates station/seed; adds reference preflight, read-only
dry-run, manifests, per-seed logs, continuation on individual failure, and whole
process-group cleanup. PBS resources use the requested `main` queue, economy
priority, account `ULHI0005`, walltimes and `-r n`. No PBS arrays are used.
The model, `train.sh`, `train.py`, source configs and old PBS scripts are unchanged.

## Submission commands (manual; not executed during preparation)

```bash
cd /glade/u/home/zel/scratch/StormSurge
qsub PBS-launcher/architecture_avgrank_multiseed_LR3e-3/CNN_GRU_48H/run_derecho.pbs
qsub PBS-launcher/architecture_avgrank_multiseed_LR3e-3/CNN_LSTM_42H/run_derecho.pbs
qsub PBS-launcher/architecture_avgrank_multiseed_LR3e-3/CNN_GRU_30H/run_derecho.pbs
qsub PBS-launcher/architecture_avgrank_multiseed_LR5e-3_Clip0.25/CNN_Attn_48H/run_derecho.pbs
qsub PBS-launcher/architecture_avgrank_multiseed_LR5e-3_Clip0.25/CNN_MLP_36H/run_derecho.pbs
qsub PBS-launcher/architecture_avgrank_multiseed_LR5e-3_Clip0.25/CNN_LSTM_30H/run_derecho.pbs
qsub PBS-launcher/architecture_avgrank_multiseed_LR5e-3_Clip0.25/CNN_Attn_24H/run_derecho.pbs
qsub PBS-launcher/architecture_avgrank_multiseed_LR5e-3_Clip0.25/CNN_LSTM_24H/run_derecho.pbs
```

## GPU-hour estimate

Measured-runtime estimate = sum of supplied four-station minutes × 3 seeds × 4 GPUs / 60.
The economy column applies the requested 0.7 factor to that estimate.

| Basis | Group A GPU-hours | Group C GPU-hours | Total GPU-hours | Total × 0.7 |
| --- | ---: | ---: | ---: | ---: |
| Supplied measured runtimes | 24.8714 | 37.0336 | 61.9050 | 43.3335 |
| Full requested walltimes | 38.0000 | 59.0000 | 97.0000 | 67.9000 |

The walltimes already cover all three seeds; four GPUs work concurrently and
do not multiply elapsed walltime. Runtime variability and actual completed job
durations determine consumption.
