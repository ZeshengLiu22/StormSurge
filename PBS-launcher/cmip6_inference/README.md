# CMIP6 future inference on Derecho or Casper

This directory implements 432 independent evaluations: nine CMIP6 sources × four
stations × seeds 42/123/2026 × CNN_GRU_MSE, PACT_MSE, PACT_Peak, and PACT_Both.
Each checkpoint evaluates its own GCM's 30 future winters, generated with
`range(2070, 2100)`, from `2070_2071` through `2099_2100`. Both scheduling options
execute this identical matrix into identical deterministic destinations.
**Choose one cluster strategy; do not submit both simultaneously.**

The existing `infer.py`, models, pressure handling, normalization, Q95, hourly
episode definitions, and aggregate/per-year metrics are reused unchanged. No
training or future-data statistics fitting occurs. Architecture, history length,
station features, and pressure representation are recovered from each checkpoint.
`--root_dir` records the original historical training graph root. Only
`--test_root_dir` is loaded, so the historical data can be unmounted.

## Files and paths

- `generate.py`: validates inputs before generating all configs, PBS files, result
  directories, and `manifest.json`. Each manifest row pins an absolute
  `best_overall.pt` path, its SHA256, saved statistics, original source root,
  future root, config, output destination, and both assigned PBS files.
- `common.py`: checkpoint resolver, metadata and output validation, graph
  inventory, resource estimates, runtime settings, and locking.
- `station_runner.py`: twelve sequential `infer.py` processes on one isolated GPU.
- `launch.py` / `run_job.sh`: PBS environment, independent station workers,
  signal forwarding, and collection of every worker's exit status.
- `preflight.py`: CPU-only validation, including completed result validation.
- `probe_memory.py`: optional one-year CPU memory measurement.
- `submit_derecho.sh` / `submit_casper.sh`: optional submission helpers; default
  behavior prints commands. Actual submission requires `--submit`.
- `tests/test_launchers.py`: synthetic CPU tests, including an actual 30-year
  `infer.py` evaluation, restart/corruption checks, GPU-mask routing, and signals.

Repository: `/glade/u/home/zel/scratch/StormSurge`.
Checkpoint root:
`/glade/u/home/zel/scratch/PACT_StormSurge_FormalRuns_1021_LR5e-3_Clip0.25/cmip6`.
Result root is its sibling `cmip6_inference`.

The inspected dataset root `StormSurge/dataset` is a symlink to
`/glade/u/home/zel/scratch/PACT-Data-Final`. Actual files follow
`<CMIP6>/graphs/2070_2071_Battery_peryear_hist48_graphs.pt`. There are 30 selected
files per station and 1,080 selected files overall. Filename filtering checks
station and year before any deserialization. Historical files are ignored.

The 432 assignment-only shell configs live under
`configs/cmip6_inference/<CMIP6>/<Station>/seed<seed>/infer_config_<Model>.sh`.
They expose the pinned manifest values for inspection; the station runner reads
the manifest directly. Do not run these through `infer.sh`, whose launcher
defaults are unnecessary here. Regenerate the configs and manifest together
instead of changing a config alone; preflight detects inconsistent generated files.

## CPU preflight and preview

From the repository root:

```bash
cd /glade/u/home/zel/scratch/StormSurge
export PYTHON_BIN=/glade/work/zel/conda-envs/derecho-ml/bin/python
export PYTHONDONTWRITEBYTECODE=1

"$PYTHON_BIN" PBS-launcher/cmip6_inference/preflight.py
"$PYTHON_BIN" -m unittest discover -s PBS-launcher/cmip6_inference/tests -v

DRY_RUN=1 bash PBS-launcher/cmip6_inference/derecho/CMIP6_AWI/run_inference.pbs
DRY_RUN=1 bash PBS-launcher/cmip6_inference/casper/CMIP6_AWI/Battery/run_inference.pbs
```

Preflight checks all 432 checkpoints using CPU deserialization, the repository's
checkpoint architecture normalization and strict state-dict loading. It verifies
the saved station, seed, model, loss variant, TRAIN normalization tensor shapes
and scales, Q95 metadata, checkpoint role, and training completion evidence.
It checks all graph inventories, paths, unique destinations, counts, exact PBS
resource templates, Bash syntax, and importable Python components. It does not
load graph tensors, touch CUDA, create result outputs, or submit jobs.
`--structural-only` skips checkpoint deserialization and hashing for a faster
structural check. `--report PATH` explicitly saves a JSON report.

PBS `DRY_RUN=1` prints 48 commands for a Derecho job or 12 for a Casper job.
It bypasses modules, GPU checks, and output creation. Real runners reject
execution outside a PBS allocation. The existing pre-deserialization year
filter in `ForcingGraphStore` is unchanged. Every command includes
`--years`, `--strict_years`, and `--save_npz`.

## Checkpoint selection and regeneration

The inspected tree contains one completed `best_overall.pt` per combination.
A completed training run must have `exit_status` equal to `0` and one final
`summary_*.json` whose selected role is `overall`. The checkpoint's saved selected
epoch must match the summary. Missing, incomplete, wrong-role, or incompatible
checkpoints fail validation. Neither `best.pt` nor `best_exceedance.pt` is used.

```bash
"$PYTHON_BIN" PBS-launcher/cmip6_inference/generate.py
```

If additional completed runs appear, generation fails on ambiguity. Preserve
the current explicit selection with:

```bash
"$PYTHON_BIN" PBS-launcher/cmip6_inference/generate.py \
  --reuse-manifest PBS-launcher/cmip6_inference/manifest.json
```

Alternatively pass `--selection selection.json`. Its format is a JSON object
mapping `CMIP6_AWI/Battery/seed42/PACT_Both` (and any other ambiguous combinations)
to each chosen absolute `.../best_overall.pt` path. No newest-file selection
exists. Existing manifests remain pinned if other training runs appear.

Generation supports `--repo`, `--checkpoint-root`, `--data-root`, `--result-root`,
`--derecho-walltime`, and `--casper-walltime`. Install the shared runner files
alongside generated PBS scripts if using a different repository. Regenerate only
when jobs are stopped: changed checkpoint hashes or graph inventories invalidate
the pinned inputs and require review before resubmission.

## Resource allocation and environment

| Strategy | Jobs | Per-job request | Initial walltime |
| --- | ---: | --- | --- |
| Derecho | 9 | `select=1:ncpus=64:mpiprocs=4:ompthreads=1:ngpus=4:mem=487GB` | `01:00:00` |
| Casper | 36 | `select=1:ncpus=32:mpiprocs=1:ngpus=1:gpu_type=a100_80gb:mem=240GB` | `02:00:00` |

Both use account `ULHI0005`, matching the existing formal training scripts, and
`#PBS -r n`. Derecho uses `preempt@desched1`; Casper uses `casper`.
Casper's `a100_80gb` selector targets its four-way A100 80GB nodes; the 32-CPU,
240GB request permits four independently scheduled jobs to share a node.
See [NCAR's Casper node table](https://ncar-hpc-docs.readthedocs.io/en/latest/compute-systems/casper/casper-node-types/).

Derecho maps the four tokens of the inherited `CUDA_VISIBLE_DEVICES`, in order,
to Battery, Boston, CBBT, and Lewes. Tokens can be numeric device IDs or GPU UUIDs.
When the variable is unset on its exclusive node, the tokens are 0,1,2,3.
Empty, duplicate, or non-four-device masks fail. Each worker receives one token,
sees exactly one CUDA device, and uses local `cuda:0`. There is no DDP,
`torchrun`, or distributed inference. Casper preserves the scheduler mask
unchanged, then checks for exactly one visible A100 with at least 75 GiB VRAM.

Python defaults to `/glade/work/zel/conda-envs/derecho-ml/bin/python`; no new
environment is created. CPU checks on Casper `crlogin1` imported PyTorch
`2.8.0+cu126` (bundled CUDA 12.6) and PyG `2.8.0.post1`. The available/default
module there is `cuda/12.9.0`, also used by the existing successful Derecho
training launchers. Its module file was verified in both clusters' `25.10/Core`
module trees. The shared shell loads this module; `CUDA_MODULE=none`
skips loading, or `CUDA_MODULE=...` selects another installed module.
PyTorch's bundled CUDA version need not equal the toolkit module version.
The allocated-node driver must support its runtime. Each station checks a tiny
CUDA operation and BF16 support before inference. Actual driver, GPU memory,
account/queue access, throughput, and total runtime still need an allocated-node
pilot on each cluster. No GPU allocation or PBS submission was made during setup.

Runtime overrides (environment variables):

| Variable | Default |
| --- | --- |
| `PYTHON_BIN` | existing `derecho-ml` Python |
| `INFER_BATCH_SIZE` | `32` |
| `INFER_AMP` | `1` |
| `INFER_AMP_DTYPE` | `bf16` (also accepts `fp16`) |
| `INFER_TF32` | `1` |
| `INFER_NUM_WORKERS` | `0` |
| `INFER_TORCH_THREADS` | `1` (also sets OMP/MKL/OpenBLAS threads) |

Worker prefetch and pinned memory are left at the existing non-enabled settings.
Defaults are conservative for Derecho's GPU memory. The shared environment
removes inherited distributed-rank variables. Submission helpers explicitly
export these overrides with `qsub -v`; they do not export the login shell's
GPU mask. Override walltime using helper `--walltime HH:MM:SS` or
`qsub -l walltime=HH:MM:SS`; PBS directives do not expand shell variables.

## CPU memory evidence

The largest source is EC-EARTH, with about 19.91 GiB of serialized graph data per
station for 30 winters. A raw-pressure probe of its `2070_2071` Battery file
loaded 600 graphs of history shape `[9, 6708, 5]`: 0.676 GiB serialized and about
1.87 GiB process RSS including imports and conversion. Reproduce with:

```bash
"$PYTHON_BIN" PBS-launcher/cmip6_inference/probe_memory.py
```

Preflight budgets `3 × selected serialized GiB + 8 GiB` per station, multiplied
by `1 + num_workers` to account conservatively for loader copies. The largest
estimate is 67.74 GiB/station, 270.97 GiB for four workers, below the 487GB Derecho
request and 240GB Casper request. Settings that exceed 80% of the station budget
are rejected. Zero workers avoids graph-list replication. The estimate is a
sizing guard, not a measured full-run peak; check actual PBS `resources_used.mem`
after the pilot. The evidence does not require streaming changes to core code.
The existing inference path concatenates physical predictions and recomputes
overall/hourly-episode metrics on the full population. It never averages yearly
RMSEs to produce the aggregate.

## Submit one strategy

Preview (does not call `qsub`):

```bash
bash PBS-launcher/cmip6_inference/submit_derecho.sh
bash PBS-launcher/cmip6_inference/submit_casper.sh
```

Submit **either** all 9 Derecho jobs:

```bash
bash PBS-launcher/cmip6_inference/submit_derecho.sh --submit
```

**Or** all 36 Casper jobs:

```bash
bash PBS-launcher/cmip6_inference/submit_casper.sh --submit
```

A Casper pilot and a larger batch override, for example:

```bash
bash PBS-launcher/cmip6_inference/submit_casper.sh \
  --source CMIP6_EC_EARTH --station Battery --walltime 02:00:00 --submit
INFER_BATCH_SIZE=16 bash PBS-launcher/cmip6_inference/submit_derecho.sh \
  --walltime 02:00:00 --submit
```

The helper runs full CPU preflight before submission, stores successful job IDs
immediately under `<result_root>/.submissions/jobs.json`, and checks recorded
jobs with `qstat` before permitting another batch. It refuses submission while
any recorded inference job is active, including jobs from the other strategy.
An uncertain `qstat` response fails closed. Do not bypass this guard by manually
submitting both PBS trees. Per-result locks also prevent concurrent writes.

## Monitoring, completion, and restart

```bash
qstat -u "$USER"
qstat -f JOBID
"$PYTHON_BIN" PBS-launcher/cmip6_inference/preflight.py --completed
```

PBS output goes to each PBS directory's `pbs_logs/`. Each execution also creates
`runlogs/<jobid>_<attempt>/`, containing independent station logs and a job status
JSON with every station's exit code. Every station runs its 12 combinations in
seed/model order. A failed combination is logged; independent later combinations
continue. Any failure makes the station and its PBS job exit nonzero.

Each deterministic result directory contains:

```text
<CMIP6>/<Station>/seed<seed>/<Model>/
  outputs/metrics.json
  outputs/metrics.md
  outputs/metrics_per_year_*.json
  outputs/predictions.npz
  outputs/config.json
  logs/<attempt>.log
  provenance.json
  status.json
  COMPLETE.json
  .run.lock
```

An advisory `flock` protects each result; lock files are persistent and should
not be deleted. The inference subprocess inherits the lock descriptor so an
unexpected parent SIGKILL cannot release its lock while inference still writes.
Outputs first go into `.attempt-<id>/outputs`, then are fully validated and
renamed atomically into `outputs/`. The completion marker is published last.
It includes output checksums and a fingerprint of the exact checkpoint, graph
inventory, combination, and runtime settings. Validation reads all NPZ members
and verifies CRCs, shapes, finite predictions, sample counts, station/year tags,
exactly 30 years, and `source_checkpoint_train` origins in both JSON and NPZ.

On SIGTERM, the supervisor signals every station, each station terminates its
inference process group, and shutdown waits are bounded. Previously completed
combinations stay intact. Failed/interrupted staging directories and separate
attempt logs remain for inspection; they are never treated as completed.
On restart, only markers whose metadata and output checksums validate are
skipped. Incomplete/corrupt outputs are recomputed; old finalized outputs are
retained in `previous_outputs/` before replacement. Changing runtime settings
also changes the fingerprint and triggers recomputation.

After the earlier batch finishes or is preempted, rerun the same submission
command. The output paths stay the same. A filesystem lock conflict reports a
failure instead of silently skipping the combination. A hard kill may leave
`status.json` saying `running`; absence of a verified marker still means pending.

For checkpoint or graph mismatches, inspect the named path, resolve the correct
completed checkpoint explicitly if needed, and regenerate while jobs are stopped.
For GPU allocation/runtime failures, inspect the worker's startup report and
the PBS allocation; do not replace Casper's GPU mask manually. For GPU OOM,
reduce `INFER_BATCH_SIZE`; for CPU RAM pressure keep `INFER_NUM_WORKERS=0` and
check the probe and PBS memory report. For timeout/preemption, resubmit with a
larger walltime if appropriate. Already verified runs remain available.
