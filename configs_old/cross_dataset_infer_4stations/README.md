# Four-station PACT cross-dataset inference

Canonical station tokens are `Boston`, `CBBT`, `Lewes`, and `Battery`, verified
against station JSON, training snapshots, split tags, and checkpoint metadata.
This workflow uses only these production roots:

- Project: `/glade/u/home/zel/scratch/PACT-Derecho/PACT/StormSurge/`
- Data: `/glade/u/home/zel/scratch/PACT-Derecho/PACT/Data/`
- Checkpoints: `/glade/u/home/zel/scratch/PACT-Derecho/PACT/FormalRuns_0925/`
- Results: `/glade/u/home/zel/scratch/PACT-Derecho/PACT/FormalRuns_0925/CrossDatasetInference_4Stations/`

`~/scratch` resolves to `/glade/derecho/scratch/zel`. Resolved paths in metadata
and the manifest refer to the same GLADE files. Pair source and target roots are
absolute; no repository-local `Data` symlink is needed. The generator maps original
training data paths to the same exact Data-relative dataset identity when auditing
relocated checkpoints. Training snapshots and checkpoint contents remain intact.
The old `configs/cross_dataset_infer` and `CrossDatasetInference_dev_BostonONLY`
are read-only references and are never written by this workflow. Exact source
checkpoint and training-snapshot hashes are retained in the checkpoint audits.

## Protocol

One `infer.py` invocation evaluates one source checkpoint against one explicitly
selected target dataset, one station, and one fixed year list. Matrix orchestration
and aggregation are outside `infer.py`. There are no station or matrix loops in
the pair inference implementation.

| Period | Sources | Targets | Years | Pairs/station | Four stations |
| --- | --- | --- | --- | ---: | ---: |
| past_only | NCEP, AWI, CNRM, EC_EARTH, MPI, MRI | same six | 2008_2009 through 2014_2015 (7 groups) | 36 | 144 |
| future_year | same six | AWI, CNRM, EC_EARTH, MPI, MRI | 2070_2071 through 2099_2100 (30 groups) | 30 | 120 |

Total: **66 pairs per station, 264 pairs**. NCEP is source-only in the future
matrix. Diagonal cells are rerun through `TEST_ROOT_DIR="${TARGET_ROOT}"` and the
same explicit years as off-diagonal cells. Stored training-test scores are never
substituted.

The source supplies model configuration, model weights, normalization, station
features, and `threshold_metadata["tau_physical"]`. Target graphs are transformed
with source TRAIN statistics. The fixed source TRAIN hourly Q95 defines extreme
and episode populations; `evaluation_threshold_origin=source_checkpoint_train`.
No target normalization or threshold is fitted. Predictions are denormalized
using source `y_std` and `y_mean`; graph targets are already physical values.
Final errors and AllRMSE are in meters.

The current repository's existing missing-label behavior is retained: hourly
metrics exclude nonfinite target hours; episode metrics exclude their entire
windows. No metric, architecture, loss, or calibration semantics were changed
for this workflow. Post-hoc summaries record valid/missing target-hour counts
without recomputing metric values.

The candidate is `G0_T0500_EP0100`: global MSE, Tail MSE weight 0.05, and
GT-aligned Episode Peak MSE weight 0.01. Only exact absolute `best_overall.pt`
paths are generated. Architecture: perceiver3/PACT, GraphSAGE, Transformer,
single head, 24 h history; site elevation and bathymetry are disabled.
Casper defaults are `USE_AMP=1`, `AMP_DTYPE=fp16`, `USE_TF32=0`, batch size 64,
one Torch thread, zero loader workers, and disabled dual diagnostics. V100 has
FP16 Tensor Cores; native BF16 and TF32 require newer GPU architectures. Batch
and precision may be overridden explicitly, e.g. `BATCH_SIZE=32` or `USE_AMP=0`.
FP16 results can differ slightly from the original H100 BF16 results.

`STRICT_YEARS=1` requires exactly the requested 7 or 30 year groups. Each output
records requested/evaluated years and counts, station, source/target, checkpoint,
normalization origin, source threshold, and architecture.

## Checkpoint and data audits

`generate_configs.py` audits all 48 station/period/source roles before writing
any pair configs. `checkpoint_audit.csv` and `checkpoint_audit.json` contain
every exact checkpoint path, SHA-256, epoch, model, normalization arrays, threshold
metadata, split years/counts, training snapshot provenance, and completion evidence.
`manifest.csv` contains exactly 264 rows and is not used by `infer.py`.

Active source areas are `single_tail_episodepeak_4x4`, `NCEP_future_transfer`, and
the four `CMIP6_<station>_QuickCheck_0929` trees. Discovery rejects missing or
multiple matching candidates instead of selecting a wildcard's first result.
All selected checkpoints must match the full training run's best VAL AllRMSE epoch.

Two Lewes training runs have `exit_status=1` from final TEST reporting after all
300 epochs completed: past-only MRI (best epoch 288), and future-year EC_EARTH
(best epoch 261). Their logs end with the old `Cannot evaluate nonfinite y_true`
error after `FINAL MULTI-CHECKPOINT RE-EVALUATION`. The audit preserves exit=1,
verifies all 300 records and the saved full-run VAL optimum, and records log
hashes. This narrow provenance rule is restricted to those exact run names;
other nonzero exits fail. The existing repository commit `dece60a` already handles
missing labels. No checkpoint is retrained, modified, or replaced.

Expected split provenance is verified from checkpoint tags:

- Past-only, all sources: TRAIN 1979–2000 (22 groups), VAL 2001–2007 (7), TEST 2008–2014 (7).
- Future NCEP: TRAIN 1979–2008 (30), VAL 2009–2014 (6), TEST empty.
- Future CMIP6: TRAIN 1979–2008 (30), VAL 2009–2014 (6), TEST 2070–2099 (30).

An empty source TEST split is valid with an explicit external target root.
Without that root, the existing clear no-held-out-TEST error is retained.

`audit_graphs.py` verifies 44 station/period/target batches through the actual
store, view, and PyG loader. Target edges and node counts remain intact;
normalization has five feature entries, independent of node count. GraphSAGE
does not use a source grid shape, and PACT uses target `to_dense_batch` masks.

## Commands

Run from the repository root:

```bash
cd /glade/u/home/zel/scratch/PACT-Derecho/PACT/StormSurge
PYTHON=/glade/work/zel/conda-envs/pact-casper-cu126/bin/python

# Audit and deterministically regenerate the pinned production configs.
"$PYTHON" configs/cross_dataset_infer_4stations/generate_configs.py
# Re-audit and compare generated files without writing.
"$PYTHON" configs/cross_dataset_infer_4stations/generate_configs.py --check
"$PYTHON" configs/cross_dataset_infer_4stations/audit_graphs.py

# One pair.
bash infer.sh configs/cross_dataset_infer_4stations/Boston/past_only/NCEP/NCEP_to_AWI.sh

# One station/period/source group (6 or 5 pairs).
bash configs/cross_dataset_infer_4stations/run_group.sh configs/cross_dataset_infer_4stations/Lewes/future_year/NCEP

# One station (66 pairs).
bash configs/cross_dataset_infer_4stations/run_station.sh Boston

# All four stations, sequentially (264 pairs), followed by the complete summary.
bash configs/cross_dataset_infer_4stations/run_all.sh && \
  "$PYTHON" configs/cross_dataset_infer_4stations/summarize_results.py

# Resolve all configs without creating inference outputs.
DRY_RUN=1 bash configs/cross_dataset_infer_4stations/run_all.sh

# Rebuild summaries from raw production artifacts only.
"$PYTHON" configs/cross_dataset_infer_4stations/summarize_results.py
```

`run_group.sh` sorts target configs, checks every config in the group before
starting, and stops immediately with the exact failed pair if inference fails.
Station order is Boston, CBBT, Lewes, Battery. Each station runs past-only then
future-year; source order is NCEP, AWI, CNRM, EC_EARTH, MPI, MRI. Targets run in
lexical order; matrix display uses the explicit source/target order above.
Every group, station, and all-stations runner supports `DRY_RUN=1`.

There is no resume, skip-completed, background scheduling, or automatic retry.
`run_all.sh` refuses an existing production station directory, so it cannot mix
another full rerun into this tree. The single-pair command remains available for
an explicit rerun; the summarizer rejects duplicate pair outputs for review.

Three isolated one-year configs live in `smoke/`. `Boston_past_only_NCEP_to_AWI.sh`
checks a real cross-dataset transfer for `2008_2009`. Two compatibility configs exercise
Lewes MRI `2013_2014` (one missing target hour) and EC_EARTH `2090_2091` (two
missing target hours). Smoke outputs live under the required result root's
`smoke/` directory and are excluded from production matrices. The 264 production
configs retain their full year lists.

## Casper environment and tmux procedure

The dedicated environment is `/glade/work/zel/conda-envs/pact-casper-cu126`.
It is cloned from the existing `torchpyg-cu126` environment using `conda create
--offline --clone`, with Python 3.10, PyTorch 2.7.1 + CUDA 12.6, and PyG 2.8.0.
This Torch build includes `sm_70` for V100. The full environment export is saved in
`casper_environment.yml`; `casper_pip_freeze.txt` records pip package versions.
`common.sh` selects its interpreter directly (`DO_CONDA=0`), so activation is
optional. To activate interactively:

```bash
module load conda
conda activate /glade/work/zel/conda-envs/pact-casper-cu126
```
 An alternate interpreter can be
selected with `S0_PYTHON_BIN=/absolute/path/to/python`.

For the tmux workflow, start tmux **on the Casper login node**, then request a
GPU shell inside it. `ULHI0006` is the project used by the current V100 VS Code
allocation; replace it if using another authorized project.

```bash
# Casper login node, outside the existing one-hour VS Code compute allocation.
# Remember this login hostname; tmux attach must reach the same login host.
hostname
tmux new -s pact4

# Inside tmux: request one V100 with enough walltime for the entire fresh run.
qsub -I -A ULHI0006 -q casper \
  -l select=1:ncpus=4:mem=64GB:ngpus=1:gpu_type=v100 \
  -l walltime=24:00:00

# After PBS assigns a GPU compute node:
cd /glade/u/home/zel/scratch/PACT-Derecho/PACT/StormSurge
bash configs/cross_dataset_infer_4stations/check_casper_runtime.sh

# One pair / one year, output under .../CrossDatasetInference_4Stations/smoke/Boston/.
bash infer.sh configs/cross_dataset_infer_4stations/smoke/Boston_past_only_NCEP_to_AWI.sh

# After a successful smoke, all 264 pairs from scratch and automatic summary:
bash configs/cross_dataset_infer_4stations/run_all.sh && \
  /glade/work/zel/conda-envs/pact-casper-cu126/bin/python \
  configs/cross_dataset_infer_4stations/summarize_results.py
```

Detach with **Ctrl+B, then D**. Reconnect to the same Casper login host and run
`tmux attach -t pact4`. Tmux preserves the interactive terminal when SSH disconnects;
it does not extend the PBS allocation. The existing one-hour VS Code allocation
is suitable for a smoke check, but the full matrix needs its own longer allocation.
Casper's published maximum walltime is 24 hours. Measure the smoke time before
starting production; 264 evaluations on V100 have not been timed in full.

A batch alternative also runs all pairs sequentially and summarizes only after
success. It continues independently of tmux/SSH, under the same PBS walltime:

```bash
cd /glade/u/home/zel/scratch/PACT-Derecho/PACT/StormSurge
qsub -A ULHI0006 configs/cross_dataset_infer_4stations/run_casper.pbs
```

Choose one production launch method for a fresh result tree.

### Four separate station PBS jobs

`run_casper_Boston.pbs`, `run_casper_CBBT.pbs`, `run_casper_Lewes.pbs`, and
`run_casper_Battery.pbs` each run one station's 66 pairs sequentially: 36
past-only followed by 30 future-year. Each requests project `ULHI0006`, queue
`casper`, 1 node, 4 CPUs, 64 GB memory, 1 MPI process, 1 OpenMP thread, 1 V100,
and walltime `02:30:00`. CPU type is unrestricted. They use the existing
`pact-casper-cu126` interpreter and force FP16 AMP with TF32 disabled.

The job starts in `/glade/u/home/zel` and then changes to the project directory.
Merged PBS stdout/stderr is written under `/glade/u/home/zel/`, using PBS's
job-ID output filename. Raw inference artifacts remain under
`CrossDatasetInference_4Stations/<station>/`. Each job atomically creates only
its own fresh station directory and fails if that station already exists.
Audit/smoke artifacts and other stations do not block it. An inference failure
stops that station immediately. The benchmark estimate is about 62 minutes per
station, but four concurrent jobs can increase shared GLADE I/O contention.

Submit the four independent jobs from a Casper login node:

```bash
cd /glade/u/home/zel/scratch/PACT-Derecho/PACT/StormSurge
qsub configs/cross_dataset_infer_4stations/run_casper_Boston.pbs
qsub configs/cross_dataset_infer_4stations/run_casper_CBBT.pbs
qsub configs/cross_dataset_infer_4stations/run_casper_Lewes.pbs
qsub configs/cross_dataset_infer_4stations/run_casper_Battery.pbs
```

The existing summarizer requires all 264 pairs. Run it once after all four jobs
finish successfully; the station jobs do not write a shared partial summary:

```bash
cd /glade/u/home/zel/scratch/PACT-Derecho/PACT/StormSurge
/glade/work/zel/conda-envs/pact-casper-cu126/bin/python \
  configs/cross_dataset_infer_4stations/summarize_results.py
```

Check all four scripts without requesting GPUs or creating production results:

```bash
for station in Boston CBBT Lewes Battery; do
  DRY_RUN=1 bash "configs/cross_dataset_infer_4stations/run_casper_${station}.pbs"
done
```

References: [NCAR Casper PBS examples](https://ncar-hpc-docs.readthedocs.io/en/latest/pbs/job-scripts/casper-job-script-examples/),
[Casper queues and limits](https://ncar-hpc-docs.readthedocs.io/en/latest/pbs/charging/),
[NVIDIA mixed precision](https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/index.html).

## Pair timing benchmark (V100, FP16, batch 64)

`benchmark_runtime.py` measures one year for all 44 station/period/target types,
then checks three complete 7-/30-year pairs in separate Python processes. It
writes only under `CrossDatasetInference_4Stations/audit/`, outside the matrices.
Run in a GPU allocation:

```bash
/glade/work/zel/conda-envs/pact-casper-cu126/bin/python \
  configs/cross_dataset_infer_4stations/benchmark_runtime.py
```

The 2026-10-06 report is in `audit/runtime_benchmark_20261006_084832/`.
`one_year_profiles.csv` and `full_pair_calibrations.csv` contain measured stage
timings; `estimated_pair_times.csv` contains all 264 estimates, and
`estimated_source_group_times.csv` contains all 48 groups. A group means one
station, period, and source: 6 past-only pairs or 5 future pairs. All 48 source
roles have the same compute structure and tensor sizes (618,885 parameters).
Only the saved `peak_prior` field varies; it is unused by SingleHead. Details are
in `source_architecture_audit.json`. The other source/target pairs are estimates.

Measured full-process times: Boston past NCEP→AWI (7 years) 14.17 s; Lewes future
NCEP→EC_EARTH (30 years) 219.02 s; Boston future MRI→CNRM (30 years) 36.56 s.
EC_EARTH's data loading took 172.49 s and its evaluation loops 41.07 s. This
benchmark observed substantial GLADE I/O variation; estimates model loading
separately and retain these target-specific full-pair observations.

At these settings, the point estimate is about 1.3 min per past source group,
9.1 min per future source group, 62 min per station, and 4.15 h for all 264 pairs.
Warm-input extrapolation is about 1.81 h, and the planning upper allowance about
6.22 h; these are scenarios, not confidence intervals. An 8-hour allocation
provides headroom for the observed I/O variability. The 264-pair matrix itself
has not been timed in full. These estimates include a process-startup allowance
of about 4.05 s per pair and exclude final matrix summarization.

`old_vs_filtered_timing.json` explains the earlier 33.727 s one-year run: the old
loader read 36 year files (13.51 GiB); the year evaluation loop took only 2.757 s.
After year filtering, Python main took 2.482 s (year loop 1.834 s) and the complete
shell command 6.51 s. Year-loop timers include loader, CUDA, CPU copies and
metrics work; they do not measure only the model's GPU forward pass.

## Results and completeness

Production pair artifacts are organized as:

```text
CrossDatasetInference_4Stations/<station>/<group>/
  <group>_<station>_<source>_to_<target>_<timestamp>/
    infer_config_used.sh
    command_used.sh
    run_infer.sh
    infer_*.log
    outputs/metrics.json
    outputs/predictions.npz
    outputs/config.json
    outputs/metrics_per_year_*.json
    outputs/metrics.md
```

The separate post-hoc utility validates every metric/NPZ metadata pair against
the manifest and checkpoint audit. It checks years, physical-unit reporting,
source tau, source statistics provenance, station, checkpoint, prediction shapes,
and reproducibility artifacts. Duplicate, invalid, or missing outputs make it
exit nonzero. Missing matrix cells remain explicit.

Under the result root's `summary/` it writes:

- `all_pairs.csv`: one row per validated result, 264 when complete.
- `<station>_past_only_AllRMSE.csv`: four 6 × 6 matrices.
- `<station>_future_year_AllRMSE.csv`: four 6 × 5 matrices.
- `AllRMSE_matrices.md`: all eight matrices, rows = source, columns = target.
- `completeness.json`: expected/completed/failed/missing counts and exact missing pairs.

AllRMSE is read directly from `metrics.json["metrics"]["all_rmse"]`. CSV retains
the stored precision; Markdown displays six decimal places. No metric is
recalculated. Completion requires 264 validated pairs, matching year sets, and
all eight matrices fully populated.

Casper migration validation on 2026-10-06 passed all 48 source roles (checkpoint,
training-snapshot, and training-metrics hashes unchanged), 44 target year audits,
44 real target batches, 285 shell syntax checks, all 264 dry runs through the exact
`/glade/u/home/zel/scratch/...` entry path, 31 focused tests, and `pip check`.
The new environment passed V100 FP16 GraphSAGE/CUDA probes and both one-year
smokes. Boston NCEP→AWI: 600 samples, AllRMSE 0.1948299753 m, Python walltime
2.482 s, process peak RAM about 1.52 GiB. Lewes future EC_EARTH: 600 samples,
2 missing target hours, AllRMSE 0.0437137445 m, walltime 3.094 s, peak RAM about
1.82 GiB. Both outputs pass the production summary's metadata and NPZ validators.
The former all-years loader filled the 64 GiB allocation during the EC_EARTH
smoke; that isolated diagnostic attempt was stopped before inference completed.
Validation details and logs are under `CrossDatasetInference_4Stations/audit/`,
including `casper_validation.json`. The production station directories remain
absent; tomorrow's command starts a fresh matrix. Only audit/smoke outputs exist.

Historical Jetstream2 validation before the Casper migration passed 70 focused tests, all 48 source
checkpoint roles, 282 shell syntax checks, all 264 dry runs (including the 12
required representatives), 44 target graph batches, and both Lewes smoke checks.
Production requires a fresh result tree. `run_all.sh` starts all 264 pairs from
scratch and refuses an existing production station directory; audit and smoke
subdirectories are allowed. A detached tmux session remains subject to PBS walltime.

The focused suite includes existing strict-year, empty-source-TEST, source
normalization/threshold, physical-unit, and missing-label tests plus
`tests/test_cross_dataset_infer_4stations.py`. A stale test fixture path was
updated to the repository's existing `configs/Legacy_Configs/baseline_ablation`.
`infer.sh`, the artifact helper, architecture, and metric definitions are unchanged.
For external/source-all evaluations with explicit `YEARS`, `infer.py` and the graph
store now filter year files before deserializing them. This avoids loading all 66
years for a single-year smoke and excludes historical years from future production.
Held-out evaluation retains full saved-TEST-tag validation. Graph tags and target
timestamps retain their identities; NPZ sample IDs index the selected in-memory store.
The existing four-station test now expects the Casper default batch size 64.
Future audit logs and inference outputs must be written only under the required
result root; `audit_graphs.py` writes to its `audit/` subdirectory.
