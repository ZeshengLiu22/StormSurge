# Four-station PACT cross-dataset inference

Canonical station tokens are `Boston`, `CBBT`, `Lewes`, and `Battery`, verified
against station JSON, training snapshots, split tags, and checkpoint metadata.
This workflow uses only these production roots:

- Configs: `/home/exouser/StormSurge/configs/cross_dataset_infer_4stations/`
- Results: `/home/exouser/media/share/PACT/FormalRuns_0925/CrossDatasetInference_4Stations/`

`/home/exouser/media` is an existing filesystem alias. Inference metadata resolves
paths to `/media/share/PACT/...`; these refer to the same files, not another result tree.
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
Runtime retains the validated Boston batch size 256, bf16 AMP, TF32, one Torch
thread, zero loader workers, and disabled dual diagnostics.

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
cd /home/exouser/StormSurge
PYTHON=/home/exouser/.conda/envs/torchpyg-cu12x/bin/python

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

Two isolated one-year compatibility configs live in `smoke/`. They exercise
Lewes MRI `2013_2014` (one missing target hour) and EC_EARTH `2090_2091` (two
missing target hours). Smoke outputs live under the required result root's
`smoke/` directory and are excluded from production matrices. The 264 production
configs retain their full year lists.

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

Pre-production validation on 2026-10-06 passed 70 focused tests, all 48 source
checkpoint roles, 282 shell syntax checks, all 264 dry runs (including the 12
required representatives), 44 target graph batches, and both Lewes smoke checks.
The production run was stopped at the user's request and the entire new result
tree was removed. There are no retained production results to resume. The next
`run_all.sh` invocation starts all 264 pairs from scratch. The source checkpoint
audits and all configs remain in this tree.

The focused suite includes existing strict-year, empty-source-TEST, source
normalization/threshold, physical-unit, and missing-label tests plus
`tests/test_cross_dataset_infer_4stations.py`. A stale test fixture path was
updated to the repository's existing `configs/Legacy_Configs/baseline_ablation`.
`infer.py`, `infer.sh`, the artifact helper, and all scientific code are unchanged.
Future audit logs and inference outputs must be written only under the required
result root; `audit_graphs.py` writes to its `audit/` subdirectory.
