# Pairwise cross-dataset inference

One config = one Source → Target cell. One `infer.py` invocation evaluates one
source checkpoint on one target. The group runner executes one source's pair
configs sequentially on one GPU. `infer.py` does not read the manifest or
implement a matrix, sweep, or aggregation.

This tree is configured for development validation. The formal production matrix
must be configured and launched separately.

## Populations and checkpoint policy

| Group | Sources | Targets | Cells | Requested target year groups |
| --- | --- | --- | ---: | --- |
| `past_only` | NCEP, AWI, CNRM, EC_EARTH, MPI, MRI | Same six | 36 | `2008_2009` through `2014_2015`, exactly 7 |
| `future_year` | Same six | AWI, CNRM, EC_EARTH, MPI, MRI | 30 | `2070_2071` through `2099_2100`, exactly 30 |

NCEP is a source only in the future matrix. All 66 configs use the final
`G0_T0500_EP0100` candidate: global MSE + tail MSE weight 0.05 + episode GT-aligned
peak MSE weight 0.01. Every checkpoint is `best_overall.pt`. Exact absolute paths
are pinned in the pair configs and [manifest.csv](manifest.csv).

Each source directory contains only its six or five pair configs. Architecture
and runtime settings live in [common.sh](common.sh); group settings define fixed
years and result roots. The old `configs/configs_infer/` tree remains unchanged
as legacy reference.

## Scientific policy

- Source checkpoint normalization is retained. Target normalization is never fitted.
- Each source model retains its own fixed TRAIN hourly Q95. Target/test data are
  never used to refit, recompute, or recalibrate tau.
- Cross-dataset Extreme and Episode metrics retain the source model's fixed TRAIN
  Q95 threshold. They measure **source-conditioned zero-shot extreme
  generalization**. NCEP → AWI uses NCEP TRAIN Q95; CNRM → AWI uses CNRM TRAIN Q95.
- AllRMSE is the initial matrix comparison metric. Extreme and Episode metrics
  continue to be exported with their existing definitions.
- Diagonal cells are re-run with explicit `TARGET_ROOT` + fixed `YEARS`, exactly
  like off-diagonal cells. Training-run test metrics are not substituted.

`STRICT_YEARS=1` passes `--strict_years`. Missing requested target groups,
duplicate requests, malformed tags, or a filtered year set different from the
request fail before any inference forward or output creation. Requested and
evaluated years and counts are exported. Older workflows retain permissive
filtering unless they opt in. Strict mode requires an explicit `--years` list.

## Run commands

Run from the repository root:

```bash
cd /home/exouser/StormSurge

# One pair (the full configured 30 target groups)
bash infer.sh \
  configs/cross_dataset_infer/future_year/NCEP/NCEP_to_AWI.sh

# Resolve one pair without running Python or creating result directories
DRY_RUN=1 bash infer.sh \
  configs/cross_dataset_infer/past_only/NCEP/NCEP_to_AWI.sh

# Dry-run one source group
DRY_RUN=1 bash configs/cross_dataset_infer/run_group.sh \
  configs/cross_dataset_infer/future_year/NCEP

# Eventually run one source group, sequentially
bash configs/cross_dataset_infer/run_group.sh \
  configs/cross_dataset_infer/future_year/NCEP
```

The group runner sorts configs, checks expected pair names/count and explicit
roots before starting, uses `USE_TMUX=0`, reports progress/timing, and stops on the
first failure. The common config defaults to foreground execution. Dry runs
print source, target, checkpoint, roots, architecture, years/count, strict mode,
and threshold origin. They do not load target data; strict selection against
loaded graphs is checked when inference runs.

## Development outputs and replay

The result roots are:

```text
/home/exouser/media/share/PACT/FormalRuns_0925/CrossDatasetInference_dev/past_only
/home/exouser/media/share/PACT/FormalRuns_0925/CrossDatasetInference_dev/future_year
```

Runs use `<group>_Boston_<source>_to_<target>_<timestamp>/outputs/`. Each run
retains a resolved `infer_config_used.sh`, `run_infer.sh`, `command_used.sh`, and
log. Outputs include `config.json`, `metrics.json`, `metrics_per_year_*.json`,
`metrics.md`, and `predictions.npz`.

Metrics JSON and prediction NPZ exports include source/target names and absolute
roots, checkpoint, requested/evaluated years and counts, station, model, encoder,
temporal block, head, history, and:

```text
normalization_origin = source_checkpoint_train
evaluation_threshold_origin = source_checkpoint_train
source_tau_physical = checkpoint["threshold_metadata"]["tau_physical"]
source_exceedance_percentile = checkpoint["threshold_metadata"]["exceedance_percentile"]
```

The original `tau_physical`, `exceedance_percentile`, `threshold_schema`, and
`metric_schema` are retained; metrics JSON preserves the complete source
`threshold_metadata`. No target threshold is fitted or reported.

A no-TEST checkpoint used without an external root fails clearly:

```text
Checkpoint has no held-out source TEST split.
Provide --test_root_dir for external transfer evaluation,
or explicitly use --scope all if source-all evaluation is intended.
```

Future-transfer NCEP intentionally has `split_tags["test"] == []`. Its
training-run “TEST” exports mirror validation, as its warning file states.
External inference uses target graphs and does not consume those exports.

## Inference audit findings

The audit found no existing model, normalization, threshold, physical-unit, or
GraphSAGE transfer correctness bug. The original year filter silently discarded
missing requests; strict mode closes that population-validation gap. The
no-TEST error and provenance reporting are also improved.

The audited path is:

1. `infer.py` reconstructs `ModelConfig` and strictly loads `model_state` from the
   source checkpoint. CLI architecture fields are checked against those values.
2. Source `normalization` tensors and `threshold_metadata.tau_physical` are read
   directly from the checkpoint. `train.py` originally fitted both from TRAIN
   indices only. Inference has no statistics/threshold fitting path.
3. With `--test_root_dir`, `ForcingGraphStore` opens only that target root and
   station; source TEST tags are not required or used for selection. `--years`
   filters target tags. The source root supplies provenance.
4. `ForcingGraphView` retains target edges and node counts and selects the last
   five forcing frames for a 24 h history. PyG batches those target graphs.
   GraphSAGE has no fixed grid dimensions; `SpatialEncoder.grid_shape` returns
   `None` for GraphSAGE. PACT uses `to_dense_batch` and a padding mask for node
   attention. Normalization has five feature entries, independent of node count.
5. `run_epoch` normalizes inputs with saved source tensors, then converts
   predictions with `prediction * source_y_std + source_y_mean`. Target labels
   already use physical water-level units. Errors are in meters; episode timing
   is in hours.
6. `evaluate_metrics(y_pred, y_true, tau_physical)` applies strict `y_true > tau`
   to target truth using source TRAIN tau. Episode populations follow the
   existing hourly continuity, split-boundary, and earliest-maximum rules.
   This source-conditioned behavior is intentional.

Representative real Boston files passed the actual CPU store/view/loader path:

| Dataset | Nodes per graph |
| --- | ---: |
| NCEP | 702 |
| AWI | 3,776 |
| CNRM | 1,677 |
| EC_EARTH | 6,708 |
| MPI | 3,776 |
| MRI | 2,597 |

CNRM → EC_EARTH, EC_EARTH → MRI, and AWI → CNRM satisfy the actual data/model shape
contract. The file audit checked one requested year per target, without model
inference; full requested populations are enforced at evaluation time.
See [graph_audit.json](graph_audit.json).

Batch size 256 matches the audited training batch and uses evaluation without
gradients. The one-year NCEP → AWI smoke passed at this size on the H100 80 GB.
Other target sizes have not received a separate real inference run in this task.
The current graph store loads all Boston files in the selected root before year
filtering, so a one-year request still incurs that CPU memory and I/O cost.

## Checkpoint audit and regeneration

[checkpoint_audit.json](checkpoint_audit.json) records all 12 source roles:
exact path, SHA-256, completion status, configuration snapshot and its hash,
epoch, architecture, normalization values, threshold metadata, complete split
year lists and counts, and successful strict model-state loading.

All checkpoints are Boston / perceiver3 (PACT) / GraphSAGE / Transformer / single /
four history steps (24 h) / z-score normalization / Q95 / overall role. All have
`exit_status=0` and the intended 0.05/0.01 loss weights. Saved split tags,
rather than directory labels alone, establish:

| Source role | TRAIN groups | VAL groups | TEST groups |
| --- | --- | --- | --- |
| All past-only sources | 1979–2000 (22) | 2001–2007 (7) | 2008–2014 (7) |
| Future-year NCEP | 1979–2008 (30) | 2009–2014 (6) | Empty (0) |
| Future-year CMIP6 | 1979–2008 (30) | 2009–2014 (6) | 2070–2099 (30) |

Exactly one candidate exists for each role in its requested active experiment
area. The older MRI future run under `FormalRuns_0925/Achieved/` is archived and
outside those areas; the active September 30 run is pinned. Regeneration fails
on additional candidates, missing completion artifacts, architecture, loss,
root, split, normalization, or threshold mismatches.

```bash
PYTHON=/home/exouser/.conda/envs/torchpyg-cu12x/bin/python

# Audit checkpoints/target filenames and compare deterministic outputs; no writes
"$PYTHON" configs/cross_dataset_infer/generate_configs.py --check

# Regenerate configs, group common files, manifest, and checkpoint audit
"$PYTHON" configs/cross_dataset_infer/generate_configs.py

# Inspect representative real target batches on CPU; no model forward
"$PYTHON" configs/cross_dataset_infer/audit_graphs.py
```

The generator audits all checkpoints before writing configs and never runs
inference. The manifest is for human audit only. `/home/exouser/media` resolves
to `/media`, so canonical checkpoint paths below begin with `/media/share/`.

### Exact resolved checkpoints

| Group | Source | Epoch | TRAIN tau (m) | Checkpoint |
| --- | --- | ---: | ---: | --- |
| past_only | NCEP | 280 | 0.31485228091478312 | `/media/share/PACT/FormalRuns_0925/single_tail_episodepeak_4x4/NCEP_Boston_G0_T0500_EP0100__20260927_191110/best_overall.pt` |
| past_only | AWI | 290 | 0.22662775516509914 | `/media/share/PACT/FormalRuns_0925/CMIP6_Boston_QuickCheck_0929/past_only/CMIP6_AWI_Boston_past_only_G0_T0500_EP0100__20260929_054511/best_overall.pt` |
| past_only | CNRM | 280 | 0.32726608216762471 | `/media/share/PACT/FormalRuns_0925/CMIP6_Boston_QuickCheck_0929/past_only/CMIP6_CNRM_Boston_past_only_G0_T0500_EP0100__20260929_054511/best_overall.pt` |
| past_only | EC_EARTH | 283 | 0.27713152468204477 | `/media/share/PACT/FormalRuns_0925/CMIP6_Boston_QuickCheck_0929/past_only/CMIP6_EC_EARTH_Boston_past_only_G0_T0500_EP0100__20260929_054511/best_overall.pt` |
| past_only | MPI | 295 | 0.24955656975507734 | `/media/share/PACT/FormalRuns_0925/CMIP6_Boston_QuickCheck_0929/past_only/CMIP6_MPI_Boston_past_only_G0_T0500_EP0100__20260929_054511/best_overall.pt` |
| past_only | MRI | 296 | 0.33672360181808447 | `/media/share/PACT/FormalRuns_0925/CMIP6_Boston_QuickCheck_0929/past_only/CMIP6_MRI_Boston_past_only_G0_T0500_EP0100__20260929_054511/best_overall.pt` |
| future_year | NCEP | 283 | 0.31119676977395994 | `/media/share/PACT/FormalRuns_0925/NCEP_future_transfer/NCEP_Boston_future_transfer_G0_T0500_EP0100__20261001_165726/best_overall.pt` |
| future_year | AWI | 273 | 0.22513962537050247 | `/media/share/PACT/FormalRuns_0925/CMIP6_Boston_QuickCheck_0929/future_year/CMIP6_AWI_Boston_future_year_G0_T0500_EP0100__20260929_054511/best_overall.pt` |
| future_year | CNRM | 286 | 0.32739297747611995 | `/media/share/PACT/FormalRuns_0925/CMIP6_Boston_QuickCheck_0929/future_year/CMIP6_CNRM_Boston_future_year_G0_T0500_EP0100__20260929_054511/best_overall.pt` |
| future_year | EC_EARTH | 278 | 0.28200886845588674 | `/media/share/PACT/FormalRuns_0925/CMIP6_Boston_QuickCheck_0929/future_year/CMIP6_EC_EARTH_Boston_future_year_G0_T0500_EP0100__20260930_142355/best_overall.pt` |
| future_year | MPI | 282 | 0.24754500091075879 | `/media/share/PACT/FormalRuns_0925/CMIP6_Boston_QuickCheck_0929/future_year/CMIP6_MPI_Boston_future_year_G0_T0500_EP0100__20260929_054511/best_overall.pt` |
| future_year | MRI | 277 | 0.33884304910898183 | `/media/share/PACT/FormalRuns_0925/CMIP6_Boston_QuickCheck_0929/future_year/CMIP6_MRI_Boston_future_year_G0_T0500_EP0100__20260930_142355/best_overall.pt` |

## Validation completed on 2026-10-01

[validation.json](validation.json) records the results:

- **64 CPU tests passed**, including seven inference-reporting tests, four
  config/launcher tests, and 53 existing dual-diagnostic, hourly-metric,
  threshold, pipeline, and config-interface tests. Tests cover no-TEST external
  transfer and explicit source-all scope, strict missing/duplicate/exact years,
  source normalization and tau retention, NPZ/JSON provenance, dry-run side
  effects, foreground sequencing, failure propagation, and snapshot replay.
- Bash syntax passed for all 73 relevant shell files. Deterministic generator
  comparison passed. Counts are 36 past-only + 30 future-year = 66; each source
  has six/five configs and each group has exactly seven/30 requested years.
- Pair dry runs passed for past-only NCEP → AWI, future-year NCEP → AWI, and
  future-year CNRM → EC_EARTH. Each resolved the intended checkpoint, roots,
  Boston / GraphSAGE / Transformer / single / 24 h, batch 256, strict years,
  and source TRAIN threshold policy.
- The NCEP future source group dry run passed all five targets without Python
  inference or result-directory creation.
- **Exactly one real smoke inference** ran: future-transfer NCEP → AWI,
  `2070_2071` only, with batch 256. It evaluated 600 samples, produced
  `metrics.json` and `predictions.npz`, and reported AllRMSE
  **0.18069831464951266 m**. Evaluation took 2.93 s; total wall time was 45.18 s.
- Both exports identify NCEP/AWI, requested and evaluated years
  `["2070_2071"]`, counts 1/1, and
  `evaluation_threshold_origin="source_checkpoint_train"`. Tau is exactly
  **0.31119676977395994 m**, equal to the NCEP future checkpoint. Metrics were
  independently recomputed from the saved predictions using that same tau.
- The smoke used a temporary config; the checked-in NCEP → AWI future config
  still requests all 30 future groups.
- No full production pair, source group, or matrix was launched.

Smoke artifacts:

```text
/home/exouser/media/share/PACT/FormalRuns_0925/CrossDatasetInference_dev/smoke/
  smoke_future_year_Boston_NCEP_to_AWI_20261001_180948/
    infer_config_used.sh
    run_infer.sh
    command_used.sh
    outputs/metrics.json
    outputs/predictions.npz
```

Focused and compatibility test commands:

```bash
PYTHONPATH=.:tests /home/exouser/.conda/envs/torchpyg-cu12x/bin/python \
  -m unittest test_inference_reporting test_cross_dataset_infer \
  test_dual_experiments test_hourly_metrics test_hourly_threshold \
  test_pipeline test_config_interfaces -v
```

Changed implementation files are `infer.py`, `infer.sh`, and
`emulator/common/inference_artifacts.sh`. Tests extend
`tests/test_inference_reporting.py` and add `tests/test_cross_dataset_infer.py`.
This config tree contains the new configs, generator, audits, manifest, runner,
and documentation. `infer_multi.sh`, the legacy config tree, training, model
code, normalization code, and metric definitions are unchanged.
