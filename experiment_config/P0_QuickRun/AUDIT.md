# P0 final configuration audit — 2026-09-10

Status: **PASS**. There are 32 fully independent configs and 32 manifest rows.
No real training, loss-weight diagnostic, LR sweep or architecture selection
was performed.

## Repository and interface inspected before generation

Current repository: `/media/volume/PACT-Data/StormSurge`, initial HEAD
`9e9a4703a1f995787a61898b7911ca54f6598d61`. The working tree was clean.
`experiment_config/` existed but was empty; the requested `P0_QuickRun/`
subdirectory was absent, so it was created at that exact location.

The audit covered:

- `train.sh`, `train.py`, `emulator/training/arguments.py`,
  `emulator/training/losses.py`, `emulator/models/heads.py`;
- normalization/statistics and training-engine output paths;
- the old common config, representative NCEP Battery P3_Base/P3_Best and
  Stable single/dual configs, and the interface retirement documentation;
- the actual `~/.bashrc` queue functions and `~/.local/bin/qsub_local_worker`.

All 133 existing historical shell configs are unchanged. Only `train.sh` was
modified among existing repository files. Model, loss, optimizer, accumulation,
split, inference, metrics and DDP implementation files were not modified.
Neither `.bashrc` nor the queue worker needed changes.

## Local environment and data verification

`ROOT_DIR=/media/share/PACT/Data/Grid4_New/NCEP/graphs` is the canonical target
of the working data link. File enumeration/stat checks found 36 readable,
nonempty graph files for each of CBBT, Lewes, Battery and Boston. Station JSONs
are present at `/media/volume/PACT-Data/StormSurge/station_json`.
No graph tensors were loaded for this configuration task.

The exact interpreter is
`/media/volume/PACT-Data/conda_envs/torchpyg-cu124/bin/python`.
The requested host interpreter check returned:

```text
PyTorch: 2.6.0+cu124
torch-geometric: 2.7.0
torch.cuda.is_available(): True
GPU: NVIDIA H100 80GB HBM3
```

The first sandbox query returned `NO CUDA`; the host query above resolved that
sandbox visibility restriction. It only imported packages and queried hardware.
Every config explicitly selects this interpreter and `DO_CONDA=0`.

## Resolved scientific settings

All configs use the requested fixed perceiver3 / GraphSAGE / Transformer
architecture, 24 h history, chronological 0.6/0.2 split and seed 42.
They use batch 256, accumulation 4, LR `5e-3`, 300 epochs, cosine scheduling,
five warmup epochs, one GPU, AMP bf16 and TF32. Elevation and bathymetry are off.

Configured frozen weights are body **1**, excess **2**, gate **0.5**, tail
**0.025** and slope **0.01**. Tail fraction is 0.05; the slope penalty uses
Charbonnier, epsilon `1e-3`, mask softness 0.10 and inactive Huber delta 0.05.
All active parsed loss weights match the fixed declarations.

The current launcher deliberately omits inactive loss arguments. Consequently,
unused parser fields retain their old defaults: single-head branch weights and
non-tail `tail_lambda` are examples. These do not affect the selected loss.
The validator records both `configured_weights` and `resolved_args`, and tests
active terms separately. This preserves the production dispatch and avoids
changing legacy commands or loss behavior merely to change unused metadata.

All dual commands resolve `head_type=dual`, `gate_mode=window`,
`dual_mode=exceedance`, `dual_loss=1`, `dual_ablation=none` and percentile 95.
All single commands resolve `dual_loss=0`. Each of the 16 single configs was
also checked with a real `SingleHead` output and the production `ForecastLoss`,
with `dual_loss_terms` patched to fail if reached. No branch loss was reached.
These checks used tiny CPU tensors, no backward pass and no optimizer.

All 32 production dry runs resolve:

```text
DISABLE_OOD = 1
x_norm = zscore
x_clip = 0
x_aug = 0
x_aug_prob = 0
x_aug_scale = 0
x_aug_bias = 0
```

Percentiles and node-sample settings required by the launcher are inert under
the current zscore branch. The normalization code clips only when `x_clip > 0`,
and the training engine receives no augmentation tuple when `x_aug=0`.

Excluded retired settings include `STABILITY_GUARD`, `STABLE_ARCH_VERSION`,
`STABLE_ARCH`, `ALPHA_INIT_LOGIT`, `TAIL_TANH_CLIP` and `GATE_BIAS_INIT`.
No unsupervised/residual head parameters or dual mechanism ablations were
introduced. The launcher's existing literal `_stable1v3` inside detailed
training log tags is preserved; it is not an architecture-selection parameter.

## Minimal launcher changes

1. Added `RUN_DIR_NAME_STYLE`, defaulting to `timestamp_runname`. P0 explicitly
   selects `runname_timestamp`, producing `<RUN_NAME>__<YYYYMMDD_HHMMSS>`.
2. Rejects a collision for the new P0 naming style before overwriting an
   existing run. Legacy naming and directory creation retain their old behavior.
3. Keeps `USE_TMUX=1` visible during `DRY_RUN=1`; an explicit dry-run guard skips
   session creation. Dry runs still create no artifact directories.
4. Records semantic name, timestamp, session and directory style in the resolved
   snapshot, and logs the selected interpreter, runtime and configured weights.
5. Passes `SESSION_NAME` and `DRY_RUN=0` explicitly into the tmux inner command,
   alongside the existing run-name/timestamp handoff. This avoids relying on
   stale environment values in an existing tmux server.
6. Saves the full inner launcher output to `launcher.log` and its final status
   to `exit_status`, inside the one result directory. Existing per-command
   training logs remain there. Added an early interpreter availability check.

Existing `PACT_RUN_NAME` precedence was already correct and remains intact.
The outer tmux invocation still creates no result directory; only its inner
invocation creates it. The existing queue status handoff remains in use.

## Artifact containment

Static inspection of `train.py` confirms that its configuration JSON,
checkpoint, JSONL metrics, summary and compressed test predictions are all
constructed from the exact supplied `output_dir`. Its fallback result-folder
creation is unused because the launcher always passes `--output_dir`.
There is no separately launched plotting/evaluation process in this invocation.

The controlled stub test verified that each actual outer/inner launcher pair
creates exactly one directory containing `config_used.sh`, `launcher.log`,
one `train_*.log`, the stub invocation metadata and `exit_status`. The real
training artifact paths were audited statically; real checkpoints/metrics were
not generated as part of validation. Task-spooler's service bookkeeping and
queue transcript are external queue infrastructure, while the full training
and launcher transcript is retained in the per-run directory.

## Validation evidence

| Check | Result |
| --- | --- |
| Shell syntax: 32 configs, launcher, `.bashrc`, worker and temporary shell helpers | PASS |
| Python syntax: generator and validation helpers | PASS |
| Independent config sourcing in an empty shell environment | 32/32 PASS |
| Production `DRY_RUN=1` → actual production argument parser | 32/32 PASS |
| Exactly one command per config | 32/32 PASS |
| Matrix | 8/station, 16 single, 16 dual, 4 of each of 8 designs |
| Frozen declarations, architecture, preprocessing and active loss values | 32/32 PASS |
| Single-head auxiliary-loss exclusion | 16/16 PASS |
| Legacy default result naming | PASS |
| Reject accumulation >1 with num_gpus=2 | PASS |
| Real queue forwarding of the selected P0 config under DRY_RUN | PASS |
| One slot held while detached tmux is active; second task stays queued | PASS |
| Failure/success exit status propagation | 23 / 0 PASS |
| One session and one folder per outer/inner invocation | PASS |
| Semantic name/timestamp/session preservation, including paths with spaces | PASS |
| P0 result-folder collision protection | PASS |
| Cleanup of private test sessions, task-spooler server and metadata | PASS |

Detailed evidence:

- [Config validation and all parsed arguments](validation/config_validation.json)
- [All 32 production dry-run outputs](validation/dry_runs.txt)
- [Queue/tmux validation](validation/queue_validation.json)
- [Actual P0 queue dry run](validation/queue_dry_run.txt)
- [Queue while one stub was running and the next was queued](validation/queue_active.txt)
- [Failure propagation](validation/queue_failure.txt)
- [Success propagation](validation/queue_success.txt)
- [Baseline provenance hashes](validation/baseline_hashes.json)

The compatibility test used the exact queue functions extracted from `.bashrc`,
the unmodified worker and current production launcher, with private sockets.
Its two process stubs used only Python's standard library. They never loaded
data, imported Torch, built models or created an optimizer. No `qsub_local`
function was exported into the worker shell. No test session/task remains,
the normal queue was untouched, and the production `All_Results/P0_QuickRun`
root was not created by validation.
