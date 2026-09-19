# 0919 direct-dual excess capacity and gate pooling

**4 stations × (1 Single + 5 Dual variants × 2 pooling modes) = 44 runs.**
CBBT, Lewes, Battery and Boston each have exactly 11 standalone configs:

| Head | Capacity | Pooling | Runs per station | Objective |
| --- | --- | --- | ---: | --- |
| Single | Normal production head | Not applicable | 1 | MSE only |
| Direct Dual | `legacy`, `c1`, `c2`, `c2r`, `c3` | `mean`, `learned` | 10 | MSE + 1×body + 2×excess + 0.5×gate |

Every Dual uses `DUAL_LOSS=1`, `DUAL_ABLATION=none`, `EXCESS_FORMULATION=direct`
and `LOSS_MODE_LIST=("mse")`. Single uses `DUAL_LOSS=0` and explicit zero branch
weights. Tail, slope, excess-amplitude, shape and peak coefficients are explicitly
zero; WMSE is never selected. There are no other experimental factors or controls.

`legacy → c1` tests width; `legacy → c2` tests depth; `c2 → c2r` tests the residual;
`c2r → c3` tests excess LayerNorm. Mean versus learned pooling is matched within
each capacity. Dual configs differ only in these two factors and run identity.

The eight corresponding Single/direct-Dual files in
`experiment_config_0916/A0_HeadCapacity` are read-only references. Configs copy
their settings explicitly; they never source an old config at runtime. Source and
reference hashes are recorded in [provenance.json](provenance.json).

The frozen protocol is LR **5e-3**, Adam/weight decay **1e-5**, 300 epochs,
batch 256 with accumulation 4, cosine scheduling, 5 warmup epochs from factor
0.1, and minimum LR 1e-6. GraphSAGE + Transformer, 24-hour history, hidden 128,
graph/Transformer depth 2, read heads 8/8, FF multiplier 4, graph/head dropout
0.05 and Transformer dropout 0.0 are unchanged. Station metadata retains six
coordinate features, with elevation/bathymetry disabled. TRAIN z-score statistics,
chronological 0.6/0.2/remainder year splits, no year shuffling/future filtering,
TRAIN 95th-percentile event threshold, no augmentation/clipping/OOD, overall VAL
RMSE checkpoint selection, and no auxiliary checkpoints all match 0916.

Dataset root: `/media/share/PACT/Data/Grid4_New/NCEP/graphs`.
Station metadata: `/media/volume/PACT-Data/StormSurge/station_json`.
The station selector changes; neither path is invented or redirected.

Every config explicitly sets **SEED=42, DETERMINISTIC=0**, matching the 0916
reference. Strict deterministic kernels were disabled by user request to improve
throughput. Execution stays one GPU (`num_gpus=1`, `CUDA_VISIBLE_DEVICES=0`),
BF16 AMP, TF32 enabled, one Torch thread, `NUM_WORKERS=0`, `PIN_MEMORY=0`,
`PERSISTENT_WORKERS=0`, `PREFETCH_FACTOR=0`, and multiprocessing `fork`.
The canonical interpreter is
`/media/volume/PACT-Data/conda_envs/torchpyg-cu124/bin/python`, with `DO_CONDA=0`.
The existing runtime disables deterministic algorithms and cuDNN determinism;
cuDNN benchmarking remains off, as in 0916. It does not set a cuBLAS workspace
constraint in this mode. Seeded initialization and matched training sample order
remain in place, but GPU training results are not guaranteed to repeat bitwise.

Two inherited details matter for interpretation. Under MSE, the launcher omits
tail/slope flags, so their inactive Python defaults remain 0.1/0.01; they do not
enter this objective. Single branch flags are also omitted and remain inactive.
The canonical Dual weights are **1/2/0.5**, not three unit weights.

**Cross-variant minibatch order is matched within each station.** Single-process
training now gives its DataLoader a private `torch.Generator` seeded once from
`args.seed`. This stream drives sample shuffling and iterator/worker seeds,
independently of parameter initialization and dropout. Its state advances across
epochs; it is never reset at epoch boundaries. All 11 architectures therefore
share the same permutation sequence for the same data, seed and loader settings.
The synthetic probe verifies all 300 epochs twice, including Single, while
perturbing global RNG consumption between epochs. DDP sampler behavior and
evaluation loader defaults are unchanged. Historical single-process training
trajectories change intentionally; compare study runs using this fixed protocol.
The two source-hash amendments are recorded in [provenance.json](provenance.json).

All run artifacts stay under
`/home/exouser/media/volume/PACT-Data/StormSurge/All_results_0919/` as
`0919_<Station>_<Variant>_<Pooling>__<TIMESTAMP>/` (Single: `0919_<Station>_Single_MSE`).
The `/home/exouser/media` prefix is the existing symlink to `/media`.
[manifest.csv](manifest.csv) has exactly 44 rows. Its `output_dir` is a timestamp
pattern, not an already-created run directory; Single factors are explicitly `NA`.

Validate or inspect without launching:

```bash
cd /home/exouser/media/volume/PACT-Data/StormSurge
/media/volume/PACT-Data/conda_envs/torchpyg-cu124/bin/python -B experiment_config_0919/generate_configs.py --check
/media/volume/PACT-Data/conda_envs/torchpyg-cu124/bin/python -B experiment_config_0919/validate_configs.py
bash experiment_config_0919/run_CBBT.sh --dry-run
# All 44 existing launcher dry runs:
bash experiment_config_0919/run_all.sh --dry-run
```

Validation checks inventory, factors, every inherited setting, optional losses,
one-process launchers, paths, 44 production-parser commands, source/reference
integrity, and unchanged result trees. It compares each run with its actual 0916
reference, including DETERMINISTIC=0. A representative C3 + Learned synthetic
forward/backward checks finite predictions and gradients with the resolved
seed/runtime settings, without data loading or optimizer steps. The report
distinguishes CPU plumbing checks from GPU BF16 smoke validation. Neither is a
claim of bitwise training repeatability; only the isolated sample-order stream
is required to match across repeats and architectures. No real training runs.

Evidence: [validation report](validation_report.md), [full resolved settings](validation_report.json),
[44 resolved commands](dry_run_commands.txt), and
[reproducibility check](validation/reproducibility.json). Individual dry-run
transcripts are in `validation/`. The fixed dry-run timestamp creates no outputs.

When explicitly choosing to submit later, `bash experiment_config_0919/run_all.sh`
queues the 44 jobs; `run_CBBT.sh`, `run_Lewes.sh`, `run_Battery.sh` and `run_Boston.sh`
queue 11 each. They validate first, require GPU 0 with BF16 support, then use the
existing `qsub_local`/tmux convention with one queue slot. No CPU or precision
fallback is allowed. [commands_all.txt](commands_all.txt) lists the same 44 queue
commands for use in an interactive shell with `qsub_local`; those direct commands
assume the documented GPU/environment preconditions. Job failures retain their
nonzero status and logs through the existing queue worker; no retry changes settings.

No study jobs were submitted or training launched while creating or validating
this suite. The later data-order fix changes only loader construction in
`train.py` and `emulator/data/loaders.py`; model, loss, optimizer, scheduler,
dataset, metric, checkpoint and existing launcher code remain unchanged.
No branch diagnostics are enabled or added.
