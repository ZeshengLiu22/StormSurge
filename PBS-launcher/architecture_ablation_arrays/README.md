# Architecture ablation arrays on Derecho

Each LR directory now has `run_CNN_array.pbs` and `run_GNN_array.pbs`.
Each array has 144 indices and each subjob starts **one configuration with four
DDP ranks**, using the existing `train.sh`, Derecho Python, and `cuda/12.9.0`.
The old station launchers and their 108 PBS scripts are retired; historical
`.OU` logs are retained. No scientific config, training script, or model code is
changed by this refactor.

## Immediate overnight commands

From the repository root:

```bash
cd /glade/u/home/zel/scratch/StormSurge

# Preview only: two independent CNN arrays, 144 experiments each.
bash PBS-launcher/architecture_ablation_arrays/sweep.sh submit \
  --groups LR3e-3 LR5e-3_Clip0.25 --encoders CNN

# Execute this separately when ready to submit. No dependency between arrays.
bash PBS-launcher/architecture_ablation_arrays/sweep.sh submit \
  --groups LR3e-3 LR5e-3_Clip0.25 --encoders CNN --submit
```

The helper defaults to those two groups and CNN, **24 concurrent subjobs per array**, and
`MAX_ATTEMPTS=3`. Submission requires the explicit `--submit` option. LR1e-3 and
all GNN arrays remain available through `--groups` / `--encoders`, but are not
selected by the overnight default. Both `submit` and `resubmit` select only
PENDING/PREEMPTED/INTERRUPTED experiments with remaining budgets.

Optional concurrency, retry budget for experiments without an existing policy,
and walltime overrides:

```bash
bash PBS-launcher/architecture_ablation_arrays/sweep.sh submit \
  --groups LR3e-3 LR5e-3_Clip0.25 --encoders CNN \
  --max-concurrent 48 --max-attempts 3 --cnn-walltime 00:30:00
```

All six PBS templates also default to `#PBS -J 0-143%24`. The limit applies
**per array**, so the two default CNN arrays may run up to **48 subjobs / 48 GPU
nodes / 192 A100 GPUs** in total if resources and policy permit. Each subjob
runs one experiment using all four GPUs through DDP. The optional `%48` override
above allows up to 96 nodes across two arrays; `--max-concurrent unlimited`
omits `%` entirely. Selective resubmission splits noncontiguous original indices into
contiguous arrays, each with its own concurrency cap; a single remaining index
uses `-J N-N`. This follows PBS's documented `X-Y[:Z]` syntax.

Direct PBS submission is also supported (the helper is preferred because it
screens out queued/completed/failed/exhausted experiments):

```bash
qsub PBS-launcher/architecture_ablation_LR3e-3/run_CNN_array.pbs
qsub PBS-launcher/architecture_ablation_LR5e-3_Clip0.25/run_CNN_array.pbs

# Optional override of the default %24 cap:
qsub -J '0-143%48' PBS-launcher/architecture_ablation_LR3e-3/run_CNN_array.pbs
```

Direct submissions still use the execution lock, persistent budget and success
verification, so duplicate subjobs cannot train the same experiment concurrently.
Keep the supplied job names/identity variables so status can identify queued
subjobs before their launcher starts.

## Mapping and dry-run validation

The encoder is fixed by the chosen array template. The mapping never depends on
directory enumeration or PBS execution order:

```text
station = [Boston, CBBT, Lewes, Battery][index // 36]
history = [0H, 6H, 12H, 18H, 24H, 30H, 36H, 42H, 48H][(index % 36) // 4]
temporal = [MLP, LSTM, GRU, Attn][index % 4]
```

Thus 0 = Boston/0H/MLP, 35 = Boston/48H/Attn, and
143 = Battery/48H/Attn. All six group/encoder combinations cover exactly 144
distinct existing configuration files, 864 total.

```bash
# Read-only mapping and config validation.
bash PBS-launcher/architecture_ablation_arrays/sweep.sh validate --mapping

# Also inspect all 864 effective train.sh commands: no CUDA imports/training.
bash PBS-launcher/architecture_ablation_arrays/sweep.sh validate --commands

# Preview one complete training command.
bash PBS-launcher/architecture_ablation_arrays/sweep.sh preview \
  --group LR3e-3 --encoder CNN --index 0

# Exercise the actual PBS wrapper without submitting or training.
DRY_RUN=1 PBS_ARRAY_INDEX=143 bash \
  PBS-launcher/architecture_ablation_LR5e-3_Clip0.25/run_CNN_array.pbs

# Submission preview without contacting PBS (queue state is unknown).
bash PBS-launcher/architecture_ablation_arrays/sweep.sh submit --offline

# CPU-only tests, including mocked attempts and detached-worker signal cleanup.
python -m unittest discover -s PBS-launcher/architecture_ablation_arrays \
  -p test_launchers.py -v
```

Offline mode cannot submit or change retry controls. Online submission fails
closed if the scheduler cannot be queried. No preview consumes attempt budget,
creates training outputs, or calls qsub.

## Persistent state, failures and artifacts

State lives on shared scratch at:

```text
PBS-launcher/architecture_ablation_arrays/state/
  <LR>/<encoder>/<station>/<history>/<temporal>/
    execution.lock
    policy.json
    state.json
    attempts/<actual-attempt-number>-<unique-id>/
      start.json
      console.log
      signal.json           # when a handled interruption occurred
      end.json
    control_events/
  submissions/<submission-token>/
```

An exclusive Lustre `flock` covers the complete experiment execution. The lock
descriptor is inherited by the training shell, so killing just the supervisor
does not release it while that shell still runs. Locks are never manually stolen.
The supervisor uses a separate process session, tracks descendants even when
torchrun workers create their own sessions, adopts orphaned descendants, forwards
SIGTERM, then escalates to SIGKILL after a 20-second cleanup allowance and reaps
children. SIGINT and SIGHUP receive the same cleanup.

The atomic, fsynced `start.json` is the attempt counter. It is published after
environment/GPU preflight and immediately before launching `train.sh`. There is
no training retry loop inside a subjob. The unavoidable crash window between
record publication and process creation conservatively consumes one attempt.
Each attempt has immutable start/end records, its own console log, and a unique
timestamp/attempt/UUID run-directory suffix. Resetting a budget retains these
records and all training artifacts; actual attempt numbers continue increasing.

The original output roots and `<history>/<station>` structure are retained.
`train.sh` still produces `launcher.log`, `exit_status`, `config_used.sh`,
checkpoints and final reports. `console.log` also survives PBS spool replacement
on a rerun. Merely finding an output directory, checkpoint, or a PBS exit code of
zero never counts as success. Verification requires:

- Clean `exit_status=0`, matching effective training arguments and output identity.
- Exactly the contiguous 300 epoch records, finite numeric metrics and final log markers.
- Consistent final summary/comparison reports for both checkpoint roles and VAL/TEST.
- Nonempty resolved config/logs/metadata and structurally complete checkpoint and prediction ZIP archives.

Checkpoint contents are not deserialized and verification does not use a GPU.
A previously successful result that stops verifying is blocked for manual
inspection. Config changes after an experiment's first attempt also require an
explicit reset; there is no silent mixing of scientific settings.

Default budget: the initial execution is attempt 1; only two further starts are
allowed. A failed/interrupted third attempt leads to **EXHAUSTED**, with a nonzero
launcher exit. Later PBS reruns exit without training. `run_count` is recorded
only for diagnostics and is never set or increased by these scripts.

Ordinary application exits, numerical failures, and environment preflight
failures are **FAILED** and are not automatically retried. A dead owner without
a final exit record is conservatively **INTERRUPTED**. Walltime is reported as
`INTERRUPTED / WALLTIME`, separately from `PREEMPTED / PBS_PREEMPTION`, ordinary
`FAILED / APPLICATION_EXIT`, and `FAILED / NUMERICAL_FAILURE`. PBS `Exit_status=-29`
and explicit termination comments refine the diagnosis when available. A
terminal PBS record must match the attempt's subjob ID and run count; comments
about queue policy, or from another execution, cannot establish preemption.
Elapsed time near the walltime limit is retained only as an estimate in
`signal.json`, not proof of walltime termination. SIGTERM
alone cannot distinguish preemption, cancellation and some other interruptions;
it remains INTERRUPTED unless evidence identifies the cause. Raw signals, PBS
IDs/run counts, exit statuses, timestamps and output directories are preserved.

## Status, selective resubmission and budget changes

```bash
# Full experiment rows plus counts separately by LR, encoder, station and history.
bash PBS-launcher/architecture_ablation_arrays/sweep.sh status
bash PBS-launcher/architecture_ablation_arrays/sweep.sh status --counts-only
bash PBS-launcher/architecture_ablation_arrays/sweep.sh status --json > sweep-status.json

# Failures and interruptions, with actual attempt counts and output paths.
bash PBS-launcher/architecture_ablation_arrays/sweep.sh status \
  --states FAILED PREEMPTED INTERRUPTED EXHAUSTED

# One experiment's immutable attempt records.
bash PBS-launcher/architecture_ablation_arrays/sweep.sh history \
  --group LR3e-3 --encoder CNN --index 0

# Preview missing/interrupted experiments from the overnight groups.
bash PBS-launcher/architecture_ablation_arrays/sweep.sh resubmit \
  --groups LR3e-3 LR5e-3_Clip0.25 --encoders CNN

# Submit only the still-eligible subset; running/queued/success/failed/exhausted are skipped.
bash PBS-launcher/architecture_ablation_arrays/sweep.sh resubmit \
  --groups LR3e-3 LR5e-3_Clip0.25 --encoders CNN --submit

# Optional restriction to original indices, and a larger walltime after investigation.
bash PBS-launcher/architecture_ablation_arrays/sweep.sh resubmit \
  --groups LR3e-3 --encoders CNN --indices '0-3,8,12-15' --cnn-walltime 00:40:00

# Explicitly extend one persisted budget to five total starts. Default is preview.
bash PBS-launcher/architecture_ablation_arrays/sweep.sh budget \
  --group LR3e-3 --encoder CNN --index 0 --max-attempts 5 --apply

# After investigating an ordinary/numerical failure, explicitly permit another start.
bash PBS-launcher/architecture_ablation_arrays/sweep.sh budget \
  --group LR3e-3 --encoder CNN --index 0 --max-attempts 5 --retry-failed --apply

# Or reset the accounting window to three NEW attempts; retain previous records/artifacts.
bash PBS-launcher/architecture_ablation_arrays/sweep.sh budget \
  --group LR3e-3 --encoder CNN --index 0 --reset --max-attempts 3 --apply
```

Changing `--max-attempts` on submission sets the initial policy only. Existing
policies require the explicit `budget` command; increasing a limit alone does
not unblock ordinary application/numerical failures. Budget controls cannot
reset verified successes or modify queued/running/reserved experiments.

A short submission lock protects duplicate checks and reservations around qsub;
it adds no execution dependency, startup sleep or serialization between arrays.
If qsub times out or its acceptance cannot be confirmed, the reservation blocks
duplicates. Inspect PBS
first, then release that specific reservation (the command refuses while any
matching experiment is queued/running):

```bash
bash PBS-launcher/architecture_ablation_arrays/sweep.sh release-reservation \
  --token TOKEN_PRINTED_BY_THE_HELPER --apply
```

PBS progress (force Derecho even when invoked from Casper, and use the native
binary to avoid the local qstat cache wrapper):

```bash
PBS_DEFAULT=desched1 /opt/pbs/bin/qstat -t -u zel
PBS_DEFAULT=desched1 /opt/pbs/bin/qstat -x -t -u zel
/opt/pbs/bin/qstat -t -f 'ARRAY_ID[].desched1'
```

## Locally verified scheduler limits and runtime evidence

Read-only queries on 2026-10-21: `qmgr -c 'print server' desched1`,
`qmgr -c 'print queue preempt' desched1`, and `qstat -Bf desched1`.
The native server reports PBS 2024.1.6.20260805092812.

| Setting | Observed value |
|---|---|
| Server `max_array_size` | 10,000 |
| Server `max_queued` | `[u:PBS_GENERIC=5000]` |
| `preempt` routing | `pcpu`, `pgpu` |
| `pgpu.kill_delay` | 600 seconds |
| Explicit server/preempt/pgpu running-job or per-user GPU concurrency cap | None shown in exposed attributes |
| Server job-history retention | 72 hours |
| Scratch filesystem locking | Lustre mounted with `flock` |
| Existing active user jobs at inspection | None |

The launcher defaults to a configurable cap of 24 running subjobs per array.
Fair share, available nodes, project allocation and scheduler hooks can still
restrict starts; absence of an exposed scheduler cap does not guarantee that
all 24 subjobs per array start together. Native array/run-count holds can
still occur before the launcher starts (e.g. infrastructure failures); this
launcher does not manipulate PBS run_count to avoid those site policies.

The copied historical logs refer to the previous 2026-10-20 experiment set, not
measurements of these new LR groups. From 144 completed runs per encoder:

| Encoder | Median | 95th percentile | Maximum | New walltime |
|---|---:|---:|---:|---:|
| CNN | 510 s | 725 s | 747 s | 30 min |
| GNN | 680 s | 1027 s | 1078 s | 35 min |

Sources: [NCAR preemption behavior](https://ncar-hpc-docs.readthedocs.io/en/latest/pbs/preemption/),
[PBS array syntax](https://ncar-hpc-docs.readthedocs.io/en/latest/pbs/job-scripts/),
[OpenPBS exit-code definitions](https://github.com/openpbs/openpbs/blob/master/src/include/job.h).

Only static checks, CPU-only simulations, read-only scheduler queries and
`train.sh` dry-runs are performed during implementation. Actual GPU allocation,
cross-node locking, site preemption delivery and live PBS reruns require a later
user-submitted run to validate end to end. No background monitoring or automatic
submission service is installed.
