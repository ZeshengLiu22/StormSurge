# Model parameter reporting

Every `train.py` and `infer.py` run prints one timestamped line at startup:

```text
[Parameters] total=... trainable=... non_trainable=... head=...
```

This applies to both baseline encoders, H=0 and positive history, and every
PACT temporal/head variant: single, direct dual, severity-shape dual, and
fixed-gate ablations. There is no enable flag or loss-dependent condition.
Training prints only on rank 0. Values describe one model replica, so DDP does
not multiply counts by world size.

`emulator/models/reporting.py::count_model_parameters` reads registered
parameters from the actual model. `total` counts all unique `Parameter`
objects, including frozen parameters; `trainable` counts those with
`requires_grad=True`; `non_trainable` is their difference. Constants stored as
buffers (TRAIN thresholds/scales, fixed gate probabilities, etc.) are excluded.
The helper does not execute a forward pass, initialize another model, consume
random numbers, or change parameter values, gradient flags or model mode.
It counts registered capacity, not only parameters receiving nonzero gradients
on a particular batch.

The complete JSON object has this structure:

```text
model_parameters:
  total, trainable, non_trainable
  head:
    total, trainable, non_trainable
  backbone:
    total, trainable, non_trainable
  head_branches:
    <branch name>:
      total, trainable, non_trainable
```

The head is each model's `model.head`; backbone counts exclude all parameters
belonging to that head. A baseline's linear head has no child branches. PACT
single reports `regression`; direct reports `body`, `excess`, and learned
`gate`; severity-shape reports `body`, `severity`, `shape`, and learned `gate`.
Fixed gates have no learned gate branch. Shared parameter objects are counted
once within each reported count; if future heads share weights between
different branches, those branch counts may overlap.

Counts are saved under `model_parameters` in:

- Every training checkpoint, including retained candidates and auxiliaries.
- `summary_<stem>.json`.
- Inference `metrics.json` and `metrics_per_year_<report_stem>.json`.

Inference recomputes counts from the reconstructed model, so checkpoints
without this metadata use the same reporting path. Counts are derived report
data, not new CLI/model config fields, and do not alter run config hashes.

At `hidden_channels=128` and `head_hidden=256`, each PACT head MLP contains
`128*256 + 256 + 256 + 1 = 33,281` parameters. With every other setting matched,
severity-shape adds one such MLP over direct, including for fixed-gate models.
Actual total capacity depends on the selected encoder, temporal block, history
configuration, feature dimensions and head width; each run reports its own
counts. The change does not add a capacity-matched model or choose an experiment.

The accompanying [severity-shape correction](SEVERITY_SHAPE.md) adds epsilon
before predicted shape normalization. Tests exercise logits below epsilon
and complete softplus underflow, exact unit shape peaks, physical severity
reconstruction, finite gradients and CPU/CUDA autocast. Parameter tests cover
48 model/encoder/history/width combinations, manually counted frozen/shared
weights and buffers, six synthetic training/inference round trips, and
rank-0-only reporting in a real two-rank Gloo run. Single/direct frozen
training references remain exact; corrected severity-shape trajectories are
equal across all five checkpoint-selection modes. No real data experiments
or new experiment configurations are part of this change.

Validation completed with **193 tests passing**, including CPU/CUDA autocast
and two-rank Gloo. See the [validation record](audit/evidence/shape_parameters_validation.json)
and [full test output](audit/evidence/shape_parameters_tests.txt).
