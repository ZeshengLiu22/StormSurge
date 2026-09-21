# True-peak Amp / Tail scale diagnosis

Read [DIAGNOSIS.md](DIAGNOSIS.md) for the measured recommendation and station
comparison. Detailed mean/median/range tables are in
[component_summary.csv](diagnosis/component_summary.csv) and
[candidate_summary.csv](diagnosis/candidate_summary.csv).

This directory is experimental. It contains a TRAIN-only scale and optimization
diagnostic, not factorial configurations or a factorial launcher. Production
training code is unchanged. Review the measured diagnosis before designing the
20-run factorial.

`diagnose_loss_weights.py` trains one fresh production direct-dual model per
station using the existing `run_epoch`, `ForecastLoss`, model, preprocessing and
threshold routines. CBBT is inspected at epochs 0, 5, 20 and 50; the other
stations are inspected at epoch 50. A 50-epoch stop retains the first 50 learning
rates of the normal **300-epoch** cosine schedule, including five warmup epochs.
There is no VAL/TEST loading, checkpoint selection or early stopping.

The protocol is NCEP, GraphSAGE + Transformer, history 24 h, hidden 128, Q95,
body/excess/gate weights 1/2/0.5, LR 0.005, batch 256, accumulation 4, seed 42,
deterministic 0, zscore, no feature augmentation or clipping, no gradient
clipping, and no auxiliary objectives during training. Active production shell
settings provide the remaining architecture and BF16/TF32 settings. Execution is
single-GPU, with effective batch 1024; the four-GPU launcher default is recorded
but is not used. Every trajectory starts from seed 42.

Sixteen batches (4096 TRAIN windows) are drawn once without replacement using a
separate seed 1042, without event balancing. Model diagnostics use training-mode
dropout with fixed batch seeds shared across checkpoints. All components use the
same forward graph, with separate full parameter derivatives obtained by
`torch.autograd.grad`. The code verifies unchanged model state, `.grad` buffers,
module modes and CPU/CUDA/shuffle RNG. It also reports gradients averaged over
four fixed batches before computing norms, to examine accumulation cancellation.
These are local gradients at a frozen checkpoint, before Adam preconditioning.

`base` means physical trajectory MSE. The trained objective is
`dual_base = base + body + 2*excess + 0.5*gate`. Both gradient denominators are
reported explicitly. `shared` covers all trainable parameters outside the head;
`temporal` is the temporal Transformer subset; `excess` is the excess MLP;
`full` covers all trainable parameters. Activation derivatives are not used as a
substitute for backbone parameter gradients.

Tail exactly retains its inclusive TRAIN threshold and fixed `tail_frac=0.05`
denominator. Amp calls the production true-peak helper: physical target argmax,
strict physical TRAIN event mask, ungated excess at that target horizon, and
fixed empirical TRAIN `q_E`. There is no predicted-max Amp implementation here.

Run from the repository root, with the existing torchpyg environment:

```bash
python -B experiment_config_0921_true_peak_amp_factorial/test_diagnose_loss_weights.py
python -B experiment_config_0921_true_peak_amp_factorial/diagnose_loss_weights.py \
  --stations CBBT --output experiment_config_0921_true_peak_amp_factorial/diagnosis
cp experiment_config_0921_true_peak_amp_factorial/diagnosis/provenance.json \
  experiment_config_0921_true_peak_amp_factorial/diagnosis/provenance_CBBT.json
# Inspect CBBT's scales first; this second command adds fresh station trajectories.
python -B experiment_config_0921_true_peak_amp_factorial/diagnose_loss_weights.py \
  --stations Lewes Battery Boston \
  --output experiment_config_0921_true_peak_amp_factorial/diagnosis
```

Station output directories must be new, preventing accidental overwrites of
trajectories. Tables can be regenerated without any training using `--report-only`.
Candidate coefficients are applied analytically to recorded losses and gradients;
no candidate-specific models are trained. See each output's `REPORT.md`,
`component_summary.csv` and `candidate_summary.csv`, plus station metadata and
fixed-batch manifests. Fixed-epoch `.pt` files are local audit artifacts and are
ignored by the repository's existing model-file rule.

The experimental tests check production-objective equivalence, physical
true-peak gradient placement, event-free zero gradients, state/RNG preservation,
full-model and accumulated gradient norms, analytical coefficient scaling and
the exclusion of deliberately unreadable VAL/TEST graph files.
