# Measured Tail / true-peak Amp scale diagnosis

**Recommend `TAIL_LAMBDA=0.025` and `EXCESS_AMP_LOSS_WEIGHT=0.003` for the subsequent factorial, subject to review.** No factorial configurations or runs have been generated. These are scale-based choices, not performance-selected settings.

The diagnostic ran on clean production `main` at `af53de756860765a9809b0c5c44bc828c2a08359`. Only this experimental directory was added; production training, loss definitions, configs and tests remain unchanged. Four fresh Dual Base trajectories were trained to epoch 50. CBBT was sampled at 0/5/20/50; Lewes, Battery and Boston at 50. Every state uses the same 16 fixed TRAIN batches (4096 windows per station), without replacement or event balancing. TRAIN has 13,224 windows and 662 strict events per station, giving `q_E=0.0500604961`. No VAL or TEST graph file was opened, and no checkpoint or coefficient was selected using their performance.

Settings: NCEP; production direct-dual GraphSAGE + Transformer; history 24 h; hidden 128; Q95; body/excess/gate=1/2/0.5; LR=0.005; batch=256; accumulation=4; training seed=42; deterministic=0; zscore; no augmentation, feature clipping, gradient clipping, slope, shape, Tail or Amp during training. One H100, effective batch 1024, BF16/TF32, production Adam and the first 50 epochs of the 300-epoch cosine schedule. There are 650 optimizer updates per station. Fixed-batch sampling uses an independent seed 1042.

`L_base` is physical trajectory MSE; the actual trained objective is `J = L_base + L_body + 2 L_excess + 0.5 L_gate`. Below, gradient percentages are means of paired batch ratios, and loss percentages divide the weighted mean loss by mean J. The CSVs separately report all requested means, medians and min–max ranges.

## Why this pair

- Tail 0.025 adds 50% extra trajectory-MSE weight to a tail window (`0.025 / 0.05`), while retaining the production inclusive threshold, whole-window error and fixed denominator. It has a visible shared-model effect without approaching the base gradient consistently. Tail 0.1 is less conservative: at CBBT epoch 20 it exceeds the shared trajectory-MSE gradient in 4 of 16 batches (it does not exceed the complete J gradient there).
- Amp 0.003 is the upper end of the requested neighborhood. At the true-peak coordinate, its direct excess gradient is about 18% of the existing 2×excess term for an event window: `0.003 * 6 / (2*q_E) = 0.1798`. Only that observed-peak coordinate receives direct Amp supervision. The loss and measured shared gradient stay small, but the excess-branch effect is meaningful. The lower Amp candidates produce proportionally weaker effects; there is no reason from these scales to target equality with the base gradient.
- Every candidate is an analytical multiplication of recorded unweighted losses and gradients. No candidate combination was retrained.

## CBBT over the clean trajectory

| Epoch | 0.025 × Tail loss | 0.003 × Amp loss | Tail shared / base | Tail shared / J | Amp shared / base | Amp shared / J |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.0041537 | 3.8034e-05 | 17.5% | 9% | 0% | 0% |
| 5 | 0.0010423 | 3.6641e-05 | 1.69% | 0.844% | 0.00121% | 0.000605% |
| 20 | 7.3992e-05 | 5.819e-06 | 20.1% | 8.57% | 1.46% | 0.626% |
| 50 | 3.599e-05 | 3.7311e-06 | 8.95% | 4.13% | 1.09% | 0.5% |

Amp has zero shared gradient at initialization because the production excess output weights initialize to zero; its excess-branch gradient is nonzero. Its epoch-5 shared gradient is also very small. The learned-state measurements therefore carry more information for scale choice than initialization alone.

## All requested candidates at CBBT epoch 50

| Term | Weight | Weighted loss mean | Loss / mean J | Shared gradient / base mean | Shared gradient / J mean | Excess gradient / J mean |
|---|---:|---:|---:|---:|---:|---:|
| tail | 0.01 | 1.4396e-05 | 0.673% | 3.58% | 1.65% | 8.23% |
| tail | 0.025 | 3.599e-05 | 1.68% | 8.95% | 4.13% | 20.6% |
| tail | 0.05 | 7.198e-05 | 3.36% | 17.9% | 8.25% | 41.2% |
| tail | 0.1 | 0.00014396 | 6.73% | 35.8% | 16.5% | 82.3% |
| amp_true | 0.0001 | 1.2437e-07 | 0.00581% | 0.0365% | 0.0167% | 0.201% |
| amp_true | 0.0003 | 3.7311e-07 | 0.0174% | 0.109% | 0.05% | 0.603% |
| amp_true | 0.0007 | 8.7058e-07 | 0.0407% | 0.255% | 0.117% | 1.41% |
| amp_true | 0.001 | 1.2437e-06 | 0.0581% | 0.365% | 0.167% | 2.01% |
| amp_true | 0.003 | 3.7311e-06 | 0.174% | 1.09% | 0.5% | 6.03% |

The complete candidate tables cover every checkpoint, parameter group and frozen accumulation group; see [candidate_summary.csv](diagnosis/candidate_summary.csv) for mean/median/min/max, and [REPORT.md](diagnosis/REPORT.md) for all CBBT states and existing 1×body, 2×excess and 0.5×gate contributions.

## Cross-station sanity at epoch 50

| Station | Tail loss | Tail / mean J | Amp loss | Amp / mean J |
|---|---:|---:|---:|---:|
| CBBT | 3.599e-05 | 1.68% | 3.7311e-06 | 0.174% |
| Lewes | 6.8073e-05 | 3.09% | 8.4919e-06 | 0.386% |
| Battery | 0.0001006 | 2.07% | 9.5291e-06 | 0.196% |
| Boston | 7.8313e-05 | 3.16% | 1.134e-05 | 0.458% |

| Station | Tail shared / base | Tail shared / J | Amp shared / base | Amp shared / J | Amp excess branch / J |
|---|---:|---:|---:|---:|---:|
| CBBT | 8.95% | 4.13% | 1.09% | 0.5% | 6.03% |
| Lewes | 20.5% | 8.68% | 2.74% | 1.16% | 6.87% |
| Battery | 6.76% | 3.05% | 1.14% | 0.531% | 5.65% |
| Boston | 14.8% | 6.81% | 1.96% | 0.903% | 6.03% |

Ratio of strongest to weakest station mean, at epoch 50 (the coefficient cancels from this comparison):

| Parameter group | Tail R/base spread | Amp R/base spread | Tail R/J spread | Amp R/J spread |
|---|---:|---:|---:|---:|
| shared | 3.03× | 2.5× | 2.85× | 2.33× |
| temporal | 4.31× | 2.49× | 4.08× | 2.31× |
| excess | 1.08× | 1.45× | 1.51× | 1.21× |
| full | 3.03× | 2.62× | 2.84× | 2.44× |

The largest spread in this table is **4.31×**. The common pair passes the requested check for a tenfold station imbalance at this sampled trained state.

The matching local accumulation calculation averages four gradient vectors before taking each norm:

| Station | Tail shared / J mean [min–max] | Amp shared / J mean [min–max] |
|---|---:|---:|
| CBBT | 3.59% [2.55%–4.73%] | 0.469% [0.271%–0.732%] |
| Lewes | 8.76% [6.74%–11.1%] | 1.33% [0.808%–1.82%] |
| Battery | 2.14% [0.903%–3.13%] | 0.448% [0.303%–0.595%] |
| Boston | 7% [6.01%–8.37%] | 0.921% [0.758%–1.26%] |

Across all 112 sampled batches/states, the sum of the two selected shared auxiliary gradient norms is at most **19.2%** of the complete J gradient norm. By the triangle inequality this also bounds their combined shared gradient; it is not a measurement from a retrained combined-loss model.

## Existing weighted dual losses at epoch 50

| Station | Base MSE | 1 × body | 2 × excess | 0.5 × gate | Complete J |
|---|---:|---:|---:|---:|---:|
| CBBT | 0.00091325 | 0.00084694 | 0.00010105 | 0.00027798 | 0.0021392 |
| Lewes | 0.00090617 | 0.00079155 | 0.00020074 | 0.00030383 | 0.0022023 |
| Battery | 0.002069 | 0.0019188 | 0.00024825 | 0.00063096 | 0.004867 |
| Boston | 0.00103 | 0.00091794 | 0.00016288 | 0.0003674 | 0.0024782 |

## Interpretation and audit

These are raw loss gradients before Adam preconditioning, measured in training mode with identical dropout realizations for the components and across checkpoints. The shared gradient includes all 585,732 trainable parameters outside the head; the temporal subset has 396,548, the excess branch 33,281, and the full model 685,575. The backbone is never detached. Separate backward passes do not perform optimizer updates or write `.grad` buffers.

At CBBT epoch 50 the average shared cosine with J is −0.215 for Tail and −0.234 for Amp. Thus even small terms can alter the optimization direction. These results support a modest coefficient pair; they do not establish improved forecasts, robustness across seeds, or behavior over all 300 epochs. Those are questions for the reviewed factorial, which remains on hold.

- Four focused experimental tests passed: objective equivalence/true-peak gradients, preservation and accumulation, TRAIN-only loading, and undefined zero-denominator handling.
- All 16 production Amp regression tests passed, including CUDA FP16/BF16 and two-rank Gloo. The first sandboxed test attempt could not bind Gloo; the complete rerun with local-interface/GPU access passed with no skips.
- All seven snapshot hashes match before/after diagnostics; CPU/CUDA/shuffle RNG checks passed. All four trajectories completed 50 epochs, with 650 updates each.
- Audit verifies TRAIN-only opened-file manifests, distinct fixed windows, disabled training auxiliaries, 3,920 raw loss/gradient records and 6,720 candidate records, and analytical scaling of every candidate row.

Reproducibility: [tool](diagnose_loss_weights.py), [instructions](README.md), [audit](diagnosis/audit.json), [unweighted summary](diagnosis/component_summary.csv), [candidate summary](diagnosis/candidate_summary.csv), [raw gradients](diagnosis/batch_gradients.csv). Per-station metadata, fixed-batch manifests, learning-rate/trajectory CSVs and seven fixed-epoch checkpoints are retained. CBBT was originally written to `diagnosis_cbbt` and then moved to `diagnosis`; `provenance_CBBT.json` preserves its original invocation, and `provenance.json` records the cross-station invocation.
