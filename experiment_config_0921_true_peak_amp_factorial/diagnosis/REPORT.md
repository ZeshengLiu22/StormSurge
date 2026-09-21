# TRAIN loss-scale diagnosis

Experimental analysis only. No factorial has been generated or launched.

`base` is physical trajectory MSE. `dual_base = base + body + 2 excess + 0.5 gate` is the actual trained objective. The requested R uses base MSE; a second ratio uses the complete Dual Base gradient, including cancellation.

Sixteen fixed, uniformly shuffled TRAIN batches of 256 by default; no event balancing or replacement. Each checkpoint uses the same windows and training-mode dropout seeds. Every component receives a separate full autograd backward through the same forward graph. No optimizer updates, parameter/buffer changes, `.grad` writes, or training RNG consumption occur during diagnosis.

`shared` includes every trainable parameter outside the head, `temporal` is its temporal Transformer subset, `excess` is head.excess, and `full` is all trainable parameters. Norms are Euclidean. Unused derivatives are zero. Zero denominators produce undefined ratios. `accum4` measures the norm of the mean gradient of four batches at a frozen state (not the mean of four norms). This is a complementary local accumulation estimate.

Production BF16 autocast/TF32; physical losses FP32; norm reduction FP64. Raw gradients precede Adam, weight decay and any optimizer preconditioning. A coefficient scales the recorded loss and gradient linearly; it does not predict the trajectory after retraining. Diagnostics do not use VAL/TEST or select checkpoints.

The single-GPU trajectory uses effective batch 1024, production Adam (weight_decay=1e-5), five-epoch warmup and the first 50 epochs of the 300-epoch cosine schedule. Early stopping is not used.

## Unweighted loss components

| Station | Epoch | Component | Mean | Median | Min | Max |
|---|---:|---|---:|---:|---:|---:|
| Battery | 50 | amp_true | 0.003176 | 0.002582 | 0.0005494 | 0.009236 |
| Battery | 50 | base | 0.002069 | 0.002037 | 0.001761 | 0.002537 |
| Battery | 50 | body | 0.001919 | 0.001905 | 0.001661 | 0.002442 |
| Battery | 50 | dual_base | 0.004867 | 0.004839 | 0.003989 | 0.006567 |
| Battery | 50 | excess | 0.0001241 | 0.0001203 | 5.313e-05 | 0.0002048 |
| Battery | 50 | gate | 0.001262 | 0.001217 | 0.0004738 | 0.002887 |
| Battery | 50 | tail | 0.004024 | 0.00394 | 0.001928 | 0.007297 |
| Boston | 50 | amp_true | 0.00378 | 0.003305 | 0.001241 | 0.009567 |
| Boston | 50 | base | 0.00103 | 0.0009967 | 0.0008679 | 0.001279 |
| Boston | 50 | body | 0.0009179 | 0.0008796 | 0.0008011 | 0.001119 |
| Boston | 50 | dual_base | 0.002478 | 0.002341 | 0.001972 | 0.003357 |
| Boston | 50 | excess | 8.144e-05 | 7e-05 | 1.822e-05 | 0.0001834 |
| Boston | 50 | gate | 0.0007348 | 0.0007977 | 0.000409 | 0.00136 |
| Boston | 50 | tail | 0.003133 | 0.002863 | 0.001023 | 0.007004 |
| CBBT | 0 | amp_true | 0.01268 | 0.01352 | 0.003834 | 0.02194 |
| CBBT | 0 | base | 0.02588 | 0.02507 | 0.02059 | 0.03082 |
| CBBT | 0 | body | 0.02286 | 0.02267 | 0.01897 | 0.02705 |
| CBBT | 0 | dual_base | 0.05194 | 0.05038 | 0.04192 | 0.06189 |
| CBBT | 0 | excess | 0.0004898 | 0.0005402 | 0.0001503 | 0.0008707 |
| CBBT | 0 | gate | 0.004457 | 0.004289 | 0.003351 | 0.0063 |
| CBBT | 0 | tail | 0.1661 | 0.1629 | 0.1084 | 0.255 |
| CBBT | 5 | amp_true | 0.01221 | 0.01298 | 0.003668 | 0.02129 |
| CBBT | 5 | base | 0.01882 | 0.01839 | 0.01686 | 0.02206 |
| CBBT | 5 | body | 0.01733 | 0.01698 | 0.01554 | 0.02101 |
| CBBT | 5 | dual_base | 0.03944 | 0.03878 | 0.03549 | 0.04543 |
| CBBT | 5 | excess | 0.0004718 | 0.0005184 | 0.0001451 | 0.0008454 |
| CBBT | 5 | gate | 0.004691 | 0.004476 | 0.003187 | 0.007144 |
| CBBT | 5 | tail | 0.04169 | 0.04495 | 0.02272 | 0.0688 |
| CBBT | 20 | amp_true | 0.00194 | 0.001892 | 0.000708 | 0.004595 |
| CBBT | 20 | base | 0.001424 | 0.001404 | 0.001216 | 0.001666 |
| CBBT | 20 | body | 0.001306 | 0.001301 | 0.00115 | 0.001538 |
| CBBT | 20 | dual_base | 0.003289 | 0.003385 | 0.002624 | 0.003886 |
| CBBT | 20 | excess | 7.69e-05 | 6.937e-05 | 2.791e-05 | 0.0001668 |
| CBBT | 20 | gate | 0.0008115 | 0.0008297 | 0.0002486 | 0.001552 |
| CBBT | 20 | tail | 0.00296 | 0.002916 | 0.001239 | 0.004824 |
| CBBT | 50 | amp_true | 0.001244 | 0.001322 | 0.0002063 | 0.002539 |
| CBBT | 50 | base | 0.0009133 | 0.0009023 | 0.0008049 | 0.001012 |
| CBBT | 50 | body | 0.0008469 | 0.0008481 | 0.0007499 | 0.0009515 |
| CBBT | 50 | dual_base | 0.002139 | 0.002105 | 0.001804 | 0.002533 |
| CBBT | 50 | excess | 5.053e-05 | 4.97e-05 | 1.209e-05 | 9.335e-05 |
| CBBT | 50 | gate | 0.000556 | 0.0005367 | 0.0001618 | 0.0009925 |
| CBBT | 50 | tail | 0.00144 | 0.001219 | 0.0005549 | 0.002653 |
| Lewes | 50 | amp_true | 0.002831 | 0.002598 | 0.0001688 | 0.005739 |
| Lewes | 50 | base | 0.0009062 | 0.0009248 | 0.0007368 | 0.00106 |
| Lewes | 50 | body | 0.0007916 | 0.0007967 | 0.0006864 | 0.0009301 |
| Lewes | 50 | dual_base | 0.002202 | 0.002194 | 0.001604 | 0.002835 |
| Lewes | 50 | excess | 0.0001004 | 0.0001031 | 1.491e-05 | 0.0001911 |
| Lewes | 50 | gate | 0.0006077 | 0.0005779 | 0.0001659 | 0.001178 |
| Lewes | 50 | tail | 0.002723 | 0.002544 | 0.000422 | 0.005462 |

## CBBT candidate and existing dual contributions

Batch summaries; gradient ratios here refer to shared parameters. Full/temporal/excess and accumulation statistics, including median and range, are in candidate_summary.csv.

| Epoch | Component | Weight | Weighted loss mean | R/base mean | R/base median | R/base min–max | R/Dual Base mean | R/Dual Base max |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | amp_true | 0.0001 | 1.268e-06 | 0 | 0 | 0–0 | 0 | 0 |
| 0 | amp_true | 0.0003 | 3.803e-06 | 0 | 0 | 0–0 | 0 | 0 |
| 0 | amp_true | 0.0007 | 8.875e-06 | 0 | 0 | 0–0 | 0 | 0 |
| 0 | amp_true | 0.001 | 1.268e-05 | 0 | 0 | 0–0 | 0 | 0 |
| 0 | amp_true | 0.003 | 3.803e-05 | 0 | 0 | 0–0 | 0 | 0 |
| 0 | body | 1 | 0.02286 | 0.9485 | 0.9463 | 0.9191–0.983 | 0.4868 | 0.4957 |
| 0 | excess | 2 | 0.0009796 | 0 | 0 | 0–0 | 0 | 0 |
| 0 | gate | 0.5 | 0.002228 | 0 | 0 | 0–0 | 0 | 0 |
| 0 | tail | 0.01 | 0.001661 | 0.07005 | 0.06869 | 0.05279–0.09856 | 0.03599 | 0.05136 |
| 0 | tail | 0.025 | 0.004154 | 0.1751 | 0.1717 | 0.132–0.2464 | 0.08997 | 0.1284 |
| 0 | tail | 0.05 | 0.008307 | 0.3502 | 0.3434 | 0.2639–0.4928 | 0.1799 | 0.2568 |
| 0 | tail | 0.1 | 0.01661 | 0.7005 | 0.6869 | 0.5279–0.9856 | 0.3599 | 0.5136 |
| 5 | amp_true | 0.0001 | 1.221e-06 | 4.038e-07 | 4.662e-07 | 1.358e-07–6.461e-07 | 2.016e-07 | 3.226e-07 |
| 5 | amp_true | 0.0003 | 3.664e-06 | 1.212e-06 | 1.399e-06 | 4.075e-07–1.938e-06 | 6.048e-07 | 9.677e-07 |
| 5 | amp_true | 0.0007 | 8.55e-06 | 2.827e-06 | 3.264e-06 | 9.508e-07–4.523e-06 | 1.411e-06 | 2.258e-06 |
| 5 | amp_true | 0.001 | 1.221e-05 | 4.038e-06 | 4.662e-06 | 1.358e-06–6.461e-06 | 2.016e-06 | 3.226e-06 |
| 5 | amp_true | 0.003 | 3.664e-05 | 1.212e-05 | 1.399e-05 | 4.075e-06–1.938e-05 | 6.048e-06 | 9.677e-06 |
| 5 | body | 1 | 0.01733 | 1.006 | 1.006 | 1.001–1.014 | 0.5023 | 0.5055 |
| 5 | excess | 2 | 0.0009436 | 0.0003126 | 0.0003725 | 9.084e-05–0.000476 | 0.000156 | 0.0002375 |
| 5 | gate | 0.5 | 0.002345 | 0.008384 | 0.008408 | 0.003785–0.01509 | 0.004186 | 0.007525 |
| 5 | tail | 0.01 | 0.0004169 | 0.006761 | 0.007036 | 0.003565–0.01267 | 0.003375 | 0.006317 |
| 5 | tail | 0.025 | 0.001042 | 0.0169 | 0.01759 | 0.008913–0.03168 | 0.008438 | 0.01579 |
| 5 | tail | 0.05 | 0.002085 | 0.0338 | 0.03518 | 0.01783–0.06336 | 0.01688 | 0.03159 |
| 5 | tail | 0.1 | 0.004169 | 0.06761 | 0.07036 | 0.03565–0.1267 | 0.03375 | 0.06317 |
| 20 | amp_true | 0.0001 | 1.94e-07 | 0.0004871 | 0.0004335 | 0.0002255–0.001202 | 0.0002087 | 0.0004691 |
| 20 | amp_true | 0.0003 | 5.819e-07 | 0.001461 | 0.001301 | 0.0006765–0.003605 | 0.000626 | 0.001407 |
| 20 | amp_true | 0.0007 | 1.358e-06 | 0.00341 | 0.003035 | 0.001578–0.008413 | 0.001461 | 0.003284 |
| 20 | amp_true | 0.001 | 1.94e-06 | 0.004871 | 0.004335 | 0.002255–0.01202 | 0.002087 | 0.004691 |
| 20 | amp_true | 0.003 | 5.819e-06 | 0.01461 | 0.01301 | 0.006765–0.03605 | 0.00626 | 0.01407 |
| 20 | body | 1 | 0.001306 | 0.8777 | 0.8832 | 0.7037–1.009 | 0.3863 | 0.5127 |
| 20 | excess | 2 | 0.0001538 | 0.2833 | 0.2237 | 0.1292–0.5626 | 0.1212 | 0.2289 |
| 20 | gate | 0.5 | 0.0004057 | 0.5207 | 0.4891 | 0.07839–1.12 | 0.2124 | 0.3848 |
| 20 | tail | 0.01 | 2.96e-05 | 0.08046 | 0.0832 | 0.03316–0.1293 | 0.03429 | 0.05045 |
| 20 | tail | 0.025 | 7.399e-05 | 0.2012 | 0.208 | 0.08289–0.3231 | 0.08571 | 0.1261 |
| 20 | tail | 0.05 | 0.000148 | 0.4023 | 0.416 | 0.1658–0.6463 | 0.1714 | 0.2523 |
| 20 | tail | 0.1 | 0.000296 | 0.8046 | 0.832 | 0.3316–1.293 | 0.3429 | 0.5045 |
| 50 | amp_true | 0.0001 | 1.244e-07 | 0.0003647 | 0.0003328 | 5.372e-05–0.0009368 | 0.0001666 | 0.0003603 |
| 50 | amp_true | 0.0003 | 3.731e-07 | 0.001094 | 0.0009984 | 0.0001612–0.00281 | 0.0004997 | 0.001081 |
| 50 | amp_true | 0.0007 | 8.706e-07 | 0.002553 | 0.00233 | 0.000376–0.006557 | 0.001166 | 0.002522 |
| 50 | amp_true | 0.001 | 1.244e-06 | 0.003647 | 0.003328 | 0.0005372–0.009368 | 0.001666 | 0.003603 |
| 50 | amp_true | 0.003 | 3.731e-06 | 0.01094 | 0.009984 | 0.001612–0.0281 | 0.004997 | 0.01081 |
| 50 | body | 1 | 0.0008469 | 1.005 | 0.9972 | 0.9584–1.088 | 0.4734 | 0.5471 |
| 50 | excess | 2 | 0.0001011 | 0.1871 | 0.1535 | 0.02663–0.7 | 0.0843 | 0.2692 |
| 50 | gate | 0.5 | 0.000278 | 0.4384 | 0.3611 | 0.1266–1.07 | 0.1999 | 0.4269 |
| 50 | tail | 0.01 | 1.44e-05 | 0.0358 | 0.03484 | 0.005207–0.09215 | 0.01651 | 0.03544 |
| 50 | tail | 0.025 | 3.599e-05 | 0.0895 | 0.08709 | 0.01302–0.2304 | 0.04127 | 0.0886 |
| 50 | tail | 0.05 | 7.198e-05 | 0.179 | 0.1742 | 0.02604–0.4608 | 0.08254 | 0.1772 |
| 50 | tail | 0.1 | 0.000144 | 0.358 | 0.3484 | 0.05207–0.9215 | 0.1651 | 0.3544 |

## Files

- component_summary.csv: all component loss/norm/ratio/cosine mean, median, minimum and maximum.
- candidate_summary.csv: analytical weights and existing weighted dual contributions for every station/state/group.
- batch_gradients.csv and candidate_batches.csv: individual batches and frozen accumulation groups.
- STATION/metadata.json, fixed_batches.json, states.csv, trajectory.csv: provenance, selected TRAIN windows, model integrity checks, learning rates and TRAIN-only trajectory progress.
- STATION/checkpoints/: fixed-epoch states, including optimizer/scheduler and RNG for audit; no best-model selection.
