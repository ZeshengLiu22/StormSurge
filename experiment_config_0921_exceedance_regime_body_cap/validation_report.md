# Validation: PASS

This is the historical pre-training validation snapshot from 2026-09-21 03:38 UTC.
Statements below about jobs not yet submitted describe that checkpoint. All 24
study runs subsequently completed; see the concluded [study outcome](README.md#study-outcome).

Base: `d665a4b0f54700a2564e54187f6c067c470d1df0`. Implementation: `06dd17ecbc527842a70edf6520a123c9edc6975b`.

- Seven focused tests passed; soft output/gradient/RNG parity, exact-cap boundaries, parameter counts, old config defaults, metadata and inference round trips.
- Full suite: **217 tests, 0 failures, 0 errors, 0 skips** (64.307 seconds), including CUDA and DDP.
- Exactly 24 standalone configs and 24 parsed production launcher dry runs passed. Only the three study factors and identity differ within a station.
- `run_all.sh --dry-run` also emitted exactly 24 training commands without queue submissions or training artifacts.
- Frozen data/threshold, normalization, loss, spatial/temporal architecture, loader, checkpoint-selection, inference and metric sources match the base byte for byte.
- Every station/condition uses the same splits and seeded initialization semantics. Random weights/RNG are identical; fitted threshold and prior-based gate bias differ between Q95/Q90 as expected.
- All six conditions match the actual TRAIN sample permutation for two consecutive epochs; epochs differ from each other.
- All 24 configurations have **685,447 parameters** (99,843 head, 585,604 backbone).
- All 24 synthetic forward/loss/backward checks passed on **NVIDIA H100 80GB HBM3**, BF16 AMP and TF32, with no optimizer steps.
- All ten requested final metrics retain their original definitions. AllRMSE/AllMAE use the full TEST split; extreme metrics use its top 5%, with exactly 193/3,848 windows per station.
- Q95 threshold fits exactly match production defaults. Q90 fits match across T2/T3/T4/T5; tail/WMSE thresholds stay fixed.
- Primary `main` remains clean at the base SHA. No merge/cherry-pick from the previous excess-risk study; no study jobs submitted; external results unchanged.

| Station | TRAIN windows | Q95 threshold (m) | Q95 events | Q90 threshold (m) | Q90 events |
| --- | ---: | ---: | ---: | ---: | ---: |
| CBBT | 13,224 | 0.212631613016 | 662 | 0.133888617158 | 1,323 |
| Lewes | 13,224 | 0.281014084816 | 662 | 0.185592353344 | 1,323 |
| Battery | 13,224 | 0.432081997395 | 662 | 0.339308053255 | 1,323 |
| Boston | 13,224 | 0.349115967751 | 662 | 0.290335148573 | 1,323 |

Split counts for every station: TRAIN 13,224 / VAL 4,208 / TEST 3,848. Full precision physical and normalized thresholds, event priors, year groups, target/split/order hashes, resolved arguments and frozen source hashes are in [validation_report.json](validation_report.json).

The first sandboxed full-suite attempt could not access local DDP/GPU communication and lacked preprocessing packages. The successful full run used temporary test-only dependencies and local process/GPU access; no canonical environment files were changed. Reproduction commands and dependency versions are in [validation/tests.json](validation/tests.json).

DETERMINISTIC=0 retains the established protocol; matched initialization and data ordering do not imply bitwise GPU training repeatability. GPU checks use synthetic inputs and no optimizer steps. No 300-epoch study training has started.

The launch commands and experiment interpretation are in [README.md](README.md).
