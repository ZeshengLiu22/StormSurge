# Boston CMIP6 — past_only

10 configs: AWI, CNRM, EC_EARTH, MPI, MRI × `T0000_EP0000` / `T0500_EP0100`.
Cane5 is excluded by request. All five selected datasets passed the data audit.

Data source: `./Data/Grid4_New_PastOnly/<dataset>/graphs`.
TRAIN_RATIO=`0.6`, VAL_RATIO=`0.2`; chronological year groups;
`SHUFFLE_YEARS=0`, `FUTURE_ONLY=0`.

| Split | Year groups | Range |
| --- | ---: | --- |
| TRAIN | 22 | 1979_1980–2000_2001 |
| VAL | 7 | 2001_2002–2007_2008 |
| TEST | 7 | 2008_2009–2014_2015 |

Results root: `/home/exouser/media/share/PACT/FormalRuns_0925/CMIP6_Boston_QuickCheck_0929/past_only`.
Exact per-run paths and the fixed batch stamp `20260929_054511` are in `manifest.csv`.
Both checkpoint roles are saved; **overall is primary**.
Formulations and unchanged training settings are in [the shared README](../README.md).

Audit: `python ../audit_configs.py past_only` (use the configured training Python).
Launch from the repository: `bash configs/CMIP6_Boston_QuickCheck_0929/past_only/launch_all.sh`.
Use `DRY_RUN=1` for a preview. Queue receipts: `_orchestration/submissions.tsv` under the results root.
