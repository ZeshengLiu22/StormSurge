# Boston CMIP6 — past_only

20 configs: AWI, CNRM, EC_EARTH, MPI, MRI × `T0000_EP0000` / `T0500_EP0100` /
`T0500_EP0000` (Tail Only) / `T0000_EP0100` (Peak Only).
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
Exact per-run paths are in `manifest.csv`: original batch stamp `20260929_054511`,
additional Tail Only / Peak Only batch stamp `20260930_233312`.
Both checkpoint roles are saved; **overall is primary**.
Formulations and unchanged training settings are in [the shared README](../README.md).

Audit from the repository with the configured training Python:

```bash
/home/exouser/.conda/envs/torchpyg-cu12x/bin/python configs/CMIP6_Boston_QuickCheck_0929/audit_configs.py past_only
```

For the 10 additional ablations in this group, copy the setup lines and the `past_only`
commands from [qsub_ablations.txt](../qsub_ablations.txt), then submit them individually.
The existing `launch_all.sh` describes only the original 10 runs; its queue receipts
remain in `_orchestration/submissions.tsv` under the results root.
