# CBBT CMIP6 — past_only

20 configurations: five GCMs × MSE, MSE + Tail, MSE + Peak, and MSE + Tail + Peak.
TRAIN 1979--2000; VAL 2001--2007; TEST 2008--2014 (22/7/7 year groups). Each year denotes the start of a paired year group.

Results: `/home/exouser/media/share/PACT/FormalRuns_0925/CMIP6_CBBT_QuickCheck_0929/past_only/<run_name>__20261003_051753`.
The prepared batch uses `PACT_RUNSTAMP=20261003_051753`; exact paths are in `manifest.csv`.

From `/home/exouser/StormSurge`, validate with:

```bash
DRY_RUN=1 bash configs/CMIP6_CBBT_QuickCheck_0929/past_only/launch_all.sh
```

Submit this group with:

```bash
bash configs/CMIP6_CBBT_QuickCheck_0929/past_only/launch_all.sh
```

For individual jobs, use `qsub_all.txt` in this folder. See [the station README](../README.md) for settings and scope.
