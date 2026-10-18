# Lewes CMIP6 — future_year

20 configurations: five GCMs × MSE, MSE + Tail, MSE + Peak, and MSE + Tail + Peak.
TRAIN 1979--2008; VAL 2009--2014; TEST 2070--2099 (30/6/30 year groups). Each year denotes the start of a paired year group.

Results: `/home/exouser/media/share/PACT/FormalRuns_0925/CMIP6_Lewes_QuickCheck_0929/future_year/<run_name>__20261003_051753`.
The prepared batch uses `PACT_RUNSTAMP=20261003_051753`; exact paths are in `manifest.csv`.

From `/home/exouser/StormSurge`, validate with:

```bash
DRY_RUN=1 bash configs/CMIP6_Lewes_QuickCheck_0929/future_year/launch_all.sh
```

Submit this group with:

```bash
bash configs/CMIP6_Lewes_QuickCheck_0929/future_year/launch_all.sh
```

For individual jobs, use `qsub_all.txt` in this folder. See [the station README](../README.md) for settings and scope.
