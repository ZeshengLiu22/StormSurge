# Boston CMIP6 quick check — 0929

Test the NCEP extreme-aware objective using **two groups of 10 configs** (20 runs).
Included in each group: `CMIP6_AWI`, `CMIP6_CNRM`, `CMIP6_EC_EARTH`, `CMIP6_MPI`, `CMIP6_MRI`.
Excluded: `CMIP6_Cane5`, by request. All five selected datasets passed completeness,
tensor/history, target/timestamp, and split-population checks.

## Groups and chronological splits

| Group | Data source | TRAIN / VAL ratios | TRAIN / VAL / TEST year groups |
| --- | --- | --- | --- |
| [past_only](past_only/README.md) | `Data/Grid4_New_PastOnly` | 0.6 / 0.2 | 22 / 7 / 7, exactly matching NCEP |
| [future_year](future_year/README.md) | `Data/Grid4_New` | 0.4545454545 / 0.0909090909 | 30 / 6 / 30; TEST is all 2070_2071–2099_2100 |

`SHUFFLE_YEARS=0` and `FUTURE_ONLY=0` in both groups. The latter keeps historical
years available for TRAIN/VAL; future years appear only in the future-year TEST split.
Past-only folders contain unused future dictionaries; the loader reads only their 36 historical graph files.

## Two loss formulations

- `T0000_EP0000`: Global MSE.
- `T0500_EP0100`: Global MSE + 0.05 × Tail-MSE + 0.01 × EpisodeGTAlignedPeak-MSE.

Tail uses strict GT > TRAIN Q95; episode peak supervises one GT peak timestamp per
physical TRAIN episode with the existing fixed TRAIN prevalence normalization.

## Common settings and checkpoints

Boston; Single; GraphSAGE + Transformer; 24 h history; hidden 128; LR 5e-3;
300 epochs; batch 256; accumulation 4; seed 42; MSE global loss; cosine scheduler;
BF16 AMP and TF32. All other architecture, Adam, warmup, normalization, loader,
and evaluation settings match the successful current Boston NCEP formal templates.
TRAIN-only fitting and the existing eleven final metrics are unchanged.
Both `best_overall.pt` and `best_exceedance.pt` are retained and reevaluated on VAL/TEST.
**Overall is primary** for reporting, aliases, and primary prediction exports.

## Exact paths and launch

Config root: `/home/exouser/StormSurge/configs/CMIP6_Boston_QuickCheck_0929`; separate `past_only/` and `future_year/` folders.

Results root: `/home/exouser/media/share/PACT/FormalRuns_0925/CMIP6_Boston_QuickCheck_0929`; separate `past_only/` and `future_year/` folders.

`manifest.csv` records all 20 exact config/result paths; each group also has its own
manifest, audit report, template diff, and dry-run commands. No NCEP config/result is modified.

From `/home/exouser/StormSurge`:

```bash
DRY_RUN=1 bash configs/CMIP6_Boston_QuickCheck_0929/launch_all.sh
bash configs/CMIP6_Boston_QuickCheck_0929/launch_all.sh
```

The launcher audits both groups before submission and uses existing `qsub_local` →
`train.sh` → tmux, one GPU job at a time. Past-only jobs are queued first.
Queue IDs are saved in each results folder's `_orchestration/submissions.tsv`.
Use `qstat_local` in an interactive shell for status. The fixed batch cannot be submitted twice.

## Submitted batch

Submitted 2026-09-29 UTC: past-only queue IDs **153–162**; future-year IDs **163–172**.
The first AWI past-only MSE run completed epoch 1 and saved both checkpoint roles.
See the results root’s `launch_status.json` for the launch verification snapshot and
`qstat_local` for current queue status.
