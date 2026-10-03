# Battery CMIP6 quick check — 0929 experiment family

40 configurations follow `CMIP6_Boston_QuickCheck_0929`: five GCMs × two time splits × four loss formulations.
Created 2026-10-03 UTC. Prepared run stamp: `20261003_051753`.

GCMs: AWI, CNRM, EC_EARTH, MPI, MRI. The dataset selection matches Boston.

| Group | Data folder | TRAIN / VAL ratios | TRAIN / VAL / TEST year groups |
| --- | --- | --- | --- |
| [past_only](past_only/README.md) | `Data/Grid4_New_PastOnly` | 0.6 / 0.2 | 22 / 7 / 7 |
| [future_year](future_year/README.md) | `Data/Grid4_New` | 0.4545454545 / 0.0909090909 | 30 / 6 / 30 |

Past-only: TRAIN 1979--2000, VAL 2001--2007, TEST 2008--2014.
Future-year: TRAIN 1979--2008, VAL 2009--2014, TEST 2070--2099.
Each year denotes the start of a paired year group. Both use chronological splits,
`SHUFFLE_YEARS=0` and `FUTURE_ONLY=0`.

| Variant | Tail MSE weight | Episode GT-aligned peak MSE weight |
| --- | ---: | ---: |
| `T0000_EP0000` | 0 | 0 |
| `T0500_EP0000` | 0.05 | 0 |
| `T0000_EP0100` | 0 | 0.01 |
| `T0500_EP0100` | 0.05 | 0.01 |

## Training and result paths

Single head; GraphSAGE + Transformer; 24 h history; hidden 128; LR 5e-3;
300 epochs; batch 256; gradient accumulation 4; seed 42; cosine scheduler;
BF16 AMP and TF32. Parameters match the station's NCEP templates and the Boston CMIP6 settings.
The primary selector is **overall**; both `best_overall.pt` and `best_exceedance.pt` are retained.

Config root: `/home/exouser/StormSurge/configs/CMIP6_Battery_QuickCheck_0929`.
Results root: `/home/exouser/media/share/PACT/FormalRuns_0925/CMIP6_Battery_QuickCheck_0929`.
Both contain `past_only/` and `future_year/` subfolders.
Each job creates `<group>/CMIP6_<GCM>_Battery_<group>_G0_<variant>__20261003_051753`.
Root and group `manifest.csv` files list exact config, template, result, and queue-label mappings.
Only the group folders are created during preparation; the launcher creates each run directory.

## Validate or submit

From `/home/exouser/StormSurge`:

```bash
# Validate all 40 configs without submitting training.
DRY_RUN=1 bash configs/CMIP6_Battery_QuickCheck_0929/launch_all.sh

# Submit all 40 jobs through qsub_local, using one queue slot and tmux.
bash configs/CMIP6_Battery_QuickCheck_0929/launch_all.sh
```

Each group also has its own `launch_all.sh` for submitting 20 jobs.
`qsub_all.txt` contains individual submission commands as an alternative to these launchers.
The launchers use the prepared run stamp and save queue receipts under
`<results>/<group>/_orchestration/20261003_051753/`; an existing batch receipt directory prevents a duplicate batch submission.

`audit_configs.py` compares every configuration with Boston and the station's NCEP template,
checks the actual training parser and year splitter, and dry-runs the launcher.
Group outputs are `config_audit.json`, `dataset_inventory.json`, and `dry_run.log`.
The data check covers station metadata, year filenames, nonempty archives, and split membership;
it does not load all graph tensors.
