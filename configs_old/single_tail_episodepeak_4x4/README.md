# Single Tail-MSE × EpisodeGTAlignedPeak-MSE 4×4

**Status: ACTIVE — current Single development factorial.**

This family extends the completed [Single G/T study](../s0_refresh/README.md)
using its G0 / Single / Global-MSE setup. Global loss is **MSE only**:

```text
GlobalMSE + lambda_T * TailMSE + lambda_P * EpisodeGTAlignedPeakMSE
```

| Factor | Weights | Filename tokens |
| --- | --- | --- |
| Tail-MSE | {0, 0.0125, 0.025, 0.05} | T0000 / T0125 / T0250 / T0500 |
| EpisodeGTAlignedPeak-MSE | {0, 0.0025, 0.005, 0.01} | EP0000 / EP0025 / EP0050 / EP0100 |

There are **16 configs/station, 64 configs total** across CBBT, Lewes, Battery,
and Boston. Tail varies slower; episode peak varies faster. Tokens encode weights
multiplied by 10,000; each config stores the actual float values. The completed
four-station gradient diagnostic retained this grid.

Tail uses all strict `GT > TRAIN Q95` hours. Episode peak supervision uses one
GT peak timestamp per physical TRAIN episode and fixed TRAIN prevalence `J/N`.
WQE, Dual objectives, forecast-window peak loss, and timing loss are outside this grid.

## Canonical anchors

| Cell | Earlier Single anchor | Objective |
| --- | --- | --- |
| `T0000_EP0000` | `G0_T0` | Plain Global MSE |
| `T0250_EP0000` | `G0_T1` | Global MSE + 0.025 Tail-MSE |

## Manifests and selection

- [All stations: 64 rows](manifest_all.csv).
- [Boston + Lewes: 32 rows](manifest_boston_lewes.csv), the first planned manual subset.
- [CBBT + Battery: 32 rows](manifest_cbbt_battery.csv).
- `PRIMARY_SELECTOR=exceedance`: minimum VAL ExceedanceRMSE.
- `SECONDARY_SELECTOR=overall`: minimum VAL AllRMSE, conventional sensitivity.

Both roles receive final VAL/TEST reevaluation with exactly 11 corrected
episode-aware metrics. `best.pt` and `best_<run-stem>.pth` are primary-checkpoint
aliases; they do not add selection roles.

## Use and result locations

No launcher was generated. These are assignment-only configs used with the
repository's `train.sh`. Preview a single config from the repository root:

```bash
DRY_RUN=1 USE_TMUX=0 bash train.sh configs/single_tail_episodepeak_4x4/train_config_NCEP_Boston_G0_T0250_EP0050.sh
```

Future runs write under
`/home/exouser/media/share/PACT/FormalRuns_0925/single_tail_episodepeak_4x4`
as `<config_id>__<timestamp>`. External audit locations are filesystem paths:

- `/home/exouser/media/share/PACT/FormalRuns_0925/single_tail_episodepeak_4x4/audit/REPORT.md`
- `/home/exouser/media/share/PACT/FormalRuns_0925/single_tail_episodepeak_4x4/audit/validation_summary.json`

Stable definitions: [episode construction and supervision](../../docs/EPISODE_PEAK.md),
[losses](../../docs/LOSSES.md), [checkpoint roles](../../docs/CHECKPOINT_SELECTION.md),
and [canonical metrics](../../docs/METRICS.md).
