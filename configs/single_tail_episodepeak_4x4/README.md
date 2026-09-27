# Single Tail-MSE x EpisodeGTAlignedPeak-MSE 4 x 4

64 assignment-only configs cloned from the current formal `configs/s0_refresh` G0 / Single / Global-MSE setup.
The completed four-station gradient diagnostic recommends **KEEP GRID AS PROPOSED**.

- Tail weights: `0, 0.0125, 0.025, 0.05`.
- EpisodeGTAlignedPeak weights: `0, 0.0025, 0.005, 0.01`.
- Station order: CBBT, Lewes, Battery, Boston. Tail varies slower; Peak varies faster.
- Tokens: T0000/T0125/T0250/T0500 and EP0000/EP0025/EP0050/EP0100. Configs store the actual float values.
- Objective: `GlobalMSE + lambda_T * TailMSE + lambda_P * EpisodeGTAlignedPeakMSE`.
- `PRIMARY_SELECTOR=exceedance`; `SECONDARY_SELECTOR=overall`.

## Manifests

- [All stations: 64 rows](manifest_all.csv).
- [Boston + Lewes: 32 rows](manifest_boston_lewes.csv), the first planned manual subset.
- [CBBT + Battery: 32 rows](manifest_cbbt_battery.csv).

## Results and audit

Future runs write under `/home/exouser/media/share/PACT/FormalRuns_0925/single_tail_episodepeak_4x4` as `<config_id>__<timestamp>`.
[Full audit report](/home/exouser/media/share/PACT/FormalRuns_0925/single_tail_episodepeak_4x4/audit/REPORT.md); [machine-readable validation](/home/exouser/media/share/PACT/FormalRuns_0925/single_tail_episodepeak_4x4/audit/validation_summary.json).

The active code retains overall/exceedance roles and produces exactly eleven corrected metrics for each final VAL/TEST reevaluation.
Existing code also writes two primary-checkpoint copies (`best.pt`, `best_<run-stem>.pth`): the two-role condition passes, while a literal two-file condition does not. See the report.
Only configs, manifests, and validation outputs were generated. No training or job submission was performed.
