# PACT storm-surge forecasting

PACT predicts six hourly storm-surge residuals from a graph of atmospheric forcing,
coordinates and optional station metadata. A forecast at time t targets t through
t+5 hours; forcing history is sampled every six hours. Single and Dual output heads
share the PACT backbone. Spatial-mean baseline models are also supported.

```text
forcing/history + graph → spatial encoder → station readout → temporal module
                        → horizon contexts → Single or Dual head → prediction (m)
```

## Current Single development

The **ACTIVE** family is **[configs/single_tail_episodepeak_4x4](configs/single_tail_episodepeak_4x4/README.md)**:
16 Tail × EpisodePeak combinations per station, 64 configs total.

```text
GlobalMSE + lambda_T * TailMSE + lambda_P * EpisodeGTAlignedPeakMSE
```

Global MSE covers all supervised hours. Tail-MSE covers strict `GT > TRAIN Q95`
hours. EpisodeGTAlignedPeak-MSE supervises one GT peak amplitude per physical
TRAIN episode, using fixed TRAIN peak prevalence. No forecast-window peak
objective is active.

Every run retains exactly two [checkpoint roles](docs/CHECKPOINT_SELECTION.md):

- **exceedance:** primary, minimum VAL ExceedanceRMSE.
- **overall:** secondary conventional sensitivity, minimum VAL AllRMSE.

Both receive timestamp-aware final VAL/TEST evaluation with exactly **11 corrected
episode-aware metrics**, defined in [METRICS](docs/METRICS.md). VAL, TEST, and OOD
reuse the threshold fitted on unique TRAIN hourly targets.

See [episode construction and supervision](docs/EPISODE_PEAK.md), the
[loss definition](docs/LOSSES.md#single-episode-gt-aligned-peak-amplitude-loss),
and the [saved TRAIN audit](docs/audits/episode_peak/audit.json).
`EPISODE_GT_ALIGNED_PEAK_WEIGHT` defaults to 0.0; the active grid varies it
alongside the Tail-MSE weight.

## Run

See the [configuration overview](configs/README.md) for experiment families,
scientific status, and directory names. The active 4×4 family provides all-station
and two-station manifests; no launcher was generated.

Use the Python 3.11 training environment specified by
[environment_training.yml](environment_training.yml). The complete preprocessing
tests also use dependencies in [environment_dataprep.yml](environment_dataprep.yml).
Place or link preprocessed graphs under `Data/Grid4_New/NCEP/graphs`.

Inspect thresholds and preview one active configuration:

```bash
python tools/audit_hourly_threshold.py --root-dir ./Data/Grid4_New/NCEP/graphs
DRY_RUN=1 USE_TMUX=0 bash train.sh configs/single_tail_episodepeak_4x4/train_config_NCEP_Boston_G0_T0250_EP0050.sh
```

Launch an explicitly chosen configuration:

```bash
USE_TMUX=0 bash train.sh configs/single_tail_episodepeak_4x4/train_config_NCEP_Boston_G0_T0250_EP0050.sh
```

The example above uses these objective controls:

```bash
HEAD_TYPE=single
DUAL_LOSS=0
LOSS_MODE_LIST=("mse")
EXCEEDANCE_PERCENTILE=95
EXCEEDANCE_LOSS_MODE="mse"
EXCEEDANCE_LOSS_WEIGHT=0.025
EXCESS_AMP_LOSS_WEIGHT=0
EPISODE_GT_ALIGNED_PEAK_WEIGHT=0.005
CHECKPOINT_SELECTION=exceedance
```

Evaluate a saved checkpoint, retaining its source TRAIN threshold:

```bash
python infer.py --ckpt /path/to/run/best_exceedance.pt \
  --root_dir ./Data/Grid4_New/NCEP/graphs --save_npz --out_dir ./All_Inference_Results/example
```

For transfer, add `--test_root_dir /path/to/CMIP6/graphs`. For Dual branch exports,
add `--dual_diagnostics`. Compare new prediction exports on shared GT episodes:

```bash
python tools/event_diagnostics.py \
  --predictions Overall=/path/to/run/test_predictions_overall.npz \
  --predictions Exceedance=/path/to/run/test_predictions_exceedance.npz \
  --output ./results/event_diagnostics
```

This writes frozen episode IDs, canonical metrics, extreme-hour branch tables,
Figures 6–8 and episode severity-bin diagnostics. Figures use the same GT episodes
for every model. Ground-truth, timestamp, station, split or threshold mismatches
raise errors.

## Documentation

The primary reading path in the [documentation index](docs/README.md) follows
current Single development:
[Formulation](docs/FORMULATION.md) → [Backbone](docs/BACKBONE.md) →
[Losses](docs/LOSSES.md) / [Episode peak supervision](docs/EPISODE_PEAK.md) →
[Checkpoint selection](docs/CHECKPOINT_SELECTION.md) →
[Metrics](docs/METRICS.md).

For **SUPPORTED LEGACY** historical/optional Dual formulations, see
[Direct Dual](docs/DUAL_EXCEEDANCE.md) and [Severity–Shape](docs/SEVERITY_SHAPE.md).

## Experiment-family status

Completed and legacy study specifications and their formal result references
remain available for provenance and reproduction. The
[configuration overview](configs/README.md#status-vocabulary) defines the status vocabulary.

| Family | Scientific status and purpose |
| --- | --- |
| [single_tail_episodepeak_4x4](configs/single_tail_episodepeak_4x4/README.md) | **ACTIVE** — current Single development factorial: Global MSE with Tail-MSE × EpisodeGTAlignedPeak-MSE weights. |
| [s0_refresh](configs/s0_refresh/README.md) | **ACHIEVED** — completed 24-run Single G × T study; established Global MSE + Tail-MSE as the reference direction and is superseded by the 4×4 for development. |
| [wqe](configs/wqe/README.md) | **ACHIEVED / ARCHIVED** — completed WQE placement study; historical/diagnostic use, outside the current 4×4. |
| [wqe_factorial_multickpt](configs/wqe_factorial_multickpt/README.md) | **ACHIEVED / LEGACY** — completed 48-run Dual G × E × T factorial / legacy in-domain study. |
| [baseline_ablation](configs/baseline_ablation/README.md) | **LEGACY / ARCHIVED — completion unverified** — original 20-cell Single/Dual, Tail, and amplitude specification. |

Saved results retain the selector and metric schema used when they were produced.
Reruns with the current implementation use the two roles and 11 final metrics
documented above.

The baseline generator creates S0 Single, D0 DualBase, D1 Exceedance, D2 Amp and
D3 ExceedanceAmp for all four stations:

```bash
python tools/generate_configs.py
DRY_RUN=1 USE_TMUX=0 bash train.sh configs/baseline_ablation/train_config_NCEP_CBBT_D0_DualBase.sh
```

Add `--include-severity-shape` to generate the optional excess parameterization.
Config generation and launcher dry runs do not start training.

To reproduce the completed global/excess WQE placement configs, run
`python tools/generate_configs.py --family wqe`. This adds W1 GlobalWQE,
W2 ExcessWQE, and W3 BothWQE for all four stations under `configs/wqe`;
existing D0 is the MSE/MSE control. See [Losses](docs/LOSSES.md) for the
shared published parameters, TRAIN scale handling, and experiment matrix.

### Factorial labels: G / E / T

G, E, and T are three independent loss choices encoded in config filenames and
run names. The digit after each letter selects that factor's option:

| Letter | Meaning / 含义 | 0 | 1 | 2 |
| --- | --- | --- | --- | --- |
| **G = Global** | 全局预测损失：final prediction versus target over all target hours | MSE | WQE | Not used |
| **E = Excess** | 原始超额分支损失：Dual raw excess before gate multiplication | MSE | WQE | Not used |
| **T = Tail** | 极端小时附加损失：final prediction versus target only where `y > tau` | Off, weight 0 | Tail-MSE, weight 0.025 | Tail-WQE, weight 0.025 |

MSE means mean squared error; WQE means weighted quantile–expectile loss.
**G0 and E0 still enable MSE; only T0 disables its term.** G and E use 0/1;
T uses 0/1/2. E supervises raw excess across every horizon of a GT Event Window;
T acts on final predictions at strict TRAIN Q95 extreme hours with fixed TRAIN
`q_H` normalization. See [Losses](docs/LOSSES.md) for the formulas.

For example, G1_E0_T2 means global WQE + raw-excess MSE + Tail-WQE at weight
0.025. Single has no raw-excess branch, so it uses G/T only: G0_T1 means global
MSE + Tail-MSE at weight 0.025. The old Single suffix S0_Single_Tail_WQE maps
to G1_T1 (global WQE + Tail-MSE).

Reproduce the completed factorial specifications and inspect their commands:

```bash
python tools/generate_configs.py --family s0_refresh
python tools/generate_configs.py --family wqe_factorial_multickpt
DRY_RUN=1 bash configs/s0_refresh/launch_all.sh
DRY_RUN=1 bash configs/wqe_factorial_multickpt/launch_all.sh
```

The [Single family](configs/s0_refresh/README.md) has 24 configs (G × T);
the [Dual family](configs/wqe_factorial_multickpt/README.md) has 48 (G × E × T).
Both include manifests and launch scripts, use their documented result roots,
and retain both checkpoint roles with primary `exceedance`.
Generation and dry runs submit no jobs.

## Verification

```bash
python -m unittest discover -s tests -v
python tools/audit_documentation.py
bash -n train.sh infer.sh infer_multi.sh
```

The suite includes numerical metric oracles, timestamp/leakage checks, branch
objectives, both checkpoint roles, train/inference round trips, distributed
reductions, preprocessing and documentation consistency. Real experiment
training is a separate explicit launch.
