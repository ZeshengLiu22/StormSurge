# PACT storm-surge forecasting

PACT predicts six hourly storm-surge residuals from a graph of atmospheric forcing,
coordinates and optional station metadata. A forecast at time t targets t through
t+5 hours; forcing history is sampled every six hours. Single and Dual output heads
share the PACT backbone. Spatial-mean baseline models are also supported.

```text
forcing/history + graph → spatial encoder → station readout → temporal module
                        → horizon contexts → Single or Dual head → prediction (m)
```

All extreme populations use one physical threshold fitted on unique TRAIN hourly
targets. The default is Q95 with strict `y > tau`; VAL, TEST and OOD reuse that
threshold. Every run retains four [checkpoint roles](docs/CHECKPOINT_SELECTION.md):
`overall`, `exceedance`, `aligned_peak`, and `bea` (Balanced Event-Aware).
BEA minimizes `0.50*AllRMSE + 0.25*ExceedanceRMSE + 0.25*GTAlignedPeakRMSE`.
All are selected using VAL and reevaluated on VAL and TEST.

## Run

See the [configuration overview](configs/README.md) for experiment families,
archive status, and directory names. The baseline ablation family is temporarily
archived; the WQE placement family is completed and archived.

Use the Python 3.11 training environment specified by
[environment_training.yml](environment_training.yml). The complete preprocessing
tests also use dependencies in [environment_dataprep.yml](environment_dataprep.yml).
Place or link preprocessed graphs under `Data/Grid4_New/NCEP/graphs`.

Inspect thresholds and populations before training:

```bash
python tools/audit_hourly_threshold.py --root-dir ./Data/Grid4_New/NCEP/graphs
python tools/generate_configs.py
DRY_RUN=1 USE_TMUX=0 bash train.sh configs/baseline_ablation/train_config_NCEP_CBBT_D0_DualBase.sh
```

The generator creates S0 Single, D0 DualBase, D1 Exceedance, D2 Amp and D3
ExceedanceAmp for CBBT, Lewes, Battery and Boston. Add
`--include-severity-shape` to generate the optional excess parameterization.
Config generation and launcher dry runs do not start training.

For the independent global/excess WQE experiment, run
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

Generate the full factorial families and inspect their commands:

```bash
python tools/generate_configs.py --family s0_refresh
python tools/generate_configs.py --family wqe_factorial_multickpt
DRY_RUN=1 bash configs/s0_refresh/launch_all.sh
DRY_RUN=1 bash configs/wqe_factorial_multickpt/launch_all.sh
```

The [Single family](configs/s0_refresh/README.md) has 24 configs (G × T);
the [Dual family](configs/wqe_factorial_multickpt/README.md) has 48 (G × E × T).
Both include manifests and launch scripts, retain their historical result roots,
and preserve all four checkpoint roles with primary `overall`.
Generation and dry runs submit no jobs.

Launch an explicitly chosen configuration:

```bash
USE_TMUX=0 bash train.sh configs/baseline_ablation/train_config_NCEP_CBBT_D0_DualBase.sh
```

The core shell controls are:

```bash
LOSS_MODE_LIST=("mse")
EXCESS_LOSS_MODE="mse"
EXCEEDANCE_PERCENTILE=95
EXCEEDANCE_LOSS_MODE="mse"
EXCEEDANCE_LOSS_WEIGHT=0.025
EXCESS_AMP_LOSS_WEIGHT=0.003
CHECKPOINT_SELECTION=overall
```

Evaluate a saved checkpoint, retaining its source TRAIN threshold:

```bash
python infer.py --ckpt /path/to/run/best_bea.pt \
  --root_dir ./Data/Grid4_New/NCEP/graphs --save_npz --out_dir ./All_Inference_Results/example
```

For transfer, add `--test_root_dir /path/to/CMIP6/graphs`. For Dual branch exports,
add `--dual_diagnostics`. Compare new prediction exports on shared GT episodes:

```bash
python tools/event_diagnostics.py \
  --predictions Overall=/path/to/run/test_predictions_overall.npz \
  --predictions bea=/path/to/run/test_predictions_bea.npz \
  --output ./results/event_diagnostics
```

This writes frozen episode IDs, canonical metrics, extreme-hour branch tables,
Figures 6–8 and episode severity-bin diagnostics. Figures use the same GT episodes
for every model. Ground-truth, timestamp, station, split or threshold mismatches
raise errors.

## Documentation

Start with the [documentation index](docs/README.md), then read
[Formulation](docs/FORMULATION.md), [Backbone](docs/BACKBONE.md),
[Direct Dual](docs/DUAL_EXCEEDANCE.md), [Severity–Shape](docs/SEVERITY_SHAPE.md),
[Losses](docs/LOSSES.md), [Checkpoint selection](docs/CHECKPOINT_SELECTION.md) and
[Metrics](docs/METRICS.md).

## Verification

```bash
python -m unittest discover -s tests -v
python tools/audit_documentation.py
bash -n train.sh infer.sh infer_multi.sh
```

The suite includes numerical metric oracles, timestamp/leakage checks, branch
objectives, all four checkpoint roles, train/inference round trips, distributed
reductions, preprocessing and documentation consistency. Real experiment
training is a separate explicit launch.
