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
threshold. Every run retains `best_overall.pt` and `best_eventaware.pt`, selected
from the same trajectory using VAL metrics. Both are reevaluated on VAL and TEST.

## Run

Use the Python 3.11 training environment specified by
[environment_training.yml](environment_training.yml). The complete preprocessing
tests also use dependencies in [environment_dataprep.yml](environment_dataprep.yml).
Place or link preprocessed graphs under `Data/Grid4_New/NCEP/graphs`.

Inspect thresholds and populations before training:

```bash
python tools/audit_hourly_threshold.py --root-dir ./Data/Grid4_New/NCEP/graphs
python tools/generate_configs.py
DRY_RUN=1 USE_TMUX=0 bash train.sh configs/current/train_config_NCEP_CBBT_D0_DualBase.sh
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

Launch an explicitly chosen configuration:

```bash
USE_TMUX=0 bash train.sh configs/current/train_config_NCEP_CBBT_D0_DualBase.sh
```

The core shell controls are:

```bash
LOSS_MODE_LIST=("mse")
EXCESS_LOSS_MODE="mse"
EXCEEDANCE_PERCENTILE=95
EXCEEDANCE_LOSS_WEIGHT=0.025
EXCESS_AMP_LOSS_WEIGHT=0.003
CHECKPOINT_SELECTION=overall
CKPT_W_ALL=0.65
CKPT_W_EXCEEDANCE=0.20
CKPT_W_PEAK=0.15
```

Evaluate a saved checkpoint, retaining its source TRAIN threshold:

```bash
python infer.py --ckpt /path/to/run/best_eventaware.pt \
  --root_dir ./Data/Grid4_New/NCEP/graphs --save_npz --out_dir ./All_Inference_Results/example
```

For transfer, add `--test_root_dir /path/to/CMIP6/graphs`. For Dual branch exports,
add `--dual_diagnostics`. Compare new prediction exports on shared GT episodes:

```bash
python tools/event_diagnostics.py \
  --predictions Overall=/path/to/run/test_predictions_overall.npz \
  --predictions EventAware=/path/to/run/test_predictions_eventaware.npz \
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
objectives, both checkpoint roles, train/inference round trips, distributed
reductions, preprocessing and documentation consistency. Real experiment
training is a separate explicit launch.
