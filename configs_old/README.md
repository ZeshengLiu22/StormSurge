# Current experiment configurations

These directories are protected experiment references. Their location under
`configs_old/` is temporary; directory names, run names, scientific splits,
checkpoint choices, and result provenance remain part of the experiment definitions.

| Family | Purpose |
| --- | --- |
| [single_tail_episodepeak_4x4](single_tail_episodepeak_4x4/README.md) | Four NCEP stations, 16 Tail × EpisodePeak weight combinations each (64 configs). |
| [CMIP6_Boston_QuickCheck_0929](CMIP6_Boston_QuickCheck_0929/README.md) | Boston: five GCMs, two temporal splits, four loss combinations. |
| [CMIP6_Battery_QuickCheck_0929](CMIP6_Battery_QuickCheck_0929/README.md) | The same 40-config CMIP6 comparison for Battery. |
| [CMIP6_CBBT_QuickCheck_0929](CMIP6_CBBT_QuickCheck_0929/README.md) | The same 40-config CMIP6 comparison for CBBT. |
| [CMIP6_Lewes_QuickCheck_0929](CMIP6_Lewes_QuickCheck_0929/README.md) | The same 40-config CMIP6 comparison for Lewes. |
| [NCEP_future_transfer](NCEP_future_transfer/README.md) | Four historical NCEP source models for future CMIP6 transfer. |
| [cross_dataset_infer](cross_dataset_infer/README.md) | Boston source-to-target development matrix and provenance. |
| [cross_dataset_infer_4stations](cross_dataset_infer_4stations/README.md) | Four-station production source-to-target matrix and launchers. |

All 228 training configs use PACT with a Single regression head and physical
Global MSE. `EXCEEDANCE_LOSS_WEIGHT` controls strict TRAIN-Q95 Tail MSE;
`EPISODE_GT_ALIGNED_PEAK_WEIGHT` controls episode GT-aligned peak MSE.
The NCEP grid varies Tail weights `{0, 0.0125, 0.025, 0.05}` and episode weights
`{0, 0.0025, 0.005, 0.01}`. Each CMIP6 comparison uses its four corner combinations.

The NCEP grid selects `exceedance` as primary; CMIP6 and future-transfer configs
select `overall`. Every run retains both VAL-selected checkpoint roles.
Past-only CMIP6 uses 22/7/7 TRAIN/VAL/TEST year groups; future-year CMIP6 uses
30/6/30. NCEP future-transfer uses 30 TRAIN and 6 VAL groups with no independent
source TEST; its compatibility TEST export mirrors VAL. Cross-dataset inference
uses explicit target years and source TRAIN normalization and Q95.

From the repository root, preview a training config with:

```bash
S0_PYTHON_BIN="$(command -v python)" DRY_RUN=1 USE_TMUX=0 bash train.sh \
  configs_old/single_tail_episodepeak_4x4/train_config_NCEP_Boston_G0_T0500_EP0100.sh
```

Use the corresponding family README for runtime overrides and inference groups.
Manifests, split inventories, source checkpoint audits, and historical validation
records remain as provenance; recorded paths and hashes describe their original
execution environment. Current scripts resolve retained config paths in this tree.
