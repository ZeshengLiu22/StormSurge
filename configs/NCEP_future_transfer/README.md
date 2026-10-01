# NCEP future transfer source checkpoint

Train a historical NCEP source model using the available year groups through
2014, with 2009–2014 reserved exclusively for validation. This is the final
candidate for future cross-dataset transfer.

## Canonical split

| Split | Year groups | Range |
|-------|-------------|-------|
| TRAIN | 30 | 1979_1980–2008_2009 |
| VAL | 6 | 2009_2010–2014_2015 |
| TEST | 0 | None |

The chronological ratios are `TRAIN_RATIO=0.8333333333333334` and
`VAL_RATIO=0.16666666666666666`, with year shuffling and future-only filtering
disabled. TRAIN supplies the optimization data, normalization statistics, and
loss thresholds. VAL alone selects checkpoints.

This run intentionally has **no source-domain held-out TEST**. Saved checkpoint
provenance remains `split_tags["test"] == []`.

During `train.py` final reporting only, when `TEST_ROOT_DIR` is empty, VAL is
mirrored as TEST for compatibility with checkpoint comparison, summary, and
prediction export code. **These mirrored metrics are not valid held-out test
results.** The run prints a warning and saves `TEST_IS_VAL_MIRROR_WARNING.txt`.
The summary and exported TEST prediction NPZs record:

```json
{
  "test_scope": "val_mirror_no_heldout_test",
  "test_is_val_mirror": true
}
```

## Model

- Boston; PACT (`perceiver3`), GraphSAGE, Transformer, Single head.
- 24 h history; hidden size 128; two graph and Transformer layers.
- MSE + Tail MSE weight 0.05.
- Episode GT-aligned Peak MSE weight 0.01.
- `overall` checkpoint selector; both checkpoint roles remain available.
- 300 epochs, learning rate 0.005, batch size 256, four gradient accumulation steps.

Architecture, runtime, and inactive Single/Dual defaults follow
`../CMIP6_Boston_QuickCheck_0929/future_year/train_config_CMIP6_AWI_Boston_future_year_G0_T0500_EP0100.sh`.
Results go under `/home/exouser/media/share/PACT/FormalRuns_0925/NCEP_future_transfer`.

## Audit and launch

Run from `/home/exouser/StormSurge`:

```bash
/home/exouser/.conda/envs/torchpyg-cu12x/bin/python configs/NCEP_future_transfer/audit_split.py

DRY_RUN=1 bash train.sh \
  configs/NCEP_future_transfer/train_config_NCEP_Boston_future_transfer_G0_T0500_EP0100.sh

# Launch the real 300-epoch run when ready:
bash train.sh \
  configs/NCEP_future_transfer/train_config_NCEP_Boston_future_transfer_G0_T0500_EP0100.sh
```

## External evaluation

The real evaluation is external cross-dataset inference on **CMIP6 2070–2099**:

```bash
/home/exouser/.conda/envs/torchpyg-cu12x/bin/python infer.py \
  --ckpt <saved-source-checkpoint> \
  --root_dir ./Data/Grid4_New/NCEP/graphs \
  --test_root_dir <CMIP6-2070-2099-graphs-root> \
  --station Boston --save_npz
```

External inference evaluates all graphs from `--test_root_dir`, independently
of the empty saved source TEST. Supply a root restricted to the target years,
or use `--years` to list the desired year tags if the root also contains other
years.
