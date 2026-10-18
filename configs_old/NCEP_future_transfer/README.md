# NCEP future transfer source checkpoints

Train a historical NCEP source model using the available year groups through
2014, with 2009–2014 reserved exclusively for validation. This is the final
candidate for future cross-dataset transfer at Boston, Battery, CBBT, and Lewes.
Boston has already completed; the other three station configs are prepared for
the same experiment. Only the station and its run identifiers differ.

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

- Boston, Battery, CBBT, Lewes; PACT (`perceiver3`), GraphSAGE, Transformer, Single head.
- 24 h history; hidden size 128; two graph and Transformer layers.
- MSE + Tail MSE weight 0.05.
- Episode GT-aligned Peak MSE weight 0.01.
- `overall` checkpoint selector; both checkpoint roles remain available.
- 300 epochs, learning rate 0.005, batch size 256, four gradient accumulation steps.

Architecture, runtime, and inactive Single/Dual defaults follow
`../CMIP6_Boston_QuickCheck_0929/future_year/train_config_CMIP6_AWI_Boston_future_year_G0_T0500_EP0100.sh`.
Results go under `/home/exouser/media/share/PACT/FormalRuns_0925/NCEP_future_transfer`.
Each run creates `NCEP_<station>_future_transfer_G0_T0500_EP0100__<timestamp>/`
there, containing the resolved config, logs, checkpoints, metrics, and predictions.
Source configs stay in this directory; the existing Boston results are retained.

## Audit and launch

Run from `/home/exouser/StormSurge`. Audit and dry-run the three remaining stations:

```bash
/home/exouser/.conda/envs/torchpyg-cu12x/bin/python \
  configs/NCEP_future_transfer/audit_split.py --stations Battery CBBT Lewes

for station in Battery CBBT Lewes; do
  DRY_RUN=1 bash train.sh \
    "configs/NCEP_future_transfer/train_config_NCEP_${station}_future_transfer_G0_T0500_EP0100.sh" || break
done
```

The audit defaults to all four stations if `--stations` is omitted. Dry runs
print commands without starting training or creating result directories.

Launch one station with the usual tmux wrapper:

```bash
bash train.sh \
  configs/NCEP_future_transfer/train_config_NCEP_Battery_future_transfer_G0_T0500_EP0100.sh
```

Or run all three remaining stations sequentially in the current shell:

```bash
for station in Battery CBBT Lewes; do
  USE_TMUX=0 bash train.sh \
    "configs/NCEP_future_transfer/train_config_NCEP_${station}_future_transfer_G0_T0500_EP0100.sh" || break
done
```

The sequential loop excludes Boston and stops if a run fails. Configs default to
300 epochs and the same Python runtime as Boston.

## External evaluation

The real evaluation is external cross-dataset inference on **CMIP6 2070–2099**:

```bash
/home/exouser/.conda/envs/torchpyg-cu12x/bin/python infer.py \
  --ckpt <saved-source-checkpoint> \
  --root_dir ./Data/Grid4_New/NCEP/graphs \
  --test_root_dir <CMIP6-2070-2099-graphs-root> \
  --station <station-matching-the-source-checkpoint> --save_npz
```

External inference evaluates all graphs from `--test_root_dir`, independently
of the empty saved source TEST. Supply a root restricted to the target years,
or use `--years` to list the desired year tags if the root also contains other
years.
