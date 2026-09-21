# Fixed factorial validation

PASS against production main `af53de756860765a9809b0c5c44bc828c2a08359`.

20/20 configs passed shell syntax, singleton axes, actual train.sh dry-run command generation, resolved shell snapshots, production argparse, manifest and all prescribed settings.

Five configs per station: S0 plus D0/D1/D2/D3. Dual pairs differ only by loss_mode/tail_lambda and/or excess_amp_loss_weight, after excluding run identity. All other arguments and runtime settings match across all Dual runs.

Parameters: Single **618,885**; Dual **685,447**. D0–D3 have identical ModelConfig, counts, and every initialized state tensor within each station. S0 retains the identical initialized backbone.

CPU construction from parsed launcher args and preserved TRAIN statistics; one production seed call per model; exact tensor comparison including buffers. TRAIN sources checked by size/mtime and filename splits; one TRAIN file per station opened for dimensions. No refitting, forward pass, optimizer, VAL/TEST file loading or diagnosis execution.

Requested USE_SITE_ELEVATION=0 gives six station features. Preserved diagnosis used elevation=1 (seven features); its model counts are not used for this factorial.

Canonical nine reporting columns verified, including the five `_top5` true-peak-time metrics. Every run selects by VAL overall RMSE, tolerance 0.01, with auxiliary checkpoints disabled. Adam weight decay remains 1e-5.

All 43 pre-existing diagnosis files passed SHA-256 preservation checks. Production tracked files are unchanged; no obsolete controls appear in configs. No jobs, training, checkpoint writes or results folders were created.

The JSON report contains every parsed argument, model count, initial-state hash and filename split. Per-run dry-run transcripts and resolved snapshots are beside this report.
