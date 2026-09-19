# 0919 configuration validation: PASS

44 configs and resolved commands passed: 11 per station; 4 Single, 20 Dual mean, 20 Dual learned.
Every config explicitly uses seed 42 and deterministic execution. All non-factor Dual settings match.
Source/reference hashes and training result trees are unchanged. No training jobs or queue submissions.

C3 + Learned synthetic repeatability: exact predictions and gradients on **cuda:0**.
GPU BF16 determinism verified: **True**.

## Protocol notes and limits

- The 0916 reference used DETERMINISTIC=0; all 0919 configs explicitly use 1 as requested.
- Canonical Dual weights are body=1, excess=2, gate=0.5, not three unit weights.
- Under mse the frozen launcher omits tail/slope flags: parser defaults 0.1/0.01 are inactive. Single branch flags are likewise omitted; dual_loss=0 disables their inactive parser defaults.
- Frozen train.py creates the model before a shuffled DataLoader with no dedicated generator. Different head construction consumes different global RNG draws. Same-config repeatability is checked; identical cross-variant minibatch order is NOT guaranteed. Enforcing it needs a separately authorized trainer change, which is outside this configuration-only task.

The sampler probe records the limitation in [reproducibility.json](validation/reproducibility.json).
This suite preserves the trainer; it does not claim identical cross-variant minibatch order.

See [manifest](manifest.csv), [resolved commands](dry_run_commands.txt), and [full report](validation_report.json). Individual launcher transcripts are in `validation/`.
