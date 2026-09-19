# 0919 configuration validation: PASS

44 configs and resolved commands passed: 11 per station; 4 Single, 20 Dual mean, 20 Dual learned.
Every config explicitly uses SEED=42 and DETERMINISTIC=0, matching 0916. All non-factor Dual settings match.
Source hashes match the amended data-order baseline; 0916 references and result trees are unchanged. No training jobs or queue submissions.

C3 + Learned synthetic smoke: finite predictions and gradients on **cuda:0**.
GPU BF16 smoke verified: **True**.
Strict deterministic kernels are disabled; bitwise training repeatability is not guaranteed.

## Protocol notes and limits

- All 0919 configs use SEED=42 and DETERMINISTIC=0, matching the 0916 reference by user request. Sample ordering remains matched; bitwise training repeatability is not guaranteed.
- Canonical Dual weights are body=1, excess=2, gate=0.5, not three unit weights.
- Under mse the frozen launcher omits tail/slope flags: parser defaults 0.1/0.01 are inactive. Single branch flags are likewise omitted; dual_loss=0 disables their inactive parser defaults.
- The authorized data-order fix gives single-process training a private generator seeded once from args.seed. Its state advances across epochs independently of model initialization/dropout. Historical single-process training trajectories change; DDP sampler behavior and evaluation loader defaults are unchanged.

The synthetic sampler probe verifies identical sequences across all 11 architectures (including Single), 300 epochs and two repeats, despite different global RNG draws. Every epoch uses a new permutation; no optimizer steps are taken. See [reproducibility.json](validation/reproducibility.json).

See [manifest](manifest.csv), [resolved commands](dry_run_commands.txt), and [full report](validation_report.json). Individual launcher transcripts are in `validation/`.
