"""The established train.py command-line interface used by the shell configs."""

import argparse
import math
import os
from pathlib import Path

from emulator.common.cli import parse_bool_int, temporal_block_name
from emulator.models import BASELINE_ARCHITECTURES, NEW_BASELINE_MODELS, STATIC_BASELINE_MODELS
from emulator.models.node_attention import NODE_ATTENTION_BACKENDS
from .checkpoint_selection import ROLES
from .losses import validate_episode_peak_config
from .weight_decay import WEIGHT_DECAY_POLICIES


def parse_args(argv=None):
    """Parse and validate the training command-line configuration.

    Validation checks architecture depth, six-hour history increments,
    optimization settings, and threshold/episode-supervision constraints. Invalid
    arguments are reported through argparse and terminate with SystemExit.

    Args:
        argv: Optional sequence of argument strings excluding the program name; None
            reads sys.argv[1:].

    Returns:
        argparse.Namespace: Parsed settings with station/filter aliases reconciled and
        station_json_dir converted to pathlib.Path.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root_dir", type=str, default="./Data/NCEP/graphs")
    parser.add_argument(
        "--pressure_mode", choices=("anomaly", "raw", "anomaly_plus_mean"), default="anomaly",
        help="Forcing pressure representation, applied before TRAIN normalization (default: anomaly).",
    )
    parser.add_argument(
        "--test_root_dir",
        type=str,
        default="",
        help="If set: train/val on root_dir, test on ALL years in test_root_dir",
    )
    parser.add_argument("--filter", type=str, default=None, help="Station key, e.g., Battery")
    parser.add_argument("--station", type=str, default=None, help="Alias for --filter (preferred)")
    parser.add_argument("--station_json_dir", type=str, default="./station_json")
    parser.add_argument(
        "--use_site_elevation",
        type=parse_bool_int,
        choices=[0, 1],
        default=1,
        help="Include finite site elevation (elevation_m/elevation/elev_m, scaled by 10) in PACT station features.",
    )
    parser.add_argument(
        "--use_bathymetry",
        type=parse_bool_int,
        choices=[0, 1],
        default=0,
        help="Include bathymetry_m / 10 in PACT station features; requires that JSON field to be finite.",
    )
    parser.add_argument("--train_ratio", type=float, default=0.6)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument(
        "--shuffle_years",
        "--shuffle_split_years",
        dest="shuffle_years",
        type=parse_bool_int,
        default=0,
        choices=[0, 1],
        help="If 1, shuffle year groups with --seed before train/val/test ratio split.",
    )
    parser.add_argument(
        "--future_only",
        "--future_only_years",
        dest="future_only",
        type=parse_bool_int,
        default=0,
        choices=[0, 1],
        help="If 1, keep only year tags with any year component > --future_year_threshold before ratio split.",
    )
    parser.add_argument(
        "--future_year_threshold",
        type=int,
        default=2030,
        help="Year threshold used by --future_only; default keeps tags containing years after 2030.",
    )
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument(
        "--grad_accum_steps",
        type=int,
        default=1,
        help="Number of microbatches per optimizer update.",
    )
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument(
        "--weight_decay_policy", choices=WEIGHT_DECAY_POLICIES,
        default=os.environ.get("WEIGHT_DECAY_POLICY", "module"),
        help="Adam decay policy (WEIGHT_DECAY_POLICY or module); legacy_global opts into blanket WD=1e-5.",
    )
    parser.add_argument("--hidden_channels", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument(
        "--dropout", type=float, default=0.05, help="Light regularization; set 0.0 if it hurts."
    )
    parser.add_argument("--history_hours", type=int, default=24)
    parser.add_argument(
        "--x_norm",
        type=str,
        default="robust",
        choices=["zscore", "robust", "mag"],
        help=(
            "Input normalization. "
            "- zscore: (x-mean)/std computed on TRAIN (distributed)\n"
            "- robust: (x-center)/scale using per-feature percentiles on TRAIN (rank0)\n"
            "          center=(p_lo+p_hi)/2, scale=(p_hi-p_lo)/2 (OOD-stable)\n"
            "- mag   : x / mag using per-feature magnitude on TRAIN (rank0), mag=p_hi percentile of |x|\n"
            "          (no centering; can help if scale shift dominates, but won’t remove mean offset)."
        ),
    )
    parser.add_argument(
        "--x_p_lo",
        type=float,
        default=1.0,
        help="Lower percentile for robust normalization (e.g., 1).",
    )
    parser.add_argument(
        "--x_p_hi",
        type=float,
        default=99.0,
        help="Upper percentile for robust normalization (e.g., 99).",
    )
    parser.add_argument(
        "--x_nodes_per_graph",
        type=int,
        default=256,
        help="Nodes sampled per TRAIN graph for robust/mag statistics; <=0 uses the original default of 256.",
    )
    parser.add_argument(
        "--x_clip",
        type=float,
        default=5.0,
        help="Clamp normalized X to [-x_clip, x_clip]. Helps OOD stability.",
    )
    parser.add_argument(
        "--x_aug",
        type=int,
        default=1,
        choices=[0, 1],
        help="Enable train-time feature scale+bias jitter (in normalized space).",
    )
    parser.add_argument(
        "--x_aug_prob",
        type=float,
        default=1.0,
        help="Probability to apply augmentation per batch (0..1).",
    )
    parser.add_argument(
        "--x_aug_scale",
        type=float,
        default=0.05,
        help="Scale jitter range. a ~ U(1-x_aug_scale, 1+x_aug_scale).",
    )
    parser.add_argument(
        "--x_aug_bias",
        type=float,
        default=0.02,
        help="Bias jitter std in normalized space. b ~ N(0, x_aug_bias).",
    )
    parser.add_argument(
        "--exceedance_percentile",
        type=float,
        default=95.0,
        help="Linear quantile percentile of unique physical TRAIN target hours; reused for every split.",
    )
    parser.add_argument(
        "--exceedance_loss_weight",
        type=float,
        default=0.0,
        help="Extra Tail loss on strict hourly y > TRAIN tau, divided by fixed TRAIN extreme-hour prevalence.",
    )
    parser.add_argument(
        "--episode_gt_aligned_peak_weight",
        type=float,
        default=0.0,
        help="Single episode GT-peak amplitude MSE weight; fixed TRAIN peak prevalence normalization.",
    )
    parser.add_argument("--scheduler", type=str, default="cosine", choices=["cosine", "rop"])
    parser.add_argument("--min_lr", type=float, default=1e-6, help="eta_min for cosine")
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument(
        "--warmup_start_factor",
        type=float,
        default=0.1,
        help="Linear warmup start lr factor (e.g., 0.1 means start at 10%% of lr)",
    )
    parser.add_argument("--rop_factor", type=float, default=0.5)
    parser.add_argument("--rop_patience", type=int, default=20)
    parser.add_argument("--rop_threshold", type=float, default=1e-4)
    parser.add_argument("--rop_cooldown", type=int, default=0)
    parser.add_argument("--rop_min_lr", type=float, default=1e-6)
    parser.add_argument(
        "--rop_metric",
        type=str,
        default="val_all_rmse",
        choices=["val_all_rmse", "val_exceedance_rmse"],
        help="Physical VAL RMSE used for ReduceLROnPlateau stepping.",
    )
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--pin_memory", action="store_true")
    parser.add_argument("--persistent_workers", action="store_true")
    parser.add_argument("--prefetch_factor", type=int, default=2)
    parser.add_argument("--mp_context", type=str, default="fork", choices=["fork", "spawn"])
    parser.add_argument("--torch_threads", type=int, default=1)
    parser.add_argument("--tf32", action="store_true")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--amp_dtype", type=str, default="bf16", choices=["bf16", "fp16"])
    parser.add_argument(
        "--model", type=str, default="baseline",
        choices=["baseline", "perceiver3", *NEW_BASELINE_MODELS],
    )
    parser.add_argument(
        "--baseline_pooling", choices=("adaptive", "global_mean"), default="adaptive",
        help="Spatial pooling for the seven shallow baselines only.",
    )
    parser.add_argument(
        "--baseline_pool_grid", type=int, default=4,
        help="Positive adaptive pooling grid side length for shallow baselines.",
    )
    parser.add_argument(
        "--baseline_stem_channels", type=int, default=32,
        help="Shallow baseline spatial stem width (and ConvLSTM recurrent width).",
    )
    parser.add_argument(
        "--encoder_type",
        type=str,
        default="GraphSAGE",
        choices=["GraphSAGE", "CNN"],
        help=(
            "Spatial encoder. GraphSAGE uses edge_index; CNN reshapes each graph "
            "from (H*W,F) to (F,H,W) using grid_H/grid_W stored in the input data."
        ),
    )
    parser.add_argument(
        "--cnn_intermediate_channel",
        type=int,
        default=29,
        help="CNN channels before the final spatial layer; output remains hidden_channels.",
    )
    parser.add_argument("--head_dropout", type=float, default=0.0)
    parser.add_argument(
        "--max_grad_norm", type=float, default=0.0, help="0 disables optional gradient clipping."
    )
    parser.add_argument(
        "--deterministic",
        type=parse_bool_int,
        choices=[0, 1],
        default=0,
        help="Enforce deterministic kernels for reproducible runs; may reduce throughput.",
    )
    parser.add_argument(
        "--temporal_block",
        type=temporal_block_name,
        default="Transformer",
        choices=["MLP", "LSTM", "GRU", "Transformer"],
        help="PACT middle temporal block; 'attn' is accepted as an alias for Transformer.",
    )
    parser.add_argument("--node_read_heads", type=int, default=8)
    parser.add_argument(
        "--node_attention_backend", choices=NODE_ATTENTION_BACKENDS,
        default=os.environ.get("NODE_ATTENTION_BACKEND", "math"),
        help="PACT station-to-node SDPA (NODE_ATTENTION_BACKEND or math); auto preserves historical dispatch. No effect on models without node attention.",
    )
    parser.add_argument("--time_read_heads", type=int, default=8)
    parser.add_argument("--transformer_layers", type=int, default=2)
    parser.add_argument("--transformer_ff_mult", type=float, default=4.0)
    parser.add_argument("--transformer_dropout", type=float, default=0.05)
    parser.add_argument(
        "--max_time_steps",
        type=int,
        default=32,
        help="PACT lag-embedding capacity, including the current step; ignored by baseline.",
    )
    parser.add_argument(
        "--checkpoint_selection",
        choices=ROLES,
        default="exceedance",
        help="Primary VAL checkpoint: exceedance (research default) or overall (sensitivity); both retained.",
    )
    parser.add_argument("--run_tag", type=str, default=None)
    parser.add_argument(
        "--output_dir",
        type=str,
        default="",
        help=(
            "Exact directory for all artifacts from this run. When omitted, "
            "train.py creates All_Results/<timestamp>_<run-name>/; run-name "
            "comes from --run_tag or the dataset/station/model arguments."
        ),
    )

    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--use_station_meta", type=parse_bool_int, choices=[0, 1], default=1)
    args = parser.parse_args(argv)
    # argparse choices do not validate defaults taken from the environment.
    if args.node_attention_backend not in NODE_ATTENTION_BACKENDS:
        parser.error(f"Invalid NODE_ATTENTION_BACKEND: {args.node_attention_backend!r}.")
    if args.weight_decay_policy not in WEIGHT_DECAY_POLICIES:
        parser.error(f"Invalid WEIGHT_DECAY_POLICY: {args.weight_decay_policy!r}.")
    if args.station is not None and args.filter is not None and args.station != args.filter:
        parser.error("--station and --filter must select the same station.")
    args.station = args.station or args.filter
    args.station_json_dir = Path(args.station_json_dir)
    if args.model in NEW_BASELINE_MODELS:
        if min(args.baseline_pool_grid, args.baseline_stem_channels, args.hidden_channels) < 1:
            parser.error("Shallow baseline pooling grid and channel dimensions must be positive.")
        if not math.isfinite(args.dropout) or not 0 <= args.dropout <= 1:
            parser.error("--dropout must be finite and between 0 and 1.")
        if args.model in STATIC_BASELINE_MODELS and args.history_hours != 0:
            parser.error(f"{args.model} requires --history_hours 0 (current forcing only).")
        args.effective_spatial_encoder, args.effective_temporal_module = BASELINE_ARCHITECTURES[args.model]
    else:
        if args.encoder_type == "GraphSAGE" and args.num_layers < 2:
            parser.error("GraphSAGE requires --num_layers >= 2.")
        if args.encoder_type == "CNN" and (args.num_layers < 1 or args.cnn_intermediate_channel < 1):
            parser.error("CNN requires positive layer count and intermediate width.")
    if args.history_hours < 0 or args.history_hours % 6:
        parser.error("--history_hours must be a nonnegative multiple of 6.")
    if args.model == "perceiver3" and args.max_time_steps < args.history_hours // 6 + 1:
        parser.error("--max_time_steps is smaller than the requested history window.")
    if (
        min(
            args.batch_size,
            args.epochs,
            args.grad_accum_steps,
            args.torch_threads,
            args.transformer_layers if args.model not in NEW_BASELINE_MODELS else 1,
        )
        < 1
    ):
        parser.error(
            "Batch size, epochs, accumulation, threads and temporal depth must be positive."
        )
    if not 0 < args.lr or args.min_lr < 0 or not 0 < args.warmup_start_factor <= 1:
        parser.error("Invalid learning-rate or warmup settings.")
    if args.warmup_epochs < 0:
        parser.error("--warmup_epochs must be nonnegative.")
    if not math.isfinite(args.exceedance_percentile) or not 0 < args.exceedance_percentile < 100:
        parser.error("Invalid TRAIN loss percentile settings.")
    for name in ("max_grad_norm", "exceedance_loss_weight"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) < 0:
            parser.error(f"--{name} must be finite and nonnegative.")
    try:
        validate_episode_peak_config(args)
        if args.episode_gt_aligned_peak_weight > 0 and args.exceedance_percentile != 95.0:
            raise ValueError("Episode peak supervision requires the fixed TRAIN Q95 threshold.")
    except ValueError as error:
        parser.error(str(error))
    return args
