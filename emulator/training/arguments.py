"""The established train.py command-line interface used by the shell configs."""

import argparse
import math
from pathlib import Path

from emulator.common.cli import head_type_name, parse_bool_int, temporal_block_name
from emulator.common.dual import DUAL_ABLATIONS, EXCESS_FORMULATIONS
from .checkpoint_selection import ROLES
from .losses import (enforce_dual_loss, validate_excess_amp_config,
                     validate_shape_config, validate_wqe_config)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root_dir", type=str, default="./Data/NCEP/graphs")
    parser.add_argument("--test_root_dir", type=str, default="", help="If set: train/val on root_dir, test on ALL years in test_root_dir")
    parser.add_argument("--filter", type=str, default=None, help="Station key, e.g., Battery")
    parser.add_argument("--station", type=str, default=None, help="Alias for --filter (preferred)")
    parser.add_argument("--station_json_dir", type=str, default="./station_json")
    parser.add_argument(
        "--use_site_elevation", type=parse_bool_int, choices=[0, 1], default=1,
        help="Include finite site elevation (elevation_m/elevation/elev_m, scaled by 10) in PACT station features.",
    )
    parser.add_argument(
        "--use_bathymetry", type=parse_bool_int, choices=[0, 1], default=0,
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
    parser.add_argument("--hidden_channels", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.05, help="Light regularization; set 0.0 if it hurts.")
    parser.add_argument("--history_hours", type=int, default=24)
    parser.add_argument("--x_norm", type=str, default="robust", choices=["zscore", "robust", "mag"],
                        help=(
                            "Input normalization. "
                            "- zscore: (x-mean)/std computed on TRAIN (distributed)\n"
                            "- robust: (x-center)/scale using per-feature percentiles on TRAIN (rank0)\n"
                            "          center=(p_lo+p_hi)/2, scale=(p_hi-p_lo)/2 (OOD-stable)\n"
                            "- mag   : x / mag using per-feature magnitude on TRAIN (rank0), mag=p_hi percentile of |x|\n"
                            "          (no centering; can help if scale shift dominates, but won’t remove mean offset)."
                        ))
    parser.add_argument("--x_p_lo", type=float, default=1.0,
                        help="Lower percentile for robust normalization (e.g., 1).")
    parser.add_argument("--x_p_hi", type=float, default=99.0,
                        help="Upper percentile for robust normalization (e.g., 99).")
    parser.add_argument("--x_nodes_per_graph", type=int, default=256,
                        help="Nodes sampled per TRAIN graph for robust/mag statistics; <=0 uses the original default of 256.")
    parser.add_argument("--x_clip", type=float, default=5.0,
                        help="Clamp normalized X to [-x_clip, x_clip]. Helps OOD stability.")
    parser.add_argument("--x_aug", type=int, default=1, choices=[0, 1],
                        help="Enable train-time feature scale+bias jitter (in normalized space).")
    parser.add_argument("--x_aug_prob", type=float, default=1.0,
                        help="Probability to apply augmentation per batch (0..1).")
    parser.add_argument("--x_aug_scale", type=float, default=0.05,
                        help="Scale jitter range. a ~ U(1-x_aug_scale, 1+x_aug_scale).")
    parser.add_argument("--x_aug_bias", type=float, default=0.02,
                        help="Bias jitter std in normalized space. b ~ N(0, x_aug_bias).")
    parser.add_argument(
        "--loss_mode",
        type=str,
        default="mse",
        choices=["mse", "wqe", "wmse", "mse_slope", "wqe_slope", "wmse_slope"],
        help="Global physical MSE, TRAIN-scale WQE, or smooth tau-weighted MSE; suffix *_slope adds slope matching.",
    )
    parser.add_argument("--wqe_quantile_tau", type=float, default=0.25,
                        help="Shared global/excess WQE quantile level, strictly between 0 and 1.")
    parser.add_argument("--wqe_expectile_tau", type=float, default=0.82,
                        help="Shared global/excess WQE expectile level, strictly between 0 and 1.")
    parser.add_argument("--wqe_quantile_weight", type=float, default=1.0 / 6.0,
                        help="Nonnegative quantile weight; WQE weights must sum to 1.")
    parser.add_argument("--wqe_expectile_weight", type=float, default=5.0 / 6.0,
                        help="Nonnegative expectile weight; WQE weights must sum to 1.")
    parser.add_argument("--wmse_alpha", type=float, default=4.0,
                        help="alpha in w(y)=1+alpha*sigmoid((y-tau)/s).")
    parser.add_argument("--wmse_s", type=float, default=0.10,
                        help="s in meters (softness) for weighted MSE. Typical: 0.05~0.2.")
    parser.add_argument("--exceedance_percentile", type=float, default=95.0,
                        help="Linear quantile percentile of unique physical TRAIN target hours; reused for every split.")
    parser.add_argument("--exceedance_loss_weight", type=float, default=0.0,
                        help="Extra MSE on strict hourly y > TRAIN tau, divided by fixed TRAIN extreme-hour prevalence.")
    parser.add_argument("--slope_lambda", type=float, default=0.01,
                        help="Weight for slope-matching smoothness loss. Typical: 0.001~0.05. Used only for *_slope modes.")
    parser.add_argument("--slope_mask_s", type=float, default=0.10,
                        help="Soft-mask sharpness in meters: w=sigmoid((tau-peak)/s). Smaller => harder mask. Typical: 0.05~0.2.")
    parser.add_argument("--slope_robust", type=str, default="charb", choices=["charb", "huber"],
                        help="Robust penalty for slope error. charb=Charbonnier; huber=Huber.")
    parser.add_argument("--slope_charb_eps", type=float, default=1e-3,
                        help="Charbonnier eps in meters for slope robust penalty.")
    parser.add_argument("--slope_huber_delta", type=float, default=0.05,
                        help="Huber delta in meters for slope robust penalty.")
    parser.add_argument("--scheduler", type=str, default="cosine", choices=["cosine", "rop"])
    parser.add_argument("--min_lr", type=float, default=1e-6, help="eta_min for cosine")
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--warmup_start_factor", type=float, default=0.1,
                        help="Linear warmup start lr factor (e.g., 0.1 means start at 10%% of lr)")
    parser.add_argument("--rop_factor", type=float, default=0.5)
    parser.add_argument("--rop_patience", type=int, default=20)
    parser.add_argument("--rop_threshold", type=float, default=1e-4)
    parser.add_argument("--rop_cooldown", type=int, default=0)
    parser.add_argument("--rop_min_lr", type=float, default=1e-6)
    parser.add_argument("--rop_metric", type=str, default="val_all_rmse",
                        choices=["val_all_rmse", "val_exceedance_rmse"],
                        help="Physical VAL RMSE used for ReduceLROnPlateau stepping.")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--pin_memory", action="store_true")
    parser.add_argument("--persistent_workers", action="store_true")
    parser.add_argument("--prefetch_factor", type=int, default=2)
    parser.add_argument("--mp_context", type=str, default="fork", choices=["fork", "spawn"])
    parser.add_argument("--torch_threads", type=int, default=1)
    parser.add_argument("--tf32", action="store_true")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--amp_dtype", type=str, default="bf16", choices=["bf16", "fp16"])
    parser.add_argument("--model", type=str, default="baseline",
                        choices=["baseline", "perceiver3"])
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
        "--cnn_intermediate_channel", type=int, default=29,
        help="CNN channels before the final spatial layer; output remains hidden_channels.",
    )
    parser.add_argument(
        "--head_type",
        type=head_type_name,
        default="dual",
        choices=["single", "dual"],
        help="PACT prediction head: single MLP or supervised exceedance dual head.",
    )
    parser.add_argument("--head_dropout", type=float, default=0.0)
    parser.add_argument("--exceedance_head_experiment", choices=("legacy", "c1", "c2", "c2r", "c3"), default=None,
                        help="Opt-in direct-dual excess decoder; omitted keeps the production head.")
    parser.add_argument("--exceedance_gate_pooling", choices=("mean", "learned"), default="mean",
                        help="Event pooling for exceedance_head_experiment only.")
    parser.add_argument("--gate_mode", choices=["window"], default="window", help="Supervised event probability for the whole forecast window.")
    parser.add_argument("--dual_mode", choices=["exceedance"], default="exceedance", help="Current supervised dual head; single-head runs ignore this setting.")
    parser.add_argument("--dual_loss", type=parse_bool_int, choices=[0, 1], default=1,
                        help="Branch supervision; required unless --dual_ablation explicitly removes it.")
    parser.add_argument("--dual_ablation", choices=tuple(DUAL_ABLATIONS), default="none",
                        help="Explicit mechanism experiment; none enforces full branch supervision.")
    parser.add_argument("--body_loss_weight", type=float, default=1.0)
    parser.add_argument("--excess_loss_weight", type=float, default=1.0)
    parser.add_argument("--excess_loss_mode", choices=["mse", "wqe"], default="mse",
                        help="Dual conditional raw-excess objective only; independent of --loss_mode and inactive for Single.")
    parser.add_argument("--excess_formulation", choices=EXCESS_FORMULATIONS, default="direct",
                        help="Direct normalized excess or physical severity times normalized temporal shape.")
    parser.add_argument("--shape_loss_weight", type=float, default=0.0,
                        help="Optional dimensionless temporal shape supervision; requires severity_shape.")
    parser.add_argument("--severity_shape_eps", type=float, default=1e-6,
                        help="Positive epsilon added before predicted shape maximum normalization; GT amplitudes are not floored.")
    parser.add_argument("--excess_amp_loss_weight", type=float, default=0.0,
                        help="Optional physical excess amplitude loss at the true target peak time; requires the supervised dual head.")
    parser.add_argument("--gate_loss_weight", type=float, default=1.0)
    parser.add_argument("--max_grad_norm", type=float, default=0.0, help="0 disables optional gradient clipping.")
    parser.add_argument("--deterministic", type=parse_bool_int, choices=[0, 1], default=0,
                        help="Enforce deterministic kernels for reproducible runs; may reduce throughput.")
    parser.add_argument(
        "--temporal_block",
        type=temporal_block_name,
        default="Transformer",
        choices=["MLP", "LSTM", "GRU", "Transformer"],
        help="PACT middle temporal block; 'attn' is accepted as an alias for Transformer.",
    )
    parser.add_argument("--node_read_heads", type=int, default=8)
    parser.add_argument("--time_read_heads", type=int, default=8)
    parser.add_argument("--transformer_layers", type=int, default=2)
    parser.add_argument("--transformer_ff_mult", type=float, default=4.0)
    parser.add_argument("--transformer_dropout", type=float, default=0.05)
    parser.add_argument("--max_time_steps", type=int, default=32,
                        help="PACT lag-embedding capacity, including the current step; ignored by baseline.")
    parser.add_argument("--checkpoint_selection", choices=ROLES, default="overall",
                        help="Primary checkpoint role; all four VAL-only roles are always retained. "
                             "Minimize AllRMSE (overall), ExceedanceRMSE (exceedance), "
                             "GTAlignedPeakRMSE (aligned_peak), or Balanced Event-Aware "
                             "0.50*AllRMSE + 0.25*ExceedanceRMSE + 0.25*GTAlignedPeakRMSE (bea).")
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
    if args.station is not None and args.filter is not None and args.station != args.filter:
        parser.error("--station and --filter must select the same station.")
    args.station = args.station or args.filter
    args.station_json_dir = Path(args.station_json_dir)
    if args.model == "baseline":
        args.head_type = "single"
    if args.exceedance_head_experiment is not None and (
            args.model != "perceiver3" or args.head_type != "dual" or args.excess_formulation != "direct"):
        parser.error("--exceedance_head_experiment requires --model perceiver3 --head_type dual --excess_formulation direct.")
    if args.encoder_type == "GraphSAGE" and args.num_layers < 2:
        parser.error("GraphSAGE requires --num_layers >= 2.")
    if args.encoder_type == "CNN" and (args.num_layers < 1 or args.cnn_intermediate_channel < 1):
        parser.error("CNN requires positive layer count and intermediate width.")
    if args.history_hours < 0 or args.history_hours % 6:
        parser.error("--history_hours must be a nonnegative multiple of 6.")
    if args.model == "perceiver3" and args.max_time_steps < args.history_hours // 6 + 1:
        parser.error("--max_time_steps is smaller than the requested history window.")
    if min(args.batch_size, args.epochs, args.grad_accum_steps, args.torch_threads, args.transformer_layers) < 1:
        parser.error("Batch size, epochs, accumulation, threads and temporal depth must be positive.")
    if not 0 < args.lr or args.min_lr < 0 or not 0 < args.warmup_start_factor <= 1:
        parser.error("Invalid learning-rate or warmup settings.")
    if args.warmup_epochs < 0:
        parser.error("--warmup_epochs must be nonnegative.")
    if not math.isfinite(args.exceedance_percentile) or not 0 < args.exceedance_percentile < 100:
        parser.error("Invalid TRAIN loss percentile settings.")
    for name in ("max_grad_norm", "body_loss_weight", "excess_loss_weight", "gate_loss_weight", "exceedance_loss_weight", "wmse_alpha", "slope_lambda"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) < 0:
            parser.error(f"--{name} must be finite and nonnegative.")
    for name in ("wmse_s", "slope_mask_s", "slope_charb_eps", "slope_huber_delta"):
        if not math.isfinite(getattr(args, name)):
            parser.error(f"--{name} must be finite; small values use the original numerical floor.")
    try:
        validate_wqe_config(args)
        validate_excess_amp_config(args, head_type=args.head_type)
        validate_shape_config(args, head_type=args.head_type, model=args.model)
    except ValueError as error:
        parser.error(str(error))
    if args.head_type == "dual":
        enforce_dual_loss(args)
    else:
        if args.dual_ablation != "none":
            parser.error("--dual_ablation requires --model perceiver3 --head_type dual.")
        args.dual_loss = 0
    return args
