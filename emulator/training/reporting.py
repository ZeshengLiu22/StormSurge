"""Human-readable physical metric tables, using the canonical metric dictionary."""

from .metrics import METRIC_GROUPS, METRIC_LABELS
from .checkpoint_selection import ROLES


_ROLE_LABELS = {
    "overall": "Overall",
    "exceedance": "Exceedance",
}
_COMPACT_METRIC_ROWS = (
    (("all_rmse", "all_mae"), 1000, "mm"),
    (("exceedance_rmse", "exceedance_mae"), 1000, "mm"),
    (("episode_peak_rmse", "episode_peak_mae", "episode_peak_bias"), 1000, "mm"),
    (("episode_gt_aligned_peak_rmse", "episode_gt_aligned_peak_mae",
      "episode_gt_aligned_peak_bias"), 1000, "mm"),
    (("episode_peak_timing_mae_hours",), 1, "h"),
)


def threshold_label(metadata):
    return (f'Extreme threshold: TRAIN hourly target Q{metadata["exceedance_percentile"]:g}; '
            f'tau = {metadata["tau_physical"]:.9f} meters; strict exceedance y > tau')


def display(value):
    return "NA" if value is None else f"{value:.9g}"


def metric_report(metrics, metadata):
    lines = [threshold_label(metadata), ""]
    for group, entries in METRIC_GROUPS.items():
        lines.extend([group, "", "| Metric | Value |", "| --- | ---: |"])
        for key in entries:
            label = METRIC_LABELS[key]
            lines.append(f"| {label} | {display(metrics.get(key))} |")
        lines.append("")
    return "\n".join(lines) + "\n"


def comparison_report(evaluations, metadata):
    lines = [threshold_label(metadata), ""]
    lines.extend(f'{_ROLE_LABELS[role]} epoch: {evaluations[role]["epoch"]}' for role in ROLES)
    lines.append("")
    for split in ("val", "test"):
        groups = METRIC_GROUPS
        for group, entries in groups.items():
            lines.extend([f"{split.upper()} — {group}", "",
                          "| Metric | " + " | ".join(_ROLE_LABELS[role] for role in ROLES) + " |",
                          "| --- | " + " | ".join("---:" for role in ROLES) + " |"])
            for key in entries:
                label = METRIC_LABELS.get(key, key)
                values = " | ".join(display(evaluations[role][split].get(key)) for role in ROLES)
                lines.append(f"| {label} | {values} |")
            lines.append("")
    return "\n".join(lines) + "\n"


def compact_comparison_report(evaluations):
    """Summarize retained roles in mm/h for the console without changing stored metrics."""
    lines = ["Selected epochs | " + " | ".join(
        f'{_ROLE_LABELS.get(role, role)}={evaluation["epoch"]}'
        for role, evaluation in evaluations.items())]
    for role, evaluation in evaluations.items():
        lines.extend(["", f'[{_ROLE_LABELS.get(role, role)} | epoch {evaluation["epoch"]}]'])
        for split in ("val", "test"):
            lines.extend(["", split.upper()])
            for keys, scale, unit in _COMPACT_METRIC_ROWS:
                values = []
                for key in keys:
                    value = evaluation[split].get(key)
                    formatted = "NA" if value is None else f"{value * scale:.2f} {unit}"
                    values.append(f"{METRIC_LABELS[key]}={formatted}")
                lines.append(" | ".join(values))
    return "\n".join(lines) + "\n"


def format_loss_components(losses):
    labels = dict(global_mse_raw="GlobalMSE", tail_mse_raw="TailMSE",
                  tail_weighted="TailWeighted", episode_gt_aligned_peak_mse_raw="EpisodeGTAlignedPeakMSE",
                  episode_gt_aligned_peak_weighted="EpisodePeakWeighted",
                  episode_peak_target_count="EpisodePeakN", p_episode_peak="TRAINPeakPrevalence",
                  total_loss="TotalLoss")
    return "Loss " + " ".join(f"{label}={display(losses.get(key))}" for key, label in labels.items())
