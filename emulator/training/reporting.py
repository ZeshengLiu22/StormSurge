"""Human-readable physical metric tables, using the canonical metric dictionary."""

from .metrics import METRIC_GROUPS, METRIC_LABELS
from .checkpoint_selection import ROLES


_ROLE_LABELS = {
    "overall": "Overall",
    "exceedance": "Exceedance",
    "aligned_peak": "Aligned-peak",
    "bea": "BEA",
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
        groups = dict(METRIC_GROUPS)
        # Include per-lead diagnostics as well as every canonical metric group.
        extra = sorted({key for role in ROLES for key in evaluations[role][split]} - set(METRIC_LABELS))
        if extra:
            groups["Additional diagnostics"] = extra
        for group, entries in groups.items():
            lines.extend([f"{split.upper()} — {group}", "",
                          "| Metric | Overall | Exceedance | AlignedPeak | BEA |",
                          "| --- | ---: | ---: | ---: | ---: |"])
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
