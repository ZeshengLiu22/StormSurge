"""Human-readable physical metric tables, using the canonical metric dictionary."""

from .metrics import METRIC_GROUPS, METRIC_LABELS
from .eventaware_checkpoints import ROLES


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
    labels = ("Overall", "Exceedance", "Aligned-peak", "Equal-composite", "Peak-priority", "Legacy event-aware")
    lines = [threshold_label(metadata), ""]
    lines.extend(f'{label} epoch: {evaluations[role]["epoch"]}' for role, label in zip(ROLES, labels))
    lines.append("")
    for split in ("val", "test"):
        groups = dict(METRIC_GROUPS)
        # Include per-lead diagnostics as well as every canonical metric group.
        extra = sorted({key for role in ROLES for key in evaluations[role][split]} - set(METRIC_LABELS))
        if extra:
            groups["Additional diagnostics"] = extra
        for group, entries in groups.items():
            lines.extend([f"{split.upper()} — {group}", "",
                          "| Metric | Overall | Exceedance | AlignedPeak | Equal | PeakPriority | EventAware |",
                          "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"])
            for key in entries:
                label = METRIC_LABELS.get(key, key)
                values = " | ".join(display(evaluations[role][split].get(key)) for role in ROLES)
                lines.append(f"| {label} | {values} |")
            lines.append("")
    return "\n".join(lines) + "\n"
