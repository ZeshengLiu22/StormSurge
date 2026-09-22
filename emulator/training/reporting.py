"""Human-readable physical metric tables, using the canonical metric dictionary."""

from .metrics import METRIC_GROUPS, METRIC_LABELS


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
    lines = [threshold_label(metadata), "", "Delta = eventaware minus overall.", "",
             f'Overall epoch: {evaluations["overall"]["epoch"]}; '
             f'event-aware epoch: {evaluations["eventaware"]["epoch"]}.', ""]
    for split in ("val", "test"):
        for group, entries in METRIC_GROUPS.items():
            lines.extend([f"{split.upper()} — {group}", "",
                          "| Metric | Overall | Event-aware | Delta |", "| --- | ---: | ---: | ---: |"])
            for key in entries:
                label = METRIC_LABELS[key]
                a, b = (evaluations[role][split].get(key) for role in ("overall", "eventaware"))
                delta = None if a is None or b is None else b - a
                lines.append(f"| {label} | {display(a)} | {display(b)} | {display(delta)} |")
            lines.append("")
    return "\n".join(lines) + "\n"
