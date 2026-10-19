"""Inference-time sample grouping helpers."""

from __future__ import annotations

import os


def infer_dataset_tag(root_dir: str) -> str:
    """Derive a dataset label from its graph-directory path.

    Only path text is inspected; the directory need not exist.

    Args:
        root_dir: Path-like value for the graph directory; None selects the fallback
            label data.

    Returns:
        str: Dataset component in Data/<dataset>/graphs, otherwise the parent of a
        trailing graphs directory, otherwise the final path component. Empty labels
        become data.
    """
    if root_dir is None:
        return "data"
    path = os.path.normpath(str(root_dir))
    parts = path.split(os.sep)
    for i in range(len(parts) - 2):
        if parts[i].lower() == "data" and parts[i + 2].lower() == "graphs":
            return parts[i + 1].strip() or "data"
    if len(parts) >= 2 and parts[-1].lower() == "graphs":
        return parts[-2].strip() or "data"
    return os.path.basename(path).strip() or "data"


def parse_year_tag(tag: str) -> tuple[int | None, int | None]:
    """Extract the leading year pair from an underscore-delimited graph tag.

    Args:
        tag: Tag beginning with two integer fields, for example 1979_1980_Battery_0.

    Returns:
        tuple: (year0, year1) as integers, or (None, None) when either field cannot be
        parsed. The years are not range-checked.
    """
    try:
        parts = str(tag).split("_")
        if len(parts) < 2:
            return (None, None)
        year0 = int(parts[0])
        year1 = int(parts[1])
        return (year0, year1)
    except Exception:
        return (None, None)


def classify_past_future(year0: int | None) -> str:
    """Assign the project period label from the first year of a graph group.

    Args:
        year0: First year as an integer, or None for an unparsed tag.

    Returns:
        str: past for 1979–2014 inclusive, future for 2070–2099 inclusive, and other for
        any remaining value.
    """
    if year0 is None:
        return "other"
    if 1979 <= year0 <= 2014:
        return "past"
    if 2070 <= year0 <= 2099:
        return "future"
    return "other"


__all__ = ["infer_dataset_tag", "parse_year_tag", "classify_past_future"]
