#!/usr/bin/env python3
"""Audit the real Boston NCEP TRAIN/VAL-only year-group membership."""

from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from emulator.data import ForcingGraphStore


def main():
    # Retain labels and tags only; load each real graph archive without keeping forcing tensors.
    store = ForcingGraphStore(REPO / "Data/Grid4_New/NCEP/graphs", "Boston", targets_only=True)
    splits = store.split(train_ratio=0.8333333333333334, val_ratio=0.16666666666666666,
                         shuffle_years=False, seed=42, future_only=False, future_year_threshold=2030)
    groups = {part: sorted({"_".join(store.graph_tags[i].split("_")[:2]) for i in indices})
              for part, indices in splits.items()}
    expected = {"train": [f"{year}_{year + 1}" for year in range(1979, 2009)],
                "val": [f"{year}_{year + 1}" for year in range(2009, 2015)], "test": []}
    assert groups == expected, f"Unexpected split membership: {groups}"
    train_groups, val_groups = set(groups["train"]), set(groups["val"])
    assert not train_groups & val_groups, "TRAIN and VAL year groups overlap"
    assert splits["test"] == [], "Canonical TEST must be empty"
    assert len(store.year_to_indices) == 36, "Expected exactly 36 available year groups"
    assert train_groups | val_groups == set(store.year_to_indices), "Missing year groups"
    train_indices, val_indices = set(splits["train"]), set(splits["val"])
    assert not train_indices & val_indices, "TRAIN and VAL graph indices overlap"
    assert train_indices | val_indices == set(range(len(store.graphs))), "Missing graphs"

    print("NCEP future-transfer split audit: PASS")
    print(f"TRAIN: 30 groups, {groups['train'][0]} -> {groups['train'][-1]}")
    print(f"VAL:    6 groups, {groups['val'][0]} -> {groups['val'][-1]}")
    print("TEST:   0 groups")
    print("Canonical TEST=None; train.py final TEST reporting will mirror VAL only.")


if __name__ == "__main__":
    main()
