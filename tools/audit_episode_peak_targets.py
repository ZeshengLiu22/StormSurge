#!/usr/bin/env python3
"""Data-only parity audit for TRAIN episode peak supervision; never trains a model."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from emulator.data import ForcingGraphStore, fit_loss_thresholds, supervised_targets
from emulator.training.episode_peaks import build_episode_peak_targets
from emulator.training.losses import episode_gt_aligned_peak_mse
from emulator.training.metrics import evaluate_metrics, gt_event_episodes


def audit_station(root, station, split_config):
    store = ForcingGraphStore(root, station, targets_only=True)
    splits = store.split(**split_config)
    fitted = fit_loss_thresholds(store, splits['train'])
    truth, times = supervised_targets(store, splits['train'])
    targets = build_episode_peak_targets(truth, times, fitted['tau_physical'])
    episodes = gt_event_episodes(truth, times, fitted['tau_physical'])
    mask = targets.is_episode_gt_peak
    # Independent oracle: one earliest chronological GT maximum per episode.
    expected = np.zeros(truth.size, dtype=bool)
    for episode in episodes:
        maximum = truth.ravel()[episode].max()
        ties = episode[truth.ravel()[episode] == maximum]
        expected[ties[np.argmin(times.ravel()[ties])]] = True
        assert mask.ravel()[episode].sum() == 1
    np.testing.assert_array_equal(mask.ravel(), expected)
    assert int(mask.sum()) == len(episodes) == fitted['train_episode_count']
    assert truth.size == np.unique(times).size
    # Position-dependent errors ensure accidental window weighting cannot pass.
    prediction = truth + (np.arange(truth.size).reshape(truth.shape) % 101 - 50) / 1000.
    metrics = evaluate_metrics(prediction, truth, fitted['tau_physical'], target_timestamps=times)
    loss = episode_gt_aligned_peak_mse(torch.from_numpy((prediction-truth)**2),
                                      torch.from_numpy(mask), targets.metadata['p_episode_peak']).item()
    expected_loss = float(np.mean((prediction.ravel()[expected]-truth.ravel()[expected])**2))
    np.testing.assert_allclose(loss, expected_loss, rtol=1e-12, atol=1e-15)
    np.testing.assert_allclose(loss, metrics['episode_gt_aligned_peak_rmse']**2, rtol=1e-12, atol=1e-15)
    selected = set(splits['train'])
    return dict(station=station, **targets.metadata,
                train_year_groups=[year for year, indices in store.year_to_indices.items() if selected.intersection(indices)],
                cross_window_episode_count=sum(len(set(episode // truth.shape[1])) > 1 for episode in episodes),
                single_hour_episode_count=sum(len(episode) == 1 for episode in episodes),
                tied_gt_peak_episode_count=sum(np.count_nonzero(truth.ravel()[episode] == truth.ravel()[episode].max()) > 1 for episode in episodes),
                final_metric_count=len(metrics), parity_loss_m2=loss,
                final_aligned_rmse_squared_m2=metrics['episode_gt_aligned_peak_rmse']**2,
                max_parity_error_m2=abs(loss - expected_loss), passed=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root-dir', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--stations', nargs='+', default=['CBBT', 'Lewes', 'Battery', 'Boston'])
    parser.add_argument('--train-ratio', type=float, default=.6)
    parser.add_argument('--val-ratio', type=float, default=.2)
    parser.add_argument('--shuffle-years', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args(argv)
    split_config = dict(train_ratio=args.train_ratio, val_ratio=args.val_ratio, shuffle_years=args.shuffle_years, seed=args.seed)
    audits = {}
    for station in args.stations:
        print(f'Auditing TRAIN episode peak targets: {station}', flush=True)
        audits[station] = audit_station(args.root_dir, station, split_config)
        values = audits[station]
        print(f'  episodes={values["episode_count"]} canonical peaks={values["canonical_peak_target_count"]} '
              f'p={values["p_episode_peak"]:.12g}; parity passed', flush=True)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(dict(source=str(Path(args.root_dir).resolve()), split_config=split_config,
                                     stations=audits, passed=True), indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
