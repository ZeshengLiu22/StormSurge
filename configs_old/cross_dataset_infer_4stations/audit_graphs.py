#!/usr/bin/env python3
"""Inspect real target batches for all stations without fitting statistics."""
import json
import sys

from generate_configs import REPO, RESULTS, STATIONS, TARGETS, YEARS, dataset_root, require

sys.path.insert(0, str(REPO))
import torch
from emulator.data import ForcingGraphStore, ForcingGraphView, build_loader


def main():
    torch.set_num_threads(1)
    rows = []
    for station in STATIONS:
        for group, targets in TARGETS.items():
            for target in targets:
                year, root = YEARS[group][0], dataset_root(group, target)
                store = ForcingGraphStore(REPO / root, station, pattern=f'{year}_{station}_*graphs.pt')
                view = ForcingGraphView(store, range(min(2, len(store.graphs))), history_steps=4)
                batch = next(iter(build_loader(view, None, 2, 0, False, False, 0, 'fork')))
                identity = f'{station}/{group}/{target}'
                require(batch.x.shape[1] == 5 and tuple(batch.x_hist.shape[1:]) == (5, 5), identity + ': feature/history mismatch')
                require(batch.y.shape[1] == 6, identity + ': target width mismatch')
                require(int(batch.edge_index.min()) >= 0 and int(batch.edge_index.max()) < len(batch.x), identity + ': invalid edges')
                sizes = sorted({int(graph.x.shape[0]) for graph in store.graphs})
                rows.append(dict(station=station, group=group, target=target, root=root,
                    inspected_year=year, graphs_in_year=len(store.graphs), node_counts=sizes,
                    batch_node_counts=batch.ptr.diff().tolist(), batch_x_hist_shape=list(batch.x_hist.shape),
                    batch_y_shape=list(batch.y.shape)))
                print(f'PASS {identity}: nodes={sizes}, history={tuple(batch.x_hist.shape)}, y={tuple(batch.y.shape)}', flush=True)
                del batch, view, store
    output = RESULTS / 'audit/graph_audit.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(rows, indent=2) + '\n')
    print(f'PASS: {len(rows)} station/target/period batches')


if __name__ == '__main__':
    main()
