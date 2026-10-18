#!/usr/bin/env python3
"""Inspect one requested Boston year per target through the real CPU data path.

This loads graph data and constructs batches; it never runs a model or fits stats.
"""
import json
import sys

from generate_configs import DESTINATION, REPO, TARGETS, YEARS, dataset_root, require

sys.path.insert(0, str(REPO))
import torch
from emulator.data import ForcingGraphStore, ForcingGraphView, build_loader


def main():
    torch.set_num_threads(1)
    rows = []
    for group, targets in TARGETS.items():
        for target in targets:
            year = YEARS[group][0]
            root = dataset_root(group, target)
            store = ForcingGraphStore(REPO / root, "Boston", pattern=f"{year}_Boston_*graphs.pt")
            view = ForcingGraphView(store, range(min(2, len(store.graphs))), history_steps=4)
            batch = next(iter(build_loader(view, None, 2, 0, False, False, 0, "fork")))
            require(batch.x.shape[1] == 5 and tuple(batch.x_hist.shape[1:]) == (5, 5),
                    f"{group}/{target}: incompatible feature/history dimensions")
            require(batch.y.shape[1] == 6, f"{group}/{target}: incompatible output width")
            require(int(batch.edge_index.max()) < len(batch.x), f"{group}/{target}: invalid edge indices")
            sizes = sorted({int(graph.x.shape[0]) for graph in store.graphs})
            row = dict(group=group, target=target, root=root, inspected_year=year,
                       graphs_in_year=len(store.graphs), node_counts=sizes,
                       batch_node_counts=batch.ptr.diff().tolist(), batch_x_shape=list(batch.x.shape),
                       batch_x_hist_shape=list(batch.x_hist.shape), batch_y_shape=list(batch.y.shape),
                       grid_H=int(store.graphs[0].grid_H), grid_W=int(store.graphs[0].grid_W))
            rows.append(row)
            print(f"{group}/{target}: nodes={sizes}, history={tuple(batch.x_hist.shape)}, y={tuple(batch.y.shape)}", flush=True)
            del batch, view, store
    (DESTINATION / "graph_audit.json").write_text(json.dumps(rows, indent=2) + "\n")
    print("PASS: representative target batches have 5 forcing features, 5 history frames, and 6 hourly targets; no inference run.")


if __name__ == "__main__":
    main()
