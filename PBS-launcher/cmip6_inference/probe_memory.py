#!/usr/bin/env python3
"""Optional CPU-only one-winter RAM probe; does not execute a model or fit statistics."""
import argparse
import json
import os
from pathlib import Path
import resource
from common import DATA_ROOT, SOURCES, STATIONS, YEARS, graph_inventory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', choices=SOURCES, default='CMIP6_EC_EARTH')
    parser.add_argument('--station', choices=STATIONS, default='Battery')
    parser.add_argument('--year', choices=YEARS, default=YEARS[0])
    parser.add_argument('--data-root', type=Path, default=DATA_ROOT)
    parser.add_argument('--pressure-mode', choices=('raw', 'anomaly', 'anomaly_plus_mean'), default='raw')
    args = parser.parse_args()
    from emulator.data import ForcingGraphStore
    root = args.data_root / args.source / 'graphs'
    inventory = graph_inventory(root, args.station)
    before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2
    store = ForcingGraphStore(root, args.station, year_filter=[args.year], pressure_mode=args.pressure_mode)
    print(json.dumps(dict(source=args.source, station=args.station, year=args.year, pressure_mode=args.pressure_mode,
                         graphs=len(store.graphs), serialized_gib=inventory[YEARS.index(args.year)]['size'] / 2**30,
                         base_peak_gib=before, peak_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2,
                         retained_rss_gib=int(Path('/proc/self/statm').read_text().split()[1]) * os.sysconf('SC_PAGE_SIZE') / 2**30,
                         shape=list(store.graphs[0].x_hist.shape)), indent=2))


if __name__ == '__main__':
    main()
