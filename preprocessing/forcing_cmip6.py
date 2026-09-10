#!/usr/bin/env python3
"""Prepare AWI/CNRM/EC_EARTH/MPI/MRI forcing with one shared transformation."""

import argparse
from pathlib import Path

import numpy as np


def prepare_forcing(forcing, latitude, longitude):
    """Preserve fort.22 north-to-south ordering and the [u,v,p',lon,lat] layout."""
    forcing = np.asarray(forcing, dtype=np.float64)
    lat_mask = (latitude >= 0) & (latitude <= 55)
    lon_mask = (longitude >= -100) & (longitude <= -40)
    if not lat_mask.any() or not lon_mask.any():
        raise ValueError("The forcing grid does not cover the selected local region.")
    cells = latitude.size * longitude.size
    if forcing.ndim != 2 or forcing.shape[1] != 3 or len(forcing) % cells:
        raise ValueError("fort.22 must contain complete timesteps with three columns.")
    values = forcing.reshape(-1, latitude.size, longitude.size, 3)[:, lat_mask][:, :, lon_mask]
    lon, lat = np.meshgrid(longitude[lon_mask], latitude[lat_mask])
    coordinates = np.broadcast_to(np.stack((lon, lat), axis=-1), (*values.shape[:-1], 2))
    result = np.concatenate((values, coordinates), axis=-1)
    pressure_mean = result[..., 2].mean(axis=(1, 2))
    result[..., 2] -= pressure_mean[:, None, None]
    return result, pressure_mean, latitude[lat_mask], longitude[lon_mask]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("AWI", "CNRM", "EC_EARTH", "MPI", "MRI"), required=True)
    parser.add_argument("--forcing_dir", type=Path, required=True)
    parser.add_argument("--grid_netcdf", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--first_year", type=int, default=1979)
    parser.add_argument("--last_year", type=int, default=2099)
    args = parser.parse_args(argv)
    import xarray as xr
    with xr.open_dataset(args.grid_netcdf, decode_times=False) as grid:
        axes = []
        for candidates in (("lat", "latitude", "nav_lat", "y"), ("lon", "longitude", "nav_lon", "x")):
            name = next((key for key in candidates if key in grid.coords), None)
            if name is None:
                name = next((key for key in candidates if key in grid.dims), None)
            if name is None or grid[name].ndim != 1:
                raise ValueError("A regular grid with 1D latitude and longitude is required.")
            axes.append(grid[name].values)
    latitude, longitude = axes
    if np.all(np.diff(latitude) > 0):
        latitude = latitude[::-1]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    lat_local = latitude[(latitude >= 0) & (latitude <= 55)]
    lon_local = longitude[(longitude >= -100) & (longitude <= -40)]
    metadata = dict(LAT_MIN=0., LAT_MAX=55., LON_MIN=-100., LON_MAX=-40.,
                    lat_local=lat_local, lon_local=lon_local, lat_forcing=latitude, lon_forcing=longitude)
    np.savez(args.out_dir / f"grid_local_cmip6_{args.model.lower()}.npz", **metadata)
    for year in range(args.first_year, args.last_year + 1):
        path = args.forcing_dir / f"fort_{year}.22"
        if not path.exists():
            print(f"WARNING: {path} not found, skipping.")
            continue
        forcing = np.loadtxt(path)
        if forcing.ndim != 2 or forcing.shape[1] != 3 or len(forcing) % (latitude.size * longitude.size):
            print(f"WARNING: incomplete or invalid forcing shape {forcing.shape} in {path}, skipping.")
            continue
        values, pressure, _, _ = prepare_forcing(forcing, latitude, longitude)
        # 3h forcing is retained for analysis; graph construction consumes 6h NPZ.
        for step, stride, suffix in ((3, 1, "_3h"), (6, 2, "")):
            np.save(args.out_dir / f"forcing_local{suffix}_{year}.npy", values[::stride])
            np.savez(args.out_dir / f"forcing_local{suffix}_{year}.npz", forcing=values[::stride],
                     p_mean_t=pressure[::stride], year=year, dt_hours=step, **metadata)


if __name__ == "__main__":
    main()
