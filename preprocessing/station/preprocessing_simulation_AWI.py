"""Extract CMIP6_AWI ADCIRC station elevations on shared simulation/tide times.

The command-line entry point reads fort.14 geometry and annual fort.63 NetCDF
files, then exports station CSVs without changing elevation units or datum.
"""

import os
from dataclasses import dataclass
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
from netCDF4 import Dataset, num2date
from tqdm import tqdm


@dataclass
class ADCIRCMesh:
    """ADCIRC mesh geometry and connectivity in source-file order.

    Attributes:
        title (str): First line of the mesh file.
        ne (int): Number of mesh elements.
        np (int): Number of mesh nodes.
        node_ids (numpy.ndarray): Integer source node identifiers, shape
            ``(np,)``; these are not zero-based array indices.
        x (numpy.ndarray): Node x coordinates, shape ``(np,)``. Station
            matching assumes longitude in degrees east.
        y (numpy.ndarray): Node y coordinates, shape ``(np,)``. Station
            matching assumes latitude in degrees north.
        depth (numpy.ndarray): Node depths, shape ``(np,)``, retaining the
            source file's units and sign convention without conversion.
        elements (list[tuple[int, int, list[int]]]): Element identifier,
            vertex count, and connectivity expressed as source node IDs.
    """
    title: str
    ne: int           # number of elements
    np: int           # number of nodes
    node_ids: np.ndarray   # (NP,)
    x: np.ndarray          # (NP,)
    y: np.ndarray          # (NP,)
    depth: np.ndarray      # (NP,)
    elements: List[Tuple[int, int, List[int]]]  # (elem_id, nvert, [node ids])


def read_fort14(path: str) -> ADCIRCMesh:
    """Read node geometry and element connectivity from an ADCIRC mesh.

    Args:
        path (str): Path to a text fort.14 file containing a title, element
            and node counts, node records, and element records.

    Returns:
        ADCIRCMesh: Mesh with node arrays in file order and unmodified
        coordinate/depth units. Boundary records after the elements are
        ignored; connectivity retains source node identifiers.

    Raises:
        OSError: The input file cannot be opened or read.
        ValueError: A count or numeric field is invalid, a required record
            is malformed, or an element has insufficient connectivity.
    """
    with open(path, "r") as f:
        title = f.readline().strip()

        ne_np_line = f.readline().split()
        if len(ne_np_line) < 2:
            raise ValueError("Second line should contain NE and NP.")
        ne, np_ = map(int, ne_np_line[:2])

        node_ids = np.empty(np_, dtype=int)
        x = np.empty(np_, dtype=float)
        y = np.empty(np_, dtype=float)
        depth = np.empty(np_, dtype=float)

        for i in range(np_):
            parts = f.readline().split()
            if len(parts) < 4:
                raise ValueError(f"Node line {i+1} is malformed: {parts}")
            node_ids[i] = int(parts[0])
            x[i] = float(parts[1])
            y[i] = float(parts[2])
            depth[i] = float(parts[3])

        elements: List[Tuple[int, int, List[int]]] = []
        for i in range(ne):
            line = f.readline()
            if not line:
                raise ValueError(f"File ended early while reading element {i+1}/{ne}")
            parts = line.split()
            if len(parts) < 5:
                raise ValueError(f"Element line {i+1} is malformed: {parts}")

            elem_id = int(parts[0])
            nvert = int(parts[1])
            conn = [int(v) for v in parts[2:2 + nvert]]

            if len(conn) != nvert:
                raise ValueError(
                    f"Element {elem_id} expected {nvert} vertices, got {len(conn)}."
                )

            elements.append((elem_id, nvert, conn))

    return ADCIRCMesh(
        title=title,
        ne=ne,
        np=np_,
        node_ids=node_ids,
        x=x,
        y=y,
        depth=depth,
        elements=elements,
    )


def find_nearest_node(mesh: ADCIRCMesh, lon: float, lat: float) -> Tuple[int, float]:
    """Find the nearest node using a local angular-distance approximation.

    Args:
        mesh (ADCIRCMesh): Nonempty mesh whose x/y coordinates are
            longitude/latitude in degrees, using the station's longitude
            convention.
        lon (float): Station longitude in degrees east.
        lat (float): Station latitude in degrees north.

    Returns:
        tuple[int, float]: Zero-based position in the mesh arrays and
        approximate angular separation in degrees. The distance is
        ``sqrt(dlat**2 + (dlon * cos(deg2rad(lat)))**2)``; longitude differences
        are not wrapped at the dateline. Equal minima select the first node.
    """
    lat0_rad = np.deg2rad(lat)
    cos_lat0 = np.cos(lat0_rad)

    dlat = mesh.y - lat
    dlon = (mesh.x - lon) * cos_lat0
    dist2 = dlat**2 + dlon**2

    idx = int(np.argmin(dist2))
    return idx, float(np.sqrt(dist2[idx]))


stations: Dict[str, Dict[str, float]] = {
    "Lewes_8557380": {
        "name": "Lewes, DE",
        "lat": 38.78278,
        "lon": -75.11916,
    },
    "Boston_8443970": {
        "name": "Boston, MA",
        "lat": 42.35392,
        "lon": -71.05033,
    },
    "CBBT_8638901": {
        "name": "CBBT, Chesapeake Channel, VA",
        "lat": 37.03290,
        "lon": -76.08330,
    },
    "Battery_8518750": {
        "name": "The Battery, NY",
        "lat": 40.70100,
        "lon": -74.01400,
    },
}

import argparse

def parse_args():
    """Parse station-extraction paths from the process command line.

    Returns:
        argparse.Namespace: Paths in ``mesh``, ``input_sim``, ``input_tide``,
        ``output_full``, and ``output_csv``. The full-output directory is
        created by ``main`` but full-field NPZ files are not exported.

    Raises:
        SystemExit: Help is requested or arguments are invalid.
    """
    parser = argparse.ArgumentParser(
        description="Preprocess ADCIRC fort.63.nc simulation and tidal data."
    )
    parser.add_argument(
        "--mesh",
        type=str,
        default="/glade/u/home/zel/scratch/Lehigh_Princeton_StormSurge_Data/CMIP6_AWI/fort.14/fort_1979.14",
        help="Path to the ADCIRC fort.14 mesh file."
    )
    parser.add_argument(
        "--input-sim",
        type=str,
        default="/glade/u/home/zel/scratch/Lehigh_Princeton_StormSurge_Data/CMIP6_AWI/fort.63",
        help="Input folder containing raw simulation fort.63.nc files."
    )
    parser.add_argument(
        "--input-tide",
        type=str,
        default="/glade/u/home/zel/scratch/Lehigh_Princeton_StormSurge_Data/Tidal/fort.63",
        help="Input folder containing raw tidal fort.63.nc files."
    )
    parser.add_argument(
        "--output-full",
        type=str,
        default="/glade/u/home/zel/scratch/Lehigh_Princeton_StormSurge_Data/Processed_Station/CMIP6_AWI/full_npz",
        help="Output folder for full time × node npz files."
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="/glade/u/home/zel/scratch/Lehigh_Princeton_StormSurge_Data/Processed_Station/CMIP6_AWI/station_CSVs",
        help="Output folder for per-station CSV files."
    )
    return parser.parse_args()


def main():

    """Export simulated and tidal elevations at the configured stations.

    Inputs are supplied through the process command line and the module's
    station coordinates. For each available season from 1979 through 2100,
    simulation and tide records are intersected by their decoded timestamps.
    Each station uses the nearest node in the supplied mesh; the same node
    ordering is assumed in both NetCDF files. Elevations retain their source
    units and datum without conversion.

    Returns:
        None: Creates the configured output directories and writes
        ``<year>_<year+1>_<station>.csv`` with ``time``, ``nc`` (simulation),
        and ``nc_tide`` (tide-only simulation) columns. Existing CSVs are
        overwritten. Missing years or years without overlapping timestamps
        are skipped with diagnostics; no full-field NPZ is written.

    Raises:
        ValueError: Mesh parsing fails, the simulation and tide node counts
            differ, or required tide timestamps are absent or invalid.
        KeyError: A required NetCDF variable is absent.
        OSError: Mesh/NetCDF inputs cannot be read or outputs cannot be written.
    """
    args = parse_args()
    mesh_path = args.mesh
    input_folder_sim = args.input_sim
    input_folder_tide = args.input_tide
    full_output_folder = args.output_full      # Directory created for the configured full-output path; no NPZ export.
    station_csv_folder = args.output_csv

    # Years to process (each fort_{year}.63.nc is assumed to cover year–year+1)
    year_list = list(range(1979, 2101))

    os.makedirs(full_output_folder, exist_ok=True)
    os.makedirs(station_csv_folder, exist_ok=True)

    print(f"Reading mesh from {mesh_path} ...")
    mesh = read_fort14(mesh_path)
    print(f"Mesh loaded: {mesh.np} nodes, {mesh.ne} elements")

    station_nodes: Dict[str, int] = {}
    print("\nNearest mesh node for each station:")
    for key, info in stations.items():
        node_idx, dist_deg = find_nearest_node(mesh, info["lon"], info["lat"])
        station_nodes[key] = node_idx
        print(
            f"  {info['name']} → node {node_idx} "
            f"(lon={mesh.x[node_idx]:.4f}, lat={mesh.y[node_idx]:.4f}), "
            f"approx distance ~ {dist_deg:.4f} deg"
        )

    for year in year_list:
        print(f"\n===== Processing simulation starting {year} (approx {year}_{year+1}) =====")

        nc_path = os.path.join(input_folder_sim, f"fort_{year}.63.nc")
        nc_tide_path = os.path.join(input_folder_tide, f"fort_{year}_Tidal.63.nc")

        if not os.path.exists(nc_path):
            print(f"  WARNING: {nc_path} not found, skipping this year.")
            continue
        if not os.path.exists(nc_tide_path):
            print(f"  WARNING: {nc_tide_path} not found, skipping this year.")
            continue

        with Dataset(nc_path) as nc, Dataset(nc_tide_path) as nc_tide:
            # Verify that mesh array positions address the simulation nodes.
            if "x" in nc.variables and "y" in nc.variables:
                x_nc = nc.variables["x"][:]
                y_nc = nc.variables["y"][:]
                if x_nc.shape[0] != mesh.np or y_nc.shape[0] != mesh.np:
                    print("  WARNING: Node count mismatch between fort.14 and fort.63!")
                else:
                    if not (np.allclose(x_nc, mesh.x) and np.allclose(y_nc, mesh.y)):
                        print("  WARNING: Node order/coords differ between fort.14 and fort.63.")
                    else:
                        print("  Node coordinates match between fort.14 and fort.63.")
            else:
                print("  No x/y variables in netCDF; cannot verify node ordering here.")

            zeta = nc.variables["zeta"]          # (Nt_main, Nnode)
            tide_zeta = nc_tide.variables["zeta"]  # (Nt_tide, Nnode)

            if zeta.shape[1] != tide_zeta.shape[1]:
                raise ValueError(
                    f"Node dimension mismatch between zeta and tide_zeta for year {year}: "
                    f"{zeta.shape} vs {tide_zeta.shape}"
                )

            # Decode each file with its own CF time units and calendar.
            time_var = nc.variables["time"]
            units = time_var.units
            calendar = getattr(time_var, "calendar", "standard")

            time_cf = num2date(time_var[:], units=units, calendar=calendar)
            time_dt = pd.to_datetime([t.isoformat() for t in time_cf])
            print(f"  nc time:   {time_dt[0]} → {time_dt[-1]}  (nt={len(time_dt)})")

            if "time" not in nc_tide.variables:
                raise ValueError(f"  nc_tide for year {year} has no 'time' variable.")

            time_tide_var = nc_tide.variables["time"]
            units_tide = time_tide_var.units
            calendar_tide = getattr(time_tide_var, "calendar", "standard")

            time_tide_cf = num2date(time_tide_var[:], units=units_tide, calendar=calendar_tide)
            time_tide_dt = pd.to_datetime([t.isoformat() for t in time_tide_cf])
            print(f"  tide time: {time_tide_dt[0]} → {time_tide_dt[-1]}  (nt={len(time_tide_dt)})")

            # Intersect decoded timestamps before extracting matched station elevations.
            idx_main = pd.Index(time_dt)
            idx_tide = pd.Index(time_tide_dt)

            common_times = idx_main.intersection(idx_tide)

            if common_times.empty:
                print("  ERROR: No overlapping times between nc and nc_tide, skipping this year.")
                continue

            main_indices = idx_main.get_indexer(common_times)
            tide_indices = idx_tide.get_indexer(common_times)

            if np.any(main_indices < 0) or np.any(tide_indices < 0):
                print("  ERROR: Failed to align time indices, skipping this year.")
                continue

            zeta_aligned = zeta[main_indices, :]
            tide_zeta_aligned = tide_zeta[tide_indices, :]
            time_aligned = pd.to_datetime(common_times)

            print(
                f"  Aligned time window: {time_aligned[0]} → {time_aligned[-1]} "
                f"(nt={len(time_aligned)})"
            )

            for key, info in stations.items():
                node_idx = station_nodes[key]

                eta = zeta_aligned[:, node_idx]          # (Nt_aligned,)
                tide_eta = tide_zeta_aligned[:, node_idx]

                df_station = pd.DataFrame({
                    "time": time_aligned,
                    "nc": eta,
                    "nc_tide": tide_eta,
                })

                # Output names use the station name without its numeric identifier.
                short_name = key.split("_")[0]
                csv_name = f"{year}_{year+1}_{short_name}.csv"
                csv_path = os.path.join(station_csv_folder, csv_name)
                df_station.to_csv(csv_path, index=False)

                print(
                    f"  {info['name']} – node {node_idx}: "
                    f"min nc={eta.min():.3f}, max nc={eta.max():.3f}, "
                    f"min nc_tide={tide_eta.min():.3f}, max nc_tide={tide_eta.max():.3f}"
                )
                print(f"    Saved station CSV: {csv_path}")
            del zeta_aligned, tide_zeta_aligned


if __name__ == "__main__":
    main()
