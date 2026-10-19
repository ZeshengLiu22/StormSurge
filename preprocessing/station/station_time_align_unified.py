#!/usr/bin/env python3
"""Align six-hourly meteorological forcing with hourly station surge labels.

Input NPZ arrays contain [u, v, pressure anomaly, longitude, latitude] features
on a model-specific grid and optional local-grid mean absolute pressure.
Forcing timestamps are reconstructed from October 25 00:00; station CSV rows
are assumed to be contiguous hours beginning October 25 01:00. Neither stored
calendar metadata nor the CSV time column determines alignment.

Each November-to-spring sample has nine forcing slices spanning 48 hours and
six hourly labels beginning at its center. Labels are simulation elevation
minus tide-only elevation in their original units and datum. Graph nodes use
row-major grid positions. Units and coordinate conventions must be consistent
across input files; no conversion is applied here.

The command-line entry point exports the peryear cutoff, defined by the last
complete day shared across selected stations and capped at March 31. The
``process_one_pair`` function also supports a fixed March 15 cutoff. Outputs
are model-specific NPZ dictionaries and lists of PyG Data objects in PT files.

Example:
    python station_time_align_unified.py --model_tag NCEP --years 1979:2014
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data


NUM_FEATURES = 5           # [u, v, pressure anomaly, longitude, latitude]
GT_PER_FORCING = 6         # 6 hourly station values per 6-hour forcing step
MAX_HISTORY_HOURS = 48     # Eight preceding six-hour forcing steps.
assert MAX_HISTORY_HOURS % 6 == 0
MAX_HISTORY_STEPS = MAX_HISTORY_HOURS // 6    # 48h -> 8 forcing steps
WINDOW_LENGTH = MAX_HISTORY_STEPS + 1         # History includes the current forcing slice.


MODEL_REGISTRY = {
    "NCEP": {
        "forcing_subdir": "Forcing_Data/Processed_Forcing_NCEP",
        "csv_subdir":     "Surge_Data/NCEP_Reanalysis_fort63_station_CSVs",
    },
    "CMIP6_AWI": {
        "forcing_subdir": "Forcing_Data/Processed_Forcing_CMIP6_AWI",
        "csv_subdir":     "Surge_Data/CMIP6_AWI_fort63_station_CSVs",
    },
    "CMIP6_CNRM": {
        "forcing_subdir": "Forcing_Data/Processed_Forcing_CMIP6_CNRM",
        "csv_subdir":     "Surge_Data/CMIP6_CNRM_fort63_station_CSVs",
    },
    "CMIP6_EC_EARTH": {
        "forcing_subdir": "Forcing_Data/Processed_Forcing_CMIP6_EC_EARTH",
        "csv_subdir":     "Surge_Data/CMIP6_EC_EARTH_fort63_station_CSVs",
    },
    "CMIP6_MPI": {
        "forcing_subdir": "Forcing_Data/Processed_Forcing_CMIP6_MPI",
        "csv_subdir":     "Surge_Data/CMIP6_MPI_fort63_station_CSVs",
    },
    "CMIP6_MRI": {
        "forcing_subdir": "Forcing_Data/Processed_Forcing_CMIP6_MRI",
        "csv_subdir":     "Surge_Data/CMIP6_MRI_fort63_station_CSVs",
    },
    "CMIP6_Cane5": {
        "forcing_subdir": "Forcing_Data/Processed_Forcing_CMIP6_Cane5",
        "csv_subdir":     "Surge_Data/CMIP6_Cane5_fort63_station_CSVs",
    }
}


@dataclass
class AlignConfig:
    """Resolved paths and selection settings for seasonal graph export.

    Attributes:
        data_root (Path): Root for resolving relative input/output paths.
        model_tag (str): Model identifier used in output subdirectory names.
        forcing_dir (Path): Directory searched recursively for forcing NPZs.
        csv_dir (Path): Directory containing annual station CSVs.
        out_root_fixed (Path): Model-specific root for fixed March 15 exports.
        out_root_peryear (Path): Model-specific root for coverage-based exports.
        stations (list[str]): Station names used in CSV/output filenames.
        years (list[int]): Selected season start years after filtering.
        exclude_years (list[int]): Season start years excluded from processing.
        edge_mode (str): Graph connectivity: ``fc``, ``grid4``, or ``grid8``.
        max_edges (int): Maximum permitted fully connected directed edges.
        debug_limit_years (int or None): Requested first-K year limit, if set.
        debug_limit_stations (int or None): Requested first-K station limit.
    """
    data_root: Path
    model_tag: str
    forcing_dir: Path
    csv_dir: Path
    out_root_fixed: Path
    out_root_peryear: Path
    stations: List[str]
    years: List[int]
    exclude_years: List[int]
    edge_mode: str
    max_edges: int
    debug_limit_years: Optional[int] = None
    debug_limit_stations: Optional[int] = None


def build_forcing_time_index(year: int, nsteps: int) -> np.ndarray:
    """Construct the assumed six-hourly forcing timeline for a season.

    Args:
        year (int): Season start year; the first timestamp is October 25 00:00.
        nsteps (int): Number of forcing records, expected to be nonnegative.

    Returns:
        numpy.ndarray: Shape ``(nsteps,)`` array of timezone-naive Python
        datetimes for nonempty input. No timestamps are read from a file;
        the Gregorian timeline is constructed from the assumed origin.
    """
    t0 = datetime(year, 10, 25, 0, 0)
    return np.array([t0 + timedelta(hours=6 * i) for i in range(nsteps)])


def build_station_time_index(year: int, nsteps: int) -> np.ndarray:
    """Construct the assumed hourly station timeline from CSV row positions.

    Args:
        year (int): Season start year; row zero represents October 25 01:00.
        nsteps (int): Number of station rows, expected to be nonnegative.

    Returns:
        numpy.ndarray: Shape ``(nsteps,)`` array of timezone-naive Python
        datetimes for nonempty input. CSV timestamp columns are not used;
        each successive row is assumed to represent the next Gregorian hour.
    """
    t0 = datetime(year, 10, 25, 1, 0)
    return np.array([t0 + timedelta(hours=i) for i in range(nsteps)])


def discover_forcing_files(forcing_dir: Path) -> Dict[int, Path]:
    """Index six-hourly forcing NPZ files by season start year.

    Args:
        forcing_dir (Path): Root directory to search recursively for names
            matching ``forcing_local_YYYY.npz``.

    Returns:
        dict[int, Path]: Year-sorted mapping of season start year to path.
        Three-hourly filenames are excluded. If multiple paths match a year,
        the last path in lexicographic order takes precedence. An unmatched
        or nonexistent directory produces an empty mapping.
    """
    candidates = sorted(forcing_dir.glob("**/forcing_local_*.npz"))
    year_to_path: Dict[int, Path] = {}

    for p in candidates:
        m = re.search(r"forcing_local_(\d{4})\.npz$", p.name)
        if m:
            year_to_path[int(m.group(1))] = p

    return dict(sorted(year_to_path.items(), key=lambda kv: kv[0]))


def load_forcing_npz(path: Path) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Read forcing and optional local pressure means from an NPZ archive.

    Args:
        path (Path): Archive containing ``forcing`` with expected shape
            ``(T, H, W, 5)`` and optionally ``p_mean_t`` with shape ``(T,)``.
            Channels are [u, v, pressure anomaly, longitude, latitude].
            Pressure means are the local-grid absolute-pressure means
            removed during preprocessing, in the input pressure units.

    Returns:
        tuple[numpy.ndarray, numpy.ndarray or None]: Float32 forcing array
        and float32 pressure-mean series, or ``None`` if absent. This loader
        converts dtype without validating shapes or converting units.

    Raises:
        KeyError: The archive lacks the required ``forcing`` key.
        OSError: The archive cannot be opened or read.
        ValueError: Archive contents cannot be loaded without pickle or
            converted to float32.
    """
    z = np.load(path, allow_pickle=False)
    if "forcing" not in z:
        raise KeyError(f"{path} missing key 'forcing'. Keys={list(z.keys())}")

    forcing = z["forcing"].astype(np.float32)
    p_mean_t = z["p_mean_t"].astype(np.float32) if "p_mean_t" in z else None
    return forcing, p_mean_t


def infer_grid_from_forcing(arr: np.ndarray) -> Tuple[int, int, int, int]:
    """Validate forcing rank/channel count and return its dimensions.

    Args:
        arr (numpy.ndarray): Forcing in ``(time, rows, columns, features)``
            order with five feature channels.

    Returns:
        tuple[int, int, int, int]: ``(T, H, W, C)`` dimensions; ``C`` is five.
        Time/grid dimensions are returned without checking positivity.

    Raises:
        ValueError: Input is not four-dimensional or has a different channel
            count from ``NUM_FEATURES``.
    """
    if arr.ndim != 4:
        raise ValueError(f"Forcing must be 4D (T,H,W,C). Got shape={arr.shape}")
    T, H, W, C = arr.shape
    if C != NUM_FEATURES:
        raise ValueError(f"Expected forcing channels C={NUM_FEATURES}, got C={C}, shape={arr.shape}")
    return T, H, W, C


def build_fully_connected_edge_index(num_nodes: int, include_self: bool = False) -> torch.Tensor:
    """Construct directed edges between every pair of graph nodes.

    Args:
        num_nodes (int): Nonnegative node count; node IDs are ``0..N-1``.
        include_self (bool): Whether to include one self-loop per node.

    Returns:
        torch.Tensor: CPU int64 source/destination indices of shape ``(2, E)``.
        ``E = N * (N - 1)`` without self-loops and ``E = N**2`` with them;
        reverse edges are included. Storage grows quadratically in N.
    """
    nodes = np.arange(num_nodes, dtype=np.int64)
    row = np.repeat(nodes, num_nodes)
    col = np.tile(nodes, num_nodes)

    if not include_self:
        mask = row != col
        row = row[mask]
        col = col[mask]

    edge_index = np.stack([row, col], axis=0)
    return torch.from_numpy(edge_index).long()


def build_grid4_edge_index(H: int, W: int) -> torch.Tensor:
    """Connect each grid cell to its horizontal and vertical neighbors.

    Args:
        H (int): Positive number of grid rows.
        W (int): Positive number of grid columns.

    Returns:
        torch.Tensor: CPU int64 directed edge indices, shape ``(2, E)`` when
        any edges exist. Nodes use row-major IDs ``r * W + c``; boundaries
        do not wrap and self-loops are omitted. A grid with no neighbors
        produces an empty tensor of shape ``(0,)``.
    """
    def nid(r: int, c: int) -> int:
        """Convert a grid coordinate to the enclosing grid's node index.

        Args:
            r (int): Zero-based row position within the grid.
            c (int): Zero-based column position within the grid.

        Returns:
            int: Row-major index ``r * W + c``, using the enclosing width W.
        """
        return r * W + c

    edges: List[Tuple[int, int]] = []
    for r in range(H):
        for c in range(W):
            u = nid(r, c)
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                rr, cc = r + dr, c + dc
                if 0 <= rr < H and 0 <= cc < W:
                    v = nid(rr, cc)
                    edges.append((u, v))

    return torch.tensor(edges, dtype=torch.long).t().contiguous()


def build_grid8_edge_index(H: int, W: int) -> torch.Tensor:
    """Connect each grid cell to adjacent axial and diagonal neighbors.

    Args:
        H (int): Positive number of grid rows.
        W (int): Positive number of grid columns.

    Returns:
        torch.Tensor: CPU int64 directed edge indices, shape ``(2, E)`` when
        any edges exist. Nodes use row-major IDs ``r * W + c``; boundaries
        do not wrap and self-loops are omitted. A grid with no neighbors
        produces an empty tensor of shape ``(0,)``.
    """
    def nid(r: int, c: int) -> int:
        """Convert a grid coordinate to the enclosing grid's node index.

        Args:
            r (int): Zero-based row position within the grid.
            c (int): Zero-based column position within the grid.

        Returns:
            int: Row-major index ``r * W + c``, using the enclosing width W.
        """
        return r * W + c

    edges: List[Tuple[int, int]] = []
    nbrs = [
        (-1, 0), (1, 0), (0, -1), (0, 1),      # grid4
        (-1, -1), (-1, 1), (1, -1), (1, 1),    # diagonals
    ]

    for r in range(H):
        for c in range(W):
            u = nid(r, c)
            for dr, dc in nbrs:
                rr, cc = r + dr, c + dc
                if 0 <= rr < H and 0 <= cc < W:
                    v = nid(rr, cc)
                    edges.append((u, v))

    return torch.tensor(edges, dtype=torch.long).t().contiguous()


def make_edge_index(edge_mode: str, H: int, W: int, max_edges: int) -> torch.Tensor:
    """Select a graph topology for the forcing grid.

    Args:
        edge_mode (str): ``fc`` for all directed non-self edges, ``grid4``
            for axial neighbors, or ``grid8`` to also include diagonals.
        H (int): Positive number of grid rows.
        W (int): Positive number of grid columns.
        max_edges (int): Upper edge-count bound applied only to ``fc``.

    Returns:
        torch.Tensor: CPU int64 edge indices for ``H * W`` row-major nodes.
        Shape is ``(2, E)`` except that a neighbor grid with no edges returns
        shape ``(0,)``. No topology includes self-loops.

    Raises:
        ValueError: The mode is unsupported or a fully connected graph would
            exceed ``max_edges``.
    """
    num_nodes = H * W

    if edge_mode == "fc":
        num_edges = num_nodes * (num_nodes - 1)
        if num_edges > max_edges:
            raise ValueError(
                f"Fully-connected graph too large: nodes={num_nodes}, edges={num_edges:,} > max_edges={max_edges:,}\n"
                f"Use --edge_mode grid4/grid8 or increase --max_edges."
            )
        return build_fully_connected_edge_index(num_nodes, include_self=False)

    if edge_mode == "grid4":
        return build_grid4_edge_index(H, W)

    if edge_mode == "grid8":
        return build_grid8_edge_index(H, W)

    raise ValueError(f"Unknown edge_mode={edge_mode}. Use 'fc', 'grid4', or 'grid8'.")


def compute_last_full_day_for_year(year: int, stations: List[str], csv_dir: Path) -> datetime.date:
    """Find the last complete station day shared across a season.

    Args:
        year (int): Season start year; the upper date cap is March 31 of
            ``year + 1`` and the earliest accepted date is November 1.
        stations (list[str]): Nonempty list of station names used in filenames.
        csv_dir (Path): Directory containing
            ``<year>_<year+1>_<station>.csv`` for every selected station.
            Each CSV must contain at least one hourly record.

    Returns:
        datetime.date: Earliest last-full-day date across stations, capped
        at March 31. Row counts imply uninterrupted hourly coverage from
        October 25 01:00; a final day is complete only if hour 23 is present.

    Raises:
        FileNotFoundError: A station CSV is missing.
        ValueError: Usable coverage ends before November 1 or no candidate
            station dates are available.
        IndexError: A header-only CSV has no rows from which to infer coverage.

    Notes:
        Reads each CSV and prints coverage diagnostics. Timestamps and
        missing values in CSV columns are not inspected; valid inputs must
        contain contiguous hourly records in chronological order.
    """
    season_cap = datetime(year + 1, 3, 31, 23, 0)
    season_start_date = datetime(year, 11, 1, 0, 0).date()

    candidate_days: List[datetime.date] = []

    for station in stations:
        csv_path = csv_dir / f"{year}_{year+1}_{station}.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing CSV for year={year}, station={station}: {csv_path}")

        df = pd.read_csv(csv_path)
        t_station = build_station_time_index(year, len(df))

        valid_mask = np.array([t <= season_cap for t in t_station], dtype=bool)
        if not valid_mask.any():
            raise ValueError(
                f"No timestamps <= {season_cap} in {csv_path}. "
                f"First={t_station[0]}, last={t_station[-1]}"
            )

        last_t = t_station[np.where(valid_mask)[0][-1]]

        if last_t.hour < 23:
            candidate = (last_t - timedelta(days=1)).date()
        else:
            candidate = last_t.date()

        if candidate < season_start_date:
            raise ValueError(
                f"{csv_path} ends before season start. candidate={candidate}, season_start={season_start_date}"
            )

        if last_t.month < 3:
            print(f"[peryear cutoff] Year {year}, {station}: ends early at {last_t} -> last full day {candidate}")
        else:
            print(f"[peryear cutoff] Year {year}, {station}: last in-cap time {last_t} -> last full day {candidate}")

        candidate_days.append(candidate)

    last_day = min(candidate_days)
    print(f"[peryear cutoff] Year {year}: common last full day (capped by Mar 31) = {last_day}")
    return last_day


def process_one_pair(
    *,
    year: int,
    station: str,
    forcing: np.ndarray,             # (T, H, W, 5)
    p_mean_t: Optional[np.ndarray],  # (T,) or None
    t_forcing: np.ndarray,           # (T,)
    edge_index: torch.Tensor,
    H: int,
    W: int,
    version: str,                    # "fixed" or "peryear"
    out_root_fixed: Path,
    out_root_peryear: Path,
    csv_dir: Path,
    last_full_day: Optional[datetime.date] = None,
) -> None:
    """Write aligned forcing arrays and graphs for one station-season pair.

    Args:
        year (int): Season start year; sample centers start November 1 00:00.
        station (str): Station name used in CSV and output filenames.
        forcing (numpy.ndarray): Six-hourly ``(T, H, W, 5)`` array with
            [u, v, pressure anomaly, longitude, latitude] channels. Physical
            units are inherited from preprocessing; coordinates are degrees.
        p_mean_t (numpy.ndarray or None): Optional ``(T,)`` local-grid mean
            absolute pressure in source units, aligned with forcing records.
        t_forcing (numpy.ndarray): ``(T,)`` timezone-naive datetimes for a
            continuous six-hourly forcing series.
        edge_index (torch.Tensor): Int64 graph connectivity of shape ``(2, E)``
            for ``H * W`` nodes in row-major grid order.
        H (int): Number of rows in the forcing grid.
        W (int): Number of columns in the forcing grid.
        version (str): ``fixed`` ends on March 15 of ``year + 1``;
            ``peryear`` uses ``last_full_day``. Final centers are at 18:00.
        out_root_fixed (Path): Model-specific output root for ``fixed``.
            Its ``dicts`` and ``graphs`` subdirectories must already exist.
        out_root_peryear (Path): Corresponding output root for ``peryear``.
        csv_dir (Path): Directory containing the station CSV with ``nc``
            (simulation) and ``nc_tide`` (tide-only) elevation columns in the
            same units and datum. Rows imply hourly coverage from October 25
            01:00; any CSV time column is ignored.
        last_full_day (datetime.date or None): Inclusive final label day,
            required for ``peryear`` and ignored for ``fixed``.

    Returns:
        None: Writes ``<year>_<year+1>_<station>_<suffix>_dict.npz`` and
        ``..._graphs.pt`` under the selected root, replacing existing files.
        Suffix is ``fixed315_hist48`` or ``peryear_hist48``. The NPZ stores
        forcing centers, aligned station values, and alignment metadata.
        The PT file stores a list of PyG ``Data`` samples with ``x`` shape
        ``(H*W, 5)``, ``x_hist`` shape ``(9, H*W, 5)``, and ``y``, ``nc``,
        and ``nc_tide`` shape ``(6,)``, all float32. ``y`` is ``nc - nc_tide``.
        Optional ``p_mean_hist`` has shape ``(9,)`` and ``p_mean_curr`` is scalar.

    Raises:
        ValueError: The version, pressure length, available history, sample
            coverage, or forcing/station time alignment is invalid.
        KeyError: The station CSV lacks ``nc`` or ``nc_tide``.
        RuntimeError: Cropped pressure or forcing histories have inconsistent
            lengths.
        OSError: A required input cannot be read or an output cannot be written.

    Notes:
        Each history includes the center and eight preceding six-hour steps
        (48 hours); the earliest context is October 30 00:00. Labels cover
        center time through center + 5 hours. Sample count is limited to
        available centers and complete six-hour station blocks, so actual
        coverage can end earlier than the requested metadata endpoint.
        The NPZ is written before graph validation completes.
    """
    num_nodes = H * W

    # Include 48 hours before the first November 1 sample center.
    t_start_center = datetime(year, 11, 1, 0, 0)
    context_start = t_start_center - timedelta(hours=6 * MAX_HISTORY_STEPS)  # => year-10-30 00:00

    # Each final center requires hourly labels through 23:00 of its day.
    if version == "fixed":
        t_end_center  = datetime(year + 1, 3, 15, 18, 0)
        t_end_station = datetime(year + 1, 3, 15, 23, 0)
        out_root = out_root_fixed
        suffix = "fixed315_hist48"

    elif version == "peryear":
        if last_full_day is None:
            raise ValueError("last_full_day must be provided for peryear version.")
        t_end_center  = datetime(last_full_day.year, last_full_day.month, last_full_day.day, 18, 0)
        t_end_station = datetime(last_full_day.year, last_full_day.month, last_full_day.day, 23, 0)
        out_root = out_root_peryear
        suffix = "peryear_hist48"

    else:
        raise ValueError(f"Unknown version={version}")

    # Retain the context interval and all selected sample centers.
    mask_force = (t_forcing >= context_start) & (t_forcing <= t_end_center)
    forcing_ctx = forcing[mask_force]
    t_forcing_ctx = t_forcing[mask_force]

    if forcing_ctx.shape[0] == 0:
        raise ValueError(
            f"No forcing in range [{context_start}, {t_end_center}] for year={year}, station={station}, version={version}"
        )

    # Pressure means must remain aligned with the cropped forcing records.
    p_mean_ctx: Optional[np.ndarray] = None
    if p_mean_t is not None:
        if p_mean_t.shape[0] != forcing.shape[0]:
            raise ValueError(
                f"p_mean_t length mismatch for year={year}: len(p_mean_t)={p_mean_t.shape[0]} vs forcing T={forcing.shape[0]}"
            )
        p_mean_ctx = p_mean_t[mask_force]
        if p_mean_ctx.shape[0] != forcing_ctx.shape[0]:
            raise RuntimeError("Internal error: p_mean_ctx and forcing_ctx time lengths differ.")

    # Center indices (within forcing_ctx) from Nov 1 00:00 to t_end_center
    center_mask = (t_forcing_ctx >= t_start_center) & (t_forcing_ctx <= t_end_center)
    center_indices_full = np.where(center_mask)[0]
    if len(center_indices_full) == 0:
        raise ValueError(f"No forcing centers found in forcing_ctx for year={year}, station={station}, version={version}")

    # Every center requires eight earlier forcing records.
    if center_indices_full[0] < MAX_HISTORY_STEPS:
        raise ValueError(
            f"Not enough forcing history after crop for year={year}, station={station}, version={version}.\n"
            f"earliest center index in forcing_ctx={center_indices_full[0]}, need >= {MAX_HISTORY_STEPS}."
        )

    # CSV row positions define hourly timestamps; the time column is ignored.
    csv_path = csv_dir / f"{year}_{year+1}_{station}.csv"
    df = pd.read_csv(csv_path)

    if not {"nc", "nc_tide"}.issubset(df.columns):
        raise KeyError(f"CSV {csv_path} must contain columns 'nc' and 'nc_tide'. Found: {list(df.columns)}")

    t_station_full = build_station_time_index(year, len(df))
    t_start_station = datetime(year, 11, 1, 0, 0)

    mask_station = (t_station_full >= t_start_station) & (t_station_full <= t_end_station)
    idx = np.where(mask_station)[0]
    if len(idx) == 0:
        raise ValueError(
            f"No station data in [{t_start_station}, {t_end_station}] for {csv_path}. "
            f"Station last time = {t_station_full[-1]}"
        )

    first_included_time = t_station_full[idx[0]]
    if first_included_time != t_start_station:
        raise ValueError(
            f"Station does not include exact Nov 1 00:00 start for {csv_path}.\n"
            f"First included={first_included_time}, expected={t_start_station}."
        )

    df_aligned = df.iloc[idx].copy().reset_index(drop=True)
    nc = df_aligned["nc"].to_numpy()
    nc_tide = df_aligned["nc_tide"].to_numpy()

    # Truncate to complete six-hour label blocks and available forcing centers.
    N_raw = len(center_indices_full)
    max_blocks_station = len(nc) // GT_PER_FORCING
    N = min(N_raw, max_blocks_station)

    if N == 0:
        raise ValueError(
            f"N=0 for year={year}, station={station}, version={version} "
            f"(N_raw={N_raw}, max_blocks_station={max_blocks_station})."
        )

    center_indices = center_indices_full[:N]
    forcing_centers = forcing_ctx[center_indices]  # (N, H, W, 5)

    nc_used = nc[: N * GT_PER_FORCING]
    nc_tide_used = nc_tide[: N * GT_PER_FORCING]

    p_mean_status = "present" if p_mean_ctx is not None else "missing"
    print(
        f"{version.upper():7s} {year} {station:8s} | "
        f"N={N:4d} | forcing_centers={tuple(forcing_centers.shape)} | "
        f"nodes={num_nodes:5d} | edges={edge_index.shape[1]:,} | station_hours={len(nc_used)} | "
        f"p_mean_t={p_mean_status} | end_center={t_end_center}"
    )

    # Store center forcing and hourly labels separately from history graphs.
    dict_obj = {
        "Forcing": forcing_centers,
        "nc": nc_used,
        "nc_tide": nc_tide_used,
        "meta_max_history_hours": MAX_HISTORY_HOURS,
        "meta_grid_H": H,
        "meta_grid_W": W,
        "meta_num_features": NUM_FEATURES,
        "meta_gt_per_forcing": GT_PER_FORCING,
        "meta_version": version,
        "meta_center_start": str(t_start_center),
        "meta_center_end": str(t_end_center),
        "meta_station_start": str(t_start_station),
        "meta_station_end": str(t_end_station),
        "meta_context_start": str(context_start),
    }

    dict_name = f"{year}_{year+1}_{station}_{suffix}_dict.npz"
    dict_path = out_root / "dicts" / dict_name
    np.savez(dict_path, **dict_obj)

    graphs: List[Data] = []

    t0_center = t_forcing_ctx[center_indices[0]]
    if t0_center != t_start_center:
        print(f"[Warning] First forcing center time = {t0_center}, expected = {t_start_center}")

    for i in range(N):
        center_idx_local = center_indices[i]
        center_time = t_forcing_ctx[center_idx_local]

        start_idx = center_idx_local - MAX_HISTORY_STEPS
        end_idx = center_idx_local + 1  # inclusive window => python slice end is exclusive

        window = forcing_ctx[start_idx:end_idx]  # (WINDOW_LENGTH, H, W, 5)
        if window.shape[0] != WINDOW_LENGTH:
            raise RuntimeError(
                f"forcing window length mismatch at i={i}: got {window.shape[0]}, expected {WINDOW_LENGTH}"
            )

        x_hist_np = window.reshape(WINDOW_LENGTH, num_nodes, NUM_FEATURES)
        x_hist = torch.from_numpy(x_hist_np).float()  # (9, num_nodes, 5)
        x_curr = x_hist[-1]                           # (num_nodes, 5)

        # A center labels its own hour and the next five hourly observations.
        s0 = i * GT_PER_FORCING
        s1 = (i + 1) * GT_PER_FORCING

        expected_center = t_start_station + timedelta(hours=s0)
        if center_time != expected_center:
            raise ValueError(
                f"Time alignment mismatch at i={i} for {year} {station} {version}:\n"
                f"  forcing center_time = {center_time}\n"
                f"  expected (station_start + {s0}h) = {expected_center}\n"
                f"Check forcing/station time origin assumptions."
            )

        nc_block = torch.from_numpy(nc_used[s0:s1]).float()               # (6,)
        nc_tide_block = torch.from_numpy(nc_tide_used[s0:s1]).float()     # (6,)
        y_block = nc_block - nc_tide_block                                # (6,)

        data = Data(
            x=x_curr,
            x_hist=x_hist,
            edge_index=edge_index,
            y=y_block,
            nc=nc_block,
            nc_tide=nc_tide_block,
            time_index=i,
            hour_start=s0,
            max_history_hours=MAX_HISTORY_HOURS,
            grid_H=H,
            grid_W=W,
            center_time=str(center_time),
        )

        # Attach local pressure means for the same nine forcing timestamps.
        if p_mean_ctx is not None:
            p_hist = p_mean_ctx[start_idx:end_idx]  # (WINDOW_LENGTH,)
            if p_hist.shape[0] != WINDOW_LENGTH:
                raise RuntimeError(
                    f"p_mean history length mismatch at i={i}: got {p_hist.shape[0]}, expected {WINDOW_LENGTH}"
                )
            p_mean_hist_t = torch.from_numpy(p_hist.astype(np.float32))  # (9,)
            data.p_mean_hist = p_mean_hist_t
            data.p_mean_curr = p_mean_hist_t[-1]  # scalar tensor

        graphs.append(data)

    graph_name = f"{year}_{year+1}_{station}_{suffix}_graphs.pt"
    graph_path = out_root / "graphs" / graph_name
    torch.save(graphs, graph_path)

    print("  Saved dict  :", dict_path)
    print("  Saved graphs:", graph_path, f"(num_graphs={len(graphs)})")


def parse_years_arg(s: str) -> List[int]:
    """Parse season start years from a command-line selection string.

    Args:
        s (str): Single year, comma-separated years, ``start:stop`` with an
            exclusive stop, or ``start-end`` with an inclusive end.

    Returns:
        list[int]: Selected years. Comma-separated order and duplicates are
        retained; a descending range produces an empty list.

    Raises:
        ValueError: A year is not an integer or a range has invalid syntax.
    """
    s = s.strip()
    if ":" in s:
        a, b = s.split(":")
        return list(range(int(a), int(b)))
    if "-" in s:
        a, b = s.split("-")
        return list(range(int(a), int(b) + 1))
    if "," in s:
        return [int(x) for x in s.split(",")]
    return [int(s)]


def resolve_dirs(
    data_root: Path,
    model_tag: str,
    forcing_dir: Optional[str],
    csv_dir: Optional[str],
) -> Tuple[Path, Path]:
    """Resolve and check forcing and station input paths.

    Args:
        data_root (Path): Base directory for relative paths.
        model_tag (str): Model registry key used when an override is absent.
        forcing_dir (str or None): Absolute/relative forcing-path override,
            or ``None`` to use the registered model default.
        csv_dir (str or None): Absolute/relative station-CSV path override,
            or ``None`` to use the registered model default.

    Returns:
        tuple[Path, Path]: Resolved forcing and CSV paths, in that order.
        Existence is checked, but directory type and contents are not.

    Raises:
        KeyError: A required default is unavailable for ``model_tag``.
        FileNotFoundError: Either resolved path does not exist.
    """
    if forcing_dir is not None:
        fdir = (data_root / forcing_dir).resolve() if not Path(forcing_dir).is_absolute() else Path(forcing_dir).resolve()
    else:
        if model_tag not in MODEL_REGISTRY:
            raise KeyError(f"Unknown model_tag={model_tag}. Add to MODEL_REGISTRY or pass --forcing_dir.")
        fdir = (data_root / MODEL_REGISTRY[model_tag]["forcing_subdir"]).resolve()

    if csv_dir is not None:
        cdir = (data_root / csv_dir).resolve() if not Path(csv_dir).is_absolute() else Path(csv_dir).resolve()
    else:
        if model_tag not in MODEL_REGISTRY:
            raise KeyError(f"Unknown model_tag={model_tag}. Add to MODEL_REGISTRY or pass --csv_dir.")
        cdir = (data_root / MODEL_REGISTRY[model_tag]["csv_subdir"]).resolve()

    if not fdir.exists():
        raise FileNotFoundError(f"Forcing dir not found: {fdir}")
    if not cdir.exists():
        raise FileNotFoundError(f"CSV dir not found: {cdir}")

    return fdir, cdir


def make_output_roots(
    data_root: Path,
    model_tag: str,
    out_root_fixed: Optional[str],
    out_root_peryear: Optional[str],
) -> Tuple[Path, Path]:
    """Create separate model output roots for both cutoff conventions.

    Args:
        data_root (Path): Base directory for relative output paths.
        model_tag (str): Model subdirectory appended to both output bases.
        out_root_fixed (str or None): Absolute/relative base for fixed-date
            exports; defaults to ``Aligned_fixed315_hist48`` under data_root.
        out_root_peryear (str or None): Absolute/relative base for coverage
            exports; defaults to ``Aligned_peryear_hist48`` under data_root.

    Returns:
        tuple[Path, Path]: Resolved, model-specific fixed and peryear roots.
        Creates each root's ``dicts`` and ``graphs`` directories as needed.

    Raises:
        OSError: An output directory cannot be created.
    """
    if out_root_fixed is not None:
        base_fixed = (data_root / out_root_fixed).resolve() if not Path(out_root_fixed).is_absolute() else Path(out_root_fixed).resolve()
    else:
        base_fixed = (data_root / "Aligned_fixed315_hist48").resolve()

    if out_root_peryear is not None:
        base_peryear = (data_root / out_root_peryear).resolve() if not Path(out_root_peryear).is_absolute() else Path(out_root_peryear).resolve()
    else:
        base_peryear = (data_root / "Aligned_peryear_hist48").resolve()

    r_fixed = (base_fixed / model_tag).resolve()
    r_peryear = (base_peryear / model_tag).resolve()

    for root in [r_fixed, r_peryear]:
        (root / "dicts").mkdir(parents=True, exist_ok=True)
        (root / "graphs").mkdir(parents=True, exist_ok=True)

    return r_fixed, r_peryear


def main() -> None:
    """Export coverage-based seasonal graphs using command-line settings.

    Reads process arguments, resolves model paths, discovers forcing files,
    computes station coverage, and exports selected station/year pairs.
    Connectivity is constructed once per year and reused for all stations.
    Both output roots are created; this entry point invokes only the
    ``peryear`` branch of ``process_one_pair``.

    Returns:
        None: Writes NPZ/PT files, prints configuration/coverage diagnostics,
        and reloads the first selected graph file if present to report shapes.

    Raises:
        SystemExit: Help is requested or command-line arguments are invalid.
        FileNotFoundError: Required paths, forcing archives, or station CSVs
            are missing. Missing year mappings are skipped after coverage
            calculation when the corresponding station CSVs are available.
        ValueError: Array shapes, station coverage, graph limits, or time
            alignment do not satisfy the preprocessing requirements.
        KeyError: A model default or required input field is absent.
    """
    ap = argparse.ArgumentParser()

    ap.add_argument("--data_root", type=str, default=".", help="Path to Lehigh_Data_Processed (default: cwd)")
    ap.add_argument("--model_tag", type=str, default="MRI", help="e.g., NCEP, CMIP6_AWI, ...")

    ap.add_argument("--forcing_dir", type=str, default=None, help="Override forcing directory (relative to data_root or absolute)")
    ap.add_argument("--csv_dir", type=str, default=None, help="Override station CSV directory (relative to data_root or absolute)")

    ap.add_argument("--out_root_fixed", type=str, default=None, help="Override base output dir for fixed (model_tag will be appended)")
    ap.add_argument("--out_root_peryear", type=str, default=None, help="Override base output dir for peryear (model_tag will be appended)")

    ap.add_argument("--stations", type=str, nargs="+", default=["Lewes", "Boston", "CBBT", "Battery"])

    ap.add_argument(
        "--years",
        type=str,
        default=None,
        help="Examples: 1979:1999 (end exclusive), 1979-1998 (inclusive), 1979,1980,1981, or 1979. "
             "If omitted, infer years from forcing files.",
    )
    ap.add_argument("--exclude_years", type=int, nargs="*", default=[])

    ap.add_argument("--edge_mode", type=str, default="grid4", choices=["fc", "grid4", "grid8"])
    ap.add_argument("--max_edges", type=int, default=50_000_000)

    ap.add_argument("--debug_limit_years", type=int, default=None, help="Process only first K years (debug)")
    ap.add_argument("--debug_limit_stations", type=int, default=None, help="Process only first K stations (debug)")

    args = ap.parse_args()

    data_root = Path(args.data_root).resolve()
    forcing_dir, csv_dir = resolve_dirs(data_root, args.model_tag, args.forcing_dir, args.csv_dir)
    out_root_fixed, out_root_peryear = make_output_roots(data_root, args.model_tag, args.out_root_fixed, args.out_root_peryear)

    year_to_path = discover_forcing_files(forcing_dir)
    if not year_to_path:
        raise FileNotFoundError(
            f"No forcing .npz files found under {forcing_dir}\n"
            f"Expected files like: forcing_local_1979.npz containing key 'forcing'."
        )

    if args.years is None:
        years = sorted(year_to_path.keys())
    else:
        years = parse_years_arg(args.years)

    exclude = set(args.exclude_years or [])
    years = [y for y in years if y not in exclude]

    if args.debug_limit_years is not None:
        years = years[: args.debug_limit_years]
    stations = list(args.stations)
    if args.debug_limit_stations is not None:
        stations = stations[: args.debug_limit_stations]

    cfg = AlignConfig(
        data_root=data_root,
        model_tag=args.model_tag,
        forcing_dir=forcing_dir,
        csv_dir=csv_dir,
        out_root_fixed=out_root_fixed,
        out_root_peryear=out_root_peryear,
        stations=stations,
        years=years,
        exclude_years=list(exclude),
        edge_mode=args.edge_mode,
        max_edges=args.max_edges,
        debug_limit_years=args.debug_limit_years,
        debug_limit_stations=args.debug_limit_stations,
    )

    print("\n================= CONFIG =================")
    print("data_root       :", cfg.data_root)
    print("model_tag       :", cfg.model_tag)
    print("forcing_dir     :", cfg.forcing_dir)
    print("csv_dir         :", cfg.csv_dir)
    print("out_root_fixed  :", cfg.out_root_fixed)
    print("out_root_peryear:", cfg.out_root_peryear)
    print("stations        :", cfg.stations)
    print("years           :", cfg.years)
    print("exclude_years   :", cfg.exclude_years)
    print("edge_mode       :", cfg.edge_mode)
    print("max_edges       :", f"{cfg.max_edges:,}")
    print("MAX_HISTORY_HRS :", MAX_HISTORY_HOURS, "=> window length", WINDOW_LENGTH)
    print("=========================================\n")

    # Use the last full station day common to all stations in each season.
    year_last_day: Dict[int, datetime.date] = {}
    for y in cfg.years:
        year_last_day[y] = compute_last_full_day_for_year(y, cfg.stations, cfg.csv_dir)

    for year in cfg.years:
        if year not in year_to_path:
            print(f"[Warning] No forcing file mapped for year={year} under {cfg.forcing_dir}, skipping.")
            continue

        forcing_path = year_to_path[year]
        forcing, p_mean_t = load_forcing_npz(forcing_path)
        T_full, H, W, C = infer_grid_from_forcing(forcing)
        t_forcing = build_forcing_time_index(year, T_full)

        # Reuse one grid topology for all station exports in this season.
        edge_index = make_edge_index(cfg.edge_mode, H, W, cfg.max_edges)

        last_day = year_last_day[year]

        print(f"\n=== Year {year} | forcing={forcing_path.name} | shape={forcing.shape} | peryear_last_day={last_day} ===")
        print(f"edge_index shape: {tuple(edge_index.shape)}  (nodes={H*W}, edges={edge_index.shape[1]:,})")

        for station in cfg.stations:
            process_one_pair(
                year=year,
                station=station,
                forcing=forcing,
                p_mean_t=p_mean_t,
                t_forcing=t_forcing,
                edge_index=edge_index,
                H=H,
                W=W,
                version="peryear",
                last_full_day=last_day,
                out_root_fixed=cfg.out_root_fixed,
                out_root_peryear=cfg.out_root_peryear,
                csv_dir=cfg.csv_dir,
            )

    # Inspect an existing output for the first selected station-season pair.
    if cfg.years and cfg.stations:
        y0 = cfg.years[0]
        s0 = cfg.stations[0]
        example = cfg.out_root_peryear / "graphs" / f"{y0}_{y0+1}_{s0}_peryear_hist48_graphs.pt"
        print("\nSanity check file:", example)
        if example.exists():
            graphs = torch.load(example, weights_only=False, map_location="cpu")
            g0 = graphs[0]
            print("Loaded graphs:", len(graphs))
            print("x shape      :", tuple(g0.x.shape))
            print("x_hist shape :", tuple(g0.x_hist.shape), f"(should be {WINDOW_LENGTH}, {g0.x.shape[0]}, {NUM_FEATURES})")
            print("y shape      :", tuple(g0.y.shape))
            print("edge_index   :", tuple(g0.edge_index.shape))
            print("grid_H,grid_W:", getattr(g0, "grid_H", None), getattr(g0, "grid_W", None))
            print("center_time  :", getattr(g0, "center_time", None))
            if hasattr(g0, "p_mean_hist"):
                print("p_mean_hist  :", tuple(g0.p_mean_hist.shape), "(should be (9,))")
                print("p_mean_curr  :", g0.p_mean_curr.item() if torch.is_tensor(g0.p_mean_curr) else g0.p_mean_curr)
            else:
                print("p_mean_hist  : <not present> (your forcing npz did not include p_mean_t)")
        else:
            print("Example graph not found yet (maybe you processed different year/station first).")


if __name__ == "__main__":
    main()
