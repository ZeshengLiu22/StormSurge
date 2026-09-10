"""Load station JSON aliases and encode the selected finite feature values."""

import json
import math
from pathlib import Path

import torch


def load_station_json(station_json_dir, station_key):
    """Look up exact, lowercase, then uppercase filenames, as in the original."""
    directory = Path(station_json_dir)
    names = dict.fromkeys((station_key, station_key.lower(), station_key.upper()))
    for name in names:
        path = directory / f"{name}.json"
        if path.exists():
            return json.loads(path.read_text())
    tried = ", ".join(f"{name}.json" for name in names)
    raise FileNotFoundError(f"Station JSON not found for {station_key!r} under {directory}; tried {tried}.")


def station_features_from_json(station, *, use_site_elevation=True, use_bathymetry=False):
    """Require valid coordinates and independently selected elevation/bathymetry."""
    if not isinstance(station, dict):
        raise ValueError("Station JSON must be an object containing station fields.")

    def required_float(keys, setting="Station metadata"):
        message = f"{setting} requires finite {keys[0]} in the station JSON (accepted keys: {', '.join(keys)})."
        for key in keys:
            if key in station:
                try:
                    value = float(station[key])
                except (TypeError, ValueError, OverflowError) as error:
                    raise ValueError(message) from error
                if not math.isfinite(value):
                    raise ValueError(message)
                return value
        raise ValueError(message)

    lat = required_float(("lat", "latitude", "Latitude"))
    lon = required_float(("lon", "longitude", "Longitude"))
    features = [lat / 90, lon / 180]
    if use_site_elevation:
        features.append(required_float(("elevation_m", "elevation", "elev_m"), "use_site_elevation=True") / 10)
    lat, lon = math.radians(lat), math.radians(lon)
    features += [math.sin(lat), math.cos(lat), math.sin(lon), math.cos(lon)]
    if use_bathymetry:
        features.append(required_float(("bathymetry_m",), "use_bathymetry=True") / 10)
    features = torch.tensor(features, dtype=torch.float32)
    if not torch.isfinite(features).all():
        raise ValueError("Station features must be finite in float32.")
    return features
