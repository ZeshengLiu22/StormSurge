"""Load station JSON aliases and encode the selected finite feature values."""

import json
import math
from pathlib import Path

import torch


def load_station_json(station_json_dir, station_key):
    """Read station metadata using exact, lowercase, then uppercase filenames.

    Args:
        station_json_dir: Directory containing one JSON file per station.
        station_key: Station identifier used to form candidate <station>.json filenames.

    Returns:
        object: Decoded JSON from the first existing candidate; schema validation is
        deferred to station_features_from_json.

    Raises:
        FileNotFoundError: No candidate station file exists.
        json.JSONDecodeError: The selected file is not valid JSON.
    """
    directory = Path(station_json_dir)
    names = dict.fromkeys((station_key, station_key.lower(), station_key.upper()))
    for name in names:
        path = directory / f"{name}.json"
        if path.exists():
            return json.loads(path.read_text())
    tried = ", ".join(f"{name}.json" for name in names)
    raise FileNotFoundError(f"Station JSON not found for {station_key!r} under {directory}; tried {tried}.")


def station_features_from_json(station, *, use_site_elevation=True, use_bathymetry=False):
    """Encode station coordinates and selected elevation fields as features.

    Args:
        station: Metadata dictionary with finite latitude and longitude in degrees.
            Accepted coordinate keys are lat/latitude/Latitude and
            lon/longitude/Longitude.
        use_site_elevation: Whether to include elevation in metres divided by 10,
            accepting elevation_m, elevation, or elev_m.
        use_bathymetry: Whether to append bathymetry_m divided by 10; its stored sign
            convention is preserved.

    Returns:
        torch.Tensor: CPU float32 vector of length 6 plus the number of enabled height
        fields. Order: lat/90, lon/180, optional elevation/10, sin(lat), cos(lat),
        sin(lon), cos(lon), optional bathymetry/10.

    Raises:
        ValueError: Metadata is not a dictionary, a selected field is absent or
            nonfinite, or encoding overflows float32.
    """
    if not isinstance(station, dict):
        raise ValueError("Station JSON must be an object containing station fields.")

    def required_float(keys, setting="Station metadata"):
        """Read the first present metadata alias as a finite float.

        Args:
            keys: Ordered field aliases to inspect in the enclosing station dictionary.
            setting: Name of the feature setting to include in validation errors.

        Returns:
            float: Value of the first present field. Invalid present fields raise rather
            than falling back to later aliases.

        Raises:
            ValueError: No alias is present or the selected value cannot be converted to a
                finite float.
        """
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
