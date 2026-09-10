"""Encode explicit station JSON fields; feature values are saved with the model."""

import math

import torch


def station_features_from_json(station, *, use_site_elevation=True, use_bathymetry=False):
    lat, lon = float(station["lat"]), float(station["lon"])
    features = [lat / 90, lon / 180]
    if use_site_elevation:
        features.append(float(station.get("elevation_m", 0.0)) / 10)
    lat, lon = math.radians(lat), math.radians(lon)
    features += [math.sin(lat), math.cos(lat), math.sin(lon), math.cos(lon)]
    if use_bathymetry:
        try:
            bathymetry = float(station["bathymetry_m"])
        except (KeyError, TypeError, ValueError):
            bathymetry = math.nan
        if not math.isfinite(bathymetry):
            raise ValueError("use_bathymetry=True requires finite bathymetry_m in the station JSON.")
        features.append(bathymetry / 10)
    if not all(math.isfinite(value) for value in features):
        raise ValueError("Station features must be finite.")
    return torch.tensor(features, dtype=torch.float32)
