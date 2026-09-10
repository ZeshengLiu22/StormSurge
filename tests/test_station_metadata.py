"""Station-feature selection, units, and optional-field isolation."""

import itertools
import json
import math
from pathlib import Path
import tempfile
import unittest

import torch

from emulator.data import load_station_json, station_features_from_json


class StationMetadataTests(unittest.TestCase):
    def test_filename_lookup_prefers_exact_then_lower_then_upper(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            names = ("Battery", "battery", "BATTERY")
            for name in names:
                (root / f"{name}.json").write_text(json.dumps({"selected": name}))
            for name in names:
                self.assertEqual(load_station_json(root, "Battery"), {"selected": name})
                (root / f"{name}.json").unlink()
            with self.assertRaisesRegex(FileNotFoundError, "Battery.json, battery.json, BATTERY.json"):
                load_station_json(root, "Battery")

    def test_invalid_exact_json_is_not_hidden_by_a_filename_alias(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "Battery.json").write_text("invalid JSON")
            (root / "battery.json").write_text(json.dumps(dict(lat=30, lon=-90, elevation_m=20)))
            with self.assertRaises(json.JSONDecodeError):
                load_station_json(root, "Battery")

    def test_all_original_field_aliases_preserve_features(self):
        expected = station_features_from_json(dict(lat=30, lon=-90, elevation_m=20, bathymetry_m=50), use_bathymetry=True)
        for lat, lon, elevation in itertools.product(
            ("lat", "latitude", "Latitude"), ("lon", "longitude", "Longitude"), ("elevation_m", "elevation", "elev_m"),
        ):
            with self.subTest(lat=lat, lon=lon, elevation=elevation):
                station = {lat: "30", lon: "-90", elevation: "20", "bathymetry_m": "50"}
                torch.testing.assert_close(station_features_from_json(station, use_bathymetry=True), expected, rtol=0, atol=0)

    def test_field_alias_priority_does_not_hide_invalid_selected_values(self):
        canonical = dict(lat=30, lon=-90, elevation_m=20)
        expected = station_features_from_json(canonical)
        keys = (("lat", "latitude", "Latitude"), ("lon", "longitude", "Longitude"), ("elevation_m", "elevation", "elev_m"))
        for aliases in keys:
            for selected in range(len(aliases)):
                with self.subTest(aliases=aliases, selected=selected):
                    station = dict(canonical)
                    value = station.pop(aliases[0])
                    station.update({key: "invalid" for key in aliases[selected + 1:]})
                    station[aliases[selected]] = value
                    torch.testing.assert_close(station_features_from_json(station), expected, rtol=0, atol=0)
                    station.update({key: value for key in aliases[selected + 1:]})
                    station[aliases[selected]] = "invalid"
                    with self.assertRaisesRegex(ValueError, f"requires finite {aliases[0]}"):
                        station_features_from_json(station)

    def test_independent_feature_switches_preserve_geography_and_units(self):
        station = dict(lat=30, lon=-90, elevation_m=20, bathymetry_m=50)
        trig = [0.5, math.sqrt(3) / 2, -1.0, 0.0]
        for elevation, bathymetry in itertools.product((False, True), repeat=2):
            with self.subTest(elevation=elevation, bathymetry=bathymetry):
                features = station_features_from_json(
                    station, use_site_elevation=elevation, use_bathymetry=bathymetry,
                )
                expected = [1 / 3, -0.5] + ([2.0] if elevation else []) + trig + ([5.0] if bathymetry else [])
                torch.testing.assert_close(features, torch.tensor(expected), rtol=1e-6, atol=1e-7)
        expected_default = torch.tensor([1 / 3, -0.5, 2.0, *trig])
        torch.testing.assert_close(station_features_from_json(station), expected_default, rtol=1e-6, atol=1e-7)

    def test_disabled_features_and_node_coordinates_do_not_affect_inputs(self):
        station = dict(lat=30, lon=-90, elevation_m=20, bathymetry_m=50)
        changed = dict(station, bathymetry_m=float("nan"), bathymetry_node=dict(latitude=0, longitude=0))
        torch.testing.assert_close(station_features_from_json(station), station_features_from_json(changed), rtol=0, atol=0)
        changed = dict(station, elevation_m=float("nan"))
        options = dict(use_site_elevation=False, use_bathymetry=True)
        torch.testing.assert_close(station_features_from_json(station, **options), station_features_from_json(changed, **options), rtol=0, atol=0)

    def test_required_fields_reject_missing_and_invalid_values(self):
        missing = object()
        for field, value in itertools.product(
            ("lat", "lon", "elevation_m", "bathymetry_m"),
            (missing, None, "invalid", float("nan"), float("inf"), -float("inf")),
        ):
            with self.subTest(field=field, value=value):
                station = dict(lat=30, lon=-90, elevation_m=20, bathymetry_m=50)
                if value is missing:
                    station.pop(field)
                else:
                    station[field] = value
                with self.assertRaisesRegex(ValueError, f"requires finite {field}"):
                    station_features_from_json(station, use_bathymetry=True)

    def test_disabled_optional_fields_can_be_missing(self):
        station = dict(lat=30, lon=-90)
        features = station_features_from_json(station, use_site_elevation=False, use_bathymetry=False)
        self.assertEqual(features.numel(), 6)
        self.assertTrue(torch.isfinite(features).all())

    def test_json_must_be_an_object_and_features_must_fit_float32(self):
        for station in (None, [], "invalid", 1):
            with self.subTest(station=station), self.assertRaisesRegex(ValueError, "must be an object"):
                station_features_from_json(station)
        with self.assertRaisesRegex(ValueError, "finite in float32"):
            station_features_from_json(dict(lat=30, lon=-90, elevation_m=1e100))


if __name__ == "__main__":
    unittest.main()
