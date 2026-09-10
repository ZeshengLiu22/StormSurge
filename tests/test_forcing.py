"""CMIP6 consolidation preserves spatial order and pressure anomaly math."""

import unittest
import numpy as np
from preprocessing.forcing_cmip6 import prepare_forcing


class ForcingTests(unittest.TestCase):
    def test_rectangular_grid_coordinates_and_pressure(self):
        lat = np.array([60., 50., 30., 0., -10.])
        lon = np.array([-120., -100., -80., -40., -20.])
        raw = np.arange(4 * 5 * 5 * 3, dtype=float).reshape(4, 5, 5, 3)
        original = raw.copy()
        actual, mean, local_lat, local_lon = prepare_forcing(raw.reshape(-1, 3), lat, lon)
        expected = raw[:, 1:4, 1:4].copy()
        pressure = expected[..., 2].mean(axis=(1, 2))
        expected[..., 2] -= pressure[:, None, None]
        np.testing.assert_array_equal(actual[..., :3], expected)
        np.testing.assert_array_equal(mean, pressure)
        np.testing.assert_array_equal(actual[0, ..., 3], np.broadcast_to(lon[1:4], (3, 3)))
        np.testing.assert_array_equal(actual[0, ..., 4], np.broadcast_to(lat[1:4, None], (3, 3)))
        np.testing.assert_array_equal(raw, original)
        np.testing.assert_array_equal(actual[::2, ..., :3], expected[::2])
