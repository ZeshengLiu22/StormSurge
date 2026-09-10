"""Small real NetCDF/CSV inputs exercise the maintained preprocessing entrypoints."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import sys

from netCDF4 import Dataset
import numpy as np
import pandas as pd
import torch

from preprocessing import forcing_cmip6, preprocessing_simulation, time_align_unified
from preprocessing import preprocessing_forcing_NCEP_Mean_Removal as forcing_ncep


class PreprocessingPipelineTests(unittest.TestCase):
    def test_cmip6_all_five_profiles_share_outputs_and_downsampling(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            grid = root / 'grid.nc'
            with Dataset(grid, 'w') as data:
                data.createDimension('lat', 2)
                data.createDimension('lon', 3)
                data.createVariable('lat', 'f8', ('lat',))[:] = [10, 50]
                data.createVariable('lon', 'f8', ('lon',))[:] = [-100, -80, -40]
            raw = np.arange(4 * 2 * 3 * 3).reshape(-1, 3)
            np.savetxt(root / 'fort_2000.22', raw)
            expected, pressure, _, _ = forcing_cmip6.prepare_forcing(raw, np.array([50, 10]), np.array([-100, -80, -40]))
            for tag in ('AWI', 'CNRM', 'EC_EARTH', 'MPI', 'MRI'):
                out = root / tag
                forcing_cmip6.main(['--model', tag, '--forcing_dir', str(root), '--grid_netcdf', str(grid),
                                   '--out_dir', str(out), '--first_year', '2000', '--last_year', '2000'])
                with np.load(out / 'forcing_local_2000.npz') as arrays:
                    np.testing.assert_array_equal(arrays['forcing'], expected[::2])
                    np.testing.assert_array_equal(arrays['p_mean_t'], pressure[::2])
                    self.assertEqual(int(arrays['dt_hours']), 6)
                np.testing.assert_array_equal(np.load(out / 'forcing_local_2000.npy'), expected[::2])
                np.testing.assert_array_equal(np.load(out / 'forcing_local_3h_2000.npy'), expected)

    def test_simulation_explicit_year_and_original_defaults(self):
        defaults = preprocessing_simulation.parse_args([])
        self.assertEqual(defaults.years, [2005])
        self.assertEqual(defaults.mesh, './scripts/fort_mesh.14')
        self.assertEqual(defaults.input_sim, './CMIP6_MPI')
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            simulation, tide = root / 'simulation', root / 'tide'
            simulation.mkdir(); tide.mkdir()
            sites = list(preprocessing_simulation.stations.values())
            coordinates = [(site['lon'], site['lat']) for site in sites]
            mesh = root / 'fort.14'
            mesh.write_text('mesh\n1 4\n' + '\n'.join(f'{i+1} {lon} {lat} 10' for i, (lon, lat) in enumerate(coordinates)) + '\n1 3 1 2 3\n')
            for directory, shift in ((simulation, 0), (tide, 1)):
                with Dataset(directory / 'fort_2001.63.nc', 'w') as data:
                    data.createDimension('time', 3); data.createDimension('node', 4)
                    clock = data.createVariable('time', 'f8', ('time',))
                    clock.units = 'hours since 2001-10-25 00:00:00'
                    clock[:] = np.arange(1, 4) + shift
                    data.createVariable('x', 'f8', ('node',))[:] = np.array(coordinates)[:, 0]
                    data.createVariable('y', 'f8', ('node',))[:] = np.array(coordinates)[:, 1]
                    data.createVariable('zeta', 'f8', ('time', 'node'))[:] = np.arange(12).reshape(3, 4) + shift * 100
            full, csv = root / 'full', root / 'csv'
            preprocessing_simulation.main(['--mesh', str(mesh), '--input-sim', str(simulation), '--input-tide', str(tide),
                                           '--output-full', str(full), '--output-csv', str(csv), '--years', '2001'])
            with np.load(full / 'fort_2001_zeta_tide_full.npz') as arrays:
                np.testing.assert_array_equal(arrays['nc'], np.arange(12).reshape(3, 4)[1:])
                np.testing.assert_array_equal(arrays['nc_tide'], np.arange(12).reshape(3, 4)[:2] + 100)
            for node, key in enumerate(preprocessing_simulation.stations):
                name = key.split('_')[0]
                frame = pd.read_csv(csv / f'2001_2002_{name}.csv')
                np.testing.assert_array_equal(frame['nc'], np.array([4, 8]) + node)
                self.assertEqual(frame['time'].iloc[0], '2001-10-25 02:00:00')

    def test_time_alignment_cli_writes_only_peryear_graphs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            forcing, csv = root / 'forcing', root / 'csv'
            forcing.mkdir(); csv.mkdir()
            np.savez(forcing / 'forcing_local_2000.npz', forcing=np.zeros((32, 2, 2, 5), dtype=np.float32))
            times = pd.date_range('2000-10-25 01:00', '2000-11-01 23:00', freq='h')
            pd.DataFrame(dict(time=times, nc=np.arange(len(times)), nc_tide=np.zeros(len(times)))).to_csv(csv / '2000_2001_Battery.csv', index=False)
            args = ['time_align_unified.py', '--data_root', str(root), '--forcing_dir', 'forcing', '--csv_dir', 'csv',
                    '--out_root_peryear', 'aligned', '--out_root_fixed', 'fixed', '--stations', 'Battery', '--years', '2000']
            with patch.object(sys, 'argv', args):
                time_align_unified.main()
            paths = list((root / 'aligned').rglob('*graphs.pt'))
            self.assertEqual(len(paths), 1)
            samples = torch.load(paths[0], weights_only=False)
            self.assertEqual(len(samples), 4)
            self.assertEqual(samples[0].center_time, '2000-11-01 00:00:00')

    def test_ncep_cli_keeps_local_grid_and_pressure_units(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw = np.arange(2 * 94 * 192 * 3, dtype=float).reshape(2, 94, 192, 3)
            np.savetxt(root / 'fort_2000.22', raw.reshape(-1, 3), fmt='%.0f')
            forcing_ncep.main(['--forcing_dir', str(root), '--out_dir', str(root / 'out'),
                              '--first_year', '2000', '--last_year', '2000'])
            lat_mask = (forcing_ncep.lat >= 0) & (forcing_ncep.lat <= 50)
            lon_mask = (forcing_ncep.lon >= -105) & (forcing_ncep.lon <= -55)
            expected = raw[:, lat_mask][:, :, lon_mask].copy()
            pressure = expected[..., 2].mean(axis=(1, 2))
            expected[..., 2] -= pressure[:, None, None]
            with np.load(root / 'out/forcing_local_2000.npz') as arrays:
                np.testing.assert_array_equal(arrays['forcing'][..., :3], expected)
                np.testing.assert_array_equal(arrays['p_mean_t'], pressure)
                self.assertEqual(int(arrays['dt_hours']), 6)
            with np.load(root / 'out/forcing_local_2000.npz') as arrays:
                np.testing.assert_array_equal(np.load(root / 'out/forcing_local_2000.npy'), arrays['forcing'])
