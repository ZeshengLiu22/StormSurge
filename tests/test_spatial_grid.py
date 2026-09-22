"""Current CNN rectangular-grid contracts retained from the useful grid audit."""
import unittest

import torch
from torch_geometric.data import Batch, Data

from emulator.models.spatial import SpatialEncoder


def grid_batch(height=2, width=3, count=3):
    return Batch.from_data_list([
        Data(x=torch.randn(height * width, 5), edge_index=torch.empty(2, 0, dtype=torch.long),
             grid_H=height, grid_W=width) for _ in range(count)])


class SpatialGridTests(unittest.TestCase):
    def test_metadata_forms_and_rectangular_layout(self):
        encoder = SpatialEncoder(5, 8, 2, 0, 'CNN')
        for convert in (int, torch.tensor, lambda x: [x], lambda x: (x,)*3,
                        lambda x: torch.full((3, 1), x), lambda x: torch.full((3,), float(x))):
            batch = grid_batch()
            batch.grid_H, batch.grid_W = convert(2), convert(3)
            before = batch.x.clone()
            self.assertEqual(encoder.grid_shape(batch), (3, 2, 3))
            output = encoder(batch.x, batch.edge_index, encoder.grid_shape(batch))
            self.assertEqual(output.shape, (18, 8))
            torch.testing.assert_close(batch.x, before)

    def test_malformed_grids_fail(self):
        encoder = SpatialEncoder(5, 8, 2, 0, 'CNN')
        for kind in ('missing', 'empty', 'nonuniform', 'nonpositive', 'wrong_product', 'unequal_sizes'):
            batch = grid_batch()
            if kind == 'missing': del batch.grid_H
            elif kind == 'empty': batch.grid_H = torch.empty(0)
            elif kind == 'nonuniform': batch.grid_H[1] = 3
            elif kind == 'nonpositive': batch.grid_H.zero_()
            elif kind == 'wrong_product': batch.grid_W.fill_(4)
            else: batch.ptr[1] -= 1
            with self.assertRaises(ValueError, msg=kind):
                encoder.grid_shape(batch)
