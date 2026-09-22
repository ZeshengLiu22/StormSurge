"""Atomic checkpoint writes retain the previous complete artifact on failure."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from emulator.training.checkpoints import atomic_save


class AtomicCheckpointTests(unittest.TestCase):
    def test_failed_serialization_preserves_existing_and_cleans_temporary(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'best_overall.pt'
            atomic_save({'epoch': 1}, path)
            def broken_save(value, destination):
                Path(destination).write_bytes(b'partial')
                raise OSError('disk full')
            with patch('emulator.training.checkpoints.torch.save', side_effect=broken_save):
                with self.assertRaisesRegex(OSError, 'disk full'):
                    atomic_save({'epoch': 2}, path)
            self.assertEqual(torch.load(path, weights_only=False), {'epoch': 1})
            self.assertEqual(list(Path(temporary).iterdir()), [path])


if __name__ == '__main__':
    unittest.main()
