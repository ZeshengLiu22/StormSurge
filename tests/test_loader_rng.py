"""Independent training shuffle RNG, epoch progression and real trainer wiring."""

import contextlib
import io
import itertools
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch.utils.data.distributed import DistributedSampler
from torch_geometric.loader import DataLoader

import train
from emulator.data import build_loader
from emulator.models import ModelConfig, build_model
from test_pipeline import make_fixture


CASES = [(None, 'mean'), *itertools.product(('legacy', 'c1', 'c2', 'c2r', 'c3'), ('mean', 'learned'))]
LOADER_OPTIONS = dict(batch_size=16, num_workers=0, pin_memory=False,
                      persistent_workers=False, prefetch_factor=0, mp_context='fork')


def order(loader):
    return tuple(int(index) for batch in loader for index in batch)


class LoaderRNGTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_all_eleven_models_match_300_epochs_despite_global_rng_draws(self):
        reference = None
        for index, (variant, pooling) in enumerate(CASES):
            with self.subTest(variant=variant, pooling=pooling):
                torch.manual_seed(42)
                build_model(ModelConfig(5, 6, hidden_channels=128,
                    head_type='single' if variant is None else 'dual', station_feat_dim=6,
                    peak_threshold_norm=[1.] * 6, exceedance_head_experiment=variant,
                    exceedance_gate_pooling=pooling))
                generator = torch.Generator().manual_seed(42)
                loader = build_loader(range(64), None, shuffle=True, generator=generator, **LOADER_OPTIONS)
                self.assertIs(loader.generator, generator)
                self.assertIs(loader.sampler.generator, generator)
                orders = []
                for epoch in range(300):
                    # Stand in for architecture-dependent dropout/other RNG use.
                    torch.rand((index + 1) * (epoch % 17 + 1))
                    before = torch.get_rng_state()
                    orders.append(order(loader))
                    torch.testing.assert_close(torch.get_rng_state(), before, rtol=0, atol=0)
                    self.assertEqual(sorted(orders[-1]), list(range(64)))
                self.assertEqual(len(set(orders)), 300, 'Do not reset the generator every epoch')
                if reference is None:
                    reference = orders
                self.assertEqual(orders, reference)
        other_seed = build_loader(range(64), None, shuffle=True,
                                  generator=torch.Generator().manual_seed(123), **LOADER_OPTIONS)
        self.assertNotEqual(order(other_seed), reference[0])

    def test_default_none_preserves_legacy_shuffle_and_evaluation_rng(self):
        for shuffle in (False, True):
            with self.subTest(shuffle=shuffle):
                torch.manual_seed(42)
                expected_loader = DataLoader(range(64), batch_size=16, shuffle=shuffle, num_workers=0)
                expected = [order(expected_loader) for _ in range(3)]
                expected_rng = torch.get_rng_state()
                torch.manual_seed(42)
                loader = build_loader(range(64), None, shuffle=shuffle, **LOADER_OPTIONS)
                self.assertIsNone(loader.generator)
                self.assertEqual([order(loader) for _ in range(3)], expected)
                torch.testing.assert_close(torch.get_rng_state(), expected_rng, rtol=0, atol=0)
                if not shuffle:
                    self.assertEqual(expected, [tuple(range(64))] * 3)

    def test_distributed_sampler_keeps_its_seed_and_epoch_order(self):
        for rank in (0, 1):
            sampler = DistributedSampler(range(64), num_replicas=2, rank=rank, shuffle=True)
            loader = build_loader(range(64), sampler, shuffle=True, **LOADER_OPTIONS)
            self.assertIsNone(loader.generator)
            self.assertIs(loader.sampler, sampler)
            self.assertEqual(sampler.seed, 0)
            for epoch in range(3):
                sampler.set_epoch(epoch)
                torch.rand(37 * (epoch + rank + 1))
                self.assertEqual(order(loader), tuple(sampler))

    def test_actual_trainer_matches_orders_and_repeats_trained_weights(self):
        previous_deterministic = torch.are_deterministic_algorithms_enabled()
        self.addCleanup(torch.use_deterministic_algorithms, previous_deterministic)
        expected_orders, repeated_state = None, None
        real_loader, real_epoch = train.build_loader, train.run_epoch
        with tempfile.TemporaryDirectory() as temporary, contextlib.redirect_stdout(io.StringIO()):
            root = Path(temporary)
            torch.manual_seed(17)
            graphs, stations = make_fixture(root)
            for index, (variant, pooling) in enumerate([*CASES, ('c3', 'learned')]):
                epoch_orders, loaders = [], []

                def loader(*args, **kwargs):
                    result = real_loader(*args, **kwargs)
                    loaders.append((bool(kwargs.get('shuffle')), result.generator))
                    return result

                def epoch(model, *args, **kwargs):
                    if kwargs.get('optimizer') is None:
                        return real_epoch(model, *args, **kwargs)
                    seen = []
                    hook = model.register_forward_pre_hook(lambda module, inputs: seen.extend(inputs[0].sample_id.tolist()))
                    try:
                        result = real_epoch(model, *args, **kwargs)
                    finally:
                        hook.remove()
                    epoch_orders.append(tuple(seen))
                    return result

                output = root / f'run_{index}'
                options = ['--root_dir', str(graphs), '--station', 'Battery', '--station_json_dir', str(stations),
                    '--output_dir', str(output), '--model', 'perceiver3', '--head_type', 'single' if variant is None else 'dual',
                    '--hidden_channels', '8', '--node_read_heads', '2', '--time_read_heads', '2', '--temporal_block', 'MLP',
                    '--history_hours', '12', '--epochs', '3', '--batch_size', '2', '--num_workers', '0', '--device', 'cpu',
                    '--deterministic', '1', '--seed', '42', '--dropout', '.2', '--head_dropout', '.2']
                if variant is not None:
                    options += ['--exceedance_head_experiment', variant, '--exceedance_gate_pooling', pooling]
                with patch.object(train, 'build_loader', side_effect=loader), patch.object(train, 'run_epoch', side_effect=epoch):
                    train.main(options)
                training_generators = [generator for shuffle, generator in loaders if shuffle]
                self.assertEqual(len(training_generators), 1)
                self.assertEqual(training_generators[0].initial_seed(), 42)
                self.assertTrue(all(generator is None for shuffle, generator in loaders if not shuffle))
                self.assertEqual(len(epoch_orders), 3)
                self.assertGreater(len(set(epoch_orders)), 1)
                if expected_orders is None:
                    expected_orders = epoch_orders
                self.assertEqual(epoch_orders, expected_orders, (variant, pooling))
                if (variant, pooling) == ('c3', 'learned'):
                    checkpoint = torch.load(next(output.glob('best_*.pth')), weights_only=False)
                    if repeated_state is None:
                        repeated_state = checkpoint['model_state']
                    else:
                        for key, value in repeated_state.items():
                            torch.testing.assert_close(checkpoint['model_state'][key], value, rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
