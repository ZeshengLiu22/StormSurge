"""Architecture-independent counts and their training/inference artifacts."""

import contextlib
import io
import itertools
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import nn

import infer
import train
from emulator.models import ModelConfig, build_model, count_model_parameters, format_parameter_counts
from test_pipeline import make_fixture


class ModelParameterTests(unittest.TestCase):
    def test_counts_deduplicate_shared_parameters_exclude_buffers_and_track_frozen_weights(self):
        model = nn.Module()
        model.backbone = nn.Linear(3, 4)  # 12 weights + 4 biases.
        model.head = nn.Linear(4, 2)  # 8 weights + 2 frozen biases.
        model.head.bias.requires_grad_(False)
        model.shared_head_alias = model.head
        model.register_buffer('statistics', torch.ones(100))
        expected = dict(total=26, trainable=24, non_trainable=2,
                        head=dict(total=10, trainable=8, non_trainable=2),
                        backbone=dict(total=16, trainable=16, non_trainable=0), head_branches={})
        self.assertEqual(count_model_parameters(model), expected)
        self.assertEqual(count_model_parameters(nn.DataParallel(model)), expected)
        self.assertEqual(format_parameter_counts(expected), '[Parameters] total=26 trainable=24 non_trainable=2 head=10')

    def test_every_model_family_history_encoder_and_head_width_has_correct_head_count(self):
        variants = [('baseline', 'single', 'direct', 'none'), ('pact', 'single', 'direct', 'none'),
                    ('pact', 'dual', 'direct', 'none'), ('pact', 'dual', 'direct', 'fixed_gate'),
                    ('pact', 'dual', 'severity_shape', 'none'), ('pact', 'dual', 'severity_shape', 'fixed_gate')]
        for (model, head, formulation, ablation), history, encoder, width in itertools.product(
                variants, (0, 2), ('GraphSAGE', 'CNN'), (None, 23)):
            with self.subTest(model=model, head=head, formulation=formulation, ablation=ablation,
                              history=history, encoder=encoder, width=width):
                config = ModelConfig(3, 4, hidden_channels=16, model=model, head_type=head, head_hidden=width,
                    temporal_hidden=19, history_steps=history, encoder_type=encoder, temporal_block='MLP',
                    node_read_heads=2, time_read_heads=2, peak_threshold_norm=[1.] * 4,
                    excess_formulation=formulation, target_y_std=[.25, .5, 2., 5.], dual_ablation=ablation)
                network = build_model(config)
                counts = count_model_parameters(network)
                if model == 'baseline':
                    head_expected = 4 * ((19 if history else 16) + 1)
                    branch_names = set()
                else:
                    branch_parameters = (width or 32) * (16 + 2) + 1
                    branch_names = {'regression'} if head == 'single' else (
                        {'body', 'severity', 'shape'} if formulation == 'severity_shape' else {'body', 'excess'})
                    if head == 'dual' and ablation != 'fixed_gate':
                        branch_names.add('gate')
                    head_expected = branch_parameters * len(branch_names)
                    self.assertTrue(all(item['total'] == branch_parameters for item in counts['head_branches'].values()))
                self.assertEqual(counts['head']['total'], head_expected)
                self.assertEqual(set(counts['head_branches']), branch_names)
                self.assertEqual(counts['trainable'], counts['total'])
                self.assertEqual(counts['non_trainable'], 0)
                self.assertEqual(counts['total'], counts['backbone']['total'] + head_expected)
                self.assertGreater(counts['backbone']['total'], 0)

    def test_severity_shape_capacity_difference_is_one_mlp_at_production_width(self):
        for ablation in ('none', 'fixed_gate'):
            models = [build_model(ModelConfig(3, 4, hidden_channels=128, head_hidden=256,
                       peak_threshold_norm=[1.] * 4, excess_formulation=formulation, target_y_std=[1.] * 4,
                       dual_ablation=ablation)) for formulation in ('direct', 'severity_shape')]
            direct, severity = map(count_model_parameters, models)
            self.assertEqual(severity['total'] - direct['total'], 33281)
            self.assertEqual(severity['head']['total'] - direct['head']['total'], 33281)
            self.assertEqual(severity['backbone'], direct['backbone'])
            self.assertEqual(severity['head_branches']['severity']['total'], 33281)

    def test_reporting_does_not_run_forward_or_change_rng_weights_training_or_grad_flags(self):
        model = build_model(ModelConfig(3, 4, hidden_channels=16, head_type='single',
                            node_read_heads=2, time_read_heads=2))
        model.head.requires_grad_(False)
        before = {name: value.clone() for name, value in model.state_dict().items()}
        rng = torch.get_rng_state().clone()
        flags = [(parameter.requires_grad, parameter.grad) for parameter in model.parameters()]
        with patch.object(model, 'forward', side_effect=AssertionError('counting must not run inference')):
            counts = count_model_parameters(model)
        torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
        self.assertTrue(model.training)
        self.assertEqual(flags, [(parameter.requires_grad, parameter.grad) for parameter in model.parameters()])
        self.assertEqual(counts['head']['trainable'], 0)
        self.assertEqual(counts['non_trainable'], counts['head']['total'])
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, before[name], rtol=0, atol=0)

    def test_training_and_inference_always_print_and_save_actual_model_counts(self):
        variants = [('baseline', 'single', 'direct', 'none', 0), ('baseline', 'single', 'direct', 'none', 12),
                    ('perceiver3', 'single', 'direct', 'none', 12), ('perceiver3', 'dual', 'direct', 'none', 12),
                    ('perceiver3', 'dual', 'severity_shape', 'none', 12),
                    ('perceiver3', 'dual', 'severity_shape', 'fixed_gate', 12)]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            torch.manual_seed(824)
            graphs, stations = make_fixture(root)
            for index, (model, head, formulation, ablation, history) in enumerate(variants):
                with self.subTest(model=model, head=head, formulation=formulation, ablation=ablation, history=history):
                    output, inferred = root / f'train_{index}', root / f'infer_{index}'
                    console = io.StringIO()
                    with contextlib.redirect_stdout(console), patch.object(train, 'run_epoch', wraps=train.run_epoch) as epochs:
                        train.main(['--root_dir', str(graphs), '--station', 'Battery', '--station_json_dir', str(stations),
                            '--output_dir', str(output), '--model', model, '--head_type', head, '--encoder_type', 'CNN',
                            '--temporal_block', 'MLP', '--history_hours', str(history), '--excess_formulation', formulation,
                            '--dual_ablation', ablation, '--hidden_channels', '16', '--node_read_heads', '2',
                            '--time_read_heads', '2', '--epochs', '1', '--num_workers', '0', '--device', 'cpu'])
                    self.assertEqual(epochs.call_count, 3)  # TRAIN, VAL, one final TEST.
                    path, = output.glob('best_*.pth')
                    checkpoint = torch.load(path, weights_only=False)
                    counts = checkpoint['model_parameters']
                    summary = json.loads(next(output.glob('summary_*.json')).read_text())
                    self.assertEqual(summary['model_parameters'], counts)
                    expected = count_model_parameters(build_model(ModelConfig(**checkpoint['model_config'])))
                    self.assertEqual(counts, expected)
                    line = format_parameter_counts(counts)
                    self.assertEqual(console.getvalue().count('[Parameters]'), 1)
                    self.assertIn(line, console.getvalue().splitlines()[0])
                    self.assertNotIn('model_parameters', checkpoint['training_config'])
                    self.assertNotIn('model_parameters', checkpoint['model_config'])
                    # Counts are computed from the actual loaded model, never required in older metadata.
                    checkpoint.pop('model_parameters')
                    torch.save(checkpoint, path)
                    inference_console = io.StringIO()
                    with contextlib.redirect_stdout(inference_console), patch.object(infer, 'run_epoch', wraps=infer.run_epoch) as passes:
                        infer.main(['--ckpt', str(path), '--root_dir', str(graphs), '--out_dir', str(inferred),
                                    '--device', 'cpu', '--num_workers', '0'])
                    self.assertEqual(passes.call_count, 1)
                    self.assertEqual(inference_console.getvalue().count('[Parameters]'), 1)
                    self.assertIn(line, inference_console.getvalue().splitlines()[0])
                    self.assertEqual(json.loads((inferred / 'metrics.json').read_text())['model_parameters'], counts)
                    report = json.loads(next(inferred.glob('metrics_per_year_*.json')).read_text())
                    self.assertEqual(report['model_parameters'], counts)


if __name__ == '__main__':
    unittest.main()
