#!/usr/bin/env python3
"""CPU argument/selection checks only: never calls train.main or infer.main."""
import argparse
import contextlib
import hashlib
import io
import json
from pathlib import Path
import sys
from types import SimpleNamespace

REPO = Path('/glade/u/home/zel/scratch/StormSurge')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(REPO))
    import torch
    import torch_geometric
    import infer
    from emulator.training.arguments import parse_args as parse_train
    from emulator.data.station_metadata import load_station_json, station_features_from_json
    torch.set_num_threads(1)
    validation = REPO/'PBS-launcher/cmip6/validation'
    inventory = json.loads((validation/'dataset_inventory.json').read_text())
    matrix = json.loads((args.out/'matrix_validation.json').read_text())
    assert matrix['status'] == 'PASS'
    for row in matrix['combinations']:
        parsed = parse_train(row['argv'])
        assert parsed.station == row['station'] and parsed.seed == row['seed']
        assert parsed.history_hours == 36 and parsed.checkpoint_selection == 'overall'
    stations = {}
    for name in ['Battery','Boston','CBBT','Lewes']:
        station = load_station_json(REPO/'station_json', name)
        features = station_features_from_json(station, use_site_elevation=False, use_bathymetry=False)
        assert features.numel() == 6 and torch.isfinite(features).all()
        stations[name] = dict(status='PASS',feature_count=features.numel(),source=str(REPO/'station_json'/f'{name}.json'))

    class FilenameStore:
        """Exercise infer.py's actual selection function without loading/evaluating graphs."""
        def __init__(self, root, station, pressure_mode, year_filter):
            assert pressure_mode == 'raw'
            assert year_filter and all(int(y[:4])>2030 for y in year_filter)
            self.graphs, self.graph_tags, self.year_to_indices = [], [], {}
            for path in sorted(Path(root).glob('*graphs.pt')):
                parts = path.stem.split('_')
                year = '_'.join(parts[:2])
                if parts[2] == station and year in year_filter:
                    self.year_to_indices[year] = [len(self.graphs)]
                    self.graphs.append(None)
                    self.graph_tags.append(path.name.removesuffix('_graphs.pt')+'_0')
            if not self.graphs:
                raise ValueError('No matching station/year files')

    checkpoint = {'training_config':{'pressure_mode':'raw'},'split_tags':{'test':['historical_test_only']}}
    original_store = infer.ForcingGraphStore
    checks = []
    try:
        infer.ForcingGraphStore = FilenameStore
        for record in inventory:
            cli = ['--ckpt','/not-created/historical/best_overall.pt',
                   '--root_dir',record['past_root'],'--test_root_dir',record['full_root'],
                   '--station',record['station'],'--years',','.join(record['future_years']),
                   '--strict_years','--pressure_mode','raw','--device','cuda','--amp','--amp_dtype','bf16',
                   '--tf32','--save_npz','--num_workers','0','--torch_threads','1']
            parsed = infer.parse_args(cli)
            assert parsed.test_root_dir and parsed.scope is None
            store, indices, requested, evaluated = infer._select_evaluation_samples(
                parsed, checkpoint, record['station'], record['full_root'], all_years=True)
            assert requested == evaluated == record['future_years'] and len(indices) == 30
            negatives = {}
            for name, years in [('missing','2100_2101'),('duplicate',record['future_years'][0]+','+record['future_years'][0]),
                                ('malformed','2099_2101'),('empty','')]:
                trial = SimpleNamespace(years=years, strict_years=True, pressure_mode='raw')
                try:
                    infer._select_evaluation_samples(trial, checkpoint, record['station'],record['full_root'],all_years=True)
                except ValueError as exc:
                    negatives[name] = str(exc)
                else:
                    raise AssertionError(f'Strict year validation accepted {name}')
            checks.append(dict(gcm=record['gcm'],station=record['station'],status='PASS',
                               year_count=len(evaluated),evaluated_years=evaluated,negative_cases=negatives))
        invalid_scope = ['--ckpt','/unused/best_overall.pt','--root_dir',inventory[0]['past_root'],
                         '--test_root_dir',inventory[0]['full_root'],'--scope','test']
        with contextlib.redirect_stderr(io.StringIO()):
            try:
                infer.parse_args(invalid_scope)
            except SystemExit as exc:
                assert exc.code == 2
            else:
                raise AssertionError('External evaluation accepted source-test scope')
    finally:
        infer.ForcingGraphStore = original_store
    train_source = (REPO/'train.py').read_text()
    infer_source = (REPO/'infer.py').read_text()
    evidence = {
        'train_statistics': 'stats_cpu = fit_statistics(\n        store,\n        splits["train"],',
        'train_threshold': 'store, splits["train"], exceedance_percentile=args.exceedance_percentile',
        'val_checkpoint_selection': 'tracker.observe(epoch, validation.metrics, checkpoint_snapshot)',
        'saved_normalization': '"normalization": stats_cpu',
        'saved_threshold': '"threshold_metadata": fitted',
        'saved_episode_metadata': '"episode_peak_metadata": episode_peak_metadata',
    }
    for key, snippet in evidence.items():
        assert snippet in train_source, key
    assert 'stats = checkpoint["normalization"]' in infer_source
    assert 'tau_physical = threshold_metadata["tau_physical"]' in infer_source
    assert 'train_hourly_q95_v1' in infer_source
    assert 'model.load_state_dict(checkpoint["model_state"], strict=True)' in infer_source
    assert 'fit_statistics(' not in infer_source and 'fit_loss_thresholds(' not in infer_source
    runtime = dict(python=sys.executable,torch=torch.__version__,pyg=torch_geometric.__version__,
                   torch_cuda_build=torch.version.cuda,nccl_compiled=torch.distributed.is_nccl_available())
    assert runtime['nccl_compiled']
    report = dict(status='PASS',training_cli_count=len(matrix['combinations']),station_metadata=stations,
                  runtime=runtime,future_selection=checks,
                  no_training_or_model_forward=True,no_future_inference=True,
                  selection_test_method='Real infer.py parser and selection function; filename-only store substitute. Real graph dimensions/times audited separately.',
                  checkpoint_limit='Strict weight loading and saved TRAIN statistics/threshold reuse are verified in source; this preparation does not run end-to-end checkpoint inference.',
                  source_evidence=evidence,
                  sha256={name:hashlib.sha256((REPO/name).read_bytes()).hexdigest() for name in ['train.py','train.sh','infer.py','emulator/data/graph_store.py','emulator/training/checkpoint_selection.py']})
    (args.out/'interface_validation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(f'PASS: {len(matrix["combinations"])} training CLI parses; {len(checks)} strict future selections; four station metadata records; TRAIN-only source provenance.', flush=True)


if __name__ == '__main__':
    main()
