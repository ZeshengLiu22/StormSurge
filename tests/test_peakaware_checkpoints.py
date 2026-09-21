"""Optional score, frozen references, artifact retention, and unchanged training trajectory."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

import train
from emulator.training.engine import EpochResult
from emulator.training.peakaware_checkpoints import (TERMS, REF_KEYS, PeakAwareTracker,
                                                     normalized_score, read_settings)
from test_checkpoint_pipeline import trajectory_run
from test_config_interfaces import dry_commands, REPO
from test_pipeline import make_fixture


def refs(path, station='Battery'):
    path.write_text(json.dumps({station:dict(zip(REF_KEYS,(1.,2.,4.))),
        '_meta':dict(split='val',metric_keys=dict(zip(REF_KEYS,TERMS)))}))
    return path


def metric(all_,top,peak, bias=-.1,under=.8):
    return dict(rmse_all=all_,mae_all=all_/2,rmse_peak5=top,mae_peak5=top/2,
        true_peak_rmse_top5=peak,true_peak_mae_top5=peak/2,true_peak_bias_top5=bias,
        true_peak_underprediction_fraction_top5=under,peak_timing_mae_steps_top5=1.)


class PeakAwareTests(unittest.TestCase):
    def test_future_config_shell_forwards_score_options_without_changing_lr(self):
        import subprocess
        import sys
        with tempfile.TemporaryDirectory() as d:
            output=Path(d)/'future.sh'
            parent=REPO/'experiment_config_0921_lr3e3/configs/train_config_0921_CBBT_D2_Amp_LR3e3.sh'
            subprocess.run([sys.executable,str(REPO/'tools/make_peakaware_config.py'),'--parent',str(parent),
                            '--output',str(output)],check=True,capture_output=True,text=True)
            before=vars(train.parse_args(dry_commands(parent)[0]))
            after=vars(train.parse_args(dry_commands(output)[0]))
            self.assertEqual(after['lr'],.003)
            self.assertEqual(after['checkpoint_selection'],'peakaware')
            self.assertEqual(after['track_peakaware'],1)
            for key in before.keys()|after.keys():
                if key in ('checkpoint_selection','track_peakaware','checkpoint_score_refs','run_tag','output_dir') or key.startswith('ckpt_score_w_'):
                    continue
                self.assertEqual(before[key],after[key],key)

    def test_defaults_match_frozen_parser_and_do_not_load_references(self):
        parsed=train.parse_args([])
        self.assertEqual(parsed.checkpoint_selection,'overall')
        self.assertFalse(hasattr(parsed,'track_peakaware'))
        self.assertFalse(hasattr(parsed,'checkpoint_score_refs'))
        with patch('pathlib.Path.read_bytes',side_effect=AssertionError('default must not open references')):
            train.parse_args(['--checkpoint_selection','overall'])

    def test_normalization_validation_only_and_frozen_file_errors(self):
        with tempfile.TemporaryDirectory() as d:
            path=refs(Path(d)/'refs.json')
            args=train.parse_args(['--station','Battery','--track_peakaware','1','--checkpoint_score_refs',str(path)])
            settings=read_settings(args)
            self.assertAlmostEqual(normalized_score(metric(1,2,4),settings),1)
            self.assertAlmostEqual(normalized_score(metric(2,2,2),settings),1.575)
            # Bias, Under%, arbitrary TEST metrics cannot influence this three-term score.
            self.assertEqual(normalized_score(dict(metric(1,2,4),test=1e99,true_peak_bias_top5=999,
                                                   true_peak_underprediction_fraction_top5=0),settings),1)
            for val in (None,float('nan'),float('inf'),-1):
                with self.assertRaises(ValueError): normalized_score(dict(metric(1,2,4),rmse_all=val),settings)
            for change in ({'split':'test'},{'metric_keys':{}}):
                doc=json.loads(path.read_text()); doc['_meta'].update(change); path.write_text(json.dumps(doc))
                with self.assertRaises(ValueError): read_settings(args)
                refs(path)
            for reference in (0,-1,float('nan')):
                doc=json.loads(path.read_text()); doc['Battery']['all_rmse']=reference; path.write_text(json.dumps(doc))
                with self.assertRaises(ValueError): read_settings(args)
                refs(path)
            args.ckpt_score_w_all=.6
            with self.assertRaises(ValueError): read_settings(args)

    def test_distinct_winners_ties_and_durable_original_overall(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); path=refs(root/'refs.json')
            args=train.parse_args(['--station','Battery','--track_peakaware','1','--checkpoint_score_refs',str(path)])
            tracker=PeakAwareTracker(root,read_settings(args))
            values=[metric(1.,2.,4.),metric(1.02,1.,2.),metric(1.02,1.,2.)]
            for epoch,v in enumerate(values,1):
                tracker.observe(epoch,v,lambda:dict(epoch=epoch,val=v,model_state={'marker':torch.tensor(epoch)}))
            self.assertEqual(tracker.best['overall']['epoch'],1)
            self.assertEqual(tracker.best['peakaware']['epoch'],2)
            self.assertGreater(tracker.degradation(),.01)
            self.assertTrue(tracker.exceeds_guardrail())
            for role,winner in [('overall',1),('peakaware',2)]:
                ckpt=torch.load(tracker.paths[role],weights_only=False)
                self.assertEqual(ckpt['epoch'],winner)
                self.assertEqual(ckpt['model_state']['marker'],winner)
                self.assertEqual(ckpt['peakaware_settings']['reference_sha256'],read_settings(args)['reference_sha256'])
            tracker.best['peakaware']['val']['rmse_all']=1.01
            self.assertFalse(tracker.exceeds_guardrail())

    def test_pipeline_retains_both_re_evaluates_reports_and_uses_val_only(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); torch.manual_seed(13)
            graphs,stations=make_fixture(root)
            path=refs(root/'refs.json')
            a,b=metric(1.,2.,4.),metric(1.02,1.,2.)
            import numpy as np
            arrays=dict(y_true=np.array([[1.,2.]],dtype='float32'),y_pred=np.array([[.9,1.8]],dtype='float32'),tags=np.array(['test']))
            for mode in ('overall','peakaware'):
                output=root/mode
                primary=a if mode=='overall' else b
                other=b if mode=='overall' else a
                fake_test=metric(999,999,999)
                sequence=[EpochResult(a),EpochResult(a),EpochResult(b),EpochResult(b),
                          EpochResult(primary),EpochResult(fake_test,arrays),EpochResult(other),EpochResult(a)]
                with patch.object(train,'run_epoch',side_effect=sequence) as run, \
                     patch.object(torch.optim.Adam,'step',side_effect=AssertionError('no experiment training')), \
                     patch.object(torch.optim.lr_scheduler.LambdaLR,'step'),contextlib.redirect_stdout(io.StringIO()):
                    train.main(['--root_dir',str(graphs),'--station','Battery','--station_json_dir',str(stations),
                        '--model','perceiver3','--head_type','single','--epochs','2','--device','cpu',
                        '--hidden_channels','16','--history_hours','12','--output_dir',str(output),
                        '--checkpoint_selection',mode,'--track_peakaware','1','--checkpoint_score_refs',str(path)])
                self.assertEqual(run.call_count,8)
                for call in run.call_args_list[4:]: self.assertNotIn('optimizer',call.kwargs)
                summary=json.loads(next(output.glob('summary_*.json')).read_text())
                self.assertEqual(summary['best_epoch'],1 if mode=='overall' else 2)
                self.assertEqual(torch.load(next(output.glob('best_*.pth')),weights_only=False)['epoch'],summary['best_epoch'])
                self.assertEqual(torch.load(output/'best_overall.pt',weights_only=False)['epoch'],1)
                self.assertEqual(torch.load(output/'best_peakaware.pt',weights_only=False)['epoch'],2)
                report=json.loads((output/'checkpoint_policy_comparison.json').read_text())
                self.assertTrue(report['selection_val_all_degradation_gt_1pct'])
                self.assertAlmostEqual(report['deltas'][0]['Delta_AllRMSE'],.02)
                self.assertEqual(summary['checkpoint_selection']['checkpoint_selection_mode'],mode)
                logged=[json.loads(s) for s in next(output.glob('metrics_*.jsonl')).read_text().splitlines()]
                self.assertAlmostEqual(logged[0]['peakaware_score'],1.)

    def test_optional_tracker_does_not_change_losses_rng_lr_or_weights(self):
        # A small existing CPU fixture checks actual forward/backward trajectories.
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); path=refs(root/'refs.json')
            (root/'baseline').mkdir(); (root/'tracked').mkdir()
            baseline=trajectory_run(root/'baseline','direct')
            original_main=train.main
            def with_tracker(argv):
                return original_main([*argv,'--track_peakaware','1','--checkpoint_score_refs',str(path)])
            with patch.object(train,'main',side_effect=with_tracker):
                actual=trajectory_run(root/'tracked','direct')
            self.assertEqual(baseline,actual)


if __name__=='__main__': unittest.main()
