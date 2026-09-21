"""GT-only membership, tied bins, missing predictions, and physical branch accounting."""
from pathlib import Path
import sys
import unittest
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from pact_diagnostic_io import align
from peak_branch_diagnosis import event_rows, quantile_bins, safe_ratio


class FixedEventTests(unittest.TestCase):
    def test_predicted_maxima_cannot_change_events_and_missing_ids_error(self):
        y=np.array([[.3,.2],[.1,.05],[.2,.21]])
        targets=dict(tags=np.array(['a','b','c']),y_true=y,sample_id=np.arange(3))
        for p in (y*.1,y*10):
            r=event_rows('CBBT','S0',dict(y_true=y,y_pred=p,tags=targets['tags']),targets,.21,np.array([1]))
            self.assertEqual([e['window_id'] for e in r],['a'])
        with self.assertRaises(ValueError):
            align(dict(y_true=y[:2],y_pred=y[:2],tags=targets['tags'][:2]),targets['tags'])

    def test_ties_bins_fallback_ratios_and_branch_reconstruction(self):
        bins,edges=quantile_bins(np.arange(60.),20)
        self.assertEqual(len(edges)-1,3)
        self.assertEqual(np.bincount(bins)[1:].tolist(),[20,20,20])
        with self.assertRaises(ValueError): quantile_bins(np.ones(100),20)
        self.assertTrue(np.isnan(safe_ratio(1.,1e-12)))
        y=np.array([[.5,.4]])
        targets=dict(y_true=y,tags=np.array(['a']),sample_id=np.array([5]))
        arrays=dict(y_true=y,y_pred=np.array([[.4,.3]]),tags=targets['tags'],body_phys=np.array([[.2,.2]]),
                    excess_phys=np.array([[.4,.2]]),gate_probability=np.array([.5]),gate_logit=np.array([0.]))
        row=event_rows('CBBT','D0',arrays,targets,.25,np.array([1]))[0]
        self.assertAlmostEqual(row['absolute_underprediction'],.1)
        self.assertAlmostEqual(row['body_deficit']+row['raw_excess_deficit']+row['gate_attenuation'],.1)
        self.assertAlmostEqual(row['reconstruction_error'],0)
        self.assertEqual(row['h_gt'],0)


if __name__=='__main__': unittest.main()
