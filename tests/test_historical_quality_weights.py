import unittest
import numpy as np
from scripts.freeze_historical_quality_weights import quality_weights,VIEWS


class TestHistoricalWeights(unittest.TestCase):
    def test_uniform(self):
        np.testing.assert_array_equal(quality_weights(dict.fromkeys(VIEWS,1.)),np.full(3,1/3))

    def test_reference_monotonic_and_named_alignment(self):
        values=dict(zip(VIEWS,[.4,1.2,.9]))
        w=quality_weights(dict(reversed(list(values.items()))))
        ref=np.exp(-np.array([.4,1.2,.9]));ref/=ref.sum()
        np.testing.assert_allclose(w,ref,atol=1e-15,rtol=0)
        self.assertGreater(w[0],w[2]);self.assertGreater(w[2],w[1])

    def test_additive_shift_invariance(self):
        a=dict(zip(VIEWS,[.4,1.2,.9]));b={k:v+100 for k,v in a.items()}
        np.testing.assert_allclose(quality_weights(a),quality_weights(b),atol=1e-14,rtol=0)

    def test_extremes_finite_no_weight_floor(self):
        w=quality_weights(dict(zip(VIEWS,[0.,1e6,1e6])))
        np.testing.assert_array_equal(w,[1,0,0])

    def test_fail_closed(self):
        for values in ({},dict.fromkeys(VIEWS[:2],1.),dict.fromkeys(list(VIEWS)+['extra'],1.)):
            with self.assertRaises(ValueError):quality_weights(values)
        for value in (np.nan,np.inf,-1.,True,'1'):
            bad=dict.fromkeys(VIEWS,1.);bad[VIEWS[0]]=value
            with self.assertRaises(ValueError):quality_weights(bad)


if __name__=='__main__':unittest.main()
