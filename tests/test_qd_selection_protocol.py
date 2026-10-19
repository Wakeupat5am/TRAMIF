import unittest
import numpy as np
from scripts.freeze_qd_selection_protocol import forecasts,select_beta


class TestSelectionProtocol(unittest.TestCase):
    def test_forecast_reference_and_zero(self):
        loss=np.array([[1.,2],[2,3],[3,4]]);d=np.array([.1,.2,.3])
        p=forecasts(loss,d)
        np.testing.assert_array_equal(p[0],loss)
        np.testing.assert_allclose(p[1],loss+.5*d[:,None])
        np.testing.assert_allclose(p[2],loss+d[:,None])

    def test_selector_and_equal_family_view_weight(self):
        p=forecasts(np.ones((3,2)),[.2,.4,.6]);y=p[1].copy()
        r=select_beta(p,y)
        self.assertEqual(r['beta'],.5);self.assertEqual(r['mae'][1],0)
        self.assertAlmostEqual(r['mae'][0],.2)

    def test_zero_winner_and_tie(self):
        p=forecasts(np.ones((3,2)),[0,0,0])
        self.assertEqual(select_beta(p,np.ones((3,2)))['beta'],0)
        p=forecasts(np.ones((3,2)),[.1,.1,.1])
        self.assertEqual(select_beta(p,np.zeros((3,2)))['beta'],0)
        # All scores within1e-12 of minimum: prefer simpler beta0.
        p=forecasts(np.ones((3,2)),[1e-13]*3)
        self.assertEqual(select_beta(p,p[2])['beta'],0)

    def test_view_family_permutation_preserves_mae(self):
        p=forecasts(np.arange(6.).reshape(3,2),[.1,.2,.3]);y=p[2]+.1
        a=select_beta(p,y);b=select_beta(p[:,::-1,::-1],y[::-1,::-1])
        np.testing.assert_allclose(a['mae'],b['mae']);self.assertEqual(a['beta'],b['beta'])

    def test_invalid_inputs(self):
        for d in ([np.nan,0,0],[-.1,0,0],[2,0,0],[0,0]):
            with self.assertRaises(ValueError):forecasts(np.ones((3,2)),d)
        for loss in (np.ones((2,2)),np.ones((3,0)),np.full((3,2),np.inf)):
            with self.assertRaises(ValueError):forecasts(loss,[0,0,0])
        p=forecasts(np.ones((3,2)),[0,0,0])
        with self.assertRaises(ValueError):select_beta(p,np.ones((3,1)))
        with self.assertRaises(ValueError):select_beta(p,np.full((3,2),np.nan))


if __name__=='__main__':unittest.main()
