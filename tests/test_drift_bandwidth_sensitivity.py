import math
import unittest
import numpy as np
from scripts.check_drift_bandwidth_sensitivity import contrasts, FACTORS
from scripts.descriptive_mmd import biased_rbf_mmd2


class TestBandwidthDiagnostic(unittest.TestCase):
    def test_fixed_factors_and_singleton_known_answer(self):
        self.assertEqual(FACTORS,(0.5,2.0))
        for factor in FACTORS:
            actual=biased_rbf_mmd2(np.array([[0.]]),np.array([[2.]]),bandwidth_h=factor)
            self.assertAlmostEqual(actual.mmd2,2-2*math.exp(-4/(2*factor)),places=14)

    def test_zero_primary_is_reported_not_divided(self):
        self.assertEqual(contrasts(0.,.1),dict(delta=.1,relative_change=None))
        self.assertAlmostEqual(contrasts(.2,.1)['relative_change'],-.5)
        with self.assertRaises(ValueError): contrasts(.1,float('nan'))


if __name__=='__main__': unittest.main()
