import unittest
import numpy as np
from scripts import multiseed_statistics as s


class TestMultiseedStatistics(unittest.TestCase):
    def test_mean_worst_before_seed_averaging(self):
        y=np.tile([0,1],6);m=np.repeat(np.arange(6),2);p=np.tile(y,(2,2,1))
        p[0,0,:2]=[1,0];p[1,0,2:4]=[1,0]
        result,counts=s.outcomes(y,p,m,[[0,1]]*6)
        self.assertEqual(counts,[2]*6)
        np.testing.assert_allclose(result[:,0,0],[5/6]*2);np.testing.assert_allclose(result[:,0,1],[0,0])
        self.assertEqual(result.mean(0)[0,1],0)

    def test_whole_groups_shared_months_deterministic_variable_size(self):
        groups=np.array(['a','a','b','c','c','c'])
        a=list(s.paired_draws(groups,5,replicates=20));b=list(s.paired_draws(groups,5,replicates=20))
        sizes=set()
        for (ss,gg,idx),other in zip(a,b):
            for x,z in zip((ss,gg,idx),other):np.testing.assert_array_equal(x,z)
            sizes.add(len(idx))
            for group_number,name in enumerate(['a','b','c']):
                multiplier=np.count_nonzero(gg==group_number)
                for member in np.flatnonzero(groups==name):self.assertEqual(np.count_nonzero(idx==member),multiplier)
        self.assertGreater(len(sizes),1)

    def test_missing_classes_keep_fixed_denominator(self):
        self.assertEqual(s.fixed_support_f1(np.array([0]),np.array([0]),[0,1]),.5)
        y=np.tile([0,1],6);m=np.repeat(np.arange(6),2);p=np.tile(y,(5,2,1))
        result,_=s.outcomes(y,p,m,[[0,1]]*6,indices=np.arange(0,12,2))
        np.testing.assert_allclose(result,.5)

    def test_pairing_and_no_model_independent_draws(self):
        y=np.tile([0,1],6);m=np.repeat(np.arange(6),2);p=np.tile(y,(5,2,1));groups=np.array(['g0','g1']*6)
        for ss,gg,idx in s.paired_draws(groups,5,replicates=10):
            result,_=s.outcomes(y,p,m,[[0,1]]*6,indices=idx,seeds=ss)
            np.testing.assert_array_equal(result[:,0]-result[:,1],0)

    def test_empty_month_and_corrupt_shapes_stop(self):
        y=np.tile([0,1],6);m=np.repeat(np.arange(6),2);p=np.tile(y,(5,2,1))
        with self.assertRaisesRegex(ValueError,'Empty'):s.outcomes(y,p,m,[[0,1]]*6,indices=np.arange(2,12))
        with self.assertRaises(ValueError):s.outcomes(y,p[:,:,:-1],m,[[0,1]]*6)
        with self.assertRaises(ValueError):s.percentile_intervals([0,np.nan])

    def test_percentile_reference_not_bootstrap_means(self):
        d=np.arange(100,dtype=float)
        result=s.percentile_intervals(d)
        np.testing.assert_allclose(result['pointwise95'],[2.475,96.525]);self.assertEqual(result['family_size'],10)


if __name__=='__main__':unittest.main()
