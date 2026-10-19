import unittest
import numpy as np
from scripts.weighted_descriptive_mmd import group_weights, weighted_mmd2
from scripts.descriptive_mmd import biased_rbf_mmd2


class TestWeightedMMD(unittest.TestCase):
    def setUp(self):
        rng=np.random.default_rng(42)
        self.x=rng.normal(size=(7,3)); self.y=rng.normal(size=(5,3))

    def test_uniform_and_singletons(self):
        a=group_weights([str(i) for i in range(7)])
        b=group_weights([str(i) for i in range(5)])
        r=weighted_mmd2(self.x,self.y,a,b,bandwidth_h=1.2)
        self.assertAlmostEqual(r.mmd2,biased_rbf_mmd2(self.x,self.y,bandwidth_h=1.2).mmd2,places=14)

    def test_dense_reference_symmetry_bounds(self):
        a=group_weights(['a']*4+['b']*3); b=group_weights(['a']*2+['c']*3)
        def k(x,y):
            return np.array([[np.exp(-sum((u-v)**2)/3/1.2/2) for v in y] for u in x])
        expected=a@k(self.x,self.x)@a+b@k(self.y,self.y)@b-2*a@k(self.x,self.y)@b
        for block in (1,3,32):
            r=weighted_mmd2(self.x,self.y,a,b,bandwidth_h=1.2,block_size=block)
            self.assertAlmostEqual(r.mmd2,expected,places=14)
            self.assertGreaterEqual(r.mmd2,0); self.assertLessEqual(r.mmd2,2)
            self.assertAlmostEqual(r.mmd2,weighted_mmd2(self.y,self.x,b,a,bandwidth_h=1.2).mmd2,places=14)

    def test_duplicate_group_member_invariance(self):
        x=self.x[:2]; y=self.y[:2]
        r=weighted_mmd2(x,y,[.5,.5],[.5,.5],bandwidth_h=1)
        duplicated=np.array([x[0],x[0],x[1]])
        s=weighted_mmd2(duplicated,y,group_weights(['g','g','h']),[.5,.5],bandwidth_h=1)
        self.assertAlmostEqual(r.mmd2,s.mmd2,places=14)

    def test_identical_and_shared_group_ids(self):
        ids=['shared']*4+['other']*3
        a=group_weights(ids)
        self.assertAlmostEqual(a[:4].sum(),.5)
        self.assertEqual(ids,['shared']*4+['other']*3)
        self.assertAlmostEqual(weighted_mmd2(self.x,self.x,a,a,bandwidth_h=1).mmd2,0,places=14)

    def test_invalid_inputs(self):
        for a in ([1]*7, [np.nan]*7, [-1]+[1/3]*6, [.2]*5, ['x']*7):
            with self.assertRaises(ValueError): weighted_mmd2(self.x,self.y,a,[.2]*5,bandwidth_h=1)
        for h in (0,-1,np.nan):
            with self.assertRaises(ValueError): weighted_mmd2(self.x,self.y,[1/7]*7,[.2]*5,bandwidth_h=h)
        with self.assertRaises(ValueError): group_weights([])
        with self.assertRaises(ValueError): weighted_mmd2(self.x,self.y[:,:2],[1/7]*7,[.2]*5,bandwidth_h=1)


if __name__=='__main__': unittest.main()
