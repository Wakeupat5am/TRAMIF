import unittest
import numpy as np
from scripts.quality_uncertainty import estimate_uncertainty, draw_group_weights


class TestQualityUncertainty(unittest.TestCase):
    def setUp(self):
        self.loss=np.array([[1,3,2,6,4],[2,6,4,12,8],[2,4,3,7,5]],dtype=float)
        self.kw=dict(sample_ids=['a','b','c','d','e'],families=['x','x','y','y','y'],
                     group_ids=['shared','g','shared','h','h'],support=['x','y'])
        self.order=['g','h','shared']
        self.draws=np.array([[1,1,1],[2,4,1],[.5,1,3]],dtype=float)

    def run_estimate(self, **changes):
        args=dict(self.kw,draws=self.draws,draw_group_order=self.order)
        args.update(changes)
        return estimate_uncertainty(self.loss,**args)

    def test_independent_reference_and_ddof(self):
        result=self.run_estimate(); expected=[]
        for draw in self.draws:
            row=[]
            for v in range(3):
                means=[]
                for c in self.kw['support']:
                    indices=[i for i,f in enumerate(self.kw['families']) if f==c]
                    w=[draw[self.order.index(self.kw['group_ids'][i])] for i in indices]
                    means.append(sum(a*self.loss[v,i] for a,i in zip(w,indices))/sum(w))
                row.append(sum(means)/2)
            expected.append(row)
        np.testing.assert_allclose(result['replicate_losses'],expected,atol=1e-14,rtol=0)
        np.testing.assert_allclose(result['uncertainty'],np.std(expected,axis=0,ddof=1))
        np.testing.assert_allclose(result['point_loss'],expected[0])
        np.testing.assert_allclose(result['covariance'],np.cov(expected,rowvar=False,ddof=1))

    def test_constant_within_classes(self):
        self.loss=np.array([[2,2,4,4,4]]*3,dtype=float)
        result=self.run_estimate()
        np.testing.assert_allclose(result['point_loss'],3,atol=1e-14)
        np.testing.assert_allclose(result['uncertainty'],0,atol=1e-14)

    def test_paired_views_and_mixed_label_group(self):
        r=self.run_estimate()
        self.assertEqual(r['group_order'],tuple(self.order))
        np.testing.assert_allclose(r['replicate_losses'][:,1],2*r['replicate_losses'][:,0])
        np.testing.assert_allclose(r['replicate_losses'][:,2],r['replicate_losses'][:,0]+1)

    def test_replicate_scaling_invariance(self):
        r=self.run_estimate(); s=self.run_estimate(draws=self.draws*np.array([2,.1,10])[:,None])
        np.testing.assert_allclose(r['replicate_losses'],s['replicate_losses'],atol=1e-14)

    def test_seed_default_reproducibility(self):
        r=estimate_uncertainty(self.loss,**self.kw)
        s=estimate_uncertainty(self.loss,**self.kw)
        self.assertEqual(r['draws'].shape,(2000,3))
        np.testing.assert_array_equal(r['draws'],s['draws'])
        np.testing.assert_array_equal(r['uncertainty'],s['uncertainty'])
        np.testing.assert_array_equal(r['draws'],np.random.Generator(np.random.PCG64(42)).exponential(1,(2000,3)))

    def test_sample_reordering_and_outside_support(self):
        p=[4,0,3,1,2]
        kw={k:[v[i] for i in p] if k!='support' else v for k,v in self.kw.items()}
        r=estimate_uncertainty(self.loss[:,p],**kw,draws=self.draws,draw_group_order=self.order)
        np.testing.assert_allclose(r['replicate_losses'],self.run_estimate()['replicate_losses'])
        kw=dict(self.kw)
        for k,val in [('sample_ids','outside'),('families','z'),('group_ids','unused')]: kw[k]=kw[k]+[val]
        s=estimate_uncertainty(np.column_stack([self.loss,[999]*3]),**kw,draws=self.draws,draw_group_order=self.order)
        np.testing.assert_array_equal(r['group_order'],s['group_order'])
        np.testing.assert_allclose(s['point_loss'],self.run_estimate()['point_loss'])

    def test_fail_closed_identity_and_support(self):
        for changes in (dict(sample_ids=['a']*5),dict(support=['x','missing']),dict(support=['x','x']),
                        dict(group_ids=['']*5),dict(draw_group_order=self.order[::-1])):
            with self.assertRaises(ValueError): self.run_estimate(**changes)

    def test_fail_closed_draws(self):
        for draws in (np.zeros((3,3)),np.full((3,3),np.nan),np.full((3,3),-1),np.ones((1,3)),np.ones((3,2))):
            with self.assertRaises(ValueError): self.run_estimate(draws=draws)

    def test_fail_closed_losses_and_overflow(self):
        for value in (-1,np.nan,np.inf):
            self.loss[0,0]=value
            with self.assertRaises(ValueError): self.run_estimate()
        self.loss=np.full((3,5),np.finfo(float).max)
        with self.assertRaises(FloatingPointError): self.run_estimate()

    def test_generator_validation(self):
        for kw in (dict(replicates=1),dict(replicates=True),dict(seed=-1),dict(seed=True)):
            with self.assertRaises(ValueError): draw_group_weights(self.order,**kw)
        with self.assertRaises(ValueError): draw_group_weights(self.order[::-1])


if __name__=='__main__': unittest.main()
