import unittest
from collections import Counter
import numpy as np
from scripts.simulate_quality_uncertainty import (
    layout,generate,reference,statistic,jackknife_se,summarize,influence,EXPECTED,
)
from scripts.quality_uncertainty import estimate_uncertainty,draw_group_weights


class TestSimulation(unittest.TestCase):
    def test_layout_counts_and_sizes(self):
        for s,counts,g in [(1,[80]*3,60),(2,[86]*3,60),(3,[80]*3,60),
                           (4,[80]*3,60),(5,[80,80,20],45),(6,[156]*3,60),(7,[80]*3,20)]:
            x=layout(s)
            self.assertEqual([x['families'].count(c) for c in x['support']],counts)
            self.assertEqual(len(set(x['group_ids'])),g)
            self.assertEqual(len(set(x['sample_ids'])),sum(counts))
        self.assertEqual(sorted(Counter(layout(2)['group_ids']).values()),sorted(([1]*10+[2]*6+[8]*3+[40])*3))
        self.assertEqual(max(Counter(layout(6)['group_ids']).values()),80)

    def test_mixed_family_identity(self):
        x=layout(7)
        for g in set(x['group_ids']):
            self.assertEqual(Counter(c for c,h in zip(x['families'],x['group_ids']) if h==g),{'c0':4,'c1':4,'c2':4})

    def test_random_stream_exact_construction(self):
        # Independent reconstruction of baseline normals in declared draw order.
        rng=np.random.Generator(np.random.PCG64(np.random.SeedSequence([20261016,1,0])))
        gs=rng.normal(size=60);gv=rng.normal(size=(60,3))
        es=rng.normal(size=240);ev=rng.normal(size=(240,3))
        z=.5*(np.repeat(gs[:,None]+gv,4,axis=0)+es[:,None]+ev)
        mu=.3+.2*np.repeat(np.arange(3),80)[:,None]+.1*np.arange(3)
        expected=(mu+np.array([.4,.7,1])*z*z).T
        losses,_=generate(1,0)
        np.testing.assert_allclose(losses,expected,rtol=1e-14,atol=1e-14)
        np.testing.assert_array_equal(losses,generate(1,0)[0])
        self.assertFalse(np.array_equal(losses,generate(1,1)[0]))

    def test_hidden_block_construction(self):
        rng=np.random.Generator(np.random.PCG64(np.random.SeedSequence([20261016,4,0])))
        group=(rng.normal(size=60)[:,None]+rng.normal(size=(60,3)))*np.sqrt(.5)
        sample=(rng.normal(size=240)[:,None]+rng.normal(size=(240,3)))*np.sqrt(.5)
        hidden=(rng.normal(size=15)[:,None]+rng.normal(size=(15,3)))*np.sqrt(.5)
        c=np.sqrt(.75)*np.repeat(hidden,4,axis=0)+np.sqrt(.25)*group
        x=np.sqrt(.5)*(np.repeat(c,4,axis=0)+sample)
        mu=.3+.2*np.repeat(np.arange(3),80)[:,None]+.1*np.arange(3)
        np.testing.assert_allclose(generate(4,0)[0],(mu+np.array([.4,.7,1])*x*x).T,rtol=1e-14,atol=1e-14)
        # Variance coefficients sum to1; cross-group latent covariance .5*.75.
        self.assertAlmostEqual(.5*(.75+.25)+.5,1)
        self.assertAlmostEqual(.5*.75,.375)

    def test_all_scenario_reference_and_expectation(self):
        np.testing.assert_allclose(EXPECTED,np.mean(.3+.2*np.arange(3)[:,None]+.1*np.arange(3),axis=0)+[.4,.7,1])
        for s in range(1,8):
            loss,spec=generate(s,0);order=sorted(set(spec['group_ids']))
            draws=draw_group_weights(order,replicates=3)
            r=estimate_uncertainty(loss,**spec,draws=draws,draw_group_order=order)
            np.testing.assert_allclose(r['replicate_losses'],reference(loss,spec,draws,order),rtol=1e-12,atol=1e-12)

    def test_jackknife_against_reference(self):
        p=np.array([1.,2,4,8]);u=np.array([.5,1,1.5,2])
        vals=[]
        for i in range(4):
            pp=[p[j] for j in range(4) if j!=i];uu=[u[j] for j in range(4) if j!=i]
            sd=np.sqrt(sum((x-sum(pp)/3)**2 for x in pp)/2)
            vals.append([sd,(sum(uu)/3)/sd,(sum(x*x for x in uu)/3)/sd**2])
        vals=np.array(vals)
        expected=np.sqrt(.75*sum((row-vals.mean(axis=0))**2 for row in vals))
        np.testing.assert_allclose(jackknife_se(p,u),expected,rtol=1e-12,atol=1e-12)

    def test_summary_triggers_and_mc_precision(self):
        p=np.tile(np.arange(300,dtype=float)[:,None],(1,3))
        sd=p[:,0].std(ddof=1);u=np.full((300,3),sd*.5)
        rows=summarize(p,u,u[:30].copy())
        for row in rows:
            self.assertAlmostEqual(row['ratio_sd'],.5)
            self.assertAlmostEqual(row['ratio_variance'],.25)
            self.assertAlmostEqual(row['mean_loss_mcse'],sd/np.sqrt(300))
            self.assertTrue(row['ratio_review_trigger']);self.assertEqual(row['seed43_trigger_count'],0)
        alt=u[:30]*2
        self.assertTrue(all(r['seed43_trigger_count']==30 for r in summarize(p,u,alt)))

    def test_influence_fixed_support(self):
        spec=dict(families=['c0','c1','c2'],group_ids=['a','b','c'],support=['c0','c1','c2'])
        rows=influence(np.ones((3,3)),spec,np.ones(3))
        self.assertTrue(all(r['delta'] is None and len(r['missing_families'])==1 for r in rows))
        loss,spec=generate(7,0)
        point=np.mean([loss[:,np.array(spec['families'])==c].mean(axis=1) for c in spec['support']],axis=0)
        rows=influence(loss,spec,point)
        self.assertTrue(all(r['removed_samples']==12 and not r['missing_families'] for r in rows))

    def test_fail_closed(self):
        for s in (0,8,True):
            with self.assertRaises(ValueError): layout(s)
        with self.assertRaises(ValueError): generate(1,300)
        with self.assertRaises(FloatingPointError): statistic(np.ones(4),np.ones(4))
        with self.assertRaises(ValueError): summarize(np.ones((2,3)),np.ones((2,3)),np.ones((2,3)))


if __name__=='__main__': unittest.main()
