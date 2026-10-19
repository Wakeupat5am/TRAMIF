import unittest
import numpy as np
from scripts.prepare_e1_outcome import validate_outcome_rows,compare_fixed


class TestOutcomePreparation(unittest.TestCase):
    def rows(self):
        return [dict(sha='id',family='a',timestamp='2020-01-01 00:00:00+00:00',stage='C',decision='KEEP',threshold='0')]

    def bundles(self):
        return [dict(ids=np.array(['a','b']),classes=np.array(['x','y','absent']),labels=np.array([0,1]),
                     logits=np.array([[2.,0.,1.],[0.,3.,1.]])+v*np.array([0.,.1,.2]),temperature=1.+v/2) for v in range(3)]

    def test_frozen_dates_retention_mapping(self):
        validate_outcome_rows(self.rows(),{'a':0})
        for k,v in [('timestamp','2020-04-01 00:00:00+00:00'),('timestamp','2020-01-01 00:00:00'),('stage','B'),('decision','DROP'),('threshold','10'),('family','missing')]:
            with self.assertRaises(ValueError):validate_outcome_rows([dict(self.rows()[0],**{k:v})],{'a':0})
        with self.assertRaises(ValueError):validate_outcome_rows(self.rows()*2,{'a':0})

    def test_probability_mixture_and_metrics(self):
        names,logs,metrics=compare_fixed(self.bundles(),[.5,.2,.3])
        expected=sum(w*np.exp(logs[v]) for v,w in enumerate([.5,.2,.3]))
        np.testing.assert_allclose(np.exp(logs[4]),expected,atol=1e-14)
        np.testing.assert_allclose(np.exp(logs[3]),np.exp(logs[:3]).mean(axis=0),atol=1e-14)
        self.assertEqual(logs.shape,(5,2,3))
        self.assertEqual(metrics[names[-1]]['supported_classes'],2)
        self.assertAlmostEqual(metrics[names[-1]]['macro_f1_supported'],1.)
        self.assertAlmostEqual(metrics[names[-1]]['macro_f1_fixed'],2/3)

    def test_labels_do_not_change_mixture(self):
        bundles=self.bundles();before=compare_fixed(bundles,[.5,.2,.3])[1]
        for b in bundles:b['labels']=np.array([1,0])
        after=compare_fixed(bundles,[.5,.2,.3])[1]
        np.testing.assert_array_equal(before,after)

    def test_alignment_fail_closed(self):
        for key in ('ids','classes','labels'):
            bundles=self.bundles();bundles[1][key]=bundles[1][key][::-1]
            with self.assertRaises(ValueError):compare_fixed(bundles,[.5,.2,.3])
        bundles=self.bundles();bundles[1]['logits'][0,0]=np.nan
        with self.assertRaises(ValueError):compare_fixed(bundles,[.5,.2,.3])

    def test_uniform_weights_reproduce_uniform(self):
        _,logs,_=compare_fixed(self.bundles(),[1/3]*3)
        np.testing.assert_allclose(logs[3],logs[4],atol=1e-14)


if __name__=='__main__':unittest.main()
