import unittest
import numpy as np
from scripts.compute_historical_quality import quality,canonical_bundle


class TestHistoricalQuality(unittest.TestCase):
    def setUp(self):
        self.mapping={'a':0,'b':1,'outside':2}
        self.rows=[dict(sha=f'id{i}',family=c) for i,c in enumerate(['a','a','a','b','outside'])]
        self.bundle=dict(logits=np.array([[2.,0,1],[0,1,2],[1,0,2],[0,2,1],[1,2,0]]),
                         labels=np.array([0,0,0,1,2]),sample_identifiers=np.array([r['sha'] for r in self.rows]),
                         class_names=np.array(list(self.mapping)))

    def test_stable_temperature_and_full_k(self):
        b=dict(self.bundle,logits=self.bundle['logits']+10000)
        result,losses,_,_=quality(b,self.rows,self.mapping,['a','b'],2.)
        expected=[]
        for z,y in zip(self.bundle['logits'],self.bundle['labels']):
            z=z/2
            expected.append(np.log(sum(np.exp(z)))-z[y])
        np.testing.assert_allclose(losses,expected,atol=1e-12,rtol=0)
        self.assertEqual(result['K'],3);self.assertEqual(result['support_families'],2)
        # Outside-support output remains in the softmax denominator.
        z=self.bundle['logits'][0]/2
        self.assertNotAlmostEqual(losses[0],np.log(sum(np.exp(z[:2])))-z[0])

    def test_balanced_and_sample_weighted_estimands(self):
        result,losses,mask,_=quality(self.bundle,self.rows,self.mapping,['a','b'],1.)
        expected=(losses[:3].mean()+losses[3])/2
        self.assertAlmostEqual(result['class_balanced_nll'],expected)
        self.assertAlmostEqual(result['Q'],-expected)
        self.assertAlmostEqual(result['standard_nll_support'],losses[:4].mean())
        self.assertAlmostEqual(result['standard_nll_all_B'],losses.mean())
        self.assertNotAlmostEqual(result['class_balanced_nll'],result['standard_nll_support'])
        self.assertEqual(mask.tolist(),[True]*4+[False])

    def test_explicit_id_join(self):
        order=[4,2,0,3,1]
        permuted={k:(v if k=='class_names' else v[order]) for k,v in self.bundle.items()}
        a,indices=canonical_bundle(permuted,self.rows,self.mapping)
        np.testing.assert_array_equal(a['logits'],self.bundle['logits'])
        np.testing.assert_array_equal(indices,[2,4,1,3,0])

    def test_fail_closed_ids_classes_labels(self):
        for key,arr in [('sample_identifiers',np.array(['id0']*5)),('sample_identifiers',np.array(['missing']+[f'id{i}' for i in range(1,5)])),
                        ('class_names',np.array(['b','a','outside'])),('labels',np.array([1,0,0,1,2]))]:
            with self.assertRaises(ValueError):quality(dict(self.bundle,**{key:arr}),self.rows,self.mapping,['a','b'],1.)
        missing={k:(v if k=='class_names' else v[:-1]) for k,v in self.bundle.items()}
        with self.assertRaises(ValueError):canonical_bundle(missing,self.rows,self.mapping)

    def test_fail_closed_nonfinite_support_temperature(self):
        for value in (np.nan,np.inf,-np.inf):
            bad=self.bundle['logits'].copy();bad[0,0]=value
            with self.assertRaises(ValueError):quality(dict(self.bundle,logits=bad),self.rows,self.mapping,['a','b'],1.)
        for t in (0.,np.nan,21.,True):
            with self.assertRaises(ValueError):quality(self.bundle,self.rows,self.mapping,['a','b'],t)
        for s in ([],['a','a'],['missing']):
            with self.assertRaises(ValueError):quality(self.bundle,self.rows,self.mapping,s,1.)
        rows=[dict(r,family='a') for r in self.rows]
        bundle=dict(self.bundle,labels=np.zeros(5,dtype=np.int64))
        with self.assertRaises(ValueError):quality(bundle,rows,self.mapping,['a','b'],1.)


if __name__=='__main__':unittest.main()
