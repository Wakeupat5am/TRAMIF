import copy
import unittest
from unittest.mock import Mock
import numpy as np
from scripts import verify_final_multiseed as v


class TestFinalMultiseedVerification(unittest.TestCase):
    def test_independent_full_K_quality_and_imbalanced_support(self):
        rows=[dict(sha=str(i),family=c) for i,c in enumerate(['a','a','a','b','c'])]
        bundle=dict(logits=np.array([[3.,1.,0.]]*5),labels=np.array([0,0,0,1,2]),sample_identifiers=np.array([str(i) for i in range(5)]),class_names=np.array(['a','b','c']))
        q,nll,mask=v.independent_q(bundle,rows,{'a':0,'b':1,'c':2},['a','b'],2.)
        expected=np.log(np.exp([1.5,.5,0]).sum())
        self.assertAlmostEqual(q['class_balanced_nll'],expected-1)
        self.assertAlmostEqual(q['standard_nll_support'],expected-1.25)
        np.testing.assert_allclose(nll,[expected-1.5]*3+[expected-.5,expected])
        self.assertEqual(mask.tolist(),[True]*4+[False])
        bundle['sample_identifiers']=bundle['sample_identifiers'][::-1]
        with self.assertRaises(ValueError):v.independent_q(bundle,rows,{'a':0,'b':1,'c':2},['a','b'],2.)

    def test_weights_independent_reference_and_wrong_seed_signals(self):
        loss=np.array([1.,2,3]);d=np.array([.1,.2,.3]);q=np.exp(-loss);q/=q.sum();qd=np.exp(-loss-d);qd/=qd.sum()
        v.frozen_weights(loss,d,q,qd)
        with self.assertRaises(ValueError):v.frozen_weights(loss+.1*np.arange(3),d,q,qd)
        with self.assertRaises(ValueError):v.frozen_weights(loss,d,q[::-1],qd)

    def test_legacy_bundle_reordered_by_explicit_ID_join(self):
        rows=[dict(sha='first',family='a'),dict(sha='second',family='b')]
        bundle=dict(logits=np.array([[0.,3.],[3.,0.]]),labels=np.array([1,0]),sample_identifiers=np.array(['second','first']),class_names=np.array(['a','b']))
        aligned=v.align_bundle(bundle,rows,{'a':0,'b':1})
        self.assertEqual(aligned['labels'].tolist(),[0,1])
        np.testing.assert_array_equal(aligned['logits'],[[3,0],[0,3]])
        bundle['sample_identifiers']=np.array(['first','first'])
        with self.assertRaises(ValueError):v.align_bundle(bundle,rows,{'a':0,'b':1})

    def test_no_automatic_latest_on_missing_or_ambiguous_candidates(self):
        with self.assertRaises(ValueError):v.one([],'missing')
        with self.assertRaises(ValueError):v.one(['a','b'],'ambiguous')
        self.assertEqual(v.one(['a'],'unique'),'a')

    def test_all_five_seeds_sample_SD_and_missing_seed_stop(self):
        rows=[]
        for seed in v.SEEDS:
            for view in v.VIEWS:rows.append(dict(seed=seed,view=view,**{k:float(seed-42) for k in ('temperature','L','mmd2','d','Q_weight','QD_weight')}))
        summary=v.summarize(rows)
        self.assertEqual(len(summary),18)
        self.assertEqual(summary[0]['mean'],2.);self.assertAlmostEqual(summary[0]['SD'],np.sqrt(2.5))
        self.assertEqual(summary[0]['min'],0.);self.assertEqual(summary[0]['max'],4.)
        with self.assertRaises(ValueError):v.summarize(rows[:-1])

    def test_saved_MMD_algebra_no_kernel_recomputation(self):
        row=dict(mean_xx=.8,mean_yy=.7,mean_xy=.6,raw_mmd2=.3,mmd2=.3)
        v.verify_drift([row]*25,.3,.15)
        with self.assertRaises(ValueError):v.verify_drift([row]*25,.4,.2)
        with self.assertRaises(ValueError):v.verify_drift([dict(row,mean_xy=1.5)]*25,.3,.15)

    def test_legacy_technical_status_is_not_new_schema_PASS(self):
        v.legacy_drift_header(dict(status='PASS_TECHNICAL_COMPUTATION',null_standardization=False,future_access=False))
        with self.assertRaises(ValueError):v.legacy_drift_header(dict(status='PASS',null_standardization=False,future_access=False))

    def test_legacy_preparation_report_and_manifest_have_distinct_pins(self):
        evidence=Mock();evidence.js.side_effect=[dict(status='PASS',observed_mmd=False),{'artifact':'hash'}]
        result=v.legacy_preparation(evidence,dict(source_input_hashes={v.PREP+'/checksums.json':'manifest_pin'}))
        self.assertEqual(result,{'artifact':'hash'})
        self.assertEqual(evidence.js.call_args_list[0].args,(v.PREP+'/report.json',v.PREP_HASH))
        self.assertEqual(evidence.js.call_args_list[1].args,(v.PREP+'/checksums.json','manifest_pin'))

    def test_train_parameter_and_bandwidth_readback(self):
        reference=[dict(sha=f'{i:064x}',family='a',timestamp='2019-12-01 00:00:00+00:00',stage='train') for i in range(2)]
        canonical=v.train_reference(reference,case='Final',families=('a',));raw={reference[0]['sha']:np.array([0.,5]),reference[1]['sha']:np.array([2.,5])}
        x=np.array([raw[r['sha']] for r in canonical]);params=dict(mean=np.array([1.,5]),std=np.array([1.,0]),keep=np.array([True,False]))
        meta=dict(reference_sha256=v.sha256(v.encoded([[r['sha'],r['family'],r['timestamp'],r['stage']] for r in canonical])),embeddings_sha256=v.sha256(x.astype('<f8').tobytes()))
        pairs=[dict(family='a',left=reference[0]['sha'],right=reference[1]['sha'])]
        band=dict(pairs=pairs,distances=[4.],bandwidth_h=4.,**meta)
        v.verify_normalization(reference,np.array([[0.,5],[2.,5]]),{'a':0},params,meta,band,pairs)
        bad=copy.deepcopy(params);bad['mean'][0]=100
        with self.assertRaises(ValueError):v.verify_normalization(reference,np.array([[0.,5],[2.,5]]),{'a':0},bad,meta,band,pairs)


if __name__=='__main__':unittest.main()
