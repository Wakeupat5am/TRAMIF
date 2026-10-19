import unittest
from unittest.mock import patch
import numpy as np
from scripts.evaluate_historical_episode import forecast_scores,fixed_qd_weights,as_bundle
from scripts.freeze_qd_selection_protocol import forecasts
from scripts.prepare_e1_outcome import compare_fixed
from scripts.quality_fusion import weighted_fusion


class TestEpisodeEvaluation(unittest.TestCase):
    def test_forecast_support_and_reference(self):
        labels=np.array([0,0,0,1]);mapping={'a':0,'b':1,'absent':2}
        logp=np.log(np.array([[.8,.1,.1],[.8,.1,.1],[.8,.1,.1],[.1,.7,.2]]))
        logs=np.stack([logp]*3);pred=forecasts(np.ones((3,3)),[.1]*3)
        result=forecast_scores(logs,labels,mapping,['a','b','absent'],pred)
        self.assertEqual(result['S_eval'],['a','b']);self.assertEqual(result['missing_S'],['absent'])
        expected=np.array([[-np.log(.8),-np.log(.7)]]*3)
        np.testing.assert_allclose(result['observed_family_nll'],expected)
        self.assertAlmostEqual(result['candidate_mae'][0],np.abs(1-expected).mean())

    def test_e2_cannot_select(self):
        lp=np.log(np.array([[[.8,.2],[.2,.8]]]*3))
        with patch('scripts.evaluate_historical_episode.select_beta',side_effect=AssertionError('No E2 selection')):
            r=forecast_scores(lp,np.array([0,1]),{'a':0,'b':1},['a','b'],forecasts(np.ones((3,2)),[.1]*3),choose=False)
            self.assertIsNone(r['selection'])

    def test_zero_exact_reproduction(self):
        loss=np.array([.4,1.1,.9]);w=np.exp(-loss-loss.min()*0);w/=w.sum()
        qd=fixed_qd_weights(loss,[.1,.2,.3],0)
        # The frozen Q method uses the same stabilized softmax.
        expected=np.exp(-loss-(-loss).max());expected/=expected.sum()
        np.testing.assert_array_equal(qd,expected)
        logs=np.log(np.array([[[.8,.2]],[[.3,.7]],[[.4,.6]]]))
        np.testing.assert_array_equal(weighted_fusion(logs,qd),weighted_fusion(logs,expected))

    def test_nonzero_weights_reference(self):
        loss=np.array([.4,1.1,.9]);d=np.array([.1,.2,.3])
        expected=np.exp(-loss-.5*d);expected/=expected.sum()
        np.testing.assert_allclose(fixed_qd_weights(loss,d,.5),expected,rtol=0,atol=1e-15)
        with self.assertRaises(ValueError):fixed_qd_weights(loss,d,2.)

    def test_id_join_and_class_guard(self):
        bundle=dict(sample_identifiers=np.array(['b','a']),class_names=np.array(['x','y']),
                    labels=np.array([1,0]),logits=np.array([[0.,1.],[2.,0.]]))
        rows=[dict(sha='a',family='x'),dict(sha='b',family='y')]
        b=as_bundle(bundle,rows,{'x':0,'y':1},1.)
        self.assertEqual(b['ids'].tolist(),['a','b'])
        with self.assertRaises(ValueError):as_bundle(bundle,rows,{'y':0,'x':1},1.)

    def test_all_five_baseline_outputs(self):
        b=dict(ids=np.array(['a']),classes=np.array(['x','y']),labels=np.array([0]),logits=np.array([[2.,0.]]),temperature=1.)
        names,logs,metrics=compare_fixed([b,b,b],[.5,.2,.3])
        self.assertEqual(len(names),5);self.assertEqual(logs.shape,(5,1,2))
        self.assertEqual(metrics[names[-1]]['accuracy'],1.)


if __name__=='__main__':unittest.main()
