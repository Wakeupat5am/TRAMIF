import unittest
import numpy as np
import torch
import tempfile
from pathlib import Path
from unittest.mock import patch
from scripts import prepare_historical_logits as preparation
from scripts.prepare_historical_logits import validate_rows,validate_bundle,fit_on_a,infer_logits
from scripts.frozen_encoder_extraction import state_hash


class TestHistoricalLogits(unittest.TestCase):
    def setUp(self):
        self.mapping={'a':0,'b':1}
        self.rows=[dict(sha='id1',family='b',timestamp='2019-11-01 00:00:00+00:00',stage='A',retained='True',group_id='g')]
        self.bundle=dict(logits=np.array([[1.,2.]],dtype=np.float32),labels=np.array([1]),sample_identifiers=np.array(['id1']),class_names=np.array(['a','b']))

    def test_valid_identity_and_episode_labels(self):
        validate_rows(self.rows,'E1','A',self.mapping)
        validate_bundle(self.bundle,self.rows,self.mapping)
        self.bundle['labels']=np.array([50])
        with self.assertRaises(ValueError):validate_bundle(self.bundle,self.rows,self.mapping)

    def test_forbidden_dates_stages_retention(self):
        for field,value in [('timestamp','2020-04-01 00:00:00+00:00'),('timestamp','2019-11-01 00:00:00'),('stage','C'),('retained','False'),('group_id','')]:
            rows=[dict(self.rows[0],**{field:value})]
            with self.assertRaises(ValueError):validate_rows(rows,'E1','A',self.mapping)

    def test_alignment_failure(self):
        for key,value in [('sample_identifiers',np.array(['wrong'])),('class_names',np.array(['b','a'])),('logits',np.array([[np.nan,0]]))]:
            with self.assertRaises(ValueError):validate_bundle(dict(self.bundle,**{key:value}),self.rows,self.mapping)

    def test_calibration_only_a(self):
        calls=[]
        def fitter(z,y): calls.append((z.numpy(),y.numpy()));return 1.5,False
        r=fit_on_a(self.bundle,self.rows,self.mapping,fitter)
        self.assertEqual(r['temperature'],1.5);self.assertEqual(len(calls),1)
        np.testing.assert_array_equal(calls[0][1],[1])
        with self.assertRaises(ValueError):fit_on_a(self.bundle,[dict(self.rows[0],stage='B')],self.mapping,fitter)
        self.assertEqual(len(calls),1)

    def test_synthetic_forward_state_and_shapes(self):
        torch.manual_seed(42)
        for k in (32,42,51):
            model=torch.nn.Sequential(torch.nn.BatchNorm2d(1),torch.nn.Flatten(),torch.nn.Linear(4096,k)).eval().requires_grad_(False)
            before=state_hash(model);batch=torch.zeros(2,1,64,64)
            a=infer_logits(model,batch,k);b=infer_logits(model,batch,k)
            self.assertEqual(a.shape,(2,k));np.testing.assert_array_equal(a,b)
            self.assertEqual(before,state_hash(model));self.assertTrue(all(p.grad is None for p in model.parameters()))
            model.train()
            with self.assertRaises(ValueError):infer_logits(model,batch,k)

    def test_prepare_synthetic_files_and_a_only_fitting(self):
        a=dict(self.rows[0],content_sha256='content-a')
        b=dict(a,sha='id2',stage='B',timestamp='2019-12-01 00:00:00+00:00',content_sha256='content-b')
        stages={'A':[a],'B':[b]};fit_calls=[]
        def loader(*args):
            model=torch.nn.Sequential(torch.nn.Flatten(),torch.nn.Linear(4096,2)).eval().requires_grad_(False)
            return model,{'K':2,'sha256':'synthetic'}
        def decoder(row,*args):return torch.zeros(1,64,64),row['content_sha256']
        def fit(bundle,rows,mapping):
            fit_calls.append([r['sha'] for r in rows])
            return {'temperature':1.,'fit_ids':fit_calls[-1]}
        with tempfile.TemporaryDirectory() as folder, \
             patch.object(preparation,'selection',return_value=(stages,self.mapping,self.mapping,['b'])), \
             patch.object(preparation,'load_encoder',side_effect=loader), \
             patch.object(preparation,'image',side_effect=decoder), \
             patch.object(preparation,'fit_on_a',side_effect=fit):
            result=preparation.prepare('E1',None,Path(folder))
            self.assertEqual(fit_calls,[['id1']]*3)
            self.assertEqual(len(result),3)
            for view in preparation.VIEWS:
                for stage in ('A','B'):
                    with np.load(Path(folder)/view/(stage+'.npz'),allow_pickle=False) as arrays:
                        validate_bundle(arrays,stages[stage],self.mapping)
        with self.assertRaises(ValueError):preparation.prepare('Final',None,Path('.'))


if __name__=='__main__':unittest.main()
