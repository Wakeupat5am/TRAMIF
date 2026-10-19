import copy
import json
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np
from scripts import process_final_seed as p


def row(i,c,stage):
    return dict(sha=f'{i:064x}',family=c,stage=stage,timestamp={'train':'2019-12','A':'2020-02','B':'2020-03'}[stage]+'-15 00:00:00+00:00',retained='True',group_id=f'g{i}')


def fixture():
    mapping={'a':0,'b':1,'c':2};support=['a','b']
    reference=[row(i+1,c,'train') for i,c in enumerate(['a','a','b','b','c','c'])]
    stages={s:[row(i+100+(s=='B')*100,c,s) for i,c in enumerate(['a','a','b','b','c'])] for s in ('A','B')}
    bundles={s:dict(logits=np.array([[1.,0,0]]*5),labels=np.array([mapping[r['family']] for r in rr]),sample_identifiers=np.array([r['sha'] for r in rr]),class_names=np.array(list(mapping))) for s,rr in stages.items()}
    matched={s:rr[:4] for s,rr in stages.items()}
    ref=np.array([[0.,0],[1,2],[2,1],[3,5],[4,3],[5,8]])
    ab={'A':np.array([[0.,0],[1,2],[2,1],[3,5]]),'B':np.array([[1.,0],[2,2],[3,1],[4,5]])}
    return stages,mapping,support,bundles,reference,ref,matched,ab


class TestFinalSeedProcessing(unittest.TestCase):
    def test_execution_guards_before_checkpoint_reads(self):
        with patch.object(p.trainer,'discover',side_effect=AssertionError('no reads')):
            with self.assertRaises(PermissionError):p.prepare_view(43,'sbsmi')
            with self.assertRaises(PermissionError):p.finalize(43,[],[])

    def test_binding_rejects_cross_seed_view_or_checkpoint(self):
        binding=dict(seed=43,view='sbsmi',checkpoint_sha256='a')
        for change in (dict(seed=42),dict(view='entropy'),dict(checkpoint_sha256='b')):
            with self.assertRaises(ValueError):p.binding_check(dict(binding,**change),binding)

    def test_composed_reference_and_determinism_A_only_temperature(self):
        args=fixture();calls=[]
        def fit(logits,labels):calls.append(labels.tolist());return 2.,False
        a=p.compute_view(*args,fitter=fit);b=p.compute_view(*args,fitter=fit)
        self.assertEqual(a[0],b[0]);np.testing.assert_array_equal(a[1],b[1])
        self.assertEqual(calls,[[0,0,1,1,2]]*2)
        self.assertEqual(a[0]['quality']['B_support_samples'],4)
        self.assertEqual(a[0]['quality']['B_retained_samples'],5)
        self.assertEqual(a[0]['quality']['K'],3)
        self.assertGreater(a[0]['d'],0)
        logits=args[3]['B']['logits']/2
        expected=np.log(np.exp(logits).sum(1))-logits[np.arange(5),[0,0,1,1,2]]
        np.testing.assert_allclose(a[1],expected)

    def test_A_B_cannot_affect_train_statistics(self):
        args=fixture();a=p.compute_view(*args,fitter=lambda z,y:(1.,False))
        changed=copy.deepcopy(args);changed[-1]['B']+=100
        b=p.compute_view(*changed,fitter=lambda z,y:(1.,False))
        np.testing.assert_array_equal(a[3].mean,b[3].mean);self.assertEqual(a[4],b[4])
        self.assertNotEqual(a[0]['d'],b[0]['d'])

    def test_alignment_nonfinite_and_B_temperature_guard(self):
        args=fixture()
        for key in ('sample_identifiers','class_names'):
            bad=copy.deepcopy(args);bad[3]['B'][key]=bad[3]['B'][key][::-1]
            with self.assertRaises(ValueError):p.compute_view(*bad,fitter=lambda z,y:(1.,False))
        bad=copy.deepcopy(args);bad[-1]['B'][0,0]=np.nan
        with self.assertRaises(ValueError):p.compute_view(*bad,fitter=lambda z,y:(1.,False))
        with self.assertRaises(ValueError):p.fit_on_a(args[3]['B'],args[0]['B'],args[1])

    def test_three_view_receipt_freeze_integration_and_content_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(p,'ROOT',Path(tmp)),patch.object(p.trainer,'ROOT',Path(tmp)),patch.object(p,'code_hashes',return_value={'synthetic':'same'}),patch.object(p.trainer,'verify_run',return_value=(None,None,{'selected':{'sha256':'cp'}})),patch.object(p.trainer,'discover',side_effect=lambda seed,view:dict(run_dir=view,complete_sha256='complete')):
            paths=[];digests=[]
            for i,view in enumerate(p.VIEWS):
                folder=Path(tmp)/p.OUTPUT/f'seed_43/{view}/synthetic';folder.mkdir(parents=True)
                binding=dict(seed=43,view=view,checkpoint_sha256='cp',run_dir=view,complete_sha256='complete')
                files={'statistics.json':dict(binding=binding,results=dict(quality=dict(class_balanced_nll=float(i+1)),d=.01*i)),
                       'selection.json':dict(ids=['a','b'],content=['hashA','hashB'],mapping={'a':0,'b':1})}
                for name,obj in files.items():p.trainer.write_new_json(folder/name,obj)
                receipt=dict(status='PASS',binding=binding,source_sha256={'synthetic':'same'},files={name:p.sha256((folder/name).read_bytes()) for name in files})
                p.trainer.write_new_json(folder/'receipt.json',receipt);paths.append(str(folder/'receipt.json'));digests.append(p.sha256((folder/'receipt.json').read_bytes()))
            result=p.finalize(43,paths,digests,True);weights=json.loads((Path(tmp)/result['weights']).read_bytes())
            self.assertEqual(weights['seed'],43);self.assertEqual(weights['beta'],1);self.assertEqual(weights['gamma'],0)
            self.assertAlmostEqual(sum(weights['Q']),1);self.assertAlmostEqual(sum(weights['QD']),1)
            folder=Path(paths[0]).parent
            (folder/'selection.json').write_bytes(p.json_bytes(dict(ids=['a','b'],content=['WRONG','hashB'],mapping={'a':0,'b':1})))
            receipt=json.loads(Path(paths[0]).read_bytes());receipt['files']['selection.json']=p.sha256((folder/'selection.json').read_bytes())
            Path(paths[0]).write_bytes(p.json_bytes(receipt));digests[0]=p.sha256(Path(paths[0]).read_bytes())
            with self.assertRaisesRegex(ValueError,'alignment'):p.finalize(43,paths,digests,True)


if __name__=='__main__':unittest.main()
