from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import zipfile
from scripts.build_combined_freeze import BoundInputs, compose_package, digest, safe_relative
from scripts.backup_combined_freeze import restore_archive
from scripts.final_multiseed_evaluator import SEEDS,VIEWS
from tests.test_d060_integration import synthetic_package


def fixtures():
    p=synthetic_package();summary=dict(status='ARTIFACT_VERIFIED',seeds=list(SEEDS),rows=[],checkpoints=[]);learned={}
    for seed in SEEDS:
        record=p['seed_records'][str(seed)];pv={}
        for i,v in enumerate(VIEWS):
            cp={**record['views'][v],'status':'ARTIFACT_VERIFIED'};summary['checkpoints'].append(cp)
            receipt=f'data/derived/synthetic/{seed}/{v}/receipt.json'
            summary['rows'].append(dict(seed=seed,view=v,L=record['L'][i],d=record['d'][i],
                Q_weight=record['weights']['Q'][i],QD_weight=record['weights']['QD'][i],
                checkpoint_status='ARTIFACT_VERIFIED',processing_status='ARTIFACT_VERIFIED',
                temperature=cp['temperature'],processing_receipt=receipt,
                normalizer_sha256='0'*64,reference_embeddings_sha256='1'*64))
            pv[v]=dict(checkpoint_sha256=cp['checkpoint_sha256'],temperature=cp['temperature'],processing_receipt=receipt)
        learned[seed]=dict(seed=seed,K=51,view_order=list(VIEWS),weights_registered=True,fit_samples=4064,
                           fit=dict(status='CONVERGED_VERIFIED',weights=record['weights']['learned_static']),provenance=pv)
    return p,summary,learned


class TestCombinedFreeze(unittest.TestCase):
    def test_d054_summary_uses_artifact_verified_not_root_receipt_status(self):
        p,s,l=fixtures()
        self.assertEqual(compose_package(s,l,p['class_to_idx'],p['size_stratification'])['K'],51)
        s['status']='PASS'
        with self.assertRaisesRegex(ValueError,'D054 completeness'):
            compose_package(s,l,p['class_to_idx'],p['size_stratification'])

    def test_compose_exact_five_seed_binding(self):
        p,s,l=fixtures();result=compose_package(s,l,p['class_to_idx'],p['size_stratification'])
        self.assertEqual(set(result['seed_records']),set(map(str,SEEDS)))
        self.assertFalse(result['Future_execution_authorized'])
        self.assertEqual(sum(len(x['views']) for x in result['seed_records'].values()),15)
        for seed in SEEDS:
            self.assertEqual(result['seed_records'][str(seed)]['weights'],p['seed_records'][str(seed)]['weights'])

    def test_cross_seed_learned_provenance_rejected(self):
        p,s,l=fixtures();l[43]['provenance']['sbsmi']['checkpoint_sha256']=l[42]['provenance']['sbsmi']['checkpoint_sha256']
        with self.assertRaisesRegex(ValueError,'Cross-seed'):compose_package(s,l,p['class_to_idx'],p['size_stratification'])

    def test_wrong_temperature_or_incomplete_fit_rejected(self):
        p,s,l=fixtures();l[43]['provenance']['entropy']['temperature']=99
        with self.assertRaises(ValueError):compose_package(s,l,p['class_to_idx'],p['size_stratification'])
        p,s,l=fixtures();l[44]['weights_registered']=False
        with self.assertRaises(ValueError):compose_package(s,l,p['class_to_idx'],p['size_stratification'])

    def test_missing_and_ambiguous_checkpoints_rejected(self):
        for missing in (True,False):
            p,s,l=fixtures()
            if missing:s['checkpoints'].pop()
            else:s['checkpoints'][0]=deepcopy(s['checkpoints'][1])
            with self.assertRaises(ValueError):compose_package(s,l,p['class_to_idx'],p['size_stratification'])

    def test_paths_reject_future_raw_escape_before_read(self):
        for name in ['../secret','C:/secret','/secret','x\\y','data/raw/file',
                     'data/derived/future_inputs/a.json','data/manifests/future.csv']:
            with self.subTest(name=name),self.assertRaises(ValueError):safe_relative(name)
        safe_relative('scripts/evaluate_frozen_future.py') # Code text is not Future data.

    def test_bound_inputs_tamper_and_conflicting_pins(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'a.json';p.write_text('{}');h=digest(p);e=BoundInputs(tmp)
            self.assertEqual(e.js('a.json',h),{})
            with self.assertRaises(ValueError):e.add('a.json','0'*64)
            p.write_text('{"changed":true}')
            with self.assertRaisesRegex(ValueError,'Hash'):e.read('a.json')

    def test_exact_archive_restore_hashes(self):
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp);archive=base/'a.zip';raw=b'checkpoint bytes: synthetic only'
            with zipfile.ZipFile(archive,'x') as z:z.writestr('data/derived/synthetic/epoch_100.pt',raw)
            manifest={'data/derived/synthetic/epoch_100.pt':hashlib.sha256(raw).hexdigest()}
            restore_archive(archive,base/'restore',manifest)
            self.assertEqual((base/'restore/data/derived/synthetic/epoch_100.pt').read_bytes(),raw)

    def test_corrupt_or_escaping_archive_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp)
            for i,name in enumerate(['../escape','ok.json']):
                archive=base/f'{i}.zip'
                with zipfile.ZipFile(archive,'x') as z:z.writestr(name,b'bad bytes')
                with self.assertRaises(ValueError):restore_archive(archive,base/f'out{i}',{name:'0'*64})


if __name__=='__main__':unittest.main()
