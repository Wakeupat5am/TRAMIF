import ast
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch
import tlsh
from models.malsbslcnet import MalSBSLCNet
from scripts.prepare_protected_future import (discover,inventory_one,retain,make_images,validate_image,
    infer_frozen,model_state_hash,MONTHS,VIEWS)
from scripts.verify_protected_future import check_logit_array,check_witnesses
from scripts.final_multiseed_evaluator import retention_decisions


def mapping():return {f'c{i:02d}':i for i in range(51)}


def row(number,month='2020-04',family='c00'):
    return dict(sha=f'{number:064x}',timestamp=month+'-15T00:00:00+00:00',family=family)


def fingerprint_row(number,month,status='TNULL',value='TNULL',content=None):
    r=row(number,month);r.update(month=month,split='future',tlsh_status=status,tlsh=value,
                                 disarmed_content_sha256=f'{number if content is None else content:064x}')
    return r


class TestProtectedFuture(unittest.TestCase):
    def test_discovery_known_unknown_benign_and_all_six_months(self):
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp);rows=[row(i+1,m) for i,m in enumerate(MONTHS)]
            rows += [row(50,family='new_family'),row(51,family='')]
            for r in rows[:-1]:(folder/(r['sha']+'.exe')).write_bytes(b'synthetic')
            result=discover(rows,mapping(),folder)
            self.assertEqual(len(result),8)
            self.assertEqual({r['kind'] for r in result},{'known_family','unknown_family','benign'})
            self.assertIsNone(next(r for r in result if r['kind']=='benign')['file_size'])

    def test_discovery_missing_known_or_other_malware_stops(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(ValueError,'Missing malware'):discover([row(1)],mapping(),Path(temp))
            with self.assertRaisesRegex(ValueError,'Missing malware'):discover([row(1,family='new')],mapping(),Path(temp))

    def test_outside_study_naive_benign_does_not_relax_future_utc(self):
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp);rows=[row(i+1,m) for i,m in enumerate(MONTHS)]
            for r in rows:(folder/(r['sha']+'.exe')).write_bytes(b'synthetic')
            outside={**row(99,family=''),'sha':'VirusShare_outside_study','timestamp':'2016-10-30 15:59:55'}
            self.assertEqual(len(discover([outside]+rows,mapping(),folder)),6)
            rows[0]['timestamp']='2020-04-15 00:00:00'
            with self.assertRaisesRegex(ValueError,'UTC'):discover(rows,mapping(),folder)

    def test_duplicate_or_invalid_metadata_stops(self):
        with tempfile.TemporaryDirectory() as temp:
            benign=row(1,family='')
            with self.assertRaisesRegex(ValueError,'Duplicate'):discover([benign,benign],mapping(),Path(temp))
            bad={**benign,'sha':'../unsafe'}
            with self.assertRaisesRegex(ValueError,'identifier'):discover([bad],mapping(),Path(temp))

    def test_inventory_reads_same_bytes_and_retains_digest(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);folder=root/'data/raw/altered';folder.mkdir(parents=True)
            r=row(1);p=folder/(r['sha']+'.exe');data=bytes(range(256))*4;p.write_bytes(data)
            r.update(file_size=len(data),input_mtime_ns=p.stat().st_mtime_ns)
            result=inventory_one((str(root),r))
            self.assertEqual(result['disarmed_content_sha256'],hashlib.sha256(data).hexdigest())
            self.assertEqual(result['tlsh'],tlsh.hash(data))
            p.write_bytes(data+b'changed')
            with self.assertRaisesRegex(ValueError,'changed'):inventory_one((str(root),r))

    def test_retention_exact_tnull_and_same_historical_policy(self):
        ref=fingerprint_row(1,'2020-03');ref['split']='validation'
        targets=[fingerprint_row(i+10,m) for i,m in enumerate(MONTHS)]
        targets.append(fingerprint_row(99,'2020-04',content=1));targets.sort(key=lambda r:(r['timestamp'],r['sha']))
        result=retain([ref],targets)
        old=retention_decisions([ref],targets,tlsh.diff)
        for t in (0,10,20):
            self.assertEqual([(r['sha'],r['decision']) for r in result[t]],[(r['sha'],r['decision']) for r in old[t]])
        keep,n=check_witnesses([ref],targets,result)
        self.assertEqual(n,3);self.assertEqual(len(keep[0]),6)

    def test_retention_valid_tlsh_cross_family_witness(self):
        value=tlsh.hash(bytes(range(256))*4)
        ref=fingerprint_row(1,'2020-03','VALID',value);ref['split']='validation'
        targets=[fingerprint_row(i+10,m) for i,m in enumerate(MONTHS)]
        targets.append(fingerprint_row(99,'2020-04','VALID',value));targets[-1]['family']='other'
        targets.sort(key=lambda r:(r['timestamp'],r['sha']))
        tables=retain([ref],targets)
        self.assertTrue(all(next(r for r in tables[t] if r['sha']==f'{99:064x}')['decision']=='EXCLUDE' for t in tables))
        check_witnesses([ref],targets,tables)

    def test_corrupt_witness_and_nonnested_decisions_stop(self):
        ref=fingerprint_row(1,'2020-03');ref['split']='validation'
        targets=[fingerprint_row(i+10,m) for i,m in enumerate(MONTHS)]
        targets.append(fingerprint_row(99,'2020-04',content=1));targets.sort(key=lambda r:(r['timestamp'],r['sha']))
        tables=retain([ref],targets)
        bad=deepcopy(tables);bad[10][1]['reference_sha']='missing';bad[10][1]['decision']='EXCLUDE'
        with self.assertRaises(ValueError):check_witnesses([ref],targets,bad)

    def test_images_preserve_same_source_and_decoder_contract(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);folder=root/'data/raw/altered';folder.mkdir(parents=True)
            r=row(1);data=bytes(range(256))*4;p=folder/(r['sha']+'.exe');p.write_bytes(data)
            r.update(file_size=len(data),input_mtime_ns=p.stat().st_mtime_ns,
                     disarmed_content_sha256=hashlib.sha256(data).hexdigest())
            out=root/'run';out.mkdir();configs={v:{'frozen':v} for v in VIEWS}
            saved=make_images((str(root),str(out),r,configs,mapping()))
            for v in VIEWS:
                tensor=validate_image(out,r,v,saved['views'][v],configs[v],mapping())
                self.assertEqual(tuple(tensor.shape),(1,64,64));self.assertEqual(tensor.dtype,torch.float32)
            with self.assertRaisesRegex(ValueError,'overwrite'):make_images((str(root),str(out),r,configs,mapping()))
            image=out/saved['views']['raw-byte']['image'];image.write_bytes(b'corrupt')
            with self.assertRaisesRegex(ValueError,'checksum'):validate_image(out,r,'raw-byte',saved['views']['raw-byte'],configs['raw-byte'],mapping())

    def test_cpu_frozen_logits_no_state_or_gradient_mutation(self):
        old_threads=torch.get_num_threads();torch.set_num_threads(1)
        try:
            model=MalSBSLCNet(51).eval().requires_grad_(False);before=model_state_hash(model)
            x=torch.full((2,1,64,64),.5,dtype=torch.float32)
            a=infer_frozen(model,x);b=infer_frozen(model,x)
            np.testing.assert_array_equal(a,b);self.assertEqual(a.shape,(2,51))
            self.assertEqual(model_state_hash(model),before);self.assertTrue(all(p.grad is None for p in model.parameters()))
            model.train()
            with self.assertRaisesRegex(ValueError,'Frozen'):infer_frozen(model,x)
        finally:torch.set_num_threads(old_threads)

    def test_logit_readback_shape_class_id_and_labels(self):
        rows=[row(1),row(2,family='c01')];m=mapping()
        a=dict(logits=np.zeros((2,51),dtype=np.float32),sample_identifiers=np.array([r['sha'] for r in rows]),
               class_names=np.array(sorted(m,key=m.get)),labels=np.array([0,1]))
        check_logit_array(a,rows,m)
        for key,value in [('logits',np.zeros((2,50),dtype=np.float32)),('labels',np.array([1,0])),
                          ('sample_identifiers',a['sample_identifiers'][::-1]),('class_names',a['class_names'][::-1])]:
            with self.subTest(key=key),self.assertRaises(ValueError):check_logit_array({**a,key:value},rows,m)

    def test_no_metric_comparison_calibration_fit_or_optimizer_calls(self):
        forbidden={'classification_metrics','evaluate_arrays','evaluate_retained','fit_temperature',
                   'argmax','softmax','backward','step'}
        root=Path(__file__).resolve().parents[1]
        for name in ('prepare_protected_future.py','verify_protected_future.py'):
            tree=ast.parse((root/'scripts'/name).read_text(encoding='utf-8'))
            calls={n.func.attr if isinstance(n.func,ast.Attribute) else n.func.id
                   for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,(ast.Attribute,ast.Name))}
            self.assertFalse(calls&forbidden,(name,calls&forbidden))


if __name__=='__main__':unittest.main()
