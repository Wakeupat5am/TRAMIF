import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from scripts.score_final_future import adapt_inputs, bound_json, prediction_logs, export_tables
from scripts.verify_final_future_scoring import reference_metrics, close, verify_summary, verify
from scripts.final_multiseed_evaluator import evaluate_retained
from scripts.uniform_fusion import classification_metrics
from scripts.replay_combined_freeze import synthetic_inputs
from tests.test_d060_integration import synthetic_package


def fixture():
    p=synthetic_package(); rows,bundles,tables=synthetic_inputs(p); records=[]; arrays={}
    for seed,items in bundles.items():
        for view,b in zip(('sbsmi','raw-byte','entropy'),items):
            records.append(dict(seed=seed,view=view,checkpoint_sha256=b['binding']['checkpoint_sha256'],
                                temperature=b['temperature'],state_unchanged=True,gradient_updates=False))
            arrays[seed,view]=dict(sample_identifiers=b['ids'],class_names=b['classes'],labels=b['labels'],
                                   logits=b['logits'].astype(np.float32))
    return p,rows,tables,records,arrays


class TestFinalFutureScoring(unittest.TestCase):
    def test_adapter_five_seed_alignment_and_retention_subset(self):
        p,rows,t,r,a=fixture(); b,d=adapt_inputs(p,rows,rows,t,r,a)
        self.assertEqual(list(b),[42,43,44,45,46]); self.assertEqual(len(d[0]),len(rows))
        self.assertEqual(prediction_logs(p,b).shape,(5,7,48,51))

    def test_primary_excluded_rows_not_required_in_logit_arrays(self):
        p,rows,t,r,a=fixture(); primary=rows[1:]
        t[0][0]['decision']='EXCLUDE'
        for x in a.values():
            for k in ('sample_identifiers','labels','logits'): x[k]=x[k][1:]
        b,d=adapt_inputs(p,primary,rows,t,r,a)
        self.assertEqual(len(d[0]),47);self.assertEqual(b[42][0]['logits'].shape,(47,51))

    def test_alignment_duplicate_nonfinite_and_cross_seed_fail_closed(self):
        for corruption in ('ids','classes','nonfinite','temperature','duplicates','retention'):
            p,rows,t,r,a=fixture()
            if corruption=='ids':a[43,'raw-byte']['sample_identifiers']=a[43,'raw-byte']['sample_identifiers'][::-1]
            if corruption=='classes':a[42,'sbsmi']['class_names']=a[42,'sbsmi']['class_names'][::-1]
            if corruption=='nonfinite':a[46,'entropy']['logits'][0,0]=np.inf
            if corruption=='temperature':r[3]['temperature']=r[0]['temperature']
            if corruption=='duplicates':r[-1]=r[0]
            if corruption=='retention':t[10][-1]['decision']='KEEP';t[20][-1]['decision']='KEEP';t[0][-1]['decision']='EXCLUDE'
            with self.subTest(corruption=corruption), self.assertRaises(ValueError):adapt_inputs(p,rows,rows,t,r,a)

    def test_hashed_input_tamper_and_path_escape(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);p=root/'a.json';p.write_text('{}')
            receipt={'files':{'a.json':hashlib.sha256(p.read_bytes()).hexdigest()}}
            self.assertEqual(bound_json(root,receipt,'a.json'),{})
            p.write_text('{"tampered":true}')
            with self.assertRaisesRegex(ValueError,'checksum'):bound_json(root,receipt,'a.json')

    def test_independent_metrics_random_imbalanced_full51(self):
        rng=np.random.default_rng(62062);z=rng.normal(size=(71,51))*3
        z-=z.max(1,keepdims=True);lp=z-np.log(np.exp(z).sum(1,keepdims=True))
        y=np.array([0]*50+[2]*15+[50]*6)
        close(reference_metrics(lp,y),classification_metrics(lp,y))

    def test_exact_ties_absent_class_and_extreme_log_probability(self):
        lp=np.full((3,51),-np.log(51));y=np.array([0,1,50])
        close(reference_metrics(lp,y),classification_metrics(lp,y))
        z=np.full((2,51),-10000.);z[:,0]=0.;y=np.array([0,50])
        m=reference_metrics(z,y);self.assertTrue(np.isfinite(m['nll']));self.assertEqual(m['nll'],5000.)
        close(m,classification_metrics(z,y))

    def test_saved_predictions_end_to_end_independent_readback(self):
        p,rows,t,r,a=fixture();bundles,decisions=adapt_inputs(p,rows,rows,t,r,a)
        results=evaluate_retained(p,rows,bundles,decisions);logs=prediction_logs(p,bundles)
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);out=root/'out';out.mkdir();raw=root/'input/logits';raw.mkdir(parents=True)
            (out/'config.json').write_text(json.dumps({'input_directory':'input'}))
            (out/'scoring_manifest.json').write_text(json.dumps(rows))
            (out/'retained_decisions.json').write_text(json.dumps({str(k):v for k,v in decisions.items()}))
            (out/'results.json').write_text(json.dumps(results))
            np.save(out/'log_probabilities.npy',logs,allow_pickle=False)
            np.savez(out/'predictions.npz',predictions=logs.argmax(3),sample_identifiers=[x['sha'] for x in rows],
                labels=a[42,'sbsmi']['labels'],class_names=a[42,'sbsmi']['class_names'],
                seeds=[42,43,44,45,46],methods=p['methods'])
            for (seed,view),v in a.items():np.savez(raw/f'seed_{seed}_{view}.npz',**v)
            export_tables(out,results,p)
            receipt=verify(root,out,p)
            self.assertEqual(receipt['monthly_metric_sets'],630)
            self.assertEqual(receipt['file_size_cells'],2520)
            self.assertEqual(receipt['pooled_metric_sets'],315)
            self.assertEqual(receipt['outcomes_and_contrasts'],66)
            results['0']['report']['summary']['contrasts'][0]['mean']+=.01
            (out/'results.json').write_text(json.dumps(results))
            with self.assertRaisesRegex(ValueError,'mismatch'):verify(root,out,p)


if __name__=='__main__':unittest.main()
