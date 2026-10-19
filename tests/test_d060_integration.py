from copy import deepcopy
import hashlib
import unittest
from unittest.mock import patch
import numpy as np
from scripts.final_multiseed_evaluator import (
    VIEWS, SEEDS, MONTHS, METHODS, evaluate_arrays, evaluate_retained, freeze_size_bins,
    size_bin_indices, retention_decisions, standard_tlsh_distance, validate_package)
from scripts.replay_combined_freeze import synthetic_inputs


def synthetic_package():
    p=dict(schema='TRAMIF_COMBINED_D060_V1',K=51,seeds=list(SEEDS),methods=list(METHODS),
           views=list(VIEWS),class_to_idx={f'c{i:02d}':i for i in range(51)},beta=1,gamma=0,
           U_used=False,size_stratification={'cutoffs':[10.,20.,30.]},seed_records={})
    for seed in SEEDS:
        loss=np.array([.3,.6,.9]);d=np.array([.02,.04,.03])
        q=np.exp(-loss);q/=q.sum();qd=np.exp(-loss-d);qd/=qd.sum()
        p['seed_records'][str(seed)]=dict(seed=seed,views_order=list(VIEWS),L=loss.tolist(),d=d.tolist(),
            views={v:dict(seed=seed,view=v,K=51,epoch=100,cutoff_exclusive='2020-02',
                checkpoint_sha256=hashlib.sha256(f'{seed}/{v}'.encode()).hexdigest(),
                temperature=1+(seed-42)*.1+i*.01,checkpoint=f'checkpoint/{seed}/{v}.pt') for i,v in enumerate(VIEWS)},
            weights=dict(Q=q.tolist(),QD=qd.tolist(),learned_static=[1.,0.,0.]))
    return p


def retention_row(number,stamp,state=None,content=None,status='VALID'):
    return dict(sha=f'{number:064x}',timestamp=stamp,family='ignored',split='validation',
                disarmed_content_sha256=f'{content if content is not None else number:064x}',
                tlsh_status=status,tlsh='TNULL' if status=='TNULL' else f'T1{state or number:070X}')


class TestD060Integration(unittest.TestCase):
    def setUp(self):
        self.package=synthetic_package()
        self.rows,self.bundles,self.decisions=synthetic_inputs(self.package)

    def test_all_five_seeds_seven_arms_and_eight_contrasts(self):
        report,logs=evaluate_arrays(self.package,self.rows,self.bundles)
        self.assertEqual(logs.shape,(5,7,48,51))
        self.assertEqual(len(report['summary']['contrasts']),8)
        self.assertFalse(report['summary']['owner_approval_required'])
        self.assertEqual(set(report['details']),set(map(str,SEEDS)))

    def test_probability_mixture_reference_full51_and_zero_weight(self):
        _,logs=evaluate_arrays(self.package,self.rows,self.bundles)
        p=np.exp(logs[0,:3]);w=self.package['seed_records']['42']['weights']
        np.testing.assert_allclose(np.exp(logs[0,3]),p.mean(0),atol=1e-15)
        np.testing.assert_allclose(np.exp(logs[0,4]),np.einsum('v,vnk->nk',w['Q'],p),atol=1e-15)
        np.testing.assert_allclose(np.exp(logs[0,5]),np.einsum('v,vnk->nk',w['QD'],p),atol=1e-15)
        np.testing.assert_array_equal(logs[0,6],logs[0,0])

    def test_mismatched_sample_order_is_rejected_not_position_fused(self):
        self.bundles[43][1]['ids']=self.bundles[43][1]['ids'][::-1]
        with self.assertRaisesRegex(ValueError,'alignment'):evaluate_arrays(self.package,self.rows,self.bundles)

    def test_class_and_label_alignment_guard(self):
        self.bundles[42][0]['classes']=self.bundles[42][0]['classes'][::-1]
        with self.assertRaisesRegex(ValueError,'alignment'):evaluate_arrays(self.package,self.rows,self.bundles)
        _,self.bundles,_=synthetic_inputs(self.package)
        self.bundles[42][0]['labels'][0]=50
        with self.assertRaisesRegex(ValueError,'alignment'):evaluate_arrays(self.package,self.rows,self.bundles)

    def test_cross_seed_checkpoint_and_temperature_guard(self):
        self.bundles[43][0]['binding']['checkpoint_sha256']=self.bundles[42][0]['binding']['checkpoint_sha256']
        with self.assertRaisesRegex(ValueError,'checkpoint'):evaluate_arrays(self.package,self.rows,self.bundles)
        _,self.bundles,_=synthetic_inputs(self.package)
        self.bundles[43][0]['temperature']=self.bundles[42][0]['temperature']
        with self.assertRaisesRegex(ValueError,'temperature'):evaluate_arrays(self.package,self.rows,self.bundles)

    def test_nonfinite_logits_and_missing_seed_stop(self):
        self.bundles[42][0]['logits'][0,0]=np.nan
        with self.assertRaises(ValueError):evaluate_arrays(self.package,self.rows,self.bundles)
        del self.bundles[46]
        with self.assertRaises(ValueError):evaluate_arrays(self.package,self.rows,self.bundles)

    def test_empty_original_month_and_duplicate_ids_stop(self):
        rows=deepcopy(self.rows);rows[0]['sha']=rows[1]['sha']
        with self.assertRaises(ValueError):evaluate_arrays(self.package,rows,self.bundles)
        rows=deepcopy(self.rows)
        for r in rows[-8:]:r['timestamp']='2020-08-15T00:00:00+00:00'
        with self.assertRaisesRegex(ValueError,'month'):evaluate_arrays(self.package,rows,self.bundles)

    def test_labels_cannot_change_frozen_probabilities(self):
        _,before=evaluate_arrays(self.package,self.rows,self.bundles)
        for i,r in enumerate(self.rows):
            label=(self.bundles[42][0]['labels'][i]+1)%3;r['family']=f'c{label:02d}'
            for seed in SEEDS:
                for b in self.bundles[seed]:b['labels'][i]=label
        _,after=evaluate_arrays(self.package,self.rows,self.bundles)
        np.testing.assert_array_equal(before,after)

    def test_train_only_quartiles_and_ties(self):
        rows=[dict(sha=str(i),timestamp='2019-10-01T00:00:00+00:00',file_size=n)
              for i,n in enumerate([10,20,30,40])]
        q=freeze_size_bins(rows)
        self.assertEqual(q['cutoffs'],[17.5,25.,32.5])
        np.testing.assert_array_equal(size_bin_indices(np.array([10,20,30,40]),[20,20,30]),[0,0,2,3])
        rows[0]['timestamp']='2020-02-01T00:00:00+00:00'
        with self.assertRaisesRegex(ValueError,'TRAIN-only'):freeze_size_bins(rows)

    def test_empty_size_bins_not_discarded_or_primary_changed(self):
        self.package['size_stratification']['cutoffs']=[1e10]*3
        report,_=evaluate_arrays(self.package,self.rows,self.bundles)
        buckets=report['details']['42']['sbsmi']['file_size'][MONTHS[0]]
        self.assertEqual([b['samples'] for b in buckets],[8,0,0,0])
        self.assertEqual([b['status'] for b in buckets][1:],['NOT_ESTIMABLE']*3)
        self.assertEqual(len(report['summary']['methods']),7)

    def test_sensitivity_subsets_all_seeds_same_ids_nested(self):
        result=evaluate_retained(self.package,self.rows,self.bundles,self.decisions)
        self.assertEqual([result[str(t)]['samples'] for t in (0,10,20)],[48,42,36])
        for t in (0,10,20):self.assertEqual(result[str(t)]['status'],'PASS')

    def test_empty_sensitivity_month_not_silently_dropped(self):
        for d in self.decisions[20][:8]:d['decision']='EXCLUDE'
        r=evaluate_retained(self.package,self.rows,self.bundles,self.decisions)
        self.assertEqual(r['20']['status'],'NOT_ESTIMABLE')
        self.assertEqual(r['20']['missing_months'],[MONTHS[0]])

    def test_bad_decision_coverage_and_nonnested_sets_fail(self):
        bad=deepcopy(self.decisions);bad[10][0]['sha']='missing'
        with self.assertRaises(ValueError):evaluate_retained(self.package,self.rows,self.bundles,bad)
        bad=deepcopy(self.decisions);bad[20][0]['decision']='KEEP'
        with self.assertRaisesRegex(ValueError,'nested'):evaluate_retained(self.package,self.rows,self.bundles,bad)

    def test_retention_excluded_bridge_and_labels_ignored(self):
        ref=[retention_row(1,'2019-10-01T00:00:00+00:00',state=1)]
        targets=[retention_row(2,'2020-02-01T00:00:00+00:00',state=9),
                 retention_row(3,'2020-02-02T00:00:00+00:00',state=17)]
        distance=lambda a,b:abs(int(a[2:],16)-int(b[2:],16))
        r=retention_decisions(ref,targets,distance)
        self.assertEqual([x['decision'] for x in r[10]],['EXCLUDE','EXCLUDE'])
        self.assertEqual(r[10][1]['reference_sha'],targets[0]['sha'])
        targets[0]['family']='different';self.assertEqual(r,retention_decisions(ref,targets,distance))

    def test_retention_parity_with_existing_historical_policy(self):
        from scripts.near_duplicate_core import apply_historical_policy
        ref=[retention_row(1,'2019-10-01T00:00:00+00:00',state=1)]
        ref[0]['split']='train'
        targets=[retention_row(i,f'2020-02-{i:02d}T00:00:00+00:00',state=i*8) for i in range(2,6)]
        distance=lambda a,b:abs(int(a[2:],16)-int(b[2:],16))
        old=apply_historical_policy(ref,targets,distance_function=distance)
        new=retention_decisions(ref,targets,distance)
        for t in (0,10,20):
            for a,b in zip(old['validation_decisions'][t],new[t]):
                for k in ('sha','decision','reason','reference_sha','tlsh_distance'):self.assertEqual(a[k],b[k])

    def test_tnull_exact_match_and_standard_length_distance(self):
        ref=[retention_row(1,'2019-10-01T00:00:00+00:00',status='TNULL')]
        target=[retention_row(2,'2020-02-01T00:00:00+00:00',content=1,status='TNULL')]
        def forbidden(*args):raise AssertionError('TNULL must not be compared')
        r=retention_decisions(ref,target,forbidden)
        self.assertTrue(all(r[t][0]['reason']=='exact_sha256' for t in r))
        with patch('tlsh.diff',return_value=11) as diff:
            self.assertEqual(standard_tlsh_distance('a','b'),11);diff.assert_called_once_with('a','b')

    def test_checkpoint_collision_or_retuned_weights_stop(self):
        self.package['seed_records']['42']['weights']['QD']=[1.,0.,0.]
        with self.assertRaisesRegex(ValueError,'weight'):validate_package(self.package)


if __name__=='__main__':unittest.main()
