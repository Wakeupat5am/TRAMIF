"""Approved cap64 and same-primary-sample group-mass diagnostics only."""
import io
import json
import math
from dataclasses import asdict
from datetime import datetime, timezone
import numpy as np
from scripts.run_frozen_descriptive_drift import (
    Evidence, ROOT, PLAN, PLAN_HASH, PREP, PREP_HASH, VIEWS,
    output_checksums, json_bytes, sha256, csv_rows, require, read_bound,
)
from scripts.descriptive_mmd import biased_rbf_mmd2
from scripts.weighted_descriptive_mmd import group_weights, weighted_mmd2
from scripts.check_drift_bandwidth_sensitivity import PRIMARY, PRIMARY_HASH, contrasts

SUBSET='data/derived/cap64_metadata_review/20261016_132758_657700/proposed_subset.csv'
SUBSET_HASH='7a64df7bb767b1c9a22807a878871578841cc6648bb6d0baed7a5e0662d8915b'


def selection_indices(rows, subset):
    ids=[r['sha'] for r in rows]
    wanted=[r['sha'] for r in subset]
    require(len(set(ids))==len(ids) and len(set(wanted))==len(wanted), 'Duplicate IDs')
    lookup={r['sha']: (i,r) for i,r in enumerate(rows)}
    indices=[]
    for row in subset:
        require(row['sha'] in lookup,'Subset outside primary')
        i,original=lookup[row['sha']]
        require(all(original.get(k)==v for k,v in row.items() if k!='case'),'Subset metadata mismatch')
        indices.append(i)
    require(bool(indices),'Empty subset')
    return indices


def execute(output):
    evidence=Evidence()
    primary=evidence.js(PRIMARY,PRIMARY_HASH)
    require(evidence.js(PREP+'/report.json',PREP_HASH)['status']=='PASS','Preparation failed')
    plans=evidence.js(PLAN+'/checksums.json',PLAN_HASH)
    checks=evidence.js(PREP+'/checksums.json')
    subset=csv_rows(evidence.read(SUBSET,SUBSET_HASH))
    for path in ('scripts/descriptive_mmd.py','scripts/mmd_measurement_preparation.py'):
        evidence.read(path,primary['source_input_hashes'][path])
    for path in ('scripts/weighted_descriptive_mmd.py','scripts/check_drift_sampling_sensitivity.py',
                 'tests/test_weighted_descriptive_mmd.py','tests/test_drift_sampling_sensitivity.py'):
        evidence.read(path)
    config=dict(arms=['cap64','group-balanced'],bandwidth='frozen primary TRAIN h',
        normalization='frozen TRAIN',support='frozen S17/25/25',block_size=32,
        group_weight='1/(represented_groups * selected_members_in_group)',
        outer_aggregation='equal-family',selection_by_score=False,primary_replacement=False)
    (output/'config.json').write_bytes(json_bytes(config))
    summaries=[]; families=[]; weight_manifest=[]
    for case in ('E1','E2','Final'):
        support=json.loads(evidence.read(PLAN+'/'+case+'/support.json',plans[case+'/support.json']))
        samples=csv_rows(evidence.read(PLAN+'/'+case+'/matched_samples.csv',plans[case+'/matched_samples.csv']))
        rows={s:[r for r in samples if r['stage']==s] for s in ('A','B')}
        case_subset={s:[r for r in subset if r['case']==case and r['stage']==s] for s in ('A','B')}
        selected={s:selection_indices(rows[s],case_subset[s]) for s in ('A','B')}
        require(all({r['family'] for r in rows[s]}==set(support) for s in rows),'Support mismatch')
        for stage in rows:
            for family in support:
                rr=[r for r in rows[stage] if r['family']==family]
                for r,w in zip(rr,group_weights([r['group_id'] for r in rr])):
                    weight_manifest.append(dict(case=case,stage=stage,family=family,sha=r['sha'],group_id=r['group_id'],weight=float(w)))
        for view in VIEWS:
            prefix=case+'/'+view+'/'
            def prepared(name):
                path=PREP+'/'+prefix+name
                blob=evidence.read(path,checks[prefix+name])
                require(sha256(blob)==primary['source_input_hashes'][path],'Primary binding mismatch')
                return blob
            meta=json.loads(prepared('parameters.json'))
            bandwidth=json.loads(prepared('bandwidth.json'))
            base=next(r for r in primary['summaries'] if (r['case'],r['view'])==(case,view))
            h=base['bandwidth_h']
            require(h==meta['bandwidth_h']==bandwidth['bandwidth_h'],'h mismatch')
            arrays={}
            for stage in rows:
                require(rows[stage]==meta['stages'][stage]['samples'],'Sample alignment mismatch')
                arr=np.load(io.BytesIO(prepared(stage+'.npy')),allow_pickle=False)
                require(arr.dtype==np.float64 and np.isfinite(arr).all(),'Invalid prepared embeddings')
                require(sha256(arr.tobytes())==meta['stages'][stage]['payload_sha256'],'Payload mismatch')
                arrays[stage]=arr
            for arm in config['arms']:
                scores=[]
                for family in support:
                    matrices={}; weights={}; counts={}
                    for stage in rows:
                        indices=[i for i in (selected[stage] if arm=='cap64' else range(len(rows[stage]))) if rows[stage][i]['family']==family]
                        groups=[rows[stage][i]['group_id'] for i in indices]
                        require(len(set(groups))>=5,'Post-cap support failure')
                        if arm=='cap64': require(len(indices)==min(64,sum(r['family']==family for r in rows[stage])),'Cap count mismatch')
                        matrices[stage]=arrays[stage][indices]
                        weights[stage]=group_weights(groups)
                        counts[stage]=dict(samples=len(indices),groups=len(set(groups)))
                    if arm=='cap64':
                        result=biased_rbf_mmd2(matrices['A'],matrices['B'],bandwidth_h=h)
                    else:
                        result=weighted_mmd2(matrices['A'],matrices['B'],weights['A'],weights['B'],bandwidth_h=h)
                    original=next(r for r in primary['per_family'] if (r['case'],r['view'],r['family'])==(case,view,family))
                    families.append(dict(case=case,view=view,arm=arm,family=family,counts=counts,
                        primary_mmd2=original['mmd2'],**asdict(result),**contrasts(original['mmd2'],result.mmd2)))
                    scores.append(result.mmd2)
                score=math.fsum(scores)/len(support)
                summaries.append(dict(case=case,view=view,arm=arm,support=len(support),bandwidth_h=h,
                    primary_mmd2=base['conditional_mmd2'],mmd2=score,**contrasts(base['conditional_mmd2'],score)))
                print(f'{case}/{view}/{arm}: {score:.10f}',flush=True)
    for path,digest in evidence.hashes.items(): read_bound(ROOT,path,digest)
    (output/'sample_weights.json').write_bytes(json_bytes(weight_manifest))
    report=dict(status='PASS_TECHNICAL_DIAGNOSTIC',config=config,summaries=summaries,per_family=families,
        source_input_hashes=evidence.hashes,future_access=False,fusion_weights_created=False,
        limitation='Group-balanced changes empirical target; no independence, null significance or predictive reliability claim.')
    (output/'report.json').write_bytes(json_bytes(report))
    (output/'checksums.json').write_bytes(json_bytes(output_checksums(output)))


def main():
    output=ROOT/'data/derived/drift_sampling_sensitivity'/datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')
    output.mkdir(parents=True,exist_ok=False)
    try:
        execute(output)
        print('Report:',output/'report.json')
    except BaseException as exc:
        (output/'FAILED.json').write_bytes(json_bytes(dict(error=type(exc).__name__,message=str(exc))))
        raise


if __name__=='__main__': main()
