"""D045 New-Q-only baseline: freeze softmax(-L), no evaluation or fitting."""
import io
import json
from datetime import datetime, timezone
import numpy as np
from scripts.frozen_encoder_extraction import Evidence,ROOT,VIEWS,json_bytes,sha256,read_bound,require

QUALITY='data/derived/historical_quality_v1/20261017_005343_157189'
QUALITY_HASH='17e887d981568bdc79a2af11488590cdac3e4abc0ab6584e30ae9c10a9c9e270'


def quality_weights(loss_by_view):
    require(isinstance(loss_by_view,dict) and set(loss_by_view)==set(VIEWS),'Three unique named views required')
    require(all(isinstance(x,(int,float,np.floating)) and not isinstance(x,(bool,np.bool_)) for x in loss_by_view.values()),'Loss must be numeric')
    loss=np.array([loss_by_view[v] for v in VIEWS],dtype=np.float64)
    require(np.isfinite(loss).all() and (loss>=0).all(),'Loss must be finite nonnegative')
    scores=-loss
    weights=np.exp(scores-scores.max());weights/=weights.sum()
    require(np.isfinite(weights).all() and (weights>=0).all() and np.isclose(weights.sum(),1,rtol=0,atol=1e-15),'Invalid weights')
    return weights


def execute(output):
    e=Evidence();completion=e.js(QUALITY+'/completion.json',QUALITY_HASH)
    require(completion['status']=='PASS' and not completion['future_access'] and
            not completion['outcome_C_access'] and not completion['U_computed'],'Q scope mismatch')
    # Current source/input bytes must still agree with the completed Q run.
    for path,h in completion['source_input_hashes'].items():e.read(path,h)
    for path in ('scripts/freeze_historical_quality_weights.py','tests/test_historical_quality_weights.py',
                 'docs/RESEARCH_PROTOCOL_V1.md'):
        e.read(path)
    checks=e.js(QUALITY+'/checksums.json')
    # Validate every Q output, then bind loss means back to the pinned completion.
    for path,h in checks.items():e.read(QUALITY+'/'+path,h)
    require(len(completion['results'])==9,'Incomplete Q batch')
    config=dict(arm='New Quality-only',formula='w=softmax(-L)',quality_coefficient=1,
                drift_used=False,uncertainty_used=False,weight_tuning=False,
                views=list(VIEWS),probability_fusion='sum_v w_v * softmax(logits_v / fixed_A_T_v)',
                evaluation_run=False,source_Q_completion_sha256=QUALITY_HASH)
    (output/'config.json').write_bytes(json_bytes(config));results=[]
    for case,k,s,n in (('E1',32,17,3130),('E2',42,25,3659),('Final',51,25,4064)):
        selection=e.js(QUALITY+'/'+case+'/selection.json',checks[case+'/selection.json'])
        rows=selection['B'];support=selection['support'];mapping=selection['class_to_idx']
        require(len(mapping)==k and len(support)==s,'Frozen mapping/support mismatch')
        loss_by_view={};provenance={}
        for view in VIEWS:
            key=case+'/'+view+'/'
            q=e.js(QUALITY+'/'+key+'quality.json',checks[key+'quality.json'])
            expected=next(r for r in completion['results'] if (r['case'],r['view'])==(case,view))
            require(all(q[field]==value for field,value in expected.items()),'Summary binding mismatch')
            with np.load(io.BytesIO(e.read(QUALITY+'/'+key+'sample_losses.npz',checks[key+'sample_losses.npz'])),allow_pickle=False) as a:
                require(a['sample_identifiers'].tolist()==[r['sha'] for r in rows],'Sample alignment mismatch')
                require(a['labels'].tolist()==[mapping[r['family']] for r in rows],'Class alignment mismatch')
                loss=a['nll'];mask=np.array([r['family'] in support for r in rows])
                require(np.array_equal(mask,a['in_support']) and int(mask.sum())==n,'Support mask mismatch')
                require(np.isfinite(loss).all() and (loss>=0).all(),'Invalid stored losses')
                family_means=[float(loss[[i for i,r in enumerate(rows) if r['family']==c]].mean()) for c in support]
                require(np.isclose(np.mean(family_means),q['class_balanced_nll'],rtol=0,atol=1e-14),'Class-balanced loss mismatch')
            loss_by_view[view]=q['class_balanced_nll'];provenance[view]=q['provenance']
        weights=quality_weights(loss_by_view)
        record=dict(case=case,K=k,support=support,class_to_idx=mapping,B_support_samples=n,
                    views=list(VIEWS),L=[loss_by_view[v] for v in VIEWS],weights=weights.tolist(),
                    provenance=provenance,selection_file=QUALITY+'/'+case+'/selection.json',
                    selection_sha256=checks[case+'/selection.json'],status='FROZEN',
                    evaluation_status='NOT_EVALUATED',formula='softmax(-L)',beta=0,gamma=0)
        (output/(case+'_weights.json')).write_bytes(json_bytes(record));results.append(record)
    for path,h in e.hashes.items():read_bound(ROOT,path,h)
    (output/'completion.json').write_bytes(json_bytes(dict(status='PASS',results=results,
        source_input_hashes=e.hashes,configuration=config,future_access=False,outcome_C_access=False)))
    (output/'checksums.json').write_bytes(json_bytes({p.name:sha256(p.read_bytes()) for p in output.iterdir() if p.is_file()}))


def main():
    output=ROOT/'data/derived/new_quality_only_v1'/datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')
    output.mkdir(parents=True,exist_ok=False)
    try:execute(output)
    except BaseException as exc:
        (output/'FAILED.json').write_bytes(json_bytes(dict(status='FAIL',error=type(exc).__name__,message=str(exc))))
        raise
    print('Completion:',output/'completion.json')


if __name__=='__main__':main()
