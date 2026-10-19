"""Authorized E1 selection then E2 held-forward evaluation; never Future Test."""
import argparse
import io
from datetime import datetime,timezone
import numpy as np
import torch
from scripts.prepare_historical_logits import (
    Evidence,ROOT,G2,G2_HASH,VIEWS,MONTHS,selection,static_checkpoint,csv_rows,
    json_bytes,sha256,read_bound,require,image,load_encoder,state_hash,infer_logits,
)
from scripts.episode_training_dataset import HISTORICAL_MANIFEST_HASH
from scripts.compute_historical_quality import canonical_bundle
from scripts.prepare_e1_outcome import FREEZE,FREEZE_HASH,compare_fixed
from scripts.freeze_qd_selection_protocol import select_beta,BETAS
from scripts.uniform_fusion import calibrated_log_probabilities,classification_metrics
from scripts.quality_fusion import weighted_fusion

LOCK='data/derived/qd_selection_protocol_v1/20261017_033146_321989'
LOCK_HASH='3f50e4aec6ec37cc7fe9fcceb912a27b499f4be3517790d0d0553b73d023e152'
LOCK_CHECK='4547cfbe4f7fd97943d68e980cc7f6a34f40c2390fcea2a585bd9206051f9fab'
E1_INPUT='data/derived/e1_outcome_preparation/20261017_032743_694605'
E1_INPUT_HASH='9cd2ceac5bc7c8c947c2461d339032a7f6d8e34c11a60c552446707b2cfb3a34'
E1_CHECK='5c6cc6192c58c7b3114b48857c28e36f8a80f86d4f0eda107a36cd8918fa6f2e'
MONTH_C={'E1':'2020-01','E2':'2020-02'}


def outcome_rows(case,e):
    require(case in MONTH_C,'Only E1/E2 historical outcomes allowed')
    ab,mapping,source,support=selection(case,e)
    checks=e.js(G2+'/checksums.json',G2_HASH)
    rows=csv_rows(e.read(f'{G2}/{case}/C_keep.csv',checks[case+'/C_keep.csv']))
    require(bool(rows) and len({r['sha'] for r in rows})==len(rows),'Empty/duplicate C')
    canonical={r['sha']:r for r in csv_rows(e.read('data/manifests/train_validation_manifest.csv',HISTORICAL_MANIFEST_HASH))}
    for r in rows:
        dt=datetime.fromisoformat(r['timestamp'])
        require(dt.utcoffset() is not None and dt.utcoffset().total_seconds()==0 and dt.strftime('%Y-%m')==MONTH_C[case],'C chronology mismatch')
        require(r['stage']=='C' and r['decision']=='KEEP' and r['threshold']=='0' and r['family'] in mapping,'C primary retention mismatch')
        require(all(r[k]==canonical[r['sha']][k] for k in ('family','timestamp')),'Canonical C mismatch')
    require(not {r['sha'] for r in rows}&{r['sha'] for rr in ab.values() for r in rr},'C/A/B IDs overlap')
    return rows,ab,mapping,source,support


def frozen_inputs(case,ab,mapping,support,e):
    lock=e.js(LOCK+'/completion.json',LOCK_HASH)
    require(lock['status']=='PASS_PRE_OUTCOME_FREEZE' and not lock['outcome_losses_read'],'Invalid pre-outcome lock')
    checks=e.js(LOCK+'/checksums.json',LOCK_CHECK)
    protocol=e.js(LOCK+'/protocol.json',checks['protocol.json'])
    require(protocol['beta_candidates']==list(BETAS) and protocol['gamma']==0 and protocol['tie_atol']==1e-12,'Selection contract mismatch')
    forecasts=e.js(LOCK+'/'+case+'_forecasts.json',checks[case+'_forecasts.json'])
    # Completion pins all source bytes; forecasts are revalidated against Q/D inputs later in verify.
    require(forecasts['support']==support and forecasts['class_to_idx']==mapping and forecasts['views']==list(VIEWS),'Forecast alignment mismatch')
    complete=e.js(FREEZE+'/completion.json',FREEZE_HASH)
    frozen=e.js(FREEZE+'/'+case+'_weights.json')
    require(frozen==next(r for r in complete['results'] if r['case']==case),'Frozen Q weights mismatch')
    require(frozen['class_to_idx']==mapping and frozen['support']==support and frozen['views']==list(VIEWS),'Q mapping/support mismatch')
    require(np.array_equal(frozen['L'],np.asarray(forecasts['per_family_B_loss']).mean(axis=1)),'Frozen loss/forecast mismatch')
    temperatures={}
    for view in VIEWS:
        cp=static_checkpoint(case,view,mapping,e);provenance=frozen['provenance'][view]
        t=e.js(provenance['temperature_file'],complete['source_input_hashes'][provenance['temperature_file']])
        require(cp==provenance['checkpoint'] and t['checkpoint']['sha256']==cp['sha256'] and t['class_to_idx']==mapping,'Checkpoint/T mismatch')
        require(t['fit_ids']==[r['sha'] for r in ab['A']] and t['fit_month']==MONTHS[case][0],'T must fit A only')
        temperatures[view]=t['temperature']
    return frozen,forecasts,temperatures,protocol


def forecast_scores(single_logs,labels,mapping,support,predictions,choose=True):
    """Freeze support intersection for scoring only; never changes forecasts."""
    require(single_logs.shape[0]==3 and labels.shape==(single_logs.shape[1],),'Outcome dimensions')
    active=[c for c in support if np.any(labels==mapping[c])]
    require(bool(active),'NOT_ESTIMABLE: empty S_eval')
    observed=np.array([[-lp[labels==mapping[c],mapping[c]].mean() for c in active] for lp in single_logs])
    idx=[support.index(c) for c in active]
    pred=np.asarray(predictions,dtype=np.float64)[:,:,idx]
    residuals=pred-observed[None,:,:]
    result=select_beta(pred,observed) if choose else None
    mae=np.abs(residuals).mean(axis=2).mean(axis=1).tolist()
    return dict(S_eval=active,missing_S=[c for c in support if c not in active],observed_family_nll=observed.tolist(),
                candidate_mae=mae,residuals=residuals.tolist(),absolute_residuals=np.abs(residuals).tolist(),
                per_view_candidate_mae=np.abs(residuals).mean(axis=2).tolist(),selection=result)


def fixed_qd_weights(loss,drift,beta):
    require(beta in BETAS,'Beta outside frozen grid')
    l=np.asarray(loss,dtype=np.float64);d=np.asarray(drift,dtype=np.float64)
    require(l.shape==d.shape==(3,) and np.isfinite(l).all() and np.isfinite(d).all()
            and (l>=0).all() and (d>=0).all() and (d<=1).all(),'Invalid frozen signals')
    score=-l-beta*d;w=np.exp(score-score.max());w/=w.sum()
    require(np.isfinite(w).all() and (w>=0).all() and np.isclose(w.sum(),1,rtol=0,atol=1e-15),'Invalid QD weights')
    return w


def load_e1(rows,mapping,temperatures,frozen,e):
    receipt=e.js(E1_INPUT+'/receipt.json',E1_INPUT_HASH);checks=e.js(E1_INPUT+'/checksums.json',E1_CHECK)
    saved=e.js(E1_INPUT+'/selection.json',checks['selection.json'])
    require(receipt['status']=='LOGITS_COMPLETE' and saved['samples']==rows and saved['class_to_idx']==mapping,'E1 outcome source mismatch')
    require(receipt['temperatures']==temperatures and not receipt['metrics_computed'],'E1 T/input state mismatch')
    bundles=[]
    for view in VIEWS:
        r=next(x for x in receipt['views'] if x['view']==view)
        require(r['state_unchanged'] and r['checkpoint']['sha256']==frozen['provenance'][view]['checkpoint']['sha256'],'E1 encoder mismatch')
        with np.load(io.BytesIO(e.read(E1_INPUT+'/'+view+'.npz',checks[view+'.npz'])),allow_pickle=False) as a:
            bundle={k:a[k].copy() for k in a.files}
        bundles.append(as_bundle(bundle,rows,mapping,temperatures[view]))
    return bundles


def as_bundle(bundle,rows,mapping,t):
    aligned,_=canonical_bundle(bundle,rows,mapping)
    return dict(logits=aligned['logits'],ids=aligned['sample_identifiers'],classes=aligned['class_names'],
                labels=aligned['labels'],temperature=t)


def extract_e2(rows,mapping,source,temperatures,frozen,e,output):
    bundles=[];reference=None
    for view in VIEWS:
        model,cp=load_encoder('E2',view,mapping,e);before=state_hash(model)
        require(cp['sha256']==frozen['provenance'][view]['checkpoint']['sha256'],'E2 wrong model')
        parts=[];contents=[]
        for start in range(0,len(rows),32):
            items=[image(row,view,source,e) for row in rows[start:start+32]]
            parts.append(infer_logits(model,torch.stack([x for x,_ in items]),len(mapping)))
            contents.extend(h for _,h in items)
        require(state_hash(model)==before and all(p.grad is None for p in model.parameters()),'E2 model state changed')
        if reference is not None:require(reference==contents,'Cross-view content mismatch')
        reference=contents
        bundle=dict(logits=np.concatenate(parts),sample_identifiers=np.array([r['sha'] for r in rows]),
                    labels=np.array([mapping[r['family']] for r in rows],dtype=np.int64),
                    class_names=np.array(sorted(mapping,key=mapping.get)))
        np.savez(output/(view+'_raw.npz'),**bundle)
        bundles.append(as_bundle(bundle,rows,mapping,temperatures[view]))
    (output/'content_hashes.json').write_bytes(json_bytes(reference))
    return bundles


def execute(case,output,e1_selection=None,e1_sha=None,execute_inference=False):
    e=Evidence()
    for path in ('scripts/evaluate_historical_episode.py','tests/test_historical_episode_evaluation.py',
                 'scripts/prepare_historical_logits.py','scripts/prepare_e1_outcome.py','scripts/compute_historical_quality.py',
                 'scripts/freeze_qd_selection_protocol.py','scripts/uniform_fusion.py','scripts/quality_fusion.py'):
        e.read(path)
    # For E2, bind beta BEFORE reading any E2 outcome metadata or predictions.
    selected=None
    if case=='E2':
        require(e1_selection and e1_sha and execute_inference,'E2 requires frozen E1 selection and authorized inference')
        selected=e.js(e1_selection,e1_sha)
        require(selected['case']=='E1' and selected['protocol_completion_sha256']==LOCK_HASH
                and selected['beta'] in BETAS and selected['gamma']==0,'E1 selection mismatch')
    rows,ab,mapping,source,support=outcome_rows(case,e)
    frozen,forecast,t,protocol=frozen_inputs(case,ab,mapping,support,e)
    (output/'config.json').write_bytes(json_bytes(dict(case=case,month=MONTH_C[case],class_to_idx=mapping,
        samples=rows,weights_Q=frozen['weights'],temperatures=t,forecast_source=LOCK+'/'+case+'_forecasts.json',
        E1_selection=e1_selection,E1_selection_sha256=e1_sha,protocol=protocol)))
    bundles=load_e1(rows,mapping,t,frozen,e) if case=='E1' else extract_e2(rows,mapping,source,t,frozen,e,output)
    names,logs,metrics=compare_fixed(bundles,frozen['weights'])
    scores=forecast_scores(logs[:3],bundles[0]['labels'],mapping,support,forecast['predicted_next_family_loss'],choose=case=='E1')
    if case=='E1':
        beta=scores['selection']['beta']
        selected=dict(case='E1',beta=beta,gamma=0,rule='D048 minimum equal-family/equal-view forecast MAE',
            protocol_completion_sha256=LOCK_HASH,forecast_source=LOCK+'/E1_forecasts.json',
            forecast_sha256=e.hashes[LOCK+'/E1_forecasts.json'],candidate_mae=scores['candidate_mae'],
            tie_candidates=scores['selection']['tie_candidates'],S_eval=scores['S_eval'],
            source_input_hashes=dict(e.hashes),scope='DEVELOPMENT_MODEL_SELECTION',frozen_before_E2=True)
        (output/'beta_selection.json').write_bytes(json_bytes(selected))
    else:beta=selected['beta']
    # Never substitute E2's candidate winner; report baseline and E1-selected arm only.
    qd_weights=fixed_qd_weights(frozen['L'],forecast['d_desc'],beta)
    qd=weighted_fusion(logs[:3],qd_weights)
    names.append('quality-plus-drift');logs=np.concatenate([logs,qd[None]],axis=0)
    metrics[names[-1]]=classification_metrics(qd,bundles[0]['labels'])
    if beta==0:require(np.array_equal(logs[-1],logs[-2]),'Beta0 failed to reproduce New Q exactly')
    np.savez_compressed(output/'predictions.npz',method_names=np.array(names),log_probabilities=logs,
        predictions=logs.argmax(axis=2),sample_identifiers=bundles[0]['ids'],labels=bundles[0]['labels'],class_names=bundles[0]['classes'])
    (output/'forecast_scores.json').write_bytes(json_bytes(scores))
    for path,h in e.hashes.items():read_bound(ROOT,path,h)
    summary=dict(status='PASS',case=case,month=MONTH_C[case],samples=len(rows),K=len(mapping),
        methods=names,metrics=metrics,beta=beta,gamma=0,Q_weights=frozen['weights'],QD_weights=qd_weights.tolist(),
        forecast_mae_Q=scores['candidate_mae'][0],forecast_mae_selected=scores['candidate_mae'][list(BETAS).index(beta)],
        S_eval=scores['S_eval'],missing_S=scores['missing_S'],
        phase_role='DEVELOPMENT_MODEL_SELECTION' if case=='E1' else 'HELD_FORWARD_NO_RETUNING',
        E2_candidate_selection_used=False,E1_selection_sha256=e1_sha,
        overlap_limitation='Episodes overlap; not independent temporal replications',source_input_hashes=e.hashes,
        weights_retrained=False,temperature_refit=False,U_used=False,future_access=False)
    (output/'report.json').write_bytes(json_bytes(summary))
    (output/'checksums.json').write_bytes(json_bytes({p.name:sha256(p.read_bytes()) for p in output.iterdir() if p.is_file()}))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--case',choices=['E1','E2'],required=True)
    parser.add_argument('--e1-selection');parser.add_argument('--e1-sha256');parser.add_argument('--execute-inference',action='store_true')
    args=parser.parse_args();torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    output=ROOT/'data/derived/historical_episode_evaluation'/args.case/datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')
    output.mkdir(parents=True,exist_ok=False)
    try:execute(args.case,output,args.e1_selection,args.e1_sha256,args.execute_inference)
    except BaseException as exc:
        (output/'FAILED.json').write_bytes(json_bytes(dict(status='FAIL',error=type(exc).__name__,message=str(exc))))
        raise
    print('Report:',output/'report.json')


if __name__=='__main__':main()
