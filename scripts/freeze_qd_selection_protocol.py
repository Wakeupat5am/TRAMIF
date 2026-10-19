"""D048: pre-outcome Q/QD forecast and selection contract, no outcome inputs."""
from datetime import datetime,timezone
import numpy as np
from scripts.frozen_encoder_extraction import Evidence,ROOT,VIEWS,json_bytes,read_bound,require,sha256

BETAS=(0.,.5,1.)
TIE_ATOL=1e-12
QROOT='data/derived/historical_quality_v1/20261017_005343_157189'
QHASH='17e887d981568bdc79a2af11488590cdac3e4abc0ab6584e30ae9c10a9c9e270'
QCHECK='fc1a55a5718fa2a0b17a796e9f2a5adb1572ce090e3063e1018e444479d829ef'
DREPORT='data/derived/observed_descriptive_drift/20261016_122644_350078/report.json'
DHASH='873d2b893d07fb41d74415299d116627c70317b3375abc6fb07d28b61019b393'


def forecasts(class_loss,descriptive_drift):
    loss=np.asarray(class_loss,dtype=np.float64);d=np.asarray(descriptive_drift,dtype=np.float64)
    require(loss.ndim==2 and loss.shape[0]==3 and loss.shape[1]>0 and np.isfinite(loss).all() and (loss>=0).all(),'Invalid per-family loss')
    require(d.shape==(3,) and np.isfinite(d).all() and (d>=0).all() and (d<=1).all(),'Invalid descriptive drift')
    result=np.stack([loss+beta*d[:,None] for beta in BETAS])
    require(np.isfinite(result).all(),'Nonfinite forecast')
    return result


def select_beta(predictions,observed_loss):
    """Pure synthetic-tested selection rule; real C not loaded in this CLI."""
    p=np.asarray(predictions,dtype=np.float64);y=np.asarray(observed_loss,dtype=np.float64)
    require(p.ndim==3 and p.shape[0]==3 and p.shape[1]==3 and p.shape[2]>0 and y.shape==p.shape[1:],'Forecast/outcome shape mismatch')
    require(np.isfinite(p).all() and np.isfinite(y).all() and (p>=0).all() and (y>=0).all(),'Invalid forecast/outcome')
    # Equal-family then equal-view; no sample-count weighted objective.
    with np.errstate(over='raise',invalid='raise'):
        scores=np.abs(p-y[None,:,:]).mean(axis=2).mean(axis=1)
    require(np.isfinite(scores).all(),'Nonfinite MAE')
    minimum=float(scores.min())
    tied=[i for i,x in enumerate(scores) if x-minimum<=TIE_ATOL]
    # gamma is0: V1's fewest terms, beta+gamma, beta rules reduce to smallest beta.
    winner=min(tied,key=lambda i:(BETAS[i]!=0,BETAS[i]))
    return dict(beta=BETAS[winner],mae=scores.tolist(),tie_candidates=[BETAS[i] for i in tied])


def execute(output):
    e=Evidence();q=e.js(QROOT+'/completion.json',QHASH);checks=e.js(QROOT+'/checksums.json',QCHECK)
    d=e.js(DREPORT,DHASH)
    require(q['status']=='PASS' and len(q['results'])==9,'Q completion mismatch')
    for path in ('scripts/freeze_qd_selection_protocol.py','tests/test_qd_selection_protocol.py',
                 'docs/RESEARCH_PROTOCOL_V1.md','docs/RESEARCH_PROTOCOL_V1_AMENDMENT_01_DESCRIPTIVE_DRIFT.md'):
        e.read(path)
    protocol=dict(status='LOCKED_PRE_OUTCOME_SCORING',a=1,beta_candidates=list(BETAS),gamma=0,
        drift='primary biased class-conditional MMD2 / 2; descriptive only',
        forecast='per_family_B_NLL + beta * view_global_d_desc',
        selector='minimum E1 equal-family/equal-view absolute error',
        support='S_eval = frozen S intersect true-support classes in C; no signal refit',
        empty_support='STOP_NOT_ESTIMABLE',tie_atol=TIE_ATOL,
        tie_order=['fewest_nonzero_terms','smallest_beta_plus_gamma','smallest_beta'],
        freeze_before='E2 outcome scoring',retune_E2=False,retune_Final=False,
        fusion_score='-L_v-beta*d_v; gamma disabled',
        controls=['three calibrated single views','Uniform','New Quality-only'],
        E1_role='retrospective model-selection development, not unbiased validation',
        E2_role='held-forward historical check, no retuning; overlapping episodes',
        Future='protected; no processing/evaluation authorized',
        U='diagnostic-only, QU/QDU not estimable in this scope',
        sensitivity='primary h/cap128/sample-level only; no choosing a sensitivity winner',
        limitations=['d_desc bounded numerically, not null-standardized or calibrated across views',
                    'negative drift penalty is a hypothesis; does not establish future reliability',
                    'forecast cannot predict loss decreases; beta0 is valid negative evidence',
                    'learned static fusion and multi-seed controls remain required for wider claims'])
    (output/'protocol.json').write_bytes(json_bytes(protocol));results=[]
    for case,k,s in (('E1',32,17),('E2',42,25),('Final',51,25)):
        key=case+'/selection.json';selection=e.js(QROOT+'/'+key,checks[key])
        support=selection['support'];mapping=selection['class_to_idx']
        require(len(support)==s and len(mapping)==k,'Frozen support mismatch')
        losses=[];drifts=[];provenance={}
        for view in VIEWS:
            key=case+'/'+view+'/quality.json';item=e.js(QROOT+'/'+key,checks[key])
            family=[r['family'] for r in item['per_family']]
            require(family==support,'Family order mismatch')
            vals=[r['mean_nll'] for r in item['per_family']]
            summary=next(r for r in q['results'] if r['case']==case and r['view']==view)
            require(item['class_balanced_nll']==summary['class_balanced_nll'] and
                    np.isclose(np.mean(vals),summary['class_balanced_nll'],rtol=0,atol=1e-14),'Quality mismatch')
            ds=next(r for r in d['summaries'] if r['case']==case and r['view']==view)
            require(ds['support']==s and 0<=ds['conditional_mmd2']<=2,'Drift/support mismatch')
            losses.append(vals);drifts.append(ds['conditional_mmd2']/2)
            provenance[view]=item['provenance']
        predicted=forecasts(losses,drifts)
        record=dict(case=case,K=k,support=support,views=list(VIEWS),class_to_idx=mapping,
            per_family_B_loss=losses,d_desc=drifts,beta_candidates=list(BETAS),
            predicted_next_family_loss=predicted.tolist(),provenance=provenance,
            beta_selected=False,outcome_scores_read=False)
        (output/(case+'_forecasts.json')).write_bytes(json_bytes(record));results.append(dict(case=case,shape=list(predicted.shape)))
    for path,h in e.hashes.items():read_bound(ROOT,path,h)
    (output/'completion.json').write_bytes(json_bytes(dict(status='PASS_PRE_OUTCOME_FREEZE',results=results,
        source_input_hashes=e.hashes,coefficient_selected=False,fusion_weights_created=False,
        outcome_losses_read=False,future_access=False)))
    (output/'checksums.json').write_bytes(json_bytes({p.name:sha256(p.read_bytes()) for p in output.iterdir() if p.is_file()}))


def main():
    output=ROOT/'data/derived/qd_selection_protocol_v1'/datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')
    output.mkdir(parents=True,exist_ok=False)
    try:execute(output)
    except BaseException as exc:
        (output/'FAILED.json').write_bytes(json_bytes(dict(error=type(exc).__name__,message=str(exc))))
        raise
    print('Completion:',output/'completion.json')


if __name__=='__main__':main()
