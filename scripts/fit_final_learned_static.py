"""D055 approved unchanged solver on five historical B/S sets; no CNN/Future."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
import argparse
from datetime import datetime,timezone
import io
import json
import math
from pathlib import Path
import numpy as np
from scripts import learned_static_solver as solver
from scripts.verify_final_multiseed import Audit,align_bundle
from scripts.frozen_encoder_extraction import ROOT,VIEWS,json_bytes,sha256,require
from scripts.prepare_historical_logits import selection
from scripts.uniform_fusion import calibrated_log_probabilities

VERIFICATION='data/derived/final_multiseed_verification/20261017_164739_728045'
RECEIPT_HASH='fcd99219417e19f61520d85c868e7a3a074ce0b32ac1727bfd48d6681aad9f86'
SEEDS=(42,43,44,45,46)


def targets(bundles,rows,mapping,support,temperatures):
    require(set(bundles)==set(temperatures)==set(VIEWS),'Three named seed-specific views required')
    family=np.array([r['family'] for r in rows]);mask=np.isin(family,support)
    require(set(family[mask])==set(support),'Frozen support missing')
    columns=[]
    for view in VIEWS:
        bundle=align_bundle(bundles[view],rows,mapping)
        require(np.isfinite(temperatures[view]) and .05<=temperatures[view]<=20,'Invalid fixed T')
        logs=calibrated_log_probabilities(bundle['logits'],temperatures[view])
        columns.append(logs[np.arange(len(rows)),bundle['labels']][mask])
    result=np.column_stack(columns)
    require(result.shape==(int(mask.sum()),3) and np.isfinite(result).all(),'Invalid mixture target logs')
    return result,family[mask].tolist(),mask


def objective_readback(logs,families,support,weights):
    """Separate logsumexp mixture and family loop, not the solver's scaled API."""
    w=np.asarray(weights,dtype=np.float64)
    require(w.shape==(3,) and np.isfinite(w).all() and (w>=0).all() and abs(w.sum()-1)<1e-12,'Invalid learned simplex')
    active=w>0;terms=logs[:,active]+np.log(w[active]);maximum=terms.max(1)
    losses=-(maximum+np.log(np.exp(terms-maximum[:,None]).sum(1)))
    return math.fsum(float(losses[np.array(families)==c].mean()) for c in support)/len(support)


def bundle(e,path,digest):
    with np.load(io.BytesIO(e.read(path,digest)),allow_pickle=False) as a:
        return {k:a[k].copy() for k in ('logits','labels','sample_identifiers','class_names')}


def failure_diagnostics(exc):
    tb=exc.__traceback__;result={}
    while tb:
        frame=tb.tb_frame
        if frame.f_globals.get('__name__')==solver.__name__ and frame.f_code.co_name=='solve':
            for key in ('iteration','residual','value','w'):
                if key in frame.f_locals:
                    v=frame.f_locals[key];result[key]=v.tolist() if isinstance(v,np.ndarray) else v
        tb=tb.tb_next
    return result


def execute(output,approved=False):
    if not approved:raise PermissionError('Owner numerical-protocol approval required')
    e=Audit();receipt=e.js(VERIFICATION+'/receipt.json',RECEIPT_HASH)
    require(receipt['status']=='PASS' and receipt['checkpoints_verified']==15 and receipt['processing_seeds_verified']==5,'D054 incomplete')
    verified=e.js(VERIFICATION+'/summary.json',receipt['files']['summary.json'])
    require(verified['seeds']==list(SEEDS) and verified['beta']==1 and verified['gamma']==0,'Wrong verified experiment')
    stages,mapping,_,support=selection('Final',e);rows=stages['B'];require(len(mapping)==51 and len(support)==25 and len(rows)==4332,'Frozen counts mismatch')
    code={p:sha256((ROOT/p).read_bytes()) for p in ('scripts/learned_static_solver.py','scripts/fit_final_learned_static.py','tests/test_learned_static_solver.py','tests/test_fit_final_learned_static.py')}
    proposed=e.js('data/derived/pre_future_engineering/verification/20261017_041031_501420/receipt.json','dab91875f12736c1f1b0820b379f633532a40c50921c5610cfb5187656127af3')
    require(code['scripts/learned_static_solver.py']==proposed['source_sha256']['scripts/learned_static_solver.py'],'Approved proposed solver source changed')
    config=dict(decision='D055',owner_authorization='Explicit owner approval of proposed numerical protocol required; CLI flag is operator acknowledgement only',solver=solver.SPEC,
        solver_empirical_approval=True,seed_order=SEEDS,view_order=VIEWS,fit_month='2020-03',temperature_fit_month='2020-02',K=51,support=support,
        objective='class-balanced mixture NLL on complete retained B/S; full K softmax',source_sha256=code,D054_receipt_sha256=RECEIPT_HASH,
        initialization='uniform for each seed',CNN_training=False,inference=False,T_refit=False,Future_access=False)
    (output/'config.json').write_bytes(json_bytes(config));(output/'selection.json').write_bytes(json_bytes(dict(B=rows,class_to_idx=mapping,support=support)))
    results=[]
    for seed in SEEDS:
        bundles={};temperatures={};provenance={}
        records=[r for r in verified['rows'] if r['seed']==seed]
        require(len(records)==3 and {r['view'] for r in records}==set(VIEWS),'Missing seed/view')
        for record in records:
            view=record['view'];path=record['processing_receipt']
            if seed==42:
                q=e.js(path,verified['source_input_hashes'][path]);path=q['provenance']['B_file']
                cp=q['provenance']['checkpoint']['sha256'];t=record['temperature']
            else:
                processing=e.js(path,verified['source_input_hashes'][path]);binding=processing['binding']
                require(binding['seed']==seed and binding['view']==view,'Cross-seed processing mismatch')
                directory=str(Path(path).parent).replace('\\','/');stats=e.js(directory+'/statistics.json',verified['source_input_hashes'][directory+'/statistics.json'])
                require(stats['binding']==binding and stats['results']['temperature']['fit_ids']==[r['sha'] for r in stages['A']],'Fixed A temperature provenance mismatch')
                path=directory+'/B.npz';cp=binding['checkpoint_sha256'];t=stats['results']['temperature']['temperature']
            checkpoint=next(x for x in verified['checkpoints'] if x['seed']==seed and x['view']==view)
            require(cp==checkpoint['checkpoint_sha256'] and t==record['temperature'],'Checkpoint/T binding mismatch')
            bundles[view]=bundle(e,path,verified['source_input_hashes'][path]);temperatures[view]=t
            provenance[view]=dict(checkpoint_sha256=cp,temperature=t,B_logits=path,B_sha256=verified['source_input_hashes'][path])
        logs,families,mask=targets(bundles,rows,mapping,support,temperatures)
        require(int(mask.sum())==4064,'B/S cohort changed');directory=output/f'seed_{seed}';directory.mkdir()
        with (directory/'fit_inputs.npz').open('xb') as f:np.savez(f,log_target=logs,families=np.array(families),sample_identifiers=np.array([r['sha'] for r,take in zip(rows,mask) if take]))
        try:fit=solver.solve(logs,families,support)
        except RuntimeError as exc:
            (directory/'NOT_CONVERGED.json').write_bytes(json_bytes(dict(seed=seed,error=str(exc),diagnostic_only=failure_diagnostics(exc),weights_registered=False)))
            raise
        independent=objective_readback(logs,families,support,fit['weights'])
        require(abs(independent-fit['objective'])<=1e-12,'Independent mixture objective mismatch')
        require(fit['KKT_projected_residual']<=solver.SPEC['projected_gradient_tolerance'],'Convergence contract not achieved')
        result=dict(seed=seed,view_order=VIEWS,fit=fit,independent_objective=independent,provenance=provenance,support=support,fit_samples=4064,
            same_B_fit_objective_only=True,heldout_evaluation=False,Future_access=False)
        (directory/'weights.json').write_bytes(json_bytes(result));results.append(result)
        print(f'seed{seed}: CONVERGED, residual={fit["KKT_projected_residual"]:.3g}',flush=True)
    for p,h in e.hashes.items():e.read(p,h)
    require(all(sha256((ROOT/p).read_bytes())==h for p,h in code.items()),'Source changed during fit')
    (output/'summary.json').write_bytes(json_bytes(dict(status='PASS',results=results,source_input_hashes=e.hashes,config_sha256=sha256((output/'config.json').read_bytes()),Future_access=False)))
    (output/'receipt.json').write_bytes(json_bytes(dict(status='PASS',seeds=SEEDS,files={p.relative_to(output).as_posix():sha256(p.read_bytes()) for p in output.rglob('*') if p.is_file()},Future_access=False)))
    return results


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--owner-approved-numerical-protocol',action='store_true');a=p.parse_args()
    if not a.owner_approved_numerical_protocol:raise PermissionError('Explicit approval flag required before I/O')
    output=ROOT/'data/derived/final_learned_static'/datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f');output.mkdir(parents=True,exist_ok=False)
    try:
        execute(output,True);print('Receipt:',output/'receipt.json')
    except BaseException as exc:
        (output/'FAILED.json').write_bytes(json_bytes(dict(status='STOP',error=type(exc).__name__,message=str(exc),retry=False,method_changed=False)))
        print('STOP evidence:',output/'FAILED.json');raise
