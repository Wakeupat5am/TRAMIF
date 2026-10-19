"""D058 owner-approved five-seed CPU fit, with immutable D057 solver binding.

The approved solver bytes are reused unchanged through their numerical entrypoint.
Only this adapter can promote a converged result after independent verification.
No model execution, calibration fit, Q/D retuning or Future data loading.
"""
import os
for _key in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS'):
    os.environ[_key] = '1'
import argparse
from collections import Counter
from datetime import datetime, timezone
from decimal import Decimal, localcontext
import json
import math
from pathlib import Path
import unittest
import numpy as np
from scripts import learned_static_solver_v2 as solver
from scripts import fit_final_learned_static as previous
from scripts.diagnose_learned_static import decimal_projection, READBACK, READBACK_SHA, FAILED_RUN

ROOT, VIEWS, SEEDS = previous.ROOT, previous.VIEWS, previous.SEEDS
SOLVER_SHA = '8cafeea83b83986111e9376813e5d23b72ac87b4ffdc8356b08096771a298452'
VERIFIED = 'data/derived/learned_static_numerical_verification/20261018_002401_874991/receipt.json'
VERIFIED_SHA = '1637bade61a3a89aea336bbeebd91ca54e39ab2aa0535f58f80ba1cb46f7d093'
SOURCE_FILES = ('scripts/learned_static_solver.py', 'scripts/learned_static_solver_v2.py',
    'scripts/fit_final_learned_static.py', 'scripts/fit_final_learned_static_v2.py',
    'scripts/verify_final_multiseed.py', 'scripts/prepare_historical_logits.py',
    'scripts/frozen_encoder_extraction.py', 'scripts/uniform_fusion.py',
    'scripts/diagnose_learned_static.py', 'tests/test_fit_final_learned_static_v2.py',
    'tests/test_learned_static_numerics.py','tests/test_learned_static_solver.py',
    'tests/test_fit_final_learned_static.py')


def write(path, value):
    with path.open('xb') as stream:
        stream.write(previous.json_bytes(value))


def source_binding():
    hashes = {p:previous.sha256((ROOT/p).read_bytes()) for p in SOURCE_FILES}
    previous.require(hashes['scripts/learned_static_solver_v2.py']==SOLVER_SHA,
                     'D057 approved solver source mismatch')
    return hashes


def preservation(e):
    receipt = e.js(READBACK.relative_to(ROOT).as_posix(), READBACK_SHA)
    for name, digest in receipt['files'].items():
        e.read((FAILED_RUN/name).relative_to(ROOT).as_posix(),digest)
    return receipt


def seed_inputs(e, verified, seed, stages, mapping, support):
    bundles, temperatures, provenance = {}, {}, {}
    records=[r for r in verified['rows'] if r['seed']==seed]
    previous.require(len(records)==3 and {r['view'] for r in records}==set(VIEWS),'Missing seed/view')
    for record in records:
        view=record['view']; path=record['processing_receipt']
        if seed==42:
            q=e.js(path,verified['source_input_hashes'][path])
            path=q['provenance']['B_file']; cp=q['provenance']['checkpoint']['sha256']; t=record['temperature']
        else:
            processing=e.js(path,verified['source_input_hashes'][path]); binding=processing['binding']
            previous.require(binding['seed']==seed and binding['view']==view,'Cross-seed processing mismatch')
            directory=str(Path(path).parent).replace('\\','/')
            stats=e.js(directory+'/statistics.json',verified['source_input_hashes'][directory+'/statistics.json'])
            previous.require(stats['binding']==binding and stats['results']['temperature']['fit_ids']==[r['sha'] for r in stages['A']], 'Fixed February T provenance mismatch')
            path=directory+'/B.npz'; cp=binding['checkpoint_sha256']; t=stats['results']['temperature']['temperature']
        checkpoint=next(x for x in verified['checkpoints'] if x['seed']==seed and x['view']==view)
        previous.require(cp==checkpoint['checkpoint_sha256'] and t==record['temperature'],'Checkpoint/T mismatch')
        bundles[view]=previous.bundle(e,path,verified['source_input_hashes'][path]); temperatures[view]=t
        provenance[view]=dict(checkpoint_sha256=cp,temperature=t,B_logits=path,B_sha256=verified['source_input_hashes'][path],processing_receipt=record['processing_receipt'])
    logs,families,mask=previous.targets(bundles,stages['B'],mapping,support,temperatures)
    ids=[r['sha'] for r,take in zip(stages['B'],mask) if take]
    previous.require(len(ids)==len(set(ids))==4064 and len(support)==25,'Frozen B/S mismatch')
    return dict(logs=logs,families=families,ids=ids,provenance=provenance)


def independent_kkt(logs, families, support, weights):
    """Scalar Decimal60, exact family balance and raw exp(log); verification only.

    No optimizer/alternative solution: independently check the same unit-projected
    stationarity definition, on the solver's implicit exact simplex point.
    """
    counts=Counter(families)
    previous.require(set(counts)==set(support),'KKT support mismatch')
    with localcontext() as ctx:
        ctx.prec=60
        w=[Decimal.from_float(float(x)) for x in weights]
        pivot=int(np.argmax(weights));w[pivot]=1-sum(x for j,x in enumerate(w) if j!=pivot)
        gradient=[Decimal(0)]*3; value=Decimal(0)
        for row,family in zip(logs,families):
            probabilities=[Decimal.from_float(float(x)).exp() for x in row]
            mixture=sum(a*b for a,b in zip(w,probabilities))
            if not mixture.is_finite() or mixture<=0:
                raise FloatingPointError('Independent zero/nonfinite mixture')
            alpha=Decimal(1)/(len(support)*counts[family])
            value-=alpha*mixture.ln()
            for j in range(3):gradient[j]-=alpha*probabilities[j]/mixture
        projected=decimal_projection([x-g for x,g in zip(w,gradient)])
        residual=max(abs(x-y) for x,y in zip(w,projected))
        multiplier=sum(x*g for x,g in zip(w,gradient))
        return dict(precision=60,projected_residual=str(residual),objective=str(value),
            gradient=[str(x) for x in gradient],gradient_minus_multiplier=[str(g-multiplier) for g in gradient],
            simplex_sum=str(sum(w)),implicit_pivot=pivot,
            implicit_storage_correction=str(w[pivot]-Decimal.from_float(float(weights[pivot]))),
            PASS=bool(residual<=Decimal('1e-10')))


def verify_fit(logs, families, support, result):
    previous.require(result['status']=='CONVERGED_CANDIDATE_NOT_APPROVED','Solver did not converge')
    weights=np.asarray(result['diagnostic_weights'],dtype=np.float64)
    solver.point(weights)
    previous.require(np.isfinite(weights).all() and (weights>=0).all(),'Invalid weights')
    residual=result['KKT_projected_residual']
    previous.require(math.isfinite(residual) and residual<=1e-10,'Convergence residual failed')
    previous.require(type(result['iterations']) is int and 0<=result['iterations']<=10000,'Iteration policy mismatch')
    objective=previous.objective_readback(logs,families,support,weights)
    previous.require(math.isfinite(result['objective']) and abs(objective-result['objective'])<=1e-12,'Objective readback mismatch')
    kkt=independent_kkt(logs,families,support,weights)
    previous.require(kkt['PASS'],'Independent KKT residual failed')
    previous.require(abs(float(kkt['objective'])-objective)<=1e-12,'Decimal objective mismatch')
    return dict(status='PASS',simplex_storage_sum=math.fsum(weights.tolist()),
        independent_objective=objective,independent_KKT=kkt)


def failure_details(exc):
    result={};tb=exc.__traceback__
    while tb:
        frame=tb.tb_frame
        if frame.f_globals.get('__name__')==solver.__name__ and frame.f_code.co_name=='solve_diagnostic':
            for key in ('iteration','residual','w','gradient','step','comparison'):
                if key in frame.f_locals:
                    value=frame.f_locals[key];result[key]=value.tolist() if isinstance(value,np.ndarray) else value
        tb=tb.tb_next
    return result


def recheck(e, sources):
    for path,digest in list(e.hashes.items()):e.read(path,digest)
    previous.require(source_binding()==sources,'Source changed during execution')


def execute(output, approved=False):
    if not approved:raise PermissionError('Explicit D058 owner production approval required')
    sources=source_binding();e=previous.Audit();preservation(e)
    d057=e.js(VERIFIED,VERIFIED_SHA)
    previous.require(d057['tests_run']==31 and d057['source_sha256']['scripts/learned_static_solver_v2.py']==SOLVER_SHA,'D057 verification mismatch')
    for name in ('scripts/learned_static_solver.py','scripts/fit_final_learned_static.py','scripts/diagnose_learned_static.py'):
        previous.require(sources[name]==d057['source_sha256'][name],'D057 dependency source changed: '+name)
    receipt=e.js(previous.VERIFICATION+'/receipt.json',previous.RECEIPT_HASH)
    previous.require(receipt['status']=='PASS' and receipt['checkpoints_verified']==15 and receipt['processing_seeds_verified']==5,'D054 incomplete')
    verified=e.js(previous.VERIFICATION+'/summary.json',receipt['files']['summary.json'])
    previous.require(verified['seeds']==list(SEEDS) and verified['beta']==1 and verified['gamma']==0,'Wrong registered seeds')
    stages,mapping,_,support=previous.selection('Final',e)
    previous.require(len(stages['A'])==3799 and len(stages['B'])==4332 and len(mapping)==51 and len(support)==25,'Frozen counts mismatch')
    config=dict(decision='D058',owner_approval='Explicit approval of unchanged D057 solver bytes and five-seed CPU execution',
        solver_sha256=SOLVER_SHA,solver_version=solver.VERSION,solver_spec=solver.SPEC,
        production_adapter_approved=True,underlying_candidate_metadata_preserved=True,
        seeds=list(SEEDS),view_order=VIEWS,K=51,fit_month='2020-03',temperature_fit_month='2020-02',
        support=support,fit_samples=4064,objective='equal-family mixture NLL',
        independent_KKT_tolerance='1e-10',objective_readback_atol=1e-12,
        source_sha256=sources,D054_receipt_sha256=previous.RECEIPT_HASH,D057_receipt_sha256=VERIFIED_SHA,
        Future_access=False,CNN_training=False,inference=False,T_refit=False,QD_retune=False,retry=False)
    write(output/'config.json',config)
    source_dir=output/'source';source_dir.mkdir()
    for name in sources:
        target=source_dir/name;target.parent.mkdir(parents=True,exist_ok=True)
        with target.open('xb') as f:f.write((ROOT/name).read_bytes())
    suite=unittest.defaultTestLoader.loadTestsFromNames(['tests.test_learned_static_numerics','tests.test_learned_static_solver','tests.test_fit_final_learned_static','tests.test_fit_final_learned_static_v2'])
    with (output/'targeted_tests.txt').open('x',encoding='utf-8') as f:
        tests=unittest.TextTestRunner(stream=f,verbosity=2).run(suite)
    previous.require(tests.wasSuccessful(),'Targeted tests failed before preflight')
    write(output/'tests_receipt.json',dict(status='PASS',run=tests.testsRun,failures=0,source_sha256=sources))
    write(output/'selection.json',dict(B=stages['B'],class_to_idx=mapping,support=support))
    # Check all five datasets and provenance BEFORE the first production solve.
    inputs={seed:seed_inputs(e,verified,seed,stages,mapping,support) for seed in SEEDS}
    for seed,data in inputs.items():
        directory=output/f'seed_{seed}';directory.mkdir()
        with (directory/'fit_inputs.npz').open('xb') as f:
            np.savez(f,log_target=data['logs'],families=np.array(data['families']),sample_identifiers=np.array(data['ids']))
    recheck(e,sources)
    previous.require(all(inputs[s]['ids']==inputs[42]['ids'] for s in SEEDS),'Cross-seed canonical IDs mismatch')
    write(output/'preflight.json',dict(status='PASS',seeds=list(SEEDS),samples_per_seed=4064,K=51,support=support,
        same_ID_order=all(inputs[s]['ids']==inputs[42]['ids'] for s in SEEDS),
        D056_preserved=True,source_input_hashes=dict(e.hashes)))
    results=[]
    for seed in SEEDS:
        data=inputs[seed];directory=output/f'seed_{seed}'
        numerical=None
        try:
            numerical=solver.solve_diagnostic(data['logs'],data['families'],support)
            verification=verify_fit(data['logs'],data['families'],support,numerical)
            recheck(e,sources)
            write(directory/'verification.json',verification)
            # Promotion occurs here only, after every numerical/provenance check.
            fit=dict(weights=numerical['diagnostic_weights'],objective=numerical['objective'],
                iterations=numerical['iterations'],KKT_projected_residual=numerical['KKT_projected_residual'],
                status='CONVERGED_VERIFIED',solver_version=solver.VERSION,solver_sha256=SOLVER_SHA)
            result=dict(seed=seed,view_order=VIEWS,fit=fit,verification=verification,
                provenance=data['provenance'],support=support,K=51,fit_samples=4064,
                input_sha256=previous.sha256((directory/'fit_inputs.npz').read_bytes()),
                config_sha256=previous.sha256((output/'config.json').read_bytes()),
                weights_registered=True,same_B_fit_objective_only=True,heldout_evaluation=False,Future_access=False)
            write(directory/'weights.json',result)
            write(directory/'COMPLETE.json',dict(status='PASS',seed=seed,files={p.name:previous.sha256(p.read_bytes()) for p in directory.iterdir() if p.is_file()}))
            results.append(result)
            print(f'seed{seed}: PASS, residual={fit["KKT_projected_residual"]:.4g}',flush=True)
        except BaseException as exc:
            write(directory/'FAILED.json',dict(status='STOP',seed=seed,error=type(exc).__name__,message=str(exc),
                diagnostics=failure_details(exc),unregistered_solver_result=numerical,weights_registered=False,
                further_seeds_not_attempted=[s for s in SEEDS if s>seed],retry=False))
            raise
    recheck(e,sources)
    write(output/'summary.json',dict(status='PASS',results=results,source_input_hashes=dict(e.hashes),
        tests_passed=tests.testsRun,source_sha256=sources,Future_access=False,research_validated=False))
    write(output/'receipt.json',dict(status='PASS',seeds=list(SEEDS),weights_registered=5,
        tests_passed=tests.testsRun,solver_sha256=SOLVER_SHA,Future_access=False,D056_preserved=True,
        files={p.relative_to(output).as_posix():previous.sha256(p.read_bytes()) for p in output.rglob('*') if p.is_file()}))
    return results


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--owner-approved-production-v2',action='store_true');a=p.parse_args()
    if not a.owner_approved_production_v2:raise PermissionError('Explicit D058 production approval required')
    output=ROOT/'data/derived/final_learned_static_v2'/datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')
    output.mkdir(parents=True,exist_ok=False)
    try:
        execute(output,True);print('Receipt:',output/'receipt.json')
    except BaseException as exc:
        write(output/'FAILED.json',dict(status='STOP',error=type(exc).__name__,message=str(exc),retry=False,method_changed=False))
        print('STOP evidence:',output/'FAILED.json');raise
