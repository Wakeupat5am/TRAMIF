"""D057 bounded verification: synthetic solves and ONE saved-failure candidate step.

No corrected empirical optimization trajectory, production fits or registration.
"""
import os
for _key in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS'):
    os.environ[_key] = '1'
import hashlib
import io
import json
import sys
import unittest
from datetime import datetime, timezone
from decimal import Decimal, localcontext
from pathlib import Path
import numpy as np
from scripts import learned_static_solver as old
from scripts import learned_static_solver_v2 as new
from scripts.diagnose_learned_static import (
    ROOT, FAILED_RUN, READBACK, READBACK_SHA, digest, write_json, DecimalReference)

REPRODUCTION = ROOT/'data/derived/learned_static_diagnosis/20261018_001906_619077'


def represented(w):
    result = [Decimal.from_float(float(x)) for x in w]
    pivot = int(np.argmax(w))
    result[pivot] = 1-sum(x for j,x in enumerate(result) if j != pivot)
    return result


def check_case(logs, families, support):
    a, alpha, offset = old.prepare(logs, families, support)
    corrected = new.solve_diagnostic(logs, families, support)
    try:
        original = old.solve(logs, families, support)
        original['weights_registered'] = False
    except RuntimeError as exc:
        original = dict(status='CONTROL_FAILURE', error=str(exc), weights_registered=False)
    with localcontext() as ctx:
        ctx.prec = 70
        w = represented(corrected['diagnostic_weights'])
        gradient = DecimalReference(logs, alpha, exact_logs=True).gradient(w)
        active = [j for j,x in enumerate(w) if x > Decimal('1e-12')]
        mu = sum(gradient[j] for j in active)/len(active)
        equality_error = max(abs(gradient[j]-mu) for j in active)
        inactive = [gradient[j]-mu for j in range(3) if j not in active]
        if equality_error > Decimal('5e-10') or any(x < Decimal('-5e-10') for x in inactive):
            raise AssertionError('Independent control KKT mismatch')
        kkt = dict(active_stationarity_error=str(equality_error), inactive_slacks=[str(x) for x in inactive])
    # The log-sum-exp readback is independent of row-max scaling and the chart.
    w = np.array(corrected['diagnostic_weights']);active = w>0
    terms = logs[:,active]+np.log(w[active]);maximum=terms.max(1)
    losses = -(maximum+np.log(np.exp(terms-maximum[:,None]).sum(1)))
    readback = float(np.mean([losses[np.asarray(families)==f].mean() for f in support]))
    if abs(readback-corrected['objective'])>2e-12:
        raise AssertionError('Independent objective readback mismatch')
    return dict(original=original, corrected=corrected, independent_KKT=kkt, independent_objective=readback)


def execute():
    if digest(READBACK) != READBACK_SHA:
        raise ValueError('D056 receipt mismatch')
    old_receipt = json.loads(READBACK.read_text())
    for name, expected in old_receipt['files'].items():
        if digest(FAILED_RUN/name) != expected:
            raise ValueError('D056 artifact mismatch')
    config = json.loads((FAILED_RUN/'config.json').read_text())
    for name, expected in config['source_sha256'].items():
        if digest(ROOT/name) != expected:
            raise ValueError('Original code changed')
    replay_receipt = json.loads((REPRODUCTION/'receipt.json').read_text())
    for name, expected in replay_receipt['files'].items():
        if digest(REPRODUCTION/name) != expected:
            raise ValueError('Reproduction evidence mismatch')
    source_names = ['scripts/learned_static_solver.py','scripts/learned_static_solver_v2.py',
                    'scripts/diagnose_learned_static.py','scripts/verify_learned_static_numerics.py',
                    'tests/test_learned_static_numerics.py','tests/test_learned_static_solver.py',
                    'tests/test_fit_final_learned_static.py','scripts/fit_final_learned_static.py']
    sources = {name:digest(ROOT/name) for name in source_names}
    output=ROOT/'data/derived/learned_static_numerical_verification'/datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')
    output.mkdir(parents=True,exist_ok=False)
    try:
        suite=unittest.defaultTestLoader.loadTestsFromNames([
            'tests.test_learned_static_numerics','tests.test_learned_static_solver','tests.test_fit_final_learned_static'])
        with (output/'targeted_tests.txt').open('x',encoding='utf-8') as stream:
            tests=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
        if not tests.wasSuccessful():
            raise AssertionError('Targeted tests failed')
        # Exactly one corrected step at the already-recorded failure. NO solve call
        # receives the saved empirical logs. The original replay is not repeated.
        with np.load(io.BytesIO((FAILED_RUN/'seed_42/fit_inputs.npz').read_bytes()),allow_pickle=False) as z:
            logs=z['log_target'];families=z['families']
        a,alpha,offset=old.prepare(logs,families,config['support'])
        failure=json.loads((FAILED_RUN/'seed_42/NOT_CONVERGED.json').read_text())['diagnostic_only']
        w=np.array(failure['w']);residual,g,anchor=new.stationarity(w,a,alpha)
        v=new.project(w-g)
        comparison=new.armijo_gap(w,v,a,alpha,anchor)
        with localcontext() as ctx:
            ctx.prec=70
            ref=DecimalReference(a,alpha);dw=represented(w);dv=represented(v)
            dg=ref.gradient(dw)
            gd=sum(x*(b-c) for x,b,c in zip(dg,dv,dw))
            delta=ref.difference(dw,dv);gap=delta-Decimal('0.0001')*gd
            if not comparison['accepted'] or not gap<0:
                raise AssertionError('Single corrected diagnostic step not justified')
            relative_gap_error=abs(Decimal.from_float(comparison['gap'])/gap-1)
            if relative_gap_error>Decimal('1e-6'):
                raise AssertionError('Float64 Armijo gap disagrees with Decimal')
            finite_differences=[]
            for direction in ([1,-1,0],[0,1,-1]):
                derivative=sum(x*y for x,y in zip(dg,direction))
                for eps in ('1e-6','1e-7','1e-8'):
                    h=Decimal(eps);plus=[x+h*y for x,y in zip(dw,direction)];minus=[x-h*y for x,y in zip(dw,direction)]
                    numerical=ref.difference(minus,plus)/(2*h)
                    finite_differences.append(dict(direction=direction,h=eps,absolute_error=str(abs(numerical-derivative))))
            single=dict(status='NOT_REGISTERED_ONE_DIAGNOSTIC_STEP',weights_registered=False,
                original_weights=failure['w'],candidate=v.tolist(),residual_before=residual,
                residual_after=new.stationarity(v,a,alpha)[0],float64=comparison,
                decimal_gap=str(gap),decimal_objective_difference=str(delta),
                relative_gap_error=str(relative_gap_error),finite_differences_70_digits=finite_differences)
        write_json(output/'single_failure_point_check.json',single)
        # Predefined controlled objectives, not a parameter/scenario search.
        rows=np.repeat(np.array([[.8,.1,.1],[.1,.8,.1],[.1,.1,.8]]),[60,25,15],axis=0)
        cases={
          'known_interior': (np.log(rows),['a']*100,['a']),
          'boundary_vertex': (np.log([[.9,.3,.1],[.8,.2,.1]]),['a','b'],['a','b']),
          'boundary_edge': (np.log([[.9,.1,.05],[.1,.9,.05]]),['a','b'],['a','b']),
          'flat': (np.log([[.3,.3,.3],[.2,.2,.2]]),['a','b'],['a','b']),
          'near_identical': (np.log([[.3+1e-8,.3,.3-1e-8],[.3-1e-8,.3,.3+1e-8]]),['a','b'],['a','b']),
          'tiny_probabilities': (np.log([[.8,.2,.1],[.1,.7,.3],[.3,.2,.6]])-1000,['a','b','c'],['a','b','c'])}
        controls={name:check_case(*args) for name,args in cases.items()}
        write_json(output/'controlled_comparison.json',controls)
        for name,h in old_receipt['files'].items():
            assert digest(FAILED_RUN/name)==h
        for name,h in sources.items():
            assert digest(ROOT/name)==h
        (output/'sources').mkdir()
        for name in sources:
            (output/'sources'/Path(name).name).write_bytes((ROOT/name).read_bytes())
        write_json(output/'receipt.json',dict(status='TARGETED_NUMERICAL_VERIFICATION_PASS',
            tests_run=tests.testsRun,failures=len(tests.failures),errors=len(tests.errors),
            controls=len(controls),diagnostic_empirical_candidate_steps=1,
            corrected_empirical_solves=0,registered_weights=0,Future_access=False,
            protocol_parameters_unchanged=True,production_integration=False,
            original_sources_and_D056_files_unchanged=True,candidate_status='NOT_APPROVED_FOR_REGISTERED_FITTING',
            source_sha256=sources,original_readback_sha256=READBACK_SHA,
            reproduction_receipt_sha256=digest(REPRODUCTION/'receipt.json'),
            files={p.relative_to(output).as_posix():digest(p) for p in output.rglob('*') if p.is_file()}))
        print(json.dumps(dict(path=str(output),tests_passed=tests.testsRun,
            single_step_gap=comparison['gap'],relative_gap_error=str(relative_gap_error),
            controls={name:dict(original=x['original']['status'],corrected=x['corrected']['status'],
                iterations=x['corrected']['iterations'],residual=x['corrected']['KKT_projected_residual']) for name,x in controls.items()}),indent=2))
    except BaseException as exc:
        write_json(output/'FAILED.json',dict(status='STOP',error=str(exc),weights_registered=False))
        raise


if __name__=='__main__':execute()
