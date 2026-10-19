"""D057 finite, read-only D056 reproduction. Never registers mixture weights.

Only the pinned seed42 input snapshot is read; no logits/encoder/Future loader.
Decimal is an independent diagnostic oracle, never an empirical solver.
"""
import os
for _key in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS'):
    os.environ[_key] = '1'
import hashlib
import io
import json
import math
import platform
import sys
from datetime import datetime, timezone
from decimal import Decimal, localcontext
from pathlib import Path

import numpy as np
from scripts import learned_static_solver as original

ROOT = Path(__file__).resolve().parents[1]
FAILED_RUN = ROOT / 'data/derived/final_learned_static/20261018_000858_569331'
READBACK = ROOT / 'data/derived/final_learned_static_failure_verification/20261018_000959_177046/receipt.json'
READBACK_SHA = '6e8db2df8af08b4bb1d7eecf377f8333b6042b75433efcc60adf0c7307cc8abc'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def replay(logs, families, support):
    """Same operations and order as original; recording has no update side effects."""
    args = original.prepare(logs, families, support)
    w = np.full(3, 1/3, dtype=np.float64)
    trace = []
    for iteration in range(original.SPEC['max_iterations'] + 1):
        value, grad = original.objective_gradient(w, *args)
        residual = float(np.max(np.abs(w-original.simplex_projection(w-grad))))
        record = dict(iteration=iteration, weights=w.tolist(), objective=value,
                      gradient=grad.tolist(), residual=residual, trials=[])
        trace.append(record)
        if residual <= original.SPEC['projected_gradient_tolerance']:
            return dict(status='CONVERGED_DIAGNOSTIC_ONLY', trace=trace)
        if iteration == original.SPEC['max_iterations']:
            break
        step = original.SPEC['step']
        for backtrack in range(original.SPEC['max_backtracks']):
            candidate = original.simplex_projection(w-step*grad)
            direction = candidate-w
            try:
                next_value, _ = original.objective_gradient(candidate, *args)
            except FloatingPointError:
                next_value = float('inf')
            gd = float(grad@direction)
            accepted = next_value <= value+original.SPEC['armijo']*gd
            record['trials'].append(dict(backtrack=backtrack, step=step,
                candidate=candidate.tolist(), delta_objective=next_value-value if math.isfinite(next_value) else None,
                directional_derivative=gd, armijo_rhs_increment=original.SPEC['armijo']*gd,
                direction_mass=math.fsum(direction.tolist()), accepted=bool(accepted)))
            if accepted:
                w = candidate
                break
            step *= original.SPEC['backtrack']
        else:
            return dict(status='ARMIJO_FAILED_REPRODUCED', trace=trace)
    return dict(status='ITERATION_LIMIT_DIAGNOSTIC_ONLY', trace=trace)


def decimal_projection(x):
    u = sorted(x, reverse=True)
    prefix = Decimal(0)
    theta = None
    for j, value in enumerate(u, 1):
        prefix += value
        trial = (prefix-1)/j
        if value > trial:
            theta = trial
    return [max(t-theta, Decimal(0)) for t in x]


class DecimalReference:
    """Independent scalar objective/gradient; context precision set by caller.

    exact_logs=True evaluates exp(log_target) independently of row-max NumPy.
    Otherwise it isolates arithmetic error on the same float64 prepared matrix.
    """
    def __init__(self, values, alpha, exact_logs=False):
        self.rows = [[Decimal.from_float(float(x)) for x in row] for row in values]
        if exact_logs:
            self.rows = [[x.exp() for x in row] for row in self.rows]
        self.alpha = [Decimal.from_float(float(x)) for x in alpha]

    def mix(self, w):
        return [sum((a*b for a, b in zip(row, w)), Decimal(0)) for row in self.rows]

    def gradient(self, w):
        mix = self.mix(w)
        return [-sum((a*row[j]/m for a, row, m in zip(self.alpha, self.rows, mix)), Decimal(0)) for j in range(3)]

    def difference(self, w, v):
        return -sum((a*(n/m).ln() for a, m, n in zip(self.alpha, self.mix(w), self.mix(v))), Decimal(0))


def numerical_evidence(logs, families, support, replayed):
    matrix, alpha, offset = original.prepare(logs, families, support)
    last = replayed['trace'][-1]
    w = np.array(last['weights'])
    grad = np.array(last['gradient'])
    evidence = dict(objective_ulp=float(np.spacing(last['objective'])),
                    exact_input_matrix='float64 prepared matrix; separate raw-log oracle below')
    with localcontext() as ctx:
        ctx.prec = 70
        ref = DecimalReference(matrix, alpha)
        dw = [Decimal.from_float(float(x)) for x in w]
        dg = ref.gradient(dw)
        evidence['sum_weights_minus_one_decimal'] = str(sum(dw)-1)
        evidence['gradient_decimal'] = [str(x) for x in dg]
        evidence['gradient_float64_max_error'] = max(abs(float(x)-y) for x, y in zip(dg, grad))
        evidence['interior_KKT_gradient_spread'] = str(max(dg)-min(dg))
        trials = []
        for j in (0, 1, 5, 20, 59):
            trial = last['trials'][j]
            dv = [Decimal.from_float(x) for x in trial['candidate']]
            direction = [b-a for a, b in zip(dw, dv)]
            exact_delta = ref.difference(dw, dv)
            exact_gd = sum(a*b for a, b in zip(dg, direction))
            mass = sum(direction)
            # Exact decomposition into a normal component and a tangent component.
            tangent_derivative = exact_gd-dg[0]*mass
            trials.append(dict(backtrack=j, delta_decimal=str(exact_delta),
                gd_decimal=str(exact_gd), mass_decimal=str(mass),
                normal_derivative=str(dg[0]*mass), tangent_derivative=str(tangent_derivative),
                armijo_pass_decimal=exact_delta <= Decimal('0.0001')*exact_gd))
        evidence['rounded_original_candidates'] = trials
        # Exact simplex representative (largest coordinate implicit) removes only
        # float64 feasibility error. Exact Euclidean projected step, no new fit.
        pivot = int(w.argmax())
        feasible = dw.copy()
        feasible[pivot] = 1-sum(x for j, x in enumerate(dw) if j != pivot)
        fg = ref.gradient(feasible)
        candidate = decimal_projection([a-b for a, b in zip(feasible, fg)])
        direction = [b-a for a, b in zip(feasible, candidate)]
        gd = sum(a*b for a, b in zip(fg, direction))
        df = ref.difference(feasible, candidate)
        evidence['exact_simplex_unit_step'] = dict(delta=str(df), gd=str(gd),
            armijo_rhs=str(Decimal('0.0001')*gd), armijo_pass=df <= Decimal('0.0001')*gd,
            projection_inequality_gap=str(gd+sum(x*x for x in direction)),
            residual=str(max(abs(x) for x in direction)), mass=str(sum(direction)),
            status='NOT_REGISTERED_SINGLE_DIAGNOSTIC_STEP')
        finite_differences = []
        for name, point in [('uniform', [Decimal(1)/3]*3), ('failure_on_simplex', feasible)]:
            g = ref.gradient(point)
            for direction in ([1, -1, 0], [0, 1, -1]):
                analytical = sum(a*b for a, b in zip(g, direction))
                for eps in ('0.001', '0.0001', '0.00001', '0.000001'):
                    h = Decimal(eps)
                    plus = [a+h*b for a, b in zip(point, direction)]
                    minus = [a-h*b for a, b in zip(point, direction)]
                    numerical = ref.difference(minus, plus)/(2*h)
                    finite_differences.append(dict(point=name, direction=direction, h=eps,
                        analytical=str(analytical), numerical=str(numerical),
                        absolute_error=str(abs(analytical-numerical))))
        evidence['finite_differences_70_digits'] = finite_differences
        raw = DecimalReference(logs, alpha, exact_logs=True)
        rawg = raw.gradient(feasible)
        evidence['raw_log_exp_vs_prepared_gradient_max_error'] = str(max(abs(a-b) for a, b in zip(rawg, fg)))
        evidence['raw_log_exp_exact_step_delta'] = str(raw.difference(feasible, candidate))
    return evidence


def execute():
    if digest(READBACK) != READBACK_SHA:
        raise ValueError('Pinned D056 readback mismatch')
    receipt = json.loads(READBACK.read_text())
    for name, expected in receipt['files'].items():
        if digest(FAILED_RUN/name) != expected:
            raise ValueError('D056 input hash mismatch: '+name)
    config = json.loads((FAILED_RUN/'config.json').read_text())
    for name, expected in config['source_sha256'].items():
        if digest(ROOT/name) != expected:
            raise ValueError('Original source snapshot mismatch: '+name)
    with np.load(io.BytesIO((FAILED_RUN/'seed_42/fit_inputs.npz').read_bytes()), allow_pickle=False) as z:
        logs, families = z['log_target'].copy(), z['families'].copy()
    expected = json.loads((FAILED_RUN/'seed_42/NOT_CONVERGED.json').read_text())['diagnostic_only']
    output = ROOT/'data/derived/learned_static_diagnosis'/datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')
    output.mkdir(parents=True, exist_ok=False)
    try:
        result = replay(logs, families, config['support'])
        last = result['trace'][-1]
        assert result['status'] == 'ARMIJO_FAILED_REPRODUCED'
        assert last['iteration'] == expected['iteration'] and last['residual'] == expected['residual']
        assert last['objective'] == expected['value'] and last['weights'] == expected['w']
        # Also call the untouched implementation, independently of instrumentation.
        try:
            original.solve(logs, families, config['support'])
        except RuntimeError as exc:
            assert str(exc) == 'Armijo failed; no empirical fallback'
        else:
            raise AssertionError('Original failure not reproduced')
        write_json(output/'trajectory.json', dict(weights_registered=False, **result))
        evidence = numerical_evidence(logs, families, config['support'], result)
        write_json(output/'numerical_evidence.json', evidence)
        sources = {p: digest(ROOT/p) for p in ('scripts/learned_static_solver.py', 'scripts/diagnose_learned_static.py')}
        for name in sources:
            (output/Path(name).name).write_bytes((ROOT/name).read_bytes())
        for name, expected_hash in receipt['files'].items():
            assert digest(FAILED_RUN/name) == expected_hash
        write_json(output/'receipt.json', dict(status='DIAGNOSTIC_REPRODUCTION_PASS',
            candidate_status='NOT_REGISTERED', weights_registered=False, registered_fitting=False,
            Future_access=False, original_files=receipt['files'], original_run=str(FAILED_RUN),
            source_sha256=sources, python=sys.version, numpy=np.__version__, platform=platform.platform(),
            source_hashes_unchanged=all(digest(ROOT/p)==h for p,h in config['source_sha256'].items()),
            files={p.name: digest(p) for p in output.iterdir() if p.is_file()}))
        print(json.dumps(dict(path=str(output), original_iteration=last['iteration'],
            original_residual=last['residual'], exact_simplex_unit_step=evidence['exact_simplex_unit_step']), indent=2))
    except BaseException as exc:
        write_json(output/'FAILED.json', dict(status='STOP', error=str(exc), weights_registered=False))
        raise


if __name__ == '__main__':
    execute()
