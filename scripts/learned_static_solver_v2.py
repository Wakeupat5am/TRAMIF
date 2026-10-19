"""D057 candidate arithmetic correction; NOT APPROVED for registered fitting.

Same convex objective, Euclidean projected gradient, Armijo and stopping contract.
An O(eps) simplex storage error is represented by an implicit largest coordinate,
not treated as a genuine change in probability mass. Armijo compares its algebraic
gap directly, using a stable log1p remainder, not two rounded O(1) objectives.
Pure numerical API. No data loader, CLI, artifact registration or legacy imports
from the production runner. All internal calculations are float64.
"""
import math
import numpy as np
from scripts import learned_static_solver as legacy

VERSION = 'D057-affine-simplex-armijo-v2-candidate'
SPEC = {**legacy.SPEC, 'version': VERSION, 'empirical_approval': False,
        'arithmetic': 'implicit largest simplex coordinate; centered gradient; direct Armijo gap'}
EPS = np.finfo(np.float64).eps


def point(weights):
    """Float64 storage plus implicit correction to largest coordinate.

    Exact represented point: p_j=w_j for j!=pivot; p_pivot=1-sum(other w).
    fsum(1,-w0,-w1,-w2) recovers the tiny storage mass error without 1-sum(w).
    No epsilon weight clipping or statistical regularization.
    """
    w = np.asarray(weights, dtype=np.float64)
    if w.shape != (3,) or not np.isfinite(w).all() or (w < 0).any():
        raise ValueError('Three finite nonnegative simplex coordinates required')
    correction = math.fsum([1., *(-float(x) for x in w)])
    if abs(correction) > 8*EPS:
        raise ValueError('Simplex storage error exceeds eight float64 eps')
    pivot = int(np.argmax(w))
    if w[pivot]+correction < 0:
        raise ValueError('Invalid implicit coordinate')
    return w, pivot, correction


def project(x):
    """Unchanged Euclidean projection, with stable storage of its largest term."""
    v = legacy.simplex_projection(x)
    pivot = int(np.argmax(v))
    v[pivot] = 1.-math.fsum(float(v[j]) for j in range(3) if j != pivot)
    point(v)
    return v


def mixture(weights, scaled):
    w, pivot, correction = point(weights)
    # Positive terms, plus only O(eps) correction to a positive largest term.
    mix = np.sum(scaled*w, axis=1, dtype=np.float64)+correction*scaled[:, pivot]
    if not np.isfinite(mix).all() or (mix <= 0).any():
        raise FloatingPointError('Zero/nonfinite target mixture')
    return mix


def centered_gradient(weights, scaled, alpha, anchor):
    """g_j-g_anchor via per-row differences, avoiding subtraction of O(1) sums."""
    mix = mixture(weights, scaled)
    g = np.zeros(3, dtype=np.float64)
    for j in range(3):
        if j != anchor:
            terms = alpha*((scaled[:, j]-scaled[:, anchor])/mix)
            if not np.isfinite(terms).all():
                raise FloatingPointError('Nonfinite gradient')
            g[j] = -math.fsum(terms.tolist())
    return g


def objective(weights, scaled, alpha, offset):
    value = offset-math.fsum((alpha*np.log(mixture(weights, scaled))).tolist())
    if not math.isfinite(value):
        raise FloatingPointError('Nonfinite objective')
    return value


def tangent_coordinates(w, v, anchor):
    """Difference of represented points in one affine simplex chart."""
    w, wp, wc = point(w)
    v, vp, vc = point(v)
    free = [j for j in range(3) if j != anchor]
    d = np.array([math.fsum([float(v[j]), -float(w[j]),
                            vc if j == vp else 0., -wc if j == wp else 0.])
                  for j in free], dtype=np.float64)
    return free, d


def log_remainder(z):
    """z-log1p(z), stable at zero; no relaxation of the Armijo inequality.

    |z|<=1e-3: terms k=2..12. Omitted tail <= |z|^13/(13*(1-|z|)),
    <=7.71e-41 at the switch; relative truncation <1.6e-34. All float64.
    Else log1p subtraction; cancellation loses at most ~1/|z| conditioning.
    """
    z = np.asarray(z, dtype=np.float64)
    if not np.isfinite(z).all() or (z <= -1).any():
        raise FloatingPointError('Invalid relative mixture change')
    out = np.empty_like(z)
    small = np.abs(z) <= 1e-3
    x = z[small]
    poly = np.full_like(x, 1/12)
    for k in range(11, 1, -1):
        poly = ((-1.)**k)/k + x*poly
    out[small] = x*x*poly
    out[~small] = z[~small]-np.log1p(z[~small])
    if not np.isfinite(out).all() or (out < 0).any():
        raise FloatingPointError('Invalid convex log remainder')
    return out


def armijo_gap(w, v, scaled, alpha, anchor):
    """F(v)-F(w)-c*g.(v-w) = (1-c)*g.d + sum alpha*(z-log1p(z)).

    z_i = (a_i.(v-w))/(a_i.w) evaluated in the simplex tangent chart.
    The additive objective offset cancels exactly. No slack, f(x) rounding
    allowance, gradient tolerance relaxation, or near-converged acceptance.
    """
    free, d = tangent_coordinates(w, v, anchor)
    g = centered_gradient(w, scaled, alpha, anchor)
    gd = math.fsum(float(g[j]*delta) for j, delta in zip(free, d))
    changes = (scaled[:, free]-scaled[:, anchor, None])*d
    z = np.sum(changes, axis=1, dtype=np.float64)/mixture(w, scaled)
    remainder = math.fsum((alpha*log_remainder(z)).tolist())
    gap = math.fsum([(1.-SPEC['armijo'])*gd, remainder])
    if not all(math.isfinite(x) for x in (gd, remainder, gap)):
        raise FloatingPointError('Nonfinite Armijo comparison')
    return dict(gap=gap, directional_derivative=gd, remainder=remainder,
                objective_difference=math.fsum([gd, remainder]),
                accepted=bool(gd < 0 and gap <= 0))


def stationarity(w, scaled, alpha):
    anchor = int(np.argmax(w))
    gradient = centered_gradient(w, scaled, alpha, anchor)
    candidate = project(w-gradient)
    free, d = tangent_coordinates(w, candidate, anchor)
    residual = max(abs(float(d[0])), abs(float(d[1])), abs(math.fsum(d.tolist())))
    return float(residual), gradient, anchor


def solve_diagnostic(log_target, families, support, *, max_iterations=SPEC['max_iterations']):
    """Synthetic/control use only; cannot produce a registered-fitting result."""
    if type(max_iterations) is not int or not 1 <= max_iterations <= SPEC['max_iterations']:
        raise ValueError('Invalid iteration budget')
    scaled, alpha, offset = legacy.prepare(log_target, families, support)
    w = np.full(3, 1/3, dtype=np.float64)
    for iteration in range(max_iterations+1):
        residual, gradient, anchor = stationarity(w, scaled, alpha)
        if residual <= SPEC['projected_gradient_tolerance']:
            return dict(status='CONVERGED_CANDIDATE_NOT_APPROVED', diagnostic_weights=w.tolist(),
                objective=objective(w, scaled, alpha, offset), iterations=iteration,
                KKT_projected_residual=residual, weights_registered=False, spec=SPEC.copy())
        if iteration == max_iterations:
            break
        step = SPEC['step']
        for _ in range(SPEC['max_backtracks']):
            candidate = project(w-step*gradient)
            # A zero/underflowed mixture is inadmissible, as in the original.
            try:
                mixture(candidate, scaled)
            except FloatingPointError:
                step *= SPEC['backtrack']
                continue
            # Nonfinite gradient/comparison errors fail closed, not silently retry.
            comparison = armijo_gap(w, candidate, scaled, alpha, anchor)
            if comparison['accepted']:
                w = candidate
                break
            step *= SPEC['backtrack']
        else:
            raise RuntimeError('Armijo failed; candidate not registered')
    raise RuntimeError('Convergence tolerance not reached; candidate not registered')
