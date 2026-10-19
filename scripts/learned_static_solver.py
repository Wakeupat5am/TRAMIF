"""Proposed deterministic convex mixture solver. Pure numerical API, no data CLI.

Synthetic verification is authorized. Empirical fitting requires owner approval
of initialization, projected-gradient rule, tolerances and deterministic ties.
Input is true-label calibrated log probability for each sample and named view;
the caller must retain the full K-class softmax before selecting true labels.
"""
import numpy as np

SPEC=dict(initialization='uniform',method='projected gradient with Armijo backtracking',
 step=1.,backtrack=.5,armijo=1e-4,max_backtracks=60,max_iterations=10000,
 projected_gradient_tolerance=1e-10,tie='uniform-start deterministic trajectory; no minimum-norm uniqueness claim',
 objective='equal-family mixture NLL',empirical_approval=False)


def simplex_projection(x):
    x=np.asarray(x,dtype=np.float64)
    if x.shape!=(3,) or not np.isfinite(x).all():raise ValueError('Three finite coordinates required')
    u=np.sort(x)[::-1];css=np.cumsum(u)-1
    rho=np.flatnonzero(u-css/np.arange(1,4)>0)[-1];theta=css[rho]/(rho+1)
    return np.maximum(x-theta,0)


def prepare(log_target,families,support):
    logs=np.asarray(log_target,dtype=np.float64);families=np.asarray(families)
    if logs.ndim!=2 or logs.shape[1]!=3 or len(logs)==0 or families.shape!=(len(logs),):raise ValueError('Expected N x 3 true-label log probabilities')
    if not np.isfinite(logs).all() or (logs>0).any():raise ValueError('Finite nonpositive log probabilities required')
    if not isinstance(support,(list,tuple)) or not support or len(set(support))!=len(support):raise ValueError('Explicit unique support required')
    if set(families)!=set(support):raise ValueError('Input must contain exactly frozen support')
    alpha=np.array([1/(len(support)*np.count_nonzero(families==c)) for c in families])
    maximum=logs.max(1);scaled=np.exp(logs-maximum[:,None])
    return scaled,alpha,-float(alpha@maximum)


def objective_gradient(weights,scaled,alpha,offset):
    w=np.asarray(weights,dtype=np.float64)
    if w.shape!=(3,) or not np.isfinite(w).all() or (w<0).any() or not np.isclose(w.sum(),1,rtol=0,atol=1e-12):raise ValueError('Invalid simplex weights')
    mix=scaled@w
    if not np.isfinite(mix).all() or (mix<=0).any():raise FloatingPointError('Zero/nonfinite target mixture')
    value=offset-float(alpha@np.log(mix));gradient=-(alpha[:,None]*scaled/mix[:,None]).sum(0)
    if not np.isfinite(value) or not np.isfinite(gradient).all():raise FloatingPointError('Nonfinite objective/gradient')
    return value,gradient


def solve(log_target,families,support,*,max_iterations=SPEC['max_iterations']):
    if type(max_iterations) is not int or max_iterations<1 or max_iterations>SPEC['max_iterations']:raise ValueError('Invalid iteration budget')
    scaled,alpha,offset=prepare(log_target,families,support);w=np.full(3,1/3,dtype=np.float64)
    for iteration in range(max_iterations+1):
        value,grad=objective_gradient(w,scaled,alpha,offset)
        residual=float(np.max(np.abs(w-simplex_projection(w-grad))))
        if residual<=SPEC['projected_gradient_tolerance']:
            return dict(weights=w.tolist(),objective=value,iterations=iteration,KKT_projected_residual=residual,status='CONVERGED',spec=SPEC.copy())
        if iteration==max_iterations:break
        step=SPEC['step'];accepted=False
        for _ in range(SPEC['max_backtracks']):
            candidate=simplex_projection(w-step*grad);direction=candidate-w
            try:next_value,_=objective_gradient(candidate,scaled,alpha,offset)
            except FloatingPointError:next_value=float('inf')
            if next_value<=value+SPEC['armijo']*float(grad@direction):
                w=candidate;accepted=True;break
            step*=SPEC['backtrack']
        if not accepted:raise RuntimeError('Armijo failed; no empirical fallback')
    raise RuntimeError('Convergence tolerance not reached; no approximate weight registration')
