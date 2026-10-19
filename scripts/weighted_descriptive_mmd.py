"""Diagnostic equal-represented-group mass; no fitting or independence claim."""
from collections import Counter
import math
import numpy as np
from scripts.descriptive_mmd import _array, _parameters, MMDResult, ROUNDOFF_ATOL


def group_weights(groups):
    groups = tuple(groups)
    if not groups or any(not isinstance(g, str) or not g for g in groups):
        raise ValueError('Expected nonempty group IDs')
    counts = Counter(groups)
    return np.array([1.0 / (len(counts)*counts[g]) for g in groups], dtype=np.float64)


def _weights(value, n):
    source = np.asarray(value)
    if source.dtype.kind not in 'fiu':
        raise ValueError('Weights must be real')
    a = np.asarray(source, dtype=np.float64)
    if a.shape != (n,) or not np.isfinite(a).all() or (a < 0).any():
        raise ValueError('Invalid weights')
    if abs(math.fsum(a) - 1.0) > ROUNDOFF_ATOL:
        raise ValueError('Weights must already sum to one')
    return a


def weighted_mmd2(x, y, a, b, *, bandwidth_h, block_size=32):
    x, y = _array(x, 'x'), _array(y, 'y')
    h, block = _parameters(bandwidth_h, block_size)
    if x.shape[1] != y.shape[1]:
        raise ValueError('Feature dimensions differ')
    a, b = _weights(a, len(x)), _weights(b, len(y))

    def kernel(u, v, wu, wv):
        sums = []
        for i in range(0, len(u), block):
            for j in range(0, len(v), block):
                with np.errstate(over='raise', invalid='raise', divide='raise', under='ignore'):
                    d = u[i:i+block, None, :] - v[None, j:j+block, :]
                    np.square(d, out=d)
                    k = np.exp(-0.5 * (d.sum(axis=2) / u.shape[1] / h))
                    sums.append(float((k * wu[i:i+block, None] * wv[None, j:j+block]).sum()))
        return math.fsum(sums)

    xx, yy, xy = kernel(x,x,a,a), kernel(y,y,b,b), kernel(x,y,a,b)
    # Normalized floating weights can deviate from unit mass by <=64 eps.
    # Allow 4 times that tolerance for the quadratic sums; never clip kernels.
    tol = 4 * ROUNDOFF_ATOL
    if any(not math.isfinite(v) or v < 0 or v > 1+tol for v in (xx,yy,xy)):
        raise FloatingPointError('Invalid weighted kernel mean')
    raw = math.fsum((xx,yy,-2*xy))
    if not math.isfinite(raw) or raw < -ROUNDOFF_ATOL or raw > 2+tol:
        raise FloatingPointError('Weighted MMD outside numerical contract')
    return MMDResult(max(0.,raw),raw,xx,yy,xy,raw<0)
