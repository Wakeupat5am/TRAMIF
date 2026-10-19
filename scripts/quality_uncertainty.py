"""Pure conditional Quality uncertainty mathematics; no I/O or model execution.

Caller must bind loss columns to historical sample IDs/calibration provenance.
This module validates array identities, not chronology or statistical independence.
It never creates fusion weights. All returned draws are diagnostic group weights.
"""
import numbers
import numpy as np

REPLICATES = 2000
SEED = 42


def _strings(values, name, unique=False):
    if isinstance(values, (str, bytes, set, frozenset)):
        raise ValueError(name + ': expected ordered sequence')
    result = tuple(values)
    if not result or any(not isinstance(x, str) or not x for x in result):
        raise ValueError(name + ': expected nonempty strings')
    if unique and len(set(result)) != len(result):
        raise ValueError(name + ': duplicate identity')
    return result


def _positive_integer(value, name, minimum):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Integral) or value < minimum:
        raise ValueError(name + ': invalid integer')
    return int(value)


def draw_group_weights(group_order, *, replicates=REPLICATES, seed=SEED):
    """Canonical group order is explicit, shared by all views and classes."""
    groups = _strings(group_order, 'group_order', unique=True)
    if groups != tuple(sorted(groups)):
        raise ValueError('group_order must be lexicographically sorted')
    count = _positive_integer(replicates, 'replicates', 2)
    seed = _positive_integer(seed, 'seed', 0)
    draws = np.random.Generator(np.random.PCG64(seed)).exponential(1.0, (count, len(groups)))
    if not np.isfinite(draws).all() or np.any(draws <= 0):
        raise FloatingPointError('Invalid random draw; no retry permitted')
    return draws


def estimate_uncertainty(losses, *, sample_ids, families, group_ids, support,
                         draws=None, draw_group_order=None):
    """losses=(3,N), already calibrated finite nonnegative per-sample NLL.

    Uses all supplied samples in support, ignoring outside-support losses only
    for balanced estimation. No cap, resampling of members or normalization over
    classes is applied. Explicit draws require explicit canonical column IDs.
    Numerical failures propagate; no replicate is removed or replaced.
    """
    ids = _strings(sample_ids, 'sample_ids', unique=True)
    family = _strings(families, 'families')
    groups = _strings(group_ids, 'group_ids')
    classes = _strings(support, 'support', unique=True)
    raw = np.asarray(losses)
    if raw.dtype.kind not in 'fiu':
        raise ValueError('losses must be real numeric')
    values = np.asarray(raw, dtype=np.float64)
    if values.shape != (3, len(ids)) or len(family) != len(ids) or len(groups) != len(ids):
        raise ValueError('Loss/identity dimensions mismatch')
    if not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError('Losses must be finite and nonnegative')
    indices = [np.array([i for i,c in enumerate(family) if c == name], dtype=np.int64) for name in classes]
    if any(len(i) == 0 for i in indices):
        raise ValueError('Missing fixed support class')
    active = sorted(set(groups[i] for idx in indices for i in idx))
    if draws is None:
        if draw_group_order is not None:
            raise ValueError('Unexpected draw order without draws')
        weights = draw_group_weights(active)
    else:
        if draw_group_order is None or tuple(draw_group_order) != tuple(active):
            raise ValueError('Draw group order mismatch')
        raw_weights = np.asarray(draws)
        if raw_weights.dtype.kind not in 'fiu':
            raise ValueError('Draws must be real numeric')
        weights = np.array(raw_weights, dtype=np.float64, copy=True)
        if weights.ndim != 2 or weights.shape[1] != len(active) or weights.shape[0] < 2:
            raise ValueError('Draw shape mismatch')
        if not np.isfinite(weights).all() or np.any(weights <= 0):
            raise ValueError('Draws must be positive and finite')
    lookup = {g:i for i,g in enumerate(active)}
    point_classes = []
    estimates = np.zeros((len(weights), 3), dtype=np.float64)
    with np.errstate(over='raise', invalid='raise', divide='raise', under='ignore'):
        for idx in indices:
            # Group totals preserve sample-level mass, including multi-label IDs.
            counts = np.zeros(len(active), dtype=np.float64)
            totals = np.zeros((len(active), 3), dtype=np.float64)
            for i in idx:
                g = lookup[groups[i]]
                counts[g] += 1
                totals[g] += values[:, i]
            denominator = weights @ counts
            numerator = weights @ totals
            if not np.isfinite(denominator).all() or np.any(denominator <= 0):
                raise FloatingPointError('Invalid class denominator')
            ratios = numerator / denominator[:, None]
            if not np.isfinite(ratios).all():
                raise FloatingPointError('Invalid replicate loss')
            estimates += ratios / len(classes)
            point_classes.append(values[:, idx].mean(axis=1))
        per_class = np.stack(point_classes, axis=1)
        point = per_class.mean(axis=1)
        mean = estimates.mean(axis=0)
        sd = estimates.std(axis=0, ddof=1)
        covariance = np.cov(estimates, rowvar=False, ddof=1)
        quantiles = np.quantile(estimates, [.025, .5, .975], axis=0, method='linear')
    if any(not np.isfinite(x).all() for x in (point, estimates, sd, covariance, quantiles)):
        raise FloatingPointError('Nonfinite estimator result')
    return dict(support=classes, group_order=tuple(active), sample_ids=ids,
                point_loss=point, per_class_loss=per_class, draws=weights,
                replicate_losses=estimates, replicate_mean=mean, uncertainty=sd,
                mean_minus_point=mean-point, covariance=covariance,
                diagnostic_quantiles=quantiles, quantile_levels=(.025,.5,.975),
                ddof=1, numpy_version=np.__version__)
