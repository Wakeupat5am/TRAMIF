"""Pure numerical V1 MMD primitives. No I/O, fitting, models or null calibration.

Inputs are already transformed embeddings. bandwidth_h is h, NOT sqrt(h):
k(x,y) = exp(-sum((x-y)**2) / p / h / 2).
"""
from collections.abc import Mapping
from dataclasses import dataclass
import math
import numbers

import numpy as np


ROUNDOFF_ATOL = 64 * np.finfo(np.float64).eps
DEFAULT_BLOCK_SIZE = 32


@dataclass(frozen=True)
class MMDResult:
    mmd2: float
    raw_mmd2: float
    mean_xx: float
    mean_yy: float
    mean_xy: float
    negative_roundoff_clipped: bool


@dataclass(frozen=True)
class ConditionalMMDResult:
    support: tuple
    per_family: tuple
    mmd2: float


def _array(value, name):
    try:
        source = np.asarray(value)
        if source.dtype.kind not in "fiu":
            raise ValueError("expected real numeric values")
        with np.errstate(over="raise", invalid="raise"):
            array = np.asarray(source, dtype=np.float64)
    except (TypeError, ValueError, FloatingPointError, OverflowError) as exc:
        raise ValueError(f"{name}: expected finite real numeric matrix") from exc
    if array.ndim != 2 or min(array.shape) < 1:
        raise ValueError(f"{name}: expected nonempty (samples, features) matrix")
    if not np.isfinite(array).all():
        raise ValueError(f"{name}: non-finite input")
    return array


def _parameters(bandwidth_h, block_size):
    if isinstance(bandwidth_h, (bool, np.bool_)) or not isinstance(bandwidth_h, numbers.Real):
        raise ValueError("bandwidth_h must be a positive finite real scalar")
    try:
        h = float(bandwidth_h)
    except (OverflowError, ValueError) as exc:
        raise ValueError("bandwidth_h not representable in float64") from exc
    if not math.isfinite(h) or h <= 0:
        raise ValueError("bandwidth_h must be positive and finite")
    if isinstance(block_size, (bool, np.bool_)) or not isinstance(block_size, numbers.Integral) or block_size < 1:
        raise ValueError("block_size must be a positive integer")
    return h, int(block_size)


def _kernel_mean(x, y, h, block_size):
    def block_sums():
        for i in range(0, len(x), block_size):
            for j in range(0, len(y), block_size):
                # Direct differences avoid catastrophic cancellation of Gram distances.
                with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
                    difference = x[i:i+block_size, None, :] - y[None, j:j+block_size, :]
                    np.square(difference, out=difference)
                    exponent = difference.sum(axis=2, dtype=np.float64)
                    exponent /= x.shape[1]
                    exponent /= h
                    exponent *= -0.5
                    np.exp(exponent, out=exponent)
                    yield float(exponent.sum(dtype=np.float64))
    value = math.fsum(block_sums()) / (len(x) * len(y))
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise FloatingPointError("Invalid RBF kernel mean")
    return value


def _finish(mean_xx, mean_yy, mean_xy):
    if any(not math.isfinite(v) or not 0 <= v <= 1 for v in (mean_xx, mean_yy, mean_xy)):
        raise FloatingPointError("Invalid kernel means")
    raw = math.fsum((mean_xx, mean_yy, -2.0 * mean_xy))
    if not math.isfinite(raw) or raw < -ROUNDOFF_ATOL or raw > 2:
        raise FloatingPointError("MMD2 outside numerical contract; do not silently clip")
    return MMDResult(max(0.0, raw), raw, mean_xx, mean_yy, mean_xy, raw < 0)


def biased_rbf_mmd2(x, y, *, bandwidth_h, block_size=DEFAULT_BLOCK_SIZE):
    """Biased empirical MMD², all ordered pairs INCLUDING self diagonals.

    Unequal sample counts are supported; feature dimensions must agree.
    Finite inputs whose intermediate arithmetic overflows fail explicitly.
    Only negative final roundoff in [-64*eps64, 0) is clipped, with diagnostics.
    No bandwidth or normalization is inferred from x/y.
    """
    h, block = _parameters(bandwidth_h, block_size)
    x, y = _array(x, "x"), _array(y, "y")
    if x.shape[1] != y.shape[1]:
        raise ValueError("Feature dimensions differ")
    return _finish(_kernel_mean(x, x, h, block),
                   _kernel_mean(y, y, h, block),
                   _kernel_mean(x, y, h, block))


def class_conditional_mmd2(x_by_family, y_by_family, *, support, bandwidth_h,
                           block_size=DEFAULT_BLOCK_SIZE):
    """Equal-family mean over EXPLICIT support; never infer/intersect/drop classes.

    Extra mapping keys outside support are not scored. Every requested class must
    exist and have nonempty matrices with the same feature width across classes.
    This numerical primitive does NOT enforce historical sample/group thresholds.
    """
    if not isinstance(x_by_family, Mapping) or not isinstance(y_by_family, Mapping):
        raise ValueError("Expected mappings from family names to embeddings")
    if isinstance(support, (str, bytes, set, frozenset)):
        raise ValueError("support must be an ordered sequence of family names")
    try:
        families = tuple(support)
    except TypeError as exc:
        raise ValueError("support must be iterable") from exc
    if not families or any(not isinstance(c, str) or not c for c in families) or len(set(families)) != len(families):
        raise ValueError("support must contain unique nonempty family names")
    h, block = _parameters(bandwidth_h, block_size)
    pairs, widths = [], set()
    for c in families:
        if c not in x_by_family or c not in y_by_family:
            raise ValueError(f"Missing support family: {c}")
        x, y = _array(x_by_family[c], c+"/x"), _array(y_by_family[c], c+"/y")
        widths.update((x.shape[1], y.shape[1]))
        pairs.append((x, y))
    if len(widths) != 1:
        raise ValueError("Feature dimensions must agree across all support families")
    results = tuple(biased_rbf_mmd2(x, y, bandwidth_h=h, block_size=block) for x, y in pairs)
    return ConditionalMMDResult(families, results, math.fsum(r.mmd2 for r in results)/len(results))
