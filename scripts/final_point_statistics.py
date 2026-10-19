"""D059 reporting contract adopted by owner D060; pure arrays, no readers or CLI.

Arithmetic verification is not statistical coverage evidence.
Inputs must already have verified canonical sample/class and checkpoint alignment.
This module aggregates monthly scores only; it never accesses Future data.
"""
import numpy as np

SEEDS = (42, 43, 44, 45, 46)
MONTHS = tuple(f"2020-{month:02d}" for month in range(4, 10))
METHODS = ("sbsmi", "raw-byte", "entropy", "uniform", "new-quality-only",
           "quality-plus-drift", "learned-static")
OUTCOMES = ("mean_monthly_macro_f1", "worst_month_macro_f1")
# Four pairs x two outcomes; owner adopted in D060.
CONTRASTS = (
    ("primary", "quality-plus-drift", "new-quality-only"),
    ("secondary", "new-quality-only", "uniform"),
    ("secondary", "quality-plus-drift", "uniform"),
    ("secondary", "quality-plus-drift", "learned-static"),
)


def _spread(values):
    return {"mean": float(values.mean()), "sample_sd": float(values.std(ddof=1)),
            "min": float(values.min()), "max": float(values.max())}


def aggregate_monthly(monthly_f1, *, seed_ids, method_names, month_names):
    """Summarize aligned [5 seeds, 7 methods, 6 months] original-data F1.

    All entries are proportions, not percentages. No missing cell, averaging of
    probabilities across seeds, seed resampling, confidence interval or p-value.
    Exact metadata order is required: callers must not silently reorder arrays.
    """
    if tuple(seed_ids) != SEEDS or any(type(x) is not int for x in seed_ids):
        raise ValueError("Exactly the five registered integer seeds in order required")
    if tuple(method_names) != METHODS or tuple(month_names) != MONTHS:
        raise ValueError("Frozen method/month order mismatch")
    raw = np.asarray(monthly_f1)
    if raw.dtype.kind not in "fiu" or raw.shape != (5, 7, 6):
        raise ValueError("Expected numeric 5 x 7 x 6 complete monthly scores")
    values = raw.astype(np.float64, copy=True)
    if not np.isfinite(values).all() or np.any(values < 0) or np.any(values > 1):
        raise ValueError("Finite F1 proportions in [0,1] required; no clipping")
    per_seed = np.stack((values.mean(axis=2), values.min(axis=2)), axis=2)
    methods = {}
    for j, name in enumerate(METHODS):
        methods[name] = {
            "outcomes": {key: {"per_seed": per_seed[:, j, k].tolist(),
                               **_spread(per_seed[:, j, k])}
                         for k, key in enumerate(OUTCOMES)},
            "monthly": {month: {"per_seed": values[:, j, m].tolist(),
                                **_spread(values[:, j, m])}
                        for m, month in enumerate(MONTHS)},
            "worst_months_per_seed": [
                [MONTHS[m] for m in np.flatnonzero(values[s, j] == values[s, j].min())]
                for s in range(5)],
        }
    contrasts = []
    for priority, left, right in CONTRASTS:
        differences = per_seed[:, METHODS.index(left)] - per_seed[:, METHODS.index(right)]
        for k, key in enumerate(OUTCOMES):
            delta = differences[:, k]
            contrasts.append(dict(priority=priority, left=left, right=right,
                                  outcome=key, per_seed=delta.tolist(), **_spread(delta)))
    return dict(schema="D060_APPROVED_POINT_SPEC_V1", owner_approval_required=False,
                seed_ids=list(SEEDS), methods=methods, contrasts=contrasts,
                interval_status="NOT_ESTABLISHED_NOT_COMPUTED", p_values=None,
                seed_spread_interpretation="descriptive; not a confidence interval")
