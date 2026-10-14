"""Additive March metrics under paired whole-group resampling. No fitting."""
import numpy as np

METRICS = ["accuracy", "nll", "brier"]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sample_metrics(log_probabilities, labels):
    logs, labels = np.asarray(log_probabilities, dtype=np.float64), np.asarray(labels)
    require(logs.ndim == 3 and np.isfinite(logs).all(), "Finite method/sample/class array required")
    m, n, k = logs.shape
    require(m > 0 and n > 0 and k > 1 and labels.shape == (n,) and labels.dtype.kind in "iu"
            and np.all((labels >= 0) & (labels < k)), "Invalid labels/dimensions")
    p = np.exp(logs)
    require(np.allclose(p.sum(2), 1, rtol=0, atol=1e-12), "Probabilities not normalized")
    correct = (logs.argmax(2) == labels[None, :]).astype(np.float64)
    nll = -logs[:, np.arange(n), labels]
    # Multiclass Brier is SUM over classes, not the mean over classes.
    brier = np.square(p - np.eye(k)[labels][None, :, :]).sum(2)
    return np.stack([correct, nll, brier], axis=2).transpose(1, 0, 2)


def aggregate_groups(values, sample_groups):
    values, sample_groups = np.asarray(values), np.asarray(sample_groups)
    require(values.ndim == 3 and len(values) > 0 and np.isfinite(values).all(), "Invalid sample metrics")
    require(sample_groups.shape == (len(values),) and sample_groups.dtype.kind in "US", "Invalid group identifiers")
    require(all(str(g).strip() for g in sample_groups), "Empty group identifier")
    groups, inverse = np.unique(sample_groups, return_inverse=True)
    require(len(groups) >= 2, "At least two active groups required")
    sizes = np.bincount(inverse)
    sums = np.zeros((len(groups), *values.shape[1:]), dtype=np.float64)
    np.add.at(sums, inverse, values)
    return groups, inverse, sizes, sums


def draw_multiplicities(group_count, replicates=2000, seed=42):
    require(isinstance(group_count, int) and group_count >= 2, "Invalid group count")
    require(isinstance(replicates, int) and replicates > 0, "Invalid replicate count")
    return np.random.default_rng(seed).multinomial(group_count, np.full(group_count, 1/group_count), size=replicates)


def resampled_metrics(draws, sizes, sums):
    draws, sizes, sums = np.asarray(draws), np.asarray(sizes), np.asarray(sums)
    require(draws.ndim == 2 and len(draws) > 0 and draws.dtype.kind in "iu" and np.all(draws >= 0), "Invalid multiplicities")
    g = draws.shape[1]
    require(sizes.shape == (g,) and sizes.dtype.kind in "iu" and np.all(sizes > 0), "Invalid group sizes")
    require(sums.ndim == 3 and sums.shape[0] == g and np.isfinite(sums).all(), "Invalid group sums")
    require(np.all(draws.sum(1) == g), "Each replicate must draw exactly G whole groups")
    counts = draws @ sizes
    values = (draws @ sums.reshape(g, -1)).reshape(len(draws), *sums.shape[1:]) / counts[:, None, None]
    require(np.all(counts > 0) and np.isfinite(values).all(), "Non-finite bootstrap metric")
    return counts, values


def paired_differences(values, quality_index, reference_indices):
    values = np.asarray(values)
    require(values.ndim == 3 and np.isfinite(values).all(), "Invalid replicate metrics")
    indices = [quality_index, *reference_indices]
    require(all(isinstance(i, int) and 0 <= i < values.shape[1] for i in indices)
            and len(set(indices)) == len(indices), "Invalid method indices")
    return values[:, quality_index, None, :] - values[:, reference_indices, :]


def percentile_interval(values):
    values = np.asarray(values)
    require(values.ndim >= 1 and len(values) >= 2 and np.isfinite(values).all(), "Finite replicates required; none may be dropped")
    return np.quantile(values, [.025, .975], axis=0, method="linear")


def distribution_diagnostic(values):
    values = np.asarray(values, dtype=float)
    require(values.ndim == 1 and np.isfinite(values).all() and len(values) >= 2, "Invalid diagnostic values")
    centered = values - values.mean()
    sd_pop = float(np.sqrt(np.mean(centered ** 2)))
    hist, edges = np.histogram(values, bins=40)
    return {"mean": float(values.mean()), "sd": float(values.std(ddof=1)),
            "skewness_moment": float(np.mean((centered / sd_pop)**3)) if sd_pop else None,
            "excess_kurtosis_moment": float(np.mean((centered / sd_pop)**4) - 3) if sd_pop else None,
            "quantiles": {str(q): float(np.quantile(values, q, method="linear"))
                          for q in (0, .001, .01, .025, .25, .5, .75, .975, .99, .999, 1)},
            "histogram_counts": hist.tolist(), "histogram_edges": edges.tolist()}


def influence_diagnostics(sizes, sums, groups, draws, differences, quality_index=4, uniform_index=3):
    """Leave-one-group-out and multiplicity associations are diagnostics only."""
    n = int(sizes.sum())
    require(np.all(sizes < n), "One group contains the entire sample")
    delta_sums = sums[:, quality_index] - sums[:, uniform_index]
    point = delta_sums.sum(0) / n
    loo = (delta_sums.sum(0)[None, :] - delta_sums) / (n - sizes[:, None])
    rows = [{"group_id": str(groups[g]), "samples": int(sizes[g]),
             **{f"loo_delta_{metric}": float(loo[g, j]) for j, metric in enumerate(METRICS)},
             **{f"shift_{metric}": float(loo[g, j] - point[j]) for j, metric in enumerate(METRICS)}} for g in range(len(groups))]
    top = sorted(range(len(groups)), key=lambda g: (-int(sizes[g]), str(groups[g])))[:10]
    most_influential = {metric: int(np.argmax(np.abs(loo[:, j] - point[j]))) for j, metric in enumerate(METRICS)}
    selected = sorted(set(top) | set(most_influential.values()))
    associations = []
    for g in selected:
        absent = draws[:, g] == 0
        stats = {}
        for j, metric in enumerate(METRICS):
            d = differences[:, j]
            stats[metric] = {
                "correlation_with_multiplicity": float(np.corrcoef(draws[:, g], d)[0, 1]) if draws[:, g].std() and d.std() else None,
                "mean_delta_when_absent": float(d[absent].mean()) if absent.any() else None,
                "mean_delta_when_present": float(d[~absent].mean()) if (~absent).any() else None,
            }
        associations.append({"group_id": str(groups[g]), "samples": int(sizes[g]),
                             "absent_replicates": int(absent.sum()), "present_replicates": int((~absent).sum()),
                             "primary_delta_associations": stats})
    return rows, {"most_influential": {metric: rows[g] for metric, g in most_influential.items()},
                  "loo_primary_sign_reversal_counts": {metric: int(np.sum(loo[:, j] * point[j] < 0)) for j, metric in enumerate(METRICS)},
                  "large_or_influential_group_associations": associations,
                  "note": "No groups excluded from estimates or CIs. LOO and associations are descriptive, not alternative estimators or causal attribution."}
