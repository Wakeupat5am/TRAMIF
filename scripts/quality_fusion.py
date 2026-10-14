"""February OOF quality mathematics. No evaluation-month input or file access."""
import numpy as np
import torch

from scripts.calibrate_sbsmi import fit_temperature, T_MIN, T_MAX
from scripts.uniform_fusion import validate_alignment, calibrated_log_probabilities


def aligned_folds(bundles, manifest):
    validate_alignment(bundles)
    first = bundles[0]
    rows = {r["sha"]: r for r in manifest}
    if len(rows) != len(manifest) or set(rows) != set(first["ids"].tolist()):
        raise ValueError("Fold/sample identity mismatch")
    folds, group_folds, content_folds = [], {}, {}
    for sid, label in zip(first["ids"], first["labels"]):
        row = rows[str(sid)]  # Explicit ID join; never assume manifest row order.
        f = int(row["fold"])
        if f not in range(5) or row["family"] != first["classes"][label]:
            raise ValueError("Fold/class mapping mismatch")
        if not str(row["timestamp"]).startswith("2020-02-"):
            raise ValueError("February only")
        if str(row["retained"]).lower() != "true":
            raise ValueError("Excluded bridge cannot be calibrated/scored")
        for key, seen in (("group_id", group_folds), ("disarmed_content_sha256", content_folds)):
            if not row[key] or (row[key] in seen and seen[row[key]] != f):
                raise ValueError("Group/content crosses folds")
            seen[row[key]] = f
        folds.append(f)
    folds = np.asarray(folds, dtype=np.int64)
    if set(folds.tolist()) != set(range(5)):
        raise ValueError("Five nonempty folds required")
    return folds


def cross_fit(bundles, manifest, temperature_fitter=fit_temperature):
    folds = aligned_folds(bundles, manifest)
    n, k = bundles[0]["logits"].shape
    logs = np.full((3, n, k), np.nan, dtype=np.float64)
    coverage = np.zeros((3, n), dtype=np.int64)
    fitted = []
    for v, batch in enumerate(bundles):
        scores = torch.tensor(batch["logits"], dtype=torch.float64, device="cpu")
        targets = torch.tensor(batch["labels"], dtype=torch.int64, device="cpu")
        for fold in range(5):
            fit, scoring = folds != fold, folds == fold
            temperature, boundary = temperature_fitter(scores[fit], targets[fit])
            if not np.isfinite(temperature) or not T_MIN <= temperature <= T_MAX:
                raise ValueError("Invalid fitted temperature")
            logs[v, scoring] = calibrated_log_probabilities(batch["logits"][scoring], temperature)
            coverage[v, scoring] += 1
            fitted.append({"view_index": v, "fold": fold, "temperature": float(temperature),
                           "boundary": bool(boundary), "fit_samples": int(fit.sum()),
                           "score_samples": int(scoring.sum())})
    if not np.all(coverage == 1) or not np.isfinite(logs).all():
        raise ValueError("OOF coverage failure")
    if not np.allclose(np.exp(logs).sum(2), 1, rtol=0, atol=1e-12):
        raise ValueError("OOF normalization failure")
    return logs, folds, fitted


def estimate_quality(logs, labels, support_indices):
    logs, labels = np.asarray(logs), np.asarray(labels)
    if logs.ndim != 3 or logs.shape[0] != 3 or not np.isfinite(logs).all():
        raise ValueError("Invalid OOF log probabilities")
    _, n, k = logs.shape
    if labels.shape != (n,) or labels.dtype.kind not in "iu" or np.any(labels < 0) or np.any(labels >= k):
        raise ValueError("Invalid labels")
    if not np.allclose(np.exp(logs).sum(2), 1, rtol=0, atol=1e-12):
        raise ValueError("Probabilities not normalized")
    support = list(support_indices)
    if (not support or len(set(support)) != len(support)
            or any(not isinstance(c, (int, np.integer)) or c < 0 or c >= k for c in support)):
        raise ValueError("Invalid fixed support")
    if any(not np.any(labels == c) for c in support):
        raise ValueError("Missing fixed support class; no denominator change allowed")
    losses = -logs[:, np.arange(n), labels]
    per_class = np.stack([losses[:, labels == c].mean(1) for c in support], axis=1)
    balanced = per_class.mean(1)
    weights = np.exp(-balanced - np.max(-balanced))
    weights /= weights.sum()
    if not np.isfinite(weights).all() or np.any(weights < 0) or not np.isclose(weights.sum(), 1, rtol=0, atol=1e-12):
        raise ValueError("Invalid weights")
    return {"class_balanced_nll": balanced.tolist(), "standard_nll_all_february": losses.mean(1).tolist(),
            "standard_nll_support_samples": losses[:, np.isin(labels, support)].mean(1).tolist(),
            "per_class_nll": per_class.tolist(), "weights": weights.tolist()}


def weighted_fusion(logs, weights):
    logs, weights = np.asarray(logs, dtype=np.float64), np.asarray(weights, dtype=np.float64)
    if (logs.ndim != 3 or logs.shape[0] != 3 or not np.isfinite(logs).all()
            or weights.shape != (3,) or not np.isfinite(weights).all() or np.any(weights < 0)
            or not np.isclose(weights.sum(), 1, rtol=0, atol=1e-12)
            or not np.allclose(np.exp(logs).sum(2), 1, rtol=0, atol=1e-12)):
        raise ValueError("Invalid mixture")
    positive = weights > 0
    return np.logaddexp.reduce(logs[positive] + np.log(weights[positive, None, None]), axis=0)
