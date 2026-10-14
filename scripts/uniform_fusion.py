"""Strict alignment and fixed probability averaging; no model fitting."""
import hashlib
import io
from pathlib import Path

import numpy as np


def load_bundle(path, expected_sha256, temperature):
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError("Prediction checksum mismatch")
    with np.load(io.BytesIO(raw), allow_pickle=False) as data:
        return {
            "ids": data["sample_identifiers"].copy(),
            "classes": data["class_names"].copy(),
            "labels": data["labels"].copy(),
            "logits": data["logits"].copy(),
            "temperature": temperature,
        }


def validate_alignment(bundles):
    if len(bundles) != 3:
        raise ValueError("Exactly three views required")
    first = bundles[0]
    for batch in bundles:
        ids, classes = np.asarray(batch["ids"]), np.asarray(batch["classes"])
        labels, logits = np.asarray(batch["labels"]), np.asarray(batch["logits"])
        if ids.ndim != 1 or classes.ndim != 1 or not len(ids) or len(classes) < 2:
            raise ValueError("Invalid sample/class dimensions")
        if ids.dtype.kind not in "US" or classes.dtype.kind not in "US":
            raise ValueError("Identifiers and classes must be strings")
        if len(set(ids.tolist())) != len(ids) or len(set(classes.tolist())) != len(classes):
            raise ValueError("Duplicate sample or class identifiers")
        if any(not str(value).strip() for value in [*ids, *classes]):
            raise ValueError("Empty identifiers")
        if logits.shape != (len(ids), len(classes)) or logits.dtype.kind != "f":
            raise ValueError("Invalid logits")
        if not np.isfinite(logits).all():
            raise ValueError("Non-finite logits")
        if labels.shape != (len(ids),) or labels.dtype.kind not in "iu":
            raise ValueError("Invalid labels")
        if np.any(labels < 0) or np.any(labels >= len(classes)):
            raise ValueError("Label outside class mapping")
        temperature = batch["temperature"]
        if isinstance(temperature, bool) or not np.isscalar(temperature) or not np.isfinite(temperature) or temperature <= 0:
            raise ValueError("Invalid temperature")
        for key in ("ids", "classes", "labels"):
            if not np.array_equal(batch[key], first[key]):
                raise ValueError("Alignment mismatch: " + key)


def calibrated_log_probabilities(logits, temperature):
    scores = np.asarray(logits, dtype=np.float64) / temperature
    scores = scores - scores.max(axis=1, keepdims=True)
    return scores - np.logaddexp.reduce(scores, axis=1, keepdims=True)


def fuse_uniform(bundles):
    validate_alignment(bundles)
    # Labels are validated for consistent evaluation, never used in weighting.
    log_probabilities = [
        calibrated_log_probabilities(batch["logits"], batch["temperature"])
        for batch in bundles
    ]
    fused = np.logaddexp.reduce(np.stack(log_probabilities), axis=0) - np.log(3.0)
    return log_probabilities, fused


def classification_metrics(log_probabilities, labels):
    logp = np.asarray(log_probabilities, dtype=np.float64)
    labels = np.asarray(labels)
    if logp.ndim != 2 or not len(logp) or not np.isfinite(logp).all():
        raise ValueError("Invalid log probabilities")
    n, k = logp.shape
    if labels.shape != (n,) or labels.dtype.kind not in "iu" or np.any(labels < 0) or np.any(labels >= k):
        raise ValueError("Invalid evaluation labels")
    probabilities = np.exp(logp)
    if not np.allclose(probabilities.sum(1), 1, rtol=0, atol=1e-12):
        raise ValueError("Probabilities not normalized")
    predictions = probabilities.argmax(1)
    confusion = np.zeros((k, k), dtype=np.int64)
    np.add.at(confusion, (labels, predictions), 1)
    support, predicted = confusion.sum(1), confusion.sum(0)
    tp = np.diag(confusion)
    supported = support > 0
    recall = tp / np.maximum(support, 1)
    f1 = 2 * tp / np.maximum(support + predicted, 1)
    correct = predictions == labels
    confidence = probabilities.max(1)
    bins = np.minimum(np.floor(confidence * 15).astype(int), 14)
    ece = 0.0
    bin_rows = []
    for index in range(15):
        mask = bins == index
        count = int(mask.sum())
        accuracy = float(correct[mask].mean()) if count else None
        mean_confidence = float(confidence[mask].mean()) if count else None
        if count:
            ece += count / n * abs(accuracy - mean_confidence)
        bin_rows.append({"bin": index, "count": count, "accuracy": accuracy, "confidence": mean_confidence})
    brier = ((probabilities - np.eye(k)[labels]) ** 2).sum(1).mean()
    return {
        "samples": n, "supported_classes": int(supported.sum()),
        "accuracy": float(correct.mean()),
        "macro_f1_supported": float(f1[supported].mean()),
        "macro_f1_fixed": float(f1.mean()),
        "balanced_accuracy": float(recall[supported].mean()),
        "nll": float(-logp[np.arange(n), labels].mean()),
        "brier": float(brier), "ece_15": float(ece), "bins": bin_rows,
        "confusion_matrix": confusion.tolist(),
        "per_class": [{"class_index": i, "support": int(support[i]),
                       "predicted_count": int(predicted[i]), "f1": float(f1[i]),
                       "recall": float(recall[i]) if support[i] else None} for i in range(k)],
    }
