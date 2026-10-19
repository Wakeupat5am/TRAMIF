"""G3-B pure preparation utilities: no I/O, models, observed MMD or fitting on A/B."""
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from fractions import Fraction
from itertools import combinations
import hashlib
import json
import math
import re

import numpy as np

CAP = 128
SEED = 42
STD_MASK = 1e-8
NAMESPACE = "TRAMIF-G3B-v1"
VIEWS = ("sbsmi", "raw-byte", "entropy")
CASES = {
    "E1": ("2019-11", "2019-11", "2019-12"),
    "E2": ("2019-12", "2019-12", "2020-01"),
    "Final": ("2020-02", "2020-02", "2020-03"),
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def encoded(value):
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def families_checked(families):
    require(isinstance(families, (list, tuple)), "Ordered families required")
    families = tuple(families)
    require(families and all(isinstance(c, str) and c for c in families), "Empty/invalid families")
    require(len(set(families)) == len(families), "Duplicate family")
    return families


def checked_rows(rows, case, families, stages):
    require(case in CASES, "Unknown case")
    families = families_checked(families)
    rows = list(rows)
    ids = set()
    end, a, b = CASES[case]
    for r in rows:
        sid = r["sha"]
        require(isinstance(sid, str) and re.fullmatch("[0-9a-f]{64}", sid), "Invalid sample ID")
        require(sid not in ids, "Duplicate sample ID")
        ids.add(sid)
        require(r["family"] in families and r["stage"] in stages, "Family/stage mismatch")
        t = datetime.fromisoformat(r["timestamp"])
        require(t.tzinfo is not None, "Timezone required")
        month = t.astimezone(timezone.utc).strftime("%Y-%m")
        require("2019-08" <= month < "2020-04", "Outside historical allowlist")
        expected = "2019-08" <= month < end if r["stage"] == "train" else month == {"A": a, "B": b}[r["stage"]]
        require(expected, "Chronological boundary violation")
        if "retained" in r:
            require(r["retained"] is True or r["retained"] == "True", "Excluded record supplied")
        if "decision" in r:
            require(r["decision"] == "KEEP", "Excluded record supplied")
    return rows


def ranking(purpose, case, family, *ids):
    require(purpose in ("train-reference", "train-pair", "matched-A", "matched-B"), "Unknown ranking purpose")
    require(case in CASES, "Unknown case")
    return (sha256(encoded([NAMESPACE, SEED, purpose, case, family, *ids])), *ids)


def train_reference(rows, *, case, families):
    families = families_checked(families)
    rows = checked_rows(rows, case, families, {"train"})
    out = []
    for c in families:
        candidates = [r for r in rows if r["family"] == c]
        require(candidates, "Missing TRAIN family: " + c)
        out.extend(sorted(candidates, key=lambda r: ranking("train-reference", case, c, r["sha"]))[:CAP])
    return out


def reference_pairs(reference, *, case, families):
    families = families_checked(families)
    reference = checked_rows(reference, case, families, {"train"})
    out = []
    for c in families:
        ids = sorted(r["sha"] for r in reference if r["family"] == c)
        require(2 <= len(ids) <= CAP, "Need 2..128 reference samples per family")
        pairs = sorted(combinations(ids, 2), key=lambda pair: ranking("train-pair", case, c, *pair))[:CAP]
        out.extend({"family": c, "left": left, "right": right} for left, right in pairs)
    return out


def cap_audit(rows, *, case, support):
    support = families_checked(support)
    rows = checked_rows(rows, case, support, {"A", "B"})
    selected, counts = [], []
    for c in support:
        sides = {s: [r for r in rows if r["family"] == c and r["stage"] == s] for s in ("A", "B")}
        n = min(len(sides["A"]), len(sides["B"]), CAP)
        for stage, before in sides.items():
            require(all(isinstance(r.get("group_id"), str) and r["group_id"] for r in before), "Missing group ID")
            groups = {r["group_id"] for r in before}
            require(len(before) >= 20 and len(groups) >= 5, "Frozen pre-cap support not met")
            after = sorted(before, key=lambda r: ranking("matched-"+stage, case, c, r["sha"]))[:n]
            after_groups = Counter(r["group_id"] for r in after)
            selected.extend(after)
            lost = len(groups) - len(after_groups)
            counts.append(dict(family=c, stage=stage, samples_before=len(before), samples_after=n,
                               groups_before=len(groups), groups_after=len(after_groups), groups_lost=lost,
                               post_cap_below5=len(after_groups) < 5, lost_at_least5=lost >= 5,
                               largest_after=max(after_groups.values()),
                               largest_after_fraction=max(after_groups.values())/n))
    return selected, counts


def verify_alignment(expected, per_view):
    """Exact order/IDs/families, never sort/intersect silently."""
    require(set(per_view) == set(VIEWS), "All three views required")
    key = lambda rows: [(r["sha"], r["family"], r["stage"]) for r in rows]
    base = key(expected)
    require(len(base) == len({t[0] for t in base}), "Duplicate expected ID")
    for view in VIEWS:
        require(key(per_view[view]) == base, "Alignment mismatch: " + view)


def _matrix(value):
    a = np.asarray(value)
    require(a.dtype.kind in "fiu", "Real embeddings required")
    with np.errstate(over="raise", invalid="raise"):
        a = np.array(a, dtype=np.float64, copy=True)
    require(a.ndim == 2 and min(a.shape) > 0 and np.isfinite(a).all(), "Invalid embedding matrix")
    return a


def _reference_matrix(reference, embeddings, case, families):
    reference = checked_rows(reference, case, families, {"train"})
    canonical = train_reference(reference, case=case, families=families)
    require(len(reference) == len(canonical), "Reference exceeds cap")
    ids = [r["sha"] for r in canonical]
    require(set(embeddings) == set(ids), "Embeddings must match TRAIN reference IDs exactly")
    matrix = _matrix([embeddings[s] for s in ids])
    binding = sha256(encoded([[r["sha"], r["family"], r["timestamp"], r["stage"]] for r in canonical]))
    content = sha256(matrix.astype("<f8", copy=False).tobytes())
    return canonical, matrix, binding, content


def _readonly(a):
    a.setflags(write=False)
    return a


@dataclass(frozen=True)
class TrainNormalizer:
    case: str
    families: tuple
    reference_sha256: str
    embeddings_sha256: str
    mean: np.ndarray
    std: np.ndarray
    keep: np.ndarray
    rank: int
    condition_number: object

    def transform(self, embeddings):
        a = _matrix(embeddings)
        require(a.shape[1] == len(self.mean), "Feature dimension mismatch")
        with np.errstate(over="raise", divide="raise", invalid="raise"):
            z = (a[:, self.keep] - self.mean[self.keep]) / self.std[self.keep]
        require(np.isfinite(z).all(), "Non-finite normalized embeddings")
        return z


def fit_train_normalizer(reference, embeddings, *, case, families):
    families = families_checked(families)
    rows, x, binding, content = _reference_matrix(reference, embeddings, case, families)
    indices = [[i for i, r in enumerate(rows) if r["family"] == c] for c in families]
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        mean = np.mean([x[idx].mean(axis=0) for idx in indices], axis=0, dtype=np.float64)
        variance = np.mean([np.square(x[idx]-mean).mean(axis=0) for idx in indices], axis=0, dtype=np.float64)
        std = np.sqrt(variance)
    require(np.isfinite(mean).all() and np.isfinite(std).all(), "Invalid TRAIN moments")
    keep = std > STD_MASK
    require(keep.any(), "All coordinates masked")
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        z = (x[:, keep] - mean[keep])/std[keep]
        row_weights = np.array([1/(len(families)*len(indices[families.index(r["family"])])) for r in rows])
        singular = np.linalg.svd(z * np.sqrt(row_weights[:, None]), compute_uv=False)
    require(np.isfinite(singular).all(), "Non-finite singular values")
    # Standard floating-point rank diagnostic; not a scientific support threshold.
    rank = int(np.sum(singular > singular[0]*max(z.shape)*np.finfo(np.float64).eps))
    cond = float(singular[0]/singular[-1]) if rank == z.shape[1] else None
    if cond is not None:
        require(math.isfinite(cond), "Non-finite conditioning")
    return TrainNormalizer(case, families, binding, content, _readonly(mean), _readonly(std), _readonly(keep), rank, cond)


def weighted_median(values, weights):
    """Lower median; exact rational cumulative masses, including zero distances."""
    values, weights = list(values), list(weights)
    require(len(values) > 0 and len(values) == len(weights), "Median lengths")
    require(all(math.isfinite(float(v)) and float(v) >= 0 for v in values), "Invalid distances")
    try:
        weights = [Fraction(w) for w in weights]
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Invalid median weights") from exc
    require(all(w > 0 for w in weights), "Positive weights required")
    half, cumulative = sum(weights)/2, Fraction(0)
    for value, weight in sorted(zip(values, weights), key=lambda pair: pair[0]):
        cumulative += weight
        if cumulative >= half:
            return float(value)
    raise AssertionError("Unreachable median")


def fit_train_bandwidth(reference, embeddings, *, normalizer):
    rows, x, binding, content = _reference_matrix(reference, embeddings, normalizer.case, normalizer.families)
    require(binding == normalizer.reference_sha256 and content == normalizer.embeddings_sha256,
            "Normalizer/reference provenance mismatch")
    z = normalizer.transform(x)
    by_id = {r["sha"]: i for i, r in enumerate(rows)}
    pairs = reference_pairs(rows, case=normalizer.case, families=normalizer.families)
    counts = Counter(p["family"] for p in pairs)
    distances, weights = [], []
    for pair in pairs:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            delta = z[by_id[pair["left"]]] - z[by_id[pair["right"]]]
            rho = float(np.square(delta).sum(dtype=np.float64)/z.shape[1])
        require(math.isfinite(rho), "Non-finite TRAIN pair distance")
        distances.append(rho)
        weights.append(Fraction(1, len(counts)*counts[pair["family"]]))
    h = weighted_median(distances, weights)
    require(h > 0 and math.isfinite(h), "Degenerate TRAIN bandwidth; no epsilon fallback")
    return {"bandwidth_h": h, "pairs": pairs, "distances": distances,
            "reference_sha256": binding, "embeddings_sha256": content,
            "pair_mass_per_family": {c: str(Fraction(1, len(counts))) for c in normalizer.families},
            "zero_distance_pairs": sum(d == 0 for d in distances)}
