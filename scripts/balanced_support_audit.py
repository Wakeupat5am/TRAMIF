"""Counts-only February/March graph audit. No models, MMD, U or weight fitting."""
from collections import Counter, defaultdict

import numpy as np
import tlsh

from scripts.near_duplicate_core import UnionFind, time_key, validate_rows

MONTHS = ("2020-02", "2020-03")
THRESHOLD = 10


def require(condition, message):
    if not condition:
        raise ValueError(message)


def month(row):
    return time_key(row)[0].strftime("%Y-%m")


def build_joint_graph(rows, retained_ids, distance_function=None, progress=None):
    rows = validate_rows(rows, "validation")
    require(bool(rows) and {month(r) for r in rows} == set(MONTHS), "Both historical months required")
    retained_ids = set(retained_ids)
    require(retained_ids and retained_ids <= {r["sha"] for r in rows}, "Unknown retained IDs")
    full, direct, local = [UnionFind(len(rows)) for _ in range(3)]
    objects = []
    for r in rows:
        obj = None
        if distance_function is None and r["tlsh_status"] == "VALID":
            obj = tlsh.Tlsh()
            obj.fromTlshStr(r["tlsh"])
        objects.append(obj)
    month_cache = [month(r) for r in rows]
    edges, counts = [], Counter()
    for i, a in enumerate(rows):
        for j in range(i):
            b = rows[j]
            cross = month_cache[i] != month_cache[j]
            exact = a["disarmed_content_sha256"] == b["disarmed_content_sha256"]
            d = None
            if a["tlsh_status"] == b["tlsh_status"] == "VALID":
                d = distance_function(a["tlsh"], b["tlsh"]) if distance_function else objects[i].diff(objects[j])
                require(isinstance(d, (int, np.integer)) and d >= 0, "Invalid TLSH distance")
            category = "cross_month" if cross else "within_month"
            counts[category + "_pairs"] += 1
            if exact or (d is not None and d <= THRESHOLD):
                full.union(i, j)
                if not cross:
                    local.union(i, j)
                both = a["sha"] in retained_ids and b["sha"] in retained_ids
                if both:
                    direct.union(i, j)
                counts[category + "_edges"] += 1
                counts[category + "_exact_edges"] += int(exact)
                counts[category + "_tlsh_edges"] += int(d is not None and d <= THRESHOLD)
                counts[category + "_retained_edges"] += int(both)
                edges.append({"sha_a": b["sha"], "sha_b": a["sha"], "cross_month": cross,
                              "content_equal": exact, "tlsh_distance": d, "both_retained": both})
        if progress and (i + 1) % 1000 == 0:
            progress(i + 1, len(rows))
    grouped = defaultdict(list)
    for i in range(len(rows)):
        grouped[full.find(i)].append(i)
    members, groups = [], []
    for number, indices in enumerate(grouped.values(), 1):
        gid = f"J{number:06d}"
        kept = [i for i in indices if rows[i]["sha"] in retained_ids]
        record = {"group_id": gid, "all_samples": len(indices), "retained_samples": len(kept),
                  "excluded_samples": len(indices) - len(kept),
                  "months_all": "|".join(sorted({month(rows[i]) for i in indices})),
                  "months_retained": "|".join(sorted({month(rows[i]) for i in kept})),
                  "families_retained": "|".join(sorted({rows[i]["family"] for i in kept})),
                  "family_count_all": len({rows[i]["family"] for i in indices}),
                  "family_count_retained": len({rows[i]["family"] for i in kept})}
        for m in MONTHS:
            record[m + "_retained"] = sum(month(rows[i]) == m for i in kept)
        groups.append(record)
        for i in indices:
            r = rows[i]
            members.append({k: r[k] for k in ("sha", "timestamp", "family", "disarmed_content_sha256")}
                           | {"month": month(r), "group_id": gid, "local_group_id": f"L{local.find(i):06d}",
                              "retained": r["sha"] in retained_ids})
    gid_by_id = {r["sha"]: r["group_id"] for r in members}
    require(all(gid_by_id[e["sha_a"]] == gid_by_id[e["sha_b"]] for e in edges), "Edge crosses groups")
    require({r["sha"] for r in members if r["retained"]} == retained_ids, "Retained coverage")
    stats = dict(counts)
    stats["retained_only_active_groups"] = len({direct.find(i) for i, r in enumerate(rows) if r["sha"] in retained_ids})
    stats["local_active_groups"] = {m: len({r["local_group_id"] for r in members if r["retained"] and r["month"] == m}) for m in MONTHS}
    return members, groups, edges, stats


def equal_side_capacity(sizes):
    """Exact best equal subsample budget after a binary whole-group assignment.

    Per-family bound only; separate family optima need not share one allocation.
    """
    require(all(isinstance(s, (int, np.integer)) and s > 0 for s in sizes), "Invalid group sizes")
    bits, total = 1, sum(sizes)
    for size in sizes:
        bits |= bits << int(size)
    return (bits & ((1 << (total // 2 + 1)) - 1)).bit_length() - 1


def class_support(members, classes):
    require(classes and len(classes) == len(set(classes)), "Invalid classes")
    kept = [r for r in members if r["retained"]]
    require(len({r["sha"] for r in kept}) == len(kept), "Duplicate retained IDs")
    require(all(r["family"] in classes and r["month"] in MONTHS for r in kept), "Unknown class/month")
    group_months = defaultdict(set)
    for r in kept:
        group_months[r["group_id"]].add(r["month"])
    result = []
    for family in classes:
        record, failures = {"family": family}, []
        for m in MONTHS:
            selected = [r for r in kept if r["family"] == family and r["month"] == m]
            counts = Counter(r["group_id"] for r in selected)
            n, g = len(selected), len(counts)
            record.update({m + "_samples": n, m + "_groups_joint": g,
                           m + "_groups_local": len({r["local_group_id"] for r in selected}),
                           m + "_largest_class_group": max(counts.values(), default=0),
                           m + "_largest_class_fraction": max(counts.values(), default=0) / n if n else None,
                           m + "_size_concentration_count": n*n / sum(s*s for s in counts.values()) if n else None,
                           m + "_cross_month_samples": sum(len(group_months[r["group_id"]]) == 2 for r in selected),
                           m + "_equal_side_capacity": equal_side_capacity(list(counts.values()))})
            if n < 20:
                failures.append(m + ":samples<20")
            if g < 5:
                failures.append(m + ":groups<5")
        record["candidate_S_star"] = not failures
        record["failure_reasons"] = "|".join(failures)
        result.append(record)
    return result


def allocation_witness(members, support):
    """One deterministic count-only greedy partition per month, NOT a null sampler."""
    if not support:
        return [], []
    assignment, summaries = [], []
    for m in MONTHS:
        counts = defaultdict(lambda: np.zeros(len(support), dtype=int))
        ci = {c: i for i, c in enumerate(support)}
        for r in members:
            if r["retained"] and r["month"] == m and r["family"] in ci:
                counts[r["group_id"]][ci[r["family"]]] += 1
        total = np.sum(list(counts.values()), axis=0)
        totals = np.zeros((2, len(support)), dtype=int)
        order = sorted(counts, key=lambda g: (-float(np.sum(counts[g] / total)), g))
        for gid in order:
            costs = []
            for side in range(2):
                trial = totals.copy()
                trial[side] += counts[gid]
                costs.append(float(np.square((trial[0] - trial[1]) / total).sum()))
            side = int(np.argmin(costs))
            totals[side] += counts[gid]
            assignment.append({"month": m, "group_id": gid, "side": side})
        require(np.array_equal(totals.sum(0), total), "Witness lost members")
        for i, family in enumerate(support):
            summaries.append({"month": m, "family": family, "side_0_samples": int(totals[0, i]),
                              "side_1_samples": int(totals[1, i]), "equal_budget": int(totals[:, i].min())})
    return assignment, summaries
