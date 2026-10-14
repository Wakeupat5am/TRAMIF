"""February-only connected-component and five-fold feasibility audit.

No prediction loading, calibration, model execution or March scoring occurs here.
"""
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import numpy as np
import tlsh

from scripts.near_duplicate_core import UnionFind, time_key, validate_rows

ROOT = Path(__file__).resolve().parents[1]
POLICY = ROOT / "data/derived/historical_near_duplicates_v1/20261009_143301_921802"
CONFIG = {
    "month": "2020-02", "threshold": 10, "folds": 5,
    "minimum_samples": 20, "minimum_groups": 5,
    "distance": "standard TLSH including length",
    "vertices": "all original February cohort samples; only retained samples scored",
    "assignment": "deterministic largest normalized class mass first; squared class/size imbalance",
    "quality_coefficient": 1, "drift_coefficient": 0, "uncertainty_penalty": 0,
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def save_csv(path, rows, fields):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def february_rows(path):
    # Shared historical CSVs contain March too. Discard non-February records at
    # ingestion; they cannot participate in grouping, support or fold assignment.
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return [row for row in csv.DictReader(stream)
                if time_key(row)[0].strftime("%Y-%m") == "2020-02"]


def build_groups(rows, retained_ids, distance_function=None):
    rows = validate_rows(rows, "validation")
    require(all(time_key(row)[0].strftime("%Y-%m") == "2020-02" for row in rows),
            "Grouping accepts February only")
    require(bool(rows), "No February vertices")
    retained_ids = set(retained_ids)
    require(retained_ids and retained_ids <= {row["sha"] for row in rows}, "Invalid retained IDs")
    union, retained_union = UnionFind(len(rows)), UnionFind(len(rows))
    objects = []
    for row in rows:
        obj = None
        if distance_function is None and row["tlsh_status"] == "VALID":
            obj = tlsh.Tlsh()
            obj.fromTlshStr(row["tlsh"])
        objects.append(obj)
    edges = []
    for i, first in enumerate(rows):
        for j in range(i):
            second = rows[j]
            exact = first["disarmed_content_sha256"] == second["disarmed_content_sha256"]
            distance = None
            if first["tlsh_status"] == second["tlsh_status"] == "VALID":
                distance = (distance_function(first["tlsh"], second["tlsh"])
                            if distance_function else objects[i].diff(objects[j]))
                require(isinstance(distance, (int, np.integer)) and distance >= 0, "Invalid distance")
            if exact or (distance is not None and distance <= CONFIG["threshold"]):
                union.union(i, j)
                if first["sha"] in retained_ids and second["sha"] in retained_ids:
                    retained_union.union(i, j)
                edges.append({"sha_a": second["sha"], "sha_b": first["sha"],
                              "content_equal": exact, "tlsh_distance": distance})
    groups = defaultdict(list)
    for i, row in enumerate(rows):
        groups[union.find(i)].append(row)
    members, summaries = [], []
    for number, (_, group) in enumerate(sorted(groups.items()), 1):
        gid = f"F{number:06d}"
        kept = [row for row in group if row["sha"] in retained_ids]
        summaries.append({"group_id": gid, "all_samples": len(group), "retained_samples": len(kept),
                          "excluded_samples": len(group) - len(kept),
                          "family_count_all": len({r["family"] for r in group}),
                          "family_count_retained": len({r["family"] for r in kept}),
                          "families": "|".join(sorted({r["family"] for r in group}))})
        for row in group:
            members.append({key: row[key] for key in ("sha", "timestamp", "family", "disarmed_content_sha256")}
                           | {"group_id": gid, "retained": row["sha"] in retained_ids})
    retained_only_groups = len({retained_union.find(i) for i, r in enumerate(rows) if r["sha"] in retained_ids})
    return members, summaries, edges, retained_only_groups


def assign_folds(members, class_names):
    """A single deterministic label-stratification heuristic; no metric tuning."""
    require(len(class_names) == len(set(class_names)), "Duplicate classes")
    lookup = {c: i for i, c in enumerate(class_names)}
    grouped = defaultdict(list)
    for row in members:
        require(row["family"] in lookup, "Family outside frozen mapping")
        if row["retained"]:
            grouped[row["group_id"]].append(row)
    require(bool(grouped), "No active groups")
    counts = {g: np.bincount([lookup[r["family"]] for r in rs], minlength=len(lookup)).astype(float)
              for g, rs in grouped.items()}
    totals = sum(counts.values())
    denominator = np.maximum(totals, 1)
    n = int(totals.sum())
    order = sorted(grouped, key=lambda g: (-float(np.max(counts[g] / denominator)),
                                         -len(grouped[g]), g))
    fold_counts = np.zeros((5, len(lookup)))
    sizes = np.zeros(5)
    assignment = {}
    for index, gid in enumerate(order):
        options = range(5) if index >= 5 else [index]
        def cost(f):
            candidate = fold_counts.copy()
            candidate[f] += counts[gid]
            candidate_sizes = sizes.copy()
            candidate_sizes[f] += len(grouped[gid])
            score = float(np.square(candidate / denominator - 0.2).sum()
                          + np.square(candidate_sizes / n - 0.2).sum())
            return score, sizes[f], f
        chosen = min(options, key=cost)
        assignment[gid] = chosen
        fold_counts[chosen] += counts[gid]
        sizes[chosen] += len(grouped[gid])
    manifest = [row | {"fold": assignment[row["group_id"]]} for row in members if row["retained"]]
    manifest.sort(key=lambda r: (r["timestamp"], r["sha"]))
    return manifest


def assess(members, groups, manifest, class_names, edges):
    retained = [r for r in members if r["retained"]]
    require(len({r["sha"] for r in manifest}) == len(manifest) == len(retained), "Fold coverage/duplicate failure")
    require({r["sha"] for r in retained} == {r["sha"] for r in manifest}, "Fold ID mismatch")
    group_folds = defaultdict(set)
    for row in manifest:
        require(row["fold"] in range(5), "Invalid fold")
        group_folds[row["group_id"]].add(row["fold"])
    require(all(len(v) == 1 for v in group_folds.values()), "Group crosses folds")
    member_groups = {r["sha"]: r["group_id"] for r in members}
    require(all(member_groups[e["sha_a"]] == member_groups[e["sha_b"]] for e in edges), "Edge crosses groups")
    by_class = defaultdict(list)
    for row in manifest:
        by_class[row["family"]].append(row)
    support = []
    for name in class_names:
        rows = by_class[name]
        gs = len({r["group_id"] for r in rows})
        fold_n = [sum(r["fold"] == f for r in rows) for f in range(5)]
        support.append({"family": name, "samples": len(rows), "groups": gs,
                        "in_support": len(rows) >= 20 and gs >= 5,
                        **{f"fold_{f}_score": fold_n[f] for f in range(5)},
                        "minimum_fit_samples": len(rows) - max(fold_n)})
    folds = []
    for f in range(5):
        score = [r for r in manifest if r["fold"] == f]
        fit = [r for r in manifest if r["fold"] != f]
        ids_overlap = {r["sha"] for r in score} & {r["sha"] for r in fit}
        group_overlap = {r["group_id"] for r in score} & {r["group_id"] for r in fit}
        content_overlap = {r["disarmed_content_sha256"] for r in score} & {r["disarmed_content_sha256"] for r in fit}
        require(not ids_overlap and not group_overlap and not content_overlap, "Fit/scoring overlap")
        folds.append({"fold": f, "fit_samples": len(fit), "score_samples": len(score),
                      "score_groups": len({r["group_id"] for r in score}),
                      "score_classes": len({r["family"] for r in score}),
                      "sample_overlap": 0, "group_overlap": 0, "content_overlap": 0})
    selected = [r for r in support if r["in_support"]]
    active_sizes = [g["retained_samples"] for g in groups if g["retained_samples"]]
    # Predeclared review gate: a component larger than the ideal fold size
    # forces imbalance that the assignment algorithm cannot fix without splitting it.
    gates = {
        "at_least_five_active_groups": len(active_sizes) >= 5,
        "all_fit_score_partitions_nonempty": all(f["fit_samples"] and f["score_samples"] for f in folds),
        "nonempty_approved_support": bool(selected),
        "supported_classes_present_in_every_fit": all(r["minimum_fit_samples"] > 0 for r in selected),
        "largest_component_within_ideal_fold_size": max(active_sizes) <= math.ceil(len(retained) / 5),
    }
    return support, folds, gates


def main():
    inventory = ROOT / "data/derived/validation_tlsh_v1/validation_tlsh.csv"
    summary = inventory.with_name("summary.json")
    config = inventory.with_name("config.json")
    retained_path = POLICY / "threshold_00/validation_keep.csv"
    mapping_path = ROOT / "docs/class_mapping.json"
    original_path = ROOT / "data/manifests/train_validation_manifest.csv"
    policy_report_path, policy_config_path = POLICY / "report.json", POLICY / "config.json"
    report = json.loads(policy_report_path.read_bytes())
    policy_config = json.loads(policy_config_path.read_bytes())
    inventory_summary = json.loads(summary.read_bytes())
    inventory_config = json.loads(config.read_bytes())
    require(sha(inventory) == inventory_summary["csv_sha256"]
            == policy_config["inventory_sources"][inventory.relative_to(ROOT).as_posix()], "Inventory hash")
    require(sha(retained_path) == report["output_csv_sha256"]["threshold_00/validation_keep.csv"], "Retention hash")
    require(sha(mapping_path) == policy_config["class_mapping_sha256"] == inventory_config["class_mapping_sha256"], "Mapping hash")
    require(sha(original_path) == policy_config["manifest_sha256"] == inventory_config["manifest_sha256"], "Manifest hash")
    require(policy_config["primary_threshold"] == 0 and version("py-tlsh") == "5.0.0", "Policy/runtime mismatch")
    rows, kept, original = february_rows(inventory), february_rows(retained_path), february_rows(original_path)
    require(len(rows) == len(original) == 3825 and len(kept) == 3799, "Unexpected February scope")
    identity = lambda rs: {r["sha"]: (r["split"], r["timestamp"], r["family"]) for r in rs}
    require(identity(rows) == identity(original), "Inventory/manifest mismatch")
    require(len(identity(kept)) == len(kept) and identity(kept).items() <= identity(original).items(), "Retention not a subset")
    mapping = json.loads(mapping_path.read_bytes())["class_to_idx"]
    require(sorted(mapping.values()) == list(range(51)), "Not frozen 51-class mapping")
    classes = sorted(mapping, key=mapping.get)
    source_paths = [inventory, summary, config, retained_path, mapping_path, original_path,
                    policy_report_path, policy_config_path, Path(__file__), ROOT / "scripts/near_duplicate_core.py",
                    ROOT / "docs/february_feasibility_protocol.md"]
    before = {p.relative_to(ROOT).as_posix(): sha(p) for p in source_paths}
    print("February vertices: 3825; retained: 3799; distance <=10; exhaustive pairs", flush=True)
    members, groups, edges, retained_only = build_groups(rows, {r["sha"] for r in kept})
    manifest = assign_folds(members, classes)
    support, folds, gates = assess(members, groups, manifest, classes, edges)
    sizes = [g["retained_samples"] for g in groups if g["retained_samples"]]
    require(all(sha(ROOT / p) == h for p, h in before.items()), "Source changed during audit")
    output = ROOT / "data/derived/february_group_audit" / datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    output.mkdir(parents=True, exist_ok=False)
    tables = {"members.csv": members, "groups.csv": groups, "fold_manifest.csv": manifest,
              "class_support.csv": support, "fold_summary.csv": folds}
    for filename, table in tables.items():
        save_csv(output / filename, table, list(table[0]))
    save_csv(output / "edges.csv", edges, ["sha_a", "sha_b", "content_equal", "tlsh_distance"])
    result = {
        "status": "FEASIBLE_ENGINEERING_ONLY" if all(gates.values()) else "STOP_REVIEW_REQUIRED",
        "config": CONFIG, "gates": gates, "source_sha256": before,
        "py_tlsh": version("py-tlsh"), "numpy": np.__version__,
        "original_samples": len(rows), "retained_samples": len(manifest), "bridge_only_samples": len(rows) - len(manifest),
        "exhaustive_pairs": len(rows) * (len(rows) - 1) // 2, "qualifying_edges": len(edges),
        "all_groups": len(groups), "active_groups": len(sizes), "retained_only_graph_groups": retained_only,
        "bridge_group_merges": retained_only - len(sizes),
        "group_size_distribution": dict(sorted(Counter(sizes).items())),
        "size_quantiles": {str(q): float(np.quantile(sizes, q)) for q in (0, 0.5, 0.9, 0.95, 0.99, 1)},
        "largest_group_fraction": max(sizes) / len(manifest),
        "multi_label_groups_all": sum(g["family_count_all"] > 1 for g in groups),
        "multi_label_groups_retained": sum(g["family_count_retained"] > 1 for g in groups),
        "classes_meeting_sample_minimum": sum(r["samples"] >= 20 for r in support),
        "support_classes": [r["family"] for r in support if r["in_support"]],
        "support_samples": sum(r["samples"] for r in support if r["in_support"]),
        "tnull": sum(r["tlsh_status"] == "TNULL" for r in rows), "folds": folds,
        "limitations": ["TLSH components are operational blocks, not proven independent units.",
                        "All direct February threshold edges considered, but false-negative similarity remains possible.",
                        "No outside-February bridge nodes; disarmed bytes only; transitive component diameter may exceed 10.",
                        "Greedy class balancing is not an optimal partition certificate.",
                        "No arbitrary maximum fold-imbalance or minimum support-coverage claim is statistically validated."],
        "march_used_for_assignment": False, "future_test_accessed": False,
        "calibration_performed": False,
        "artifact_sha256": {p.name: sha(p) for p in sorted(output.glob("*.csv"))},
    }
    save_json(output / "report.json", result)
    print(json.dumps({k: result[k] for k in ("status", "gates", "all_groups", "active_groups", "bridge_group_merges",
                                           "size_quantiles", "multi_label_groups_retained", "support_classes", "support_samples", "folds")}, indent=2))
    print("Report:", output / "report.json")


if __name__ == "__main__":
    main()
