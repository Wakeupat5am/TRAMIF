"""March connected components and support-only pilot; NO metric bootstrap."""
import csv
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import numpy as np
import tlsh

from scripts.february_group_audit import ROOT, POLICY, require, sha, save_json, save_csv
from scripts.near_duplicate_core import UnionFind, time_key, validate_rows

EVALUATION = ROOT / "data/derived/quality_march_evaluation/20261013_125046_074164"
THRESHOLD = 10
PILOT_REPLICATES = 500
SEED = 42


def march_rows(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return [r for r in csv.DictReader(stream) if time_key(r)[0].strftime("%Y-%m") == "2020-03"]


def march_components(rows, retained_ids, distance_function=None):
    rows = validate_rows(rows, "validation")
    require(bool(rows) and all(time_key(r)[0].strftime("%Y-%m") == "2020-03" for r in rows), "March only")
    retained_ids = set(retained_ids)
    require(bool(retained_ids) and retained_ids <= {r["sha"] for r in rows}, "Unknown retained IDs")
    full, retained_only = UnionFind(len(rows)), UnionFind(len(rows))
    objects = []
    for r in rows:
        obj = None
        if distance_function is None and r["tlsh_status"] == "VALID":
            obj = tlsh.Tlsh()
            obj.fromTlshStr(r["tlsh"])
        objects.append(obj)
    edges = []
    for i, a in enumerate(rows):
        for j in range(i):
            b = rows[j]
            exact = a["disarmed_content_sha256"] == b["disarmed_content_sha256"]
            d = None
            if a["tlsh_status"] == b["tlsh_status"] == "VALID":
                d = distance_function(a["tlsh"], b["tlsh"]) if distance_function else objects[i].diff(objects[j])
                require(isinstance(d, (int, np.integer)) and d >= 0, "Invalid TLSH distance")
            if exact or (d is not None and d <= THRESHOLD):
                full.union(i, j)
                if a["sha"] in retained_ids and b["sha"] in retained_ids:
                    retained_only.union(i, j)
                edges.append({"sha_a": b["sha"], "sha_b": a["sha"], "content_equal": exact, "tlsh_distance": d})
    groups = defaultdict(list)
    for i, r in enumerate(rows):
        groups[full.find(i)].append(r)
    members, summary = [], []
    for index, (_, group) in enumerate(sorted(groups.items()), 1):
        gid = f"M{index:06d}"
        kept = [r for r in group if r["sha"] in retained_ids]
        summary.append({"group_id": gid, "all_samples": len(group), "retained_samples": len(kept),
                        "excluded_samples": len(group) - len(kept),
                        "family_count_all": len({r["family"] for r in group}),
                        "family_count_retained": len({r["family"] for r in kept}),
                        "families_all": "|".join(sorted({r["family"] for r in group})),
                        "families_retained": "|".join(sorted({r["family"] for r in kept}))})
        for r in group:
            members.append({key: r[key] for key in ("sha", "timestamp", "family", "disarmed_content_sha256")}
                           | {"group_id": gid, "retained": r["sha"] in retained_ids})
    ids_to_group = {r["sha"]: r["group_id"] for r in members}
    require(all(ids_to_group[e["sha_a"]] == ids_to_group[e["sha_b"]] for e in edges), "Edge crosses groups")
    require({r["sha"] for r in members if r["retained"]} == retained_ids, "Retained coverage")
    direct_groups = len({retained_only.find(i) for i, r in enumerate(rows) if r["sha"] in retained_ids})
    return members, summary, edges, direct_groups


def support_matrix(members, classes):
    require(len(classes) == len(set(classes)) and bool(classes), "Invalid classes")
    retained = [r for r in members if r["retained"]]
    require(bool(retained) and len({r["sha"] for r in retained}) == len(retained), "Duplicate/empty retained samples")
    groups = sorted({r["group_id"] for r in retained})
    gi, ci = {g: i for i, g in enumerate(groups)}, {c: i for i, c in enumerate(classes)}
    matrix = np.zeros((len(groups), len(classes)), dtype=np.int64)
    for r in retained:
        require(r["family"] in ci, "Unknown family")
        matrix[gi[r["group_id"]], ci[r["family"]]] += 1
    return groups, matrix


def support_pilot(matrix, classes, replicates=500, seed=42):
    matrix = np.asarray(matrix)
    require(matrix.ndim == 2 and matrix.shape[0] > 0 and matrix.shape[1] == len(classes)
            and matrix.dtype.kind in "iu" and np.all(matrix >= 0) and np.all(matrix.sum(1) > 0), "Invalid group counts")
    require(isinstance(replicates, int) and 1 <= replicates <= 500, "Pilot budget must be 1..500")
    g = len(matrix)
    # Multiplicities of G draws with replacement; all class memberships stay together.
    draws = np.random.default_rng(seed).multinomial(g, np.full(g, 1/g), size=replicates)
    counts = draws @ matrix
    original = matrix.sum(0)
    supported = original > 0  # fixed before any draw
    missing = (counts == 0) & supported[None, :]
    require(np.all(draws.sum(1) == g), "Invalid draw counts")
    rows = []
    for c, family in enumerate(classes):
        sizes = matrix[:, c]
        m, n = int((sizes > 0).sum()), int(original[c])
        absence = float((1 - m/g) ** g) if m else None
        rows.append({"family": family, "samples": n, "groups": m,
                     "effective_groups_size_only": float(n*n / np.square(sizes).sum()) if n else None,
                     "largest_group_class_samples": int(sizes.max()),
                     "largest_group_class_fraction": float(sizes.max() / n) if n else None,
                     "analytic_missing_probability": absence,
                     "pilot_missing_replicates": int(missing[:, c].sum()) if n else None,
                     "pilot_missing_fraction": float(missing[:, c].mean()) if n else None})
    reps = [{"replicate": b, "sample_count": int(counts[b].sum()),
             "distinct_sampled_groups": int((draws[b] > 0).sum()),
             "missing_supported_classes": int(missing[b].sum()),
             "missing_families": "|".join(classes[c] for c in np.flatnonzero(missing[b]))}
            for b in range(replicates)]
    return rows, reps, draws, counts, missing


def main():
    inventory = ROOT / "data/derived/validation_tlsh_v1/validation_tlsh.csv"
    inv_summary, inv_config = inventory.with_name("summary.json"), inventory.with_name("config.json")
    retained_path = POLICY / "threshold_00/validation_keep.csv"
    original_path, mapping_path = ROOT / "data/manifests/train_validation_manifest.csv", ROOT / "docs/class_mapping.json"
    policy_config = json.loads((POLICY / "config.json").read_bytes())
    policy_report = json.loads((POLICY / "report.json").read_bytes())
    summary, config = json.loads(inv_summary.read_bytes()), json.loads(inv_config.read_bytes())
    require(sha(inventory) == summary["csv_sha256"] == policy_config["inventory_sources"][inventory.relative_to(ROOT).as_posix()], "Inventory hash")
    for p in (inv_summary, inv_config):
        require(sha(p) == policy_config["inventory_sources"][p.relative_to(ROOT).as_posix()], "Inventory provenance")
    require(sha(retained_path) == policy_report["output_csv_sha256"]["threshold_00/validation_keep.csv"], "Retained hash")
    require(sha(original_path) == policy_config["manifest_sha256"] == config["manifest_sha256"], "Original manifest hash")
    require(sha(mapping_path) == policy_config["class_mapping_sha256"] == config["class_mapping_sha256"], "Mapping hash")
    require(policy_config["primary_threshold"] == 0 and version("py-tlsh") == "5.0.0", "Policy/runtime mismatch")
    rows, retained, original = march_rows(inventory), march_rows(retained_path), march_rows(original_path)
    require(len(rows) == len(original) == 4438 and len(retained) == 4332, "March counts")
    identity = lambda rs: {r["sha"]: (r["split"], r["timestamp"], r["family"]) for r in rs}
    require(identity(rows) == identity(original), "Original March identity mismatch")
    require(len(identity(retained)) == len(retained) and identity(retained).items() <= identity(original).items(), "Invalid retention")
    mapping = json.loads(mapping_path.read_bytes())["class_to_idx"]
    require(sorted(mapping.values()) == list(range(51)), "Class mapping")
    classes = sorted(mapping, key=mapping.get)
    evaluation = json.loads((EVALUATION / "report.json").read_bytes())
    require(sha(EVALUATION / "predictions.npz") == evaluation["artifact_sha256"]["predictions.npz"], "Evaluation hash")
    with np.load(EVALUATION / "predictions.npz", allow_pickle=False) as data:
        ids, labels = data["sample_identifiers"].tolist(), data["labels"].tolist()
        require(data["class_names"].tolist() == classes and len(ids) == len(set(ids)) == 4332, "Evaluation identity")
        require(dict(zip(ids, labels)) == {r["sha"]: mapping[r["family"]] for r in retained}, "Audit/evaluation selection mismatch")
    paths = [inventory, inv_summary, inv_config, retained_path, original_path, mapping_path,
             POLICY / "config.json", POLICY / "report.json", EVALUATION / "report.json", EVALUATION / "predictions.npz",
             Path(__file__), ROOT / "scripts/near_duplicate_core.py", ROOT / "scripts/february_group_audit.py"]
    before = {p.relative_to(ROOT).as_posix(): sha(p) for p in paths}
    print("March-only exhaustive graph: 4438 vertices, 4332 retained; threshold 10", flush=True)
    members, groups, edges, direct_groups = march_components(rows, {r["sha"] for r in retained})
    group_ids, matrix = support_matrix(members, classes)
    support, reps, draws, counts, missing = support_pilot(matrix, classes, PILOT_REPLICATES, SEED)
    sizes = matrix.sum(1)
    require(int(matrix.sum()) == 4332 and int((matrix.sum(0) > 0).sum()) == 47, "Support totals")
    require(all(sha(ROOT / p) == digest for p, digest in before.items()), "Inputs changed")
    output = ROOT / "data/derived/march_group_feasibility" / datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    output.mkdir(parents=True, exist_ok=False)
    for name, table in (("members", members), ("groups", groups), ("class_support", support), ("pilot_replicates", reps)):
        save_csv(output / f"{name}.csv", table, list(table[0]))
    save_csv(output / "edges.csv", edges, ["sha_a", "sha_b", "content_equal", "tlsh_distance"])
    np.savez_compressed(output / "pilot_counts.npz", group_ids=np.array(group_ids), class_names=np.array(classes),
                        group_class_counts=matrix, sampled_group_multiplicities=draws,
                        replicate_class_counts=counts, missing_original_supported_classes=missing)
    result = {"status": "GROUPING_AND_SUPPORT_PILOT_ONLY", "month": "2020-03", "threshold": THRESHOLD,
              "primary_policy_threshold": 0, "original_samples": 4438, "retained_samples": 4332,
              "bridge_only_vertices": 106, "all_groups": len(groups), "active_groups": len(sizes),
              "retained_only_graph_groups": direct_groups, "bridge_group_merges": direct_groups - len(sizes),
              "exhaustive_pairs": len(rows)*(len(rows)-1)//2, "qualifying_edges": len(edges),
              "all_group_size_distribution": dict(sorted(Counter(g["all_samples"] for g in groups).items())),
              "active_group_size_distribution": dict(sorted(Counter(int(s) for s in sizes).items())),
              "active_size_quantiles": {str(q): float(np.quantile(sizes, q)) for q in (0, .5, .9, .95, .99, 1)},
              "largest_retained_group": int(sizes.max()), "largest_group_fraction": float(sizes.max()/sizes.sum()),
              "effective_groups_size_only": float(sizes.sum()**2 / np.square(sizes).sum()),
              "effective_groups_definition": "(sum n_g)^2 / sum n_g^2; descriptive size concentration, NOT independent sample size",
              "multi_label_groups_all": sum(g["family_count_all"] > 1 for g in groups),
              "multi_label_groups_retained": sum(g["family_count_retained"] > 1 for g in groups),
              "tnull": sum(r["tlsh_status"] == "TNULL" for r in rows),
              "supported_classes": [r["family"] for r in support if r["samples"]],
              "absent_classes": [r["family"] for r in support if not r["samples"]],
              "classes_with_fewer_than_five_groups": [r["family"] for r in support if 0 < r["groups"] < 5],
              "pilot_replicates": 500, "seed": 42, "pilot_method": "G whole-group draws with replacement using multinomial multiplicities",
              "replicates_missing_any_supported_class": int(missing.any(1).sum()),
              "missing_any_fraction": float(missing.any(1).mean()),
              "missing_class_count_distribution": dict(sorted(Counter(int(x) for x in missing.sum(1)).items())),
              "replicates_discarded": 0, "retry_draws": 0, "metric_denominators_changed": False,
              "metric_bootstrap_performed": False, "confidence_intervals_computed": False,
              "future_test_used": False, "sample_selection_changed": False, "weights_changed": False,
              "source_sha256": before, "numpy_version": np.__version__, "py_tlsh_version": version("py-tlsh"),
              "limitations": ["Operational components, not established independent/exchangeable units.",
                              "Transitive closure; component diameter may exceed 10; TLSH can miss similarity.",
                              "No outside-March bridges; disarmed content only.",
                              "Pilot estimates support feasibility only, not a metric uncertainty report.",
                              "Resampled sample count varies with group sizes; rare/mixed-label groups cannot be split by class."],
              "artifact_sha256": {p.name: sha(p) for p in sorted(output.iterdir())}}
    save_json(output / "report.json", result)
    save_json(output / "checksums.json", {p.name: sha(p) for p in sorted(output.iterdir())})
    print(json.dumps({k: result[k] for k in ("all_groups", "active_groups", "bridge_group_merges", "active_size_quantiles",
                                           "effective_groups_size_only", "multi_label_groups_retained", "classes_with_fewer_than_five_groups",
                                           "replicates_missing_any_supported_class", "missing_any_fraction")}, indent=2))
    print("Report:", output / "report.json")


if __name__ == "__main__":
    main()
