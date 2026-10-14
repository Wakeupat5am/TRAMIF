"""BALANCED Phase 1: read historical inventories; write a new counts-only audit."""
import csv
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import numpy as np

from scripts.balanced_support_audit import (
    MONTHS, build_joint_graph, class_support, allocation_witness, month, require,
)
from scripts.february_group_audit import ROOT, POLICY, sha, save_json, save_csv

FREEZE = ROOT / "data/derived/quality_weight_freeze/20261013_124031_275438"
EVAL = ROOT / "data/derived/quality_march_evaluation/20261013_125046_074164"
BOOT = ROOT / "data/derived/march_paired_bootstrap/20261013_131843_476287"
FEB = ROOT / "data/derived/february_group_audit/20261013_090516_568369"
MAR = ROOT / "data/derived/march_group_feasibility/20261013_125459_149338"


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def preprocessing_identity(config):
    """Match dataset loaders' semantic config ID, separate from file-byte SHA."""
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode("utf-8")).hexdigest()


def check_prediction_ids(path, retained, classes, mapping):
    with np.load(path, allow_pickle=False) as data:
        ids = data["sample_identifiers"].tolist()
        labels = data["labels"].tolist()
        require(len(ids) == len(labels) and data["labels"].ndim == 1, "Prediction label shape")
        require(data["class_names"].tolist() == classes, "Prediction class order")
        require(len(ids) == len(set(ids)) == len(retained), "Prediction duplicate/count mismatch")
        require(dict(zip(ids, labels)) == {r["sha"]: mapping[r["family"]] for r in retained}, "Prediction IDs/labels mismatch")
    return ids


def main():
    sources = {}

    def track(path, expected=None):
        value = sha(path)
        if expected is not None:
            require(value == expected, "Provenance mismatch: " + str(path))
        sources[path.relative_to(ROOT).as_posix()] = value
        return value

    def load(path, expected=None):
        track(path, expected)
        return json.loads(path.read_bytes())

    # Legacy containers are read and hashed only, including checkpoint bytes.
    for folder in (FREEZE, EVAL, BOOT, MAR):
        for name, expected in load(folder / "checksums.json").items():
            require(Path(name).name == name, "Non-local artifact filename")
            track(folder / name, expected)
    frozen = load(FREEZE / "weights.json", "3fbaf039b608c4df195922b6fc29a2ff141b6fbdd1a4527a5a8c87f96eecc0ba")
    feb_report = load(FEB / "report.json")
    for name, expected in feb_report["artifact_sha256"].items():
        track(FEB / name, expected)
    registry = load(ROOT / "docs/baseline_artifacts.json", "06c7ae980024d62ffe01245a859159c091606f5060b29dc0c57508decb824a95")
    policy = load(POLICY / "config.json")
    policy_report = load(POLICY / "report.json")
    inventory = ROOT / "data/derived/validation_tlsh_v1/validation_tlsh.csv"
    inv_summary = load(inventory.with_name("summary.json"), policy["inventory_sources"][inventory.with_name("summary.json").relative_to(ROOT).as_posix()])
    inv_config = load(inventory.with_name("config.json"), policy["inventory_sources"][inventory.with_name("config.json").relative_to(ROOT).as_posix()])
    track(inventory, inv_summary["csv_sha256"])
    require(sources[inventory.relative_to(ROOT).as_posix()] == policy["inventory_sources"][inventory.relative_to(ROOT).as_posix()], "Inventory policy binding")
    kept_path = POLICY / "threshold_00/validation_keep.csv"
    track(kept_path, policy_report["output_csv_sha256"]["threshold_00/validation_keep.csv"])
    original_path = ROOT / "data/manifests/train_validation_manifest.csv"
    track(original_path, policy["manifest_sha256"])
    require(policy["manifest_sha256"] == inv_config["manifest_sha256"], "Original manifest binding")
    mapping = load(ROOT / "docs/class_mapping.json", policy["class_mapping_sha256"])["class_to_idx"]
    require(policy["class_mapping_sha256"] == inv_config["class_mapping_sha256"] and sorted(mapping.values()) == list(range(51)), "Mapping binding")
    classes = sorted(mapping, key=mapping.get)
    require(classes == frozen["all_classes"] and policy["primary_threshold"] == 0 and version("py-tlsh") == "5.0.0", "Frozen settings")
    rows, kept = read_csv(inventory), read_csv(kept_path)
    original_all = read_csv(original_path)
    require(all(r["split"] in ("train", "validation") for r in original_all), "Unexpected manifest scope")
    original = [r for r in original_all if r["split"] == "validation"]
    ident = lambda rs: {r["sha"]: (r["split"], r["timestamp"], r["family"]) for r in rs}
    require(len(rows) == len(original) == 8263 and len(ident(rows)) == len(rows) and ident(rows) == ident(original), "Inventory identity")
    require(len(kept) == len(ident(kept)) == 8131 and ident(kept).items() <= ident(original).items(), "Retention identity")
    require(Counter(month(r) for r in rows) == {"2020-02": 3825, "2020-03": 4438}, "Original months")
    require(Counter(month(r) for r in kept) == {"2020-02": 3799, "2020-03": 4332}, "Retained months")
    require(all(r["family"] in mapping for r in rows), "Unknown inventory family")

    consistency, canonical = [], {}
    dataset_names = {"sbsmi": "sbsmi_dataset_v1", "raw-byte": "raw_byte_dataset_v1", "entropy": "entropy_dataset_v1"}
    require([e["view"] for e in registry["entries"]] == ["sbsmi", "raw-byte", "entropy"], "Registry views")
    for entry in registry["entries"]:
        track(ROOT / entry["checkpoint"], entry["checkpoint_sha256"])
        temp = load(ROOT / entry["temperature_file"], entry["temperature_file_sha256"])
        require(temp["fit_month"] == "2020-02" and temp["fit_samples"] == 3799
                and temp["checkpoint_sha256"] == entry["checkpoint_sha256"]
                and temp["temperature"] == entry["temperature"] and temp["class_to_idx"] == mapping, "Temperature scope")
        configs = set()
        for item in entry["evaluations"]:
            path = ROOT / item["report"]
            report = load(path, item["report_sha256"])
            track(path.with_name("predictions.npz"), item["predictions_sha256"])
            m = item["month"]
            require(m in MONTHS and report["checkpoint_sha256"] == entry["checkpoint_sha256"]
                    and report["class_to_idx"] == mapping and report["training_seed"] == 42
                    and not report["future_test_used"] and report["selection"]["threshold"] == 0, "Evaluation scope")
            require(report["selection"]["policy_config_sha256"] == sha(POLICY / "config.json")
                    and report["selection"]["retained_manifest_sha256"] == sha(kept_path), "Selection binding")
            selected = [r for r in kept if month(r) == m]
            ids = check_prediction_ids(path.with_name("predictions.npz"), selected, classes, mapping)
            require(m not in canonical or canonical[m] == ids, "Views have different sample order")
            canonical[m] = ids
            configs.add(report["dataset_config_sha256"])
        require(len(configs) == 1, "Within-view preprocessing differs across months")
        config_path = ROOT / "data/derived" / dataset_names[entry["view"]] / "config.json"
        dataset_config = load(config_path)
        require(preprocessing_identity(dataset_config) == next(iter(configs)), "Preprocessing identity mismatch")
        require(dataset_config["manifest_sha256"] == policy["manifest_sha256"]
                and dataset_config["class_mapping_sha256"] == policy["class_mapping_sha256"], "Preprocessing inputs mismatch")
        consistency.append({"view": entry["view"], "checkpoint_sha256": entry["checkpoint_sha256"],
                            "temperature": entry["temperature"], "dataset_config_sha256": next(iter(configs)),
                            "feature_layer_proposal": "head_norm output before dropout, eval mode",
                            "feature_dimensions_from_source": 2048 if entry["view"] == "sbsmi" else 512,
                            "models_or_images_executed": False})
    for p in (Path(__file__), ROOT / "scripts/balanced_support_audit.py", ROOT / "scripts/near_duplicate_core.py",
              ROOT / "models/malsbslcnet.py", ROOT / "models/compact_view_net.py", ROOT / "docs/balanced_research_scope.md"):
        track(p)
    print("Frozen inventory, retention, 51-class mapping, checkpoints, temperatures and six prediction alignments: PASS", flush=True)
    members, groups, edges, stats = build_joint_graph(rows, {r["sha"] for r in kept},
        progress=lambda n, total: print(f"Joint graph: {n}/{total} vertices", flush=True))
    require(stats["local_active_groups"] == {"2020-02": 2032, "2020-03": 2234}, "Prior local grouping counts not reproduced")
    # Verify the prior local partitions themselves, not just counts (group names differ).
    for m, folder in zip(MONTHS, (FEB, MAR)):
        old = read_csv(folder / "members.csv")
        old_partition = {r["sha"]: r["group_id"] for r in old}
        a_to_b, b_to_a = {}, {}
        for r in members:
            if r["month"] == m:
                a, b = r["local_group_id"], old_partition[r["sha"]]
                require(a_to_b.get(a, b) == b and b_to_a.get(b, a) == a, "Local partitions disagree")
                a_to_b[a], b_to_a[b] = b, a
    support = class_support(members, classes)
    selected_support = [r["family"] for r in support if r["candidate_S_star"]]
    assignment, witness = allocation_witness(members, selected_support)
    active = [g for g in groups if g["retained_samples"]]
    sizes = np.array([g["retained_samples"] for g in active], dtype=np.int64)
    retained_ids = {r["sha"] for r in kept}
    for key in ("cross_month_exact_edges", "within_month_exact_edges"):
        stats.setdefault(key, 0)
    stats.update({"all_groups": len(groups), "active_groups": len(active),
                  "bridge_merges": stats["retained_only_active_groups"] - len(active),
                  "cross_month_groups_all": sum(g["months_all"] == "2020-02|2020-03" for g in groups),
                  "cross_month_groups_retained": sum(g["months_retained"] == "2020-02|2020-03" for g in groups),
                  "retained_samples_in_cross_month_groups": sum(g["retained_samples"] for g in groups if g["months_retained"] == "2020-02|2020-03"),
                  "multi_label_groups_all": sum(g["family_count_all"] > 1 for g in groups),
                  "multi_label_groups_retained": sum(g["family_count_retained"] > 1 for g in groups),
                  "size_distribution_retained": dict(sorted(Counter(int(s) for s in sizes).items())),
                  "size_quantiles_retained": {str(q): float(np.quantile(sizes, q)) for q in (0, .5, .9, .95, .99, 1)},
                  "largest_retained_group": int(sizes.max()), "largest_fraction": float(sizes.max()/sizes.sum()),
                  "size_concentration_count": float(sizes.sum()**2/np.square(sizes).sum())})
    require(sum(g["retained_samples"] for g in groups) == 8131 and len(retained_ids) == 8131, "Changed retained selection")
    require(all(sha(ROOT / p) == value for p, value in sources.items()), "Inputs changed during audit")
    output = ROOT / "data/derived/balanced_phase1_audit" / datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    output.mkdir(parents=True, exist_ok=False)
    (output / "scope_snapshot.md").write_bytes((ROOT / "docs/balanced_research_scope.md").read_bytes())
    for name, records in (("members", members), ("groups", groups), ("class_support", support), ("source_consistency", consistency)):
        save_csv(output / (name + ".csv"), records, list(records[0]))
    save_csv(output / "edges.csv", edges, ["sha_a", "sha_b", "cross_month", "content_equal", "tlsh_distance", "both_retained"])
    save_csv(output / "null_allocation_witness.csv", assignment, ["month", "group_id", "side"])
    save_csv(output / "null_witness_support.csv", witness, ["month", "family", "side_0_samples", "side_1_samples", "equal_budget"])
    result = {"status": "PHASE1_COUNTS_AUDIT_COMPLETE_ESTIMATORS_NOT_APPROVED", "graph_threshold": 10,
              "primary_retention_threshold": 0, "minimum_samples_per_month": 20, "minimum_joint_groups_per_month": 5,
              "original_samples": len(rows), "retained_samples": len(kept), "excluded_bridge_vertices": len(rows)-len(kept),
              "exhaustive_pairs": len(rows)*(len(rows)-1)//2, "grouping": stats,
              "candidate_S_star": selected_support,
              "failed_support_classes": [r["family"] for r in support if not r["candidate_S_star"]],
              "support_samples": {m: sum(r[m+"_samples"] for r in support if r["candidate_S_star"]) for m in MONTHS},
              "joint_groups_per_month": {m: len({r["group_id"] for r in members if r["retained"] and r["month"] == m}) for m in MONTHS},
              "local_partitions_reproduced": True, "sample_order_and_class_mapping_aligned": True,
              "null_capacity_status": "Per-family exact binary allocation capacity; one shared greedy witness, not an approved null estimator",
              "witness_min_equal_budget": min((r["equal_budget"] for r in witness), default=None),
              "witness_capacity_by_month": {m: {str(n): sum(r["equal_budget"] >= n for r in witness if r["month"] == m) for n in (1,10,20,64,128)} for m in MONTHS},
              "feature_extraction_performed": False, "MMD_computed": False, "U_estimator_executed": False,
              "new_weights_computed": False, "future_test_accessed": False, "primary_dataset_changed": False,
              "source_consistency": consistency, "source_sha256": sources,
              "numpy_version": np.__version__, "py_tlsh_version": version("py-tlsh"),
              "limitations": ["Operational/transitive components do not establish independent statistical units.",
                              "No outside-February/March bridge nodes; exact/TLSH audit uses saved disarmed-content inventory, not rehashed binaries.",
                              "Per-family binary capacity is not a simultaneous partition guarantee; greedy witness is not optimal.",
                              "Count feasibility is not adequate MMD power, null exchangeability or reliable uncertainty coverage.",
                              "No estimator, coefficients or fallback are selected."]}
    save_json(output / "report.json", result)
    save_json(output / "checksums.json", {p.name: sha(p) for p in sorted(output.iterdir())})
    print(json.dumps({k: result[k] for k in ("status", "grouping", "candidate_S_star", "support_samples", "joint_groups_per_month", "witness_capacity_by_month")}, indent=2))
    print("Report:", output / "report.json")


if __name__ == "__main__":
    main()
