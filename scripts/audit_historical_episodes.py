"""G2 only: historical metadata, chronological retention and graph feasibility.

No raw metadata/binaries/images/checkpoints/predictions or model imports.
All data reads use a finite historical allowlist. Legacy modules stay unchanged.
"""
import csv
import hashlib
import io
import json
import subprocess
from collections import Counter, defaultdict
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import tlsh

from scripts.near_duplicate_core import UnionFind, time_key, validate_rows

ROOT = Path(__file__).resolve().parents[1]
POLICY = "data/derived/historical_near_duplicates_v1/20261009_143301_921802"
PINNED = {
    "data/manifests/train_validation_manifest.csv": "757251a8986c8808c304d980ff534c4cb8c557f54fcfe3ff168947b5808e362d",
    "data/manifests/train_validation_zip_index.csv": "8557c07fdb3abc7c9bb9e364f35faf0aa737f11d98ebd553f90cfae2d820b9b4",
    "docs/class_mapping.json": "a38b074ce5b977db8e7139b7539fc8f290cb970b7b4b7279edaad150eef007be",
    "docs/RESEARCH_PROTOCOL_V1.md": "e64664c4c499ef762b4a52c45889f0d05db6741ec5021b249f2072585f816546",
}
EPISODES = {
    "E1": {"train_end": "2019-11", "A": "2019-11", "B": "2019-12", "C": "2020-01"},
    "E2": {"train_end": "2019-12", "A": "2019-12", "B": "2020-01", "C": "2020-02"},
}
DATA_ALLOWLIST = set(PINNED) | {
    "docs/train_family_cohort.csv", "docs/near_duplicate_protocol.md",
    f"{POLICY}/config.json", f"{POLICY}/policy_snapshot.md",
} | {
    f"data/derived/{split}_tlsh_v1/{name}"
    for split in ("train", "validation")
    for name in (f"{split}_tlsh.csv", "config.json", "summary.json")
} | {
    f"data/derived/{view}_dataset_v1/config.json"
    for view in ("sbsmi", "raw_byte", "entropy")
}
SOURCE_ALLOWLIST = {
    "scripts/audit_historical_episodes.py", "tests/test_historical_episode_audit.py",
    "scripts/near_duplicate_core.py", "scripts/build_train_val_manifest.py",
    "scripts/inspect_metadata.py", "scripts/train_sbsmi.py",
    "scripts/train_float_view.py", "scripts/historical_policy_dataset.py",
    "scripts/sbsmi_dataset.py", "scripts/float_image_dataset.py",
    "models/malsbslcnet.py", "models/compact_view_net.py",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def month(row):
    return time_key(row)[0].strftime("%Y-%m")


def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8")


def save_csv(path, rows, fields=None):
    fields = fields or (list(rows[0]) if rows else ["empty"])
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


class Inputs:
    def __init__(self, root=ROOT):
        self.root = root
        self.hashes = {}

    def read(self, relative, expected=None):
        require(relative in DATA_ALLOWLIST | SOURCE_ALLOWLIST, "Path outside G2 allowlist")
        path = (self.root / relative).resolve()
        require(path.is_relative_to(self.root.resolve()), "Path escapes workspace")
        data = path.read_bytes()
        actual = digest(data)
        expected = expected or PINNED.get(relative)
        require(expected is None or actual == expected, "Input checksum mismatch: " + relative)
        self.hashes[relative] = actual
        return data

    def json(self, relative, expected=None):
        return json.loads(self.read(relative, expected))

    def csv(self, relative, expected=None):
        return list(csv.DictReader(io.StringIO(self.read(relative, expected).decode("utf-8-sig"))))


def select_families(rows, train_end, minimum=50):
    counts, months = defaultdict(set), defaultdict(set)
    for row in rows:
        if "2019-08" <= month(row) < train_end:
            counts[row["family"]].add(row["sha"])
            months[row["family"]].add(month(row))
    candidates = sorted({r["family"] for r in rows})
    records = [{"family": c, "prefix_unique_sha": len(counts[c]),
                "prefix_months": len(months[c]),
                "eligible": len(counts[c]) >= minimum and len(months[c]) >= 3}
               for c in candidates]
    return [r["family"] for r in records if r["eligible"]], records


def stage(row, spec):
    m = month(row)
    if "2019-08" <= m < spec["train_end"]:
        return "train"
    return next((s for s in ("A", "B", "C") if m == spec[s]), None)


def load_inputs(inputs):
    inputs.read("docs/RESEARCH_PROTOCOL_V1.md")
    policy = inputs.json(f"{POLICY}/config.json")
    require(policy["primary_threshold"] == 0 and policy["sensitivity_thresholds"] == [10, 20],
            "Policy thresholds changed")
    inputs.read(f"{POLICY}/policy_snapshot.md", policy["policy_sha256"])
    inputs.read("docs/near_duplicate_protocol.md", policy["policy_sha256"])
    manifest = inputs.csv("data/manifests/train_validation_manifest.csv", policy["manifest_sha256"])
    index = inputs.csv("data/manifests/train_validation_zip_index.csv")
    mapping = inputs.json("docs/class_mapping.json", policy["class_mapping_sha256"])["class_to_idx"]
    require(sorted(mapping.values()) == list(range(51)), "Final mapping changed")
    for rows in (manifest, index):
        require(len(rows) == 26324 and len({r["sha"] for r in rows}) == len(rows), "Manifest coverage")
        for r in rows:
            m = month(r)
            require("2019-08" <= m < "2020-04", "Nonhistorical manifest row")
            require(r["family"] in mapping, "Unknown manifest family")
            require(r["split"] == ("train" if m < "2020-02" else "validation"), "Split/date mismatch")
    by_id = {r["sha"]: r for r in manifest}
    index_by_id = {r["sha"]: r for r in index}
    require(set(by_id) == set(index_by_id), "Index IDs differ")
    inventory = []
    for split, count in (("train", 18061), ("validation", 8263)):
        base = f"data/derived/{split}_tlsh_v1"
        config = inputs.json(f"{base}/config.json", policy["inventory_sources"][f"{base}/config.json"])
        summary = inputs.json(f"{base}/summary.json", policy["inventory_sources"][f"{base}/summary.json"])
        require(config["py_tlsh"] == version("py-tlsh") == "5.0.0", "TLSH version mismatch")
        config_id = digest(json.dumps(config, sort_keys=True).encode())
        require(summary["config_sha256"] == config_id and summary["errors"] == 0, "Inventory config/errors")
        require(summary[f"complete_{split}_inventory"] is True, "Incomplete inventory")
        require(config["zip_index_sha256"] == PINNED["data/manifests/train_validation_zip_index.csv"],
                "Index provenance mismatch")
        path = f"{base}/{split}_tlsh.csv"
        require(summary["csv_sha256"] == policy["inventory_sources"][path], "Inventory hash bindings")
        rows = validate_rows(inputs.csv(path, summary["csv_sha256"]), split)
        require(len(rows) == summary["processed_samples"] == count, "Inventory count mismatch")
        require({r["sha"] for r in rows} == {r["sha"] for r in manifest if r["split"] == split},
                "Inventory ID coverage mismatch")
        for r in rows:
            for key in ("sha", "split", "timestamp", "family"):
                require(r[key] == by_id[r["sha"]][key] == index_by_id[r["sha"]][key],
                        "Identity mismatch")
            require(int(r["file_size"]) == int(index_by_id[r["sha"]]["file_size"]), "Size mismatch")
            require(r["crc32"] == index_by_id[r["sha"]]["zip_crc32"], "CRC provenance mismatch")
            require(r["config_sha256"] == config_id, "Row config mismatch")
        require(sum(r["tlsh_status"] == "VALID" for r in rows) == summary["valid_tlsh_hashes"],
                "TLSH valid count")
        require(sum(r["tlsh_status"] == "TNULL" for r in rows) == summary["tnull_samples"],
                "TLSH null count")
        inventory.extend(rows)
    require(len({r["sha"] for r in inventory}) == len(inventory), "Cross-inventory duplicate ID")
    train = [r for r in inventory if r["split"] == "train"]
    cohort = inputs.csv("docs/train_family_cohort.csv")
    require({r["family"] for r in cohort} == set(mapping), "Cohort differs from mapping")
    for c in cohort:
        rs = [r for r in train if r["family"] == c["family"]]
        require(len({r["sha"] for r in rs}) == int(c["unique_train_sha"])
                and len({month(r) for r in rs}) == int(c["train_months"]), "Cohort counts differ")
        require(len(rs) >= 50 and len({month(r) for r in rs}) >= 3, "Cohort eligibility")
    views = {}
    for view in ("sbsmi", "raw_byte", "entropy"):
        cfg = inputs.json(f"data/derived/{view}_dataset_v1/config.json")
        for key, path in (("manifest_sha256", "data/manifests/train_validation_manifest.csv"),
                          ("class_mapping_sha256", "docs/class_mapping.json"),
                          ("zip_index_sha256", "data/manifests/train_validation_zip_index.csv")):
            require(cfg[key] == PINNED[path], "View provenance mismatch")
        views[view] = "same canonical IDs/order via shared manifest; image files not opened/rehashed"
    for path in sorted(SOURCE_ALLOWLIST):
        inputs.read(path)
    return sorted(inventory, key=time_key), mapping, views


def pair_edges(rows, distance_function=None, progress=None):
    """All unordered pairs, saved edges <=20 or exact; no approximate search."""
    objects = []
    for row in rows:
        obj = None
        if distance_function is None and row["tlsh_status"] == "VALID":
            obj = tlsh.Tlsh()
            obj.fromTlshStr(row["tlsh"])
        objects.append(obj)
    contents = [r["disarmed_content_sha256"] for r in rows]
    edges = []
    for i in range(len(rows)):
        for j in range(i):
            exact = contents[i] == contents[j]
            d = None
            if rows[i]["tlsh_status"] == rows[j]["tlsh_status"] == "VALID":
                d = (distance_function(rows[j]["tlsh"], rows[i]["tlsh"])
                     if distance_function else objects[i].diff(objects[j]))
                if d < 0:
                    raise ValueError("Negative TLSH distance")
            if exact or (d is not None and d <= 20):
                edges.append((j, i, d, exact))
        if progress and (i + 1) % 1000 == 0:
            progress(i + 1, len(rows))
    return edges


def retained_decisions(rows, edges, spec, threshold):
    """Train removes exact copies only; A/B/C refer to ALL earlier episode rows."""
    prior = defaultdict(list)
    for j, i, d, exact in edges:
        if exact or (d is not None and d <= threshold):
            prior[i].append((j, d, exact))
    decisions = []
    for i, row in enumerate(rows):
        s = stage(row, spec)
        require(s is not None, "Row outside episode")
        candidates = sorted(prior[i])
        exacts = [x for x in candidates if x[2]]
        witness = (exacts[0] if exacts else (candidates[0] if candidates and s != "train" else None))
        ref, d, exact = witness if witness else (None, None, False)
        if ref is not None:
            require(time_key(rows[ref]) < time_key(row), "Non-earlier witness")
        decisions.append({"sha": row["sha"], "timestamp": row["timestamp"], "family": row["family"],
                          "stage": s, "threshold": threshold,
                          "decision": "KEEP" if witness is None else "EXCLUDE",
                          "reason": ("exact_content" if exact else "tlsh") if witness else "retained",
                          "reference_sha": rows[ref]["sha"] if ref is not None else "",
                          "reference_stage": stage(rows[ref], spec) if ref is not None else "",
                          "tlsh_distance": d})
    return decisions


def graph_members(rows, edges, nodes, kept, spec, threshold):
    """Induced graph: nodes outside scope NEVER serve as bridges."""
    nodes = set(nodes)
    uf = UnionFind(len(rows))
    for j, i, d, exact in edges:
        if j in nodes and i in nodes and (exact or (d is not None and d <= threshold)):
            uf.union(j, i)
    members = []
    for i in sorted(nodes):
        r = rows[i]
        members.append({"sha": r["sha"], "family": r["family"], "timestamp": r["timestamp"],
                        "stage": stage(r, spec), "content_sha256": r["disarmed_content_sha256"],
                        "retained": r["sha"] in kept, "group_id": rows[uf.find(i)]["sha"]})
    return members


def graph_summary(members):
    groups = defaultdict(list)
    for row in members:
        groups[row["group_id"]].append(row)
    kept_groups = [[r for r in rs if r["retained"]] for rs in groups.values()]
    kept_groups = [rs for rs in kept_groups if rs]
    sizes = sorted(map(len, kept_groups))
    return {
        "all_groups": len(groups), "active_groups": len(sizes),
        "largest_retained_group": max(sizes, default=0),
        "retained_size_histogram": dict(sorted(Counter(sizes).items())),
        "multi_label_active_groups": sum(len({r["family"] for r in rs}) > 1 for rs in kept_groups),
        "cross_stage_active_groups": sum(len({r["stage"] for r in rs}) > 1 for rs in kept_groups),
        "samples_in_cross_stage_groups": sum(len(rs) for rs in kept_groups if len({r["stage"] for r in rs}) > 1),
        "size_concentration_index_not_independent_count":
            sum(sizes)**2 / sum(s*s for s in sizes) if sizes else None,
    }


def support_table(members, classes, outcomes):
    table = []
    for c in classes:
        rec, errors = {"family": c}, []
        for s in ("A", "B"):
            rs = [r for r in members if r["retained"] and r["stage"] == s and r["family"] == c]
            sizes = Counter(r["group_id"] for r in rs)
            rec[s + "_samples"] = len(rs)
            rec[s + "_groups"] = len(sizes)
            rec[s + "_largest_class_group"] = max(sizes.values(), default=0)
            rec[s + "_largest_class_group_fraction"] = max(sizes.values(), default=0)/len(rs) if rs else None
            if len(rs) < 20:
                errors.append(s + ":samples<20")
            if len(sizes) < 5:
                errors.append(s + ":groups<5")
        rec["eligible_S"] = not errors
        rec["failure_reasons"] = "|".join(errors)
        rec["C_samples"] = outcomes.get(c, 0)
        rec["S_eval"] = not errors and rec["C_samples"] > 0
        table.append(rec)
    return table


def audit_episode(inventory, edges, spec):
    families, eligibility = select_families(inventory, spec["train_end"])
    indices = [i for i, r in enumerate(inventory) if r["family"] in families and stage(r, spec)]
    remap = {old: new for new, old in enumerate(indices)}
    rows = [inventory[i] for i in indices]
    ee = [(remap[j], remap[i], d, exact) for j, i, d, exact in edges if j in remap and i in remap]
    all_decisions, thresholds, graphs = [], {}, {}
    for t in (0, 10, 20):
        ds = retained_decisions(rows, ee, spec, t)
        all_decisions += ds
        kept = {r["sha"] for r in ds if r["decision"] == "KEEP"}
        if t == 0:
            primary_kept = kept
            primary_decisions = ds
        counts = {}
        for s in ("train", "A", "B", "C"):
            rs = [r for r in rows if stage(r, spec) == s]
            kr = [r for r in rs if r["sha"] in kept]
            counts[s] = {"original": len(rs), "retained": len(kr), "excluded": len(rs)-len(kr),
                         "unique_sha": len({r["sha"] for r in kr}),
                         "unique_content": len({r["disarmed_content_sha256"] for r in kr}),
                         "families_retained": len({r["family"] for r in kr}),
                         "tnull_original": sum(r["tlsh_status"] == "TNULL" for r in rs),
                         "min_timestamp": min((r["timestamp"] for r in rs), default=None),
                         "max_timestamp": max((r["timestamp"] for r in rs), default=None)}
        thresholds[str(t)] = counts
    # Shared SIGNAL graph uses only A/B, including excluded bridge nodes.
    ab_nodes = [i for i, r in enumerate(rows) if stage(r, spec) in ("A", "B")]
    ab = graph_members(rows, ee, ab_nodes, primary_kept, spec, 10)
    outcomes = Counter(r["family"] for r in primary_decisions if r["stage"] == "C" and r["decision"] == "KEEP")
    support = support_table(ab, families, outcomes)
    for t in (0, 10, 20):
        for scope in ("train", "AB", "whole_episode"):
            nodes = ([i for i, r in enumerate(rows) if stage(r, spec) == "train"] if scope == "train"
                     else ab_nodes if scope == "AB" else range(len(rows)))
            ms = graph_members(rows, ee, nodes, primary_kept, spec, t)
            graphs[f"{scope}_{t:02d}"] = ms
    # Independently verify every kept row against all prohibited earlier edges.
    stages = [stage(r, spec) for r in rows]
    ds_by_t = {t: {r["sha"]: r for r in all_decisions if r["threshold"] == t} for t in (0, 10, 20)}
    violations = []
    cross_edges = Counter()
    retained_cross_edges = Counter()
    for j, i, d, exact in ee:
        for t in (0, 10, 20):
            matches = exact or (d is not None and d <= t)
            if matches and stages[j] != stages[i]:
                cross_edges[f"{t}:{stages[j]}->{stages[i]}"] += 1
                if rows[j]["sha"] in primary_kept and rows[i]["sha"] in primary_kept:
                    retained_cross_edges[f"{t}:{stages[j]}->{stages[i]}"] += 1
            forbidden = exact or (matches and stages[i] != "train")
            if forbidden and ds_by_t[t][rows[i]["sha"]]["decision"] == "KEEP":
                violations.append((t, rows[i]["sha"]))
    require(not violations, "Retained row has prohibited earlier match")
    for i in range(len(rows)):
        decisions = [ds_by_t[t][rows[i]["sha"]]["decision"] == "KEEP" for t in (0, 10, 20)]
        require(decisions[0] >= decisions[1] >= decisions[2], "Non-nested retention")
    original_contents = Counter(r["disarmed_content_sha256"] for r in rows)
    stage_ids = {s: {r["sha"] for r in rows if stage(r, spec) == s} for s in ("train", "A", "B", "C")}
    require(sum(map(len, stage_ids.values())) == len(rows), "Stage ID overlap")
    for s in ("train", "A", "B", "C"):
        require(len(stage_ids[s]) == sum(stage(r, spec) == s for r in rows), "Repeated ID within stage")
    supported = [r["family"] for r in support if r["eligible_S"]]
    criteria = {"K_at_least_2": len(families) >= 2, "S_at_least_2": len(supported) >= 2,
                "nonempty_train_A_B_C": all(thresholds["0"][s]["retained"] > 0 for s in stage_ids),
                "chronological_decisions_and_coverage": True, "nested_retention": True,
                "no_prohibited_retained_edges": not violations,
                "no_C_vertices_in_signal_graph": all(r["stage"] in ("A", "B") for r in ab)}
    summary = {
        "computational_feasibility": "PASS" if all(criteria.values()) else "FAIL",
        "criteria": criteria, "families": families, "K": len(families), "S": supported,
        "S_count": len(supported), "S_eval_count": sum(r["S_eval"] for r in support),
        "threshold_counts": thresholds, "duplicate_content_groups_original": sum(n > 1 for n in original_contents.values()),
        "cross_stage_edges_original": dict(sorted(cross_edges.items())),
        "cross_stage_edges_primary_retained": dict(sorted(retained_cross_edges.items())),
        "graphs": {key: graph_summary(ms) for key, ms in graphs.items()},
        "signals_support_uses_primary_retention_and_AB_graph10": True,
        "support_coverage": {s: sum(r[s+"_samples"] for r in support if r["eligible_S"]) for s in ("A", "B")},
        "training_ready": False, "scientific_validity": "UNKNOWN",
        "label_availability": "UNKNOWN; no historical label-release timestamps in allowlisted inputs",
        "notes": ["K selected using prefix only; final-51 candidate envelope justified by monotonic eligibility.",
                  "Groups at 10/20 are operational, not statistically independent.",
                  "Whole-episode graph includes C for leakage diagnosis ONLY; it never selects S.",
                  "Three view configs share canonical source IDs; images not opened or rehashed.",
                  "Legacy training APIs require adapters; no prefix models trained."],
    }
    return summary, rows, eligibility, all_decisions, graphs, support, ee


def main():
    output = ROOT / "data/derived/historical_episode_feasibility" / datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    output.mkdir(parents=True, exist_ok=False)
    inputs = Inputs()
    save_json(output / "allowlist.json", {"data": sorted(DATA_ALLOWLIST), "source": sorted(SOURCE_ALLOWLIST),
                                        "forbidden": ["data/raw/*", "predictions", "checkpoints", "Future Test"],
                                        "purpose": "G2 historical counts/retention/groups only"})
    started = perf_counter()
    try:
        inventory, mapping, views = load_inputs(inputs)
        original_counts = dict(sorted(Counter(month(r) for r in inventory).items()))
        # No March nodes even though the validated inventory includes historical March.
        candidate_sets = {e: set(select_families(inventory, s["train_end"])[0]) for e, s in EPISODES.items()}
        rows = [r for r in inventory if any(r["family"] in candidate_sets[e] and stage(r, s)
                                          for e, s in EPISODES.items())]
        print("Provenance/allowlist: PASS", flush=True)
        print("Prefix eligible families:", {e: len(c) for e, c in candidate_sets.items()}, flush=True)
        print("Pair audit vertices:", len(rows), "| all pairs:", len(rows)*(len(rows)-1)//2, flush=True)
        edges = pair_edges(rows, progress=lambda n, total: print(f"Compared rows: {n}/{total}", flush=True))
        save_csv(output / "pair_vertices.csv", [{"index": i, **{k: r[k] for k in ("sha", "timestamp", "family", "disarmed_content_sha256", "tlsh", "tlsh_status")}} for i, r in enumerate(rows)])
        save_csv(output / "edges_le20.csv", [{"earlier_index": j, "later_index": i, "tlsh_distance": d, "content_equal": exact}
                                           for j, i, d, exact in edges],
                 ["earlier_index", "later_index", "tlsh_distance", "content_equal"])
        results = {}
        for e, spec in EPISODES.items():
            summary, erows, eligibility, decisions, graphs, support, ee = audit_episode(rows, edges, spec)
            # Candidate audit includes all 51 families, even those absent from union pair vertices.
            _, eligibility = select_families(inventory, spec["train_end"])
            folder = output / e
            folder.mkdir()
            save_json(folder / "episode_spec.json", spec)
            save_json(folder / "class_mapping.json", {c: i for i, c in enumerate(summary["families"])})
            save_csv(folder / "eligibility.csv", eligibility)
            save_csv(folder / "decisions.csv", decisions)
            save_csv(folder / "support.csv", support)
            for key, members in graphs.items():
                save_csv(folder / (key + "_members.csv"), members)
            for s in ("train", "A", "B", "C"):
                selected = [r for r in decisions if r["stage"] == s and r["threshold"] == 0 and r["decision"] == "KEEP"]
                save_csv(folder / (s + "_keep.csv"), selected)
                # One canonical ID list, explicitly shared by the three future loaders.
                save_json(folder / (s + "_view_alignment.json"),
                          {"views": ["sbsmi", "raw-byte", "entropy"], "canonical_ids": [r["sha"] for r in selected],
                           "source_manifest_hash": PINNED["data/manifests/train_validation_manifest.csv"],
                           "verification": "source manifest/config identity; NOT image-content verification"})
            save_json(folder / "report.json", summary)
            results[e] = summary
            print(e, summary["computational_feasibility"], "K=", summary["K"], "S=", summary["S_count"], flush=True)
        # Verify inputs unchanged, without opening any extra paths.
        for path, expected in list(inputs.hashes.items()):
            inputs.read(path, expected)
        report = {"scope": "G2 only; no training/inference/Q/D/U/metrics/Future", "date_local": "2026-10-15 Asia/Saigon",
                  "source_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                  "source_hashes": inputs.hashes, "original_historical_month_counts": original_counts,
                  "input_samples": len(inventory), "pair_vertices": len(rows),
                  "pair_count": len(rows)*(len(rows)-1)//2, "edges_le20": len(edges), "views": views,
                  "candidate_envelope_proof": "Prefix unique SHA count/month count cannot exceed Aug-Jan counts. Every prefix-eligible family must be final-eligible under identical >=50/>=3 criteria. Frozen cohort totals match inventories; source selection code inspected, raw metadata not reopened.",
                  "provenance_limit": "Relies on existing frozen historical export and saved content hashes; does not rehash binaries or independently re-extract full raw metadata.",
                  "episodes": results, "elapsed_seconds": perf_counter()-started}
        save_json(output / "report.json", report)
        checks = {p.relative_to(output).as_posix(): digest(p.read_bytes()) for p in sorted(output.rglob("*")) if p.is_file()}
        save_json(output / "checksums.json", checks)
        print("Artifacts:", output, flush=True)
        print("Elapsed seconds:", round(report["elapsed_seconds"], 2), flush=True)
    except Exception as exc:
        save_json(output / "failure.json", {"status": "UNKNOWN", "error": str(exc), "verified_sources": inputs.hashes,
                                            "dependent_processing_stopped": True})
        print("Audit stopped; failure artifact:", output, flush=True)
        raise


if __name__ == "__main__":
    main()
