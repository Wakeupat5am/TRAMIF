"""Verify a completed G2 run; read only historical allowlisted inputs/artifacts.

Reconstruct connected components by graph traversal, independently of UnionFind.
Do not repeat exhaustive TLSH search or read any model/prediction/image files.
"""
import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from scripts.audit_historical_episodes import ROOT, Inputs, EPISODES, stage, month, time_key


def check(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def verify(run):
    run = run.resolve()
    check(run.is_relative_to((ROOT / "data/derived/historical_episode_feasibility").resolve()), "Wrong run root")
    hashes = json.loads((run / "checksums.json").read_text(encoding="utf-8"))
    for name, expected in hashes.items():
        path = (run / name).resolve()
        check(path.is_relative_to(run), "Artifact path escape")
        check(sha(path) == expected, "Artifact checksum: " + name)
    report = json.loads((run / "report.json").read_text(encoding="utf-8"))
    inputs = Inputs()
    for name, expected in report["source_hashes"].items():
        inputs.read(name, expected)
    vertices = read_csv(run / "pair_vertices.csv")
    check(all("2019-08" <= month(r) <= "2020-02" for r in vertices), "Vertex date outside episode union")
    check(vertices == sorted(vertices, key=time_key), "Vertex order")
    by_id = {r["sha"]: r for r in vertices}
    check(len(by_id) == len(vertices), "Vertex uniqueness")
    edges = read_csv(run / "edges_le20.csv")
    checks = {}
    for episode, spec in EPISODES.items():
        folder = run / episode
        sr = report["episodes"][episode]
        mapping = json.loads((folder / "class_mapping.json").read_text())
        check(mapping == {c: i for i, c in enumerate(sr["families"])}, "Episode class order")
        decisions = read_csv(folder / "decisions.csv")
        expected_ids = {r["sha"] for r in vertices if r["family"] in mapping and stage(r, spec)}
        for threshold in (0, 10, 20):
            ds = [r for r in decisions if int(r["threshold"]) == threshold]
            check(len(ds) == len(expected_ids) and {r["sha"] for r in ds} == expected_ids, "Decision coverage")
            for s in ("train", "A", "B", "C"):
                kept = [r for r in ds if r["stage"] == s and r["decision"] == "KEEP"]
                check(len(kept) == sr["threshold_counts"][str(threshold)][s]["retained"], "Retention count")
                if threshold == 0:
                    check(read_csv(folder / (s + "_keep.csv")) == kept, "Retained export differs")
                    alignment = json.loads((folder / (s + "_view_alignment.json")).read_text())
                    check(alignment["canonical_ids"] == [r["sha"] for r in kept], "Canonical view order")
            for r in ds:
                check(stage(by_id[r["sha"]], spec) == r["stage"], "Stage boundary")
                if r["reference_sha"]:
                    check(time_key(by_id[r["reference_sha"]]) < time_key(by_id[r["sha"]]), "Chronology")
        for threshold in (0, 10, 20):
            for scope in ("train", "AB", "whole_episode"):
                ms = read_csv(folder / f"{scope}_{threshold:02d}_members.csv")
                nodes = {r["sha"] for r in ms}
                allowed = {s for s in expected_ids if
                           scope == "whole_episode" or
                           (scope == "AB" and stage(by_id[s], spec) in ("A", "B")) or
                           (scope == "train" and stage(by_id[s], spec) == "train")}
                check(nodes == allowed, "Graph scope vertices")
                adj = defaultdict(set)
                for edge in edges:
                    a = vertices[int(edge["earlier_index"])]["sha"]
                    b = vertices[int(edge["later_index"])]["sha"]
                    match = edge["content_equal"] == "True" or (edge["tlsh_distance"] and int(edge["tlsh_distance"]) <= threshold)
                    if a in nodes and b in nodes and match:
                        adj[a].add(b)
                        adj[b].add(a)
                saved = {r["sha"]: r["group_id"] for r in ms}
                unseen, group_labels = set(nodes), set()
                while unseen:
                    root = next(iter(unseen))
                    pending, component = [root], {root}
                    unseen.remove(root)
                    while pending:
                        for other in adj[pending.pop()]:
                            if other in unseen:
                                unseen.remove(other)
                                component.add(other)
                                pending.append(other)
                    labels = {saved[s] for s in component}
                    check(len(labels) == 1 and not labels & group_labels, "Component partition mismatch")
                    group_labels.update(labels)
        ab = read_csv(folder / "AB_10_members.csv")
        support = read_csv(folder / "support.csv")
        selected = []
        for r in support:
            eligible = True
            for s in ("A", "B"):
                rs = [x for x in ab if x["retained"] == "True" and x["stage"] == s and x["family"] == r["family"]]
                groups = {x["group_id"] for x in rs}
                check(len(rs) == int(r[s+"_samples"]) and len(groups) == int(r[s+"_groups"]), "Support mismatch")
                eligible &= len(rs) >= 20 and len(groups) >= 5
            check(eligible == (r["eligible_S"] == "True"), "Support eligibility mismatch")
            if eligible:
                selected.append(r["family"])
        check(selected == sr["S"], "Signal support changed")
        checks[episode] = {"status": "PASS", "graph_partitions_reconstructed": 9,
                           "canonical_manifest_alignment": "PASS; not pixel verification",
                           "K": len(mapping), "S": len(selected)}
    return {"status": "PASS", "artifact_checksums": len(hashes), "source_checksums": len(inputs.hashes),
            "episodes": checks, "run": str(run), "run_report_sha256": sha(run / "report.json"),
            "verifier_source_sha256": sha(Path(__file__)), "no_new_TLSH_search_or_model_processing": True}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.run_dir)
    out = ROOT / "data/derived/historical_episode_feasibility_verification" / datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    out.mkdir(parents=True, exist_ok=False)
    (out / "report.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    print("Verification artifact:", out)


if __name__ == "__main__":
    main()
