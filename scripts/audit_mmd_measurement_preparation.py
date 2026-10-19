"""G3-B finite-allowlist metadata audit. Never rebuilds graphs or reads embeddings."""
import csv
from collections import Counter, defaultdict
from datetime import datetime, timezone
import io
import json
from pathlib import Path
import subprocess

from scripts.mmd_measurement_preparation import (
    CAP, SEED, STD_MASK, NAMESPACE, CASES, VIEWS, require, encoded, sha256,
    checked_rows, train_reference, reference_pairs, cap_audit, verify_alignment,
)

ROOT = Path(__file__).resolve().parents[1]
H = "data/derived/historical_episode_feasibility/20261014_233013_930021"
F = "data/derived/balanced_phase1_audit/20261013_233825_921873"
PINNED = {
    H+"/checksums.json": "c1cd9cabb29c16c60f88e5355dca0f0e0086fcbaf86578235e94fb55f92c45ba",
    F+"/checksums.json": "c790952dbc7601420e016a19a38cba806db0947092c20325e16fc077932be09e",
    "docs/RESEARCH_PROTOCOL_V1.md": "e64664c4c499ef762b4a52c45889f0d05db6741ec5021b249f2072585f816546",
}
ARTIFACTS = {H+"/report.json", F+"/report.json", F+"/members.csv", F+"/class_support.csv"}
for case in ("E1", "E2"):
    ARTIFACTS.update(H+"/"+case+"/"+f for f in (
        "class_mapping.json", "episode_spec.json", "support.csv", "AB_10_members.csv",
        "train_keep.csv", "A_keep.csv", "B_keep.csv",
        "train_view_alignment.json", "A_view_alignment.json", "B_view_alignment.json"))
SOURCES = {"data/manifests/train_validation_manifest.csv", "docs/class_mapping.json"} | {
    f"data/derived/{v}_dataset_v1/config.json" for v in ("sbsmi", "raw_byte", "entropy")}
CODE = {"scripts/mmd_measurement_preparation.py", "scripts/audit_mmd_measurement_preparation.py",
        "scripts/descriptive_mmd.py", "tests/test_mmd_measurement_preparation.py",
        "tests/test_g3b_historical_metadata.py"}
ALLOWLIST = set(PINNED) | ARTIFACTS | SOURCES | CODE | {
    "docs/RESEARCH_PROTOCOL_V1_AMENDMENT_01_DESCRIPTIVE_DRIFT.md"}


class Inputs:
    def __init__(self, root=ROOT):
        self.root = Path(root).resolve()
        self.hashes = {}

    def read(self, path, expected=None):
        require(path in ALLOWLIST, "Outside finite G3-B allowlist")
        target = (self.root/path).resolve()
        require(target.is_relative_to(self.root), "Path escape")
        blob = target.read_bytes()
        digest = sha256(blob)
        expected = expected or PINNED.get(path)
        require(expected is not None or path in CODE or path.endswith("AMENDMENT_01_DESCRIPTIVE_DRIFT.md"),
                "Missing expected provenance hash")
        require(expected is None or digest == expected, "Checksum mismatch: "+path)
        self.hashes[path] = digest
        return blob

    def json(self, path, expected=None):
        return json.loads(self.read(path, expected))

    def csv(self, path, expected):
        return list(csv.DictReader(io.StringIO(self.read(path, expected).decode("utf-8-sig"))))


def ordered_mapping(mapping):
    require(mapping and sorted(mapping.values()) == list(range(len(mapping))), "Invalid class mapping")
    return sorted(mapping, key=mapping.get)


def retained_members(rows, *, case, families):
    require(len(rows) == len({r["sha"] for r in rows}), "Duplicate membership IDs")
    for r in rows:
        require(r["retained"] in ("True", "False"), "Invalid retained flag")
        require(bool(r["group_id"]), "Missing saved group ID")
    # Validate dates/IDs even for excluded bridges without treating them as selected.
    checked_rows([{k: v for k, v in r.items() if k != "retained"} for r in rows], case, families, {"A", "B"})
    return [r for r in rows if r["retained"] == "True"]


def incidence(rows):
    groups = defaultdict(list)
    for r in rows:
        groups[r["group_id"]].append(r)
    return {"groups": len(groups),
            "cross_month_groups": sum(len({r["stage"] for r in g}) == 2 for g in groups.values()),
            "multi_label_groups": sum(len({r["family"] for r in g}) > 1 for g in groups.values()),
            "largest_group": max((len(g) for g in groups.values()), default=0)}


def load_cases(inputs):
    inputs.read("docs/RESEARCH_PROTOCOL_V1.md")
    inputs.read("docs/RESEARCH_PROTOCOL_V1_AMENDMENT_01_DESCRIPTIVE_DRIFT.md")
    hc, fc = inputs.json(H+"/checksums.json"), inputs.json(F+"/checksums.json")
    hr = inputs.json(H+"/report.json", hc["report.json"])
    fr = inputs.json(F+"/report.json", fc["report.json"])
    require(hc["report.json"] == "8f232f09d7ddcec2e5b71ba68d0627c1a96c2359631c858e67f329c8832f4f1e", "G2 report binding")
    require(fc["report.json"] == "c86639b6eeae08e80cbe83ae858d6ae2e099c7c5f0f3d6a0e7fcbb8033a6668f", "Phase1 report binding")
    expected = hr["source_hashes"]
    manifest = inputs.csv("data/manifests/train_validation_manifest.csv", expected["data/manifests/train_validation_manifest.csv"])
    mapping = inputs.json("docs/class_mapping.json", expected["docs/class_mapping.json"])["class_to_idx"]
    final_families = ordered_mapping(mapping)
    require(len(final_families) == 51 and len(manifest) == 26324, "Final cohort changed")
    canonical = {r["sha"]: r for r in manifest}
    normalized = []
    for r in manifest:
        month = datetime.fromisoformat(r["timestamp"]).astimezone(timezone.utc).strftime("%Y-%m")
        require(r["split"] == ("train" if month < "2020-02" else "validation"), "Split/date mismatch")
        normalized.append(dict(r, stage="train" if r["split"] == "train" else ("A" if month == "2020-02" else "B")))
    checked_rows(normalized, "Final", final_families, {"train", "A", "B"})
    for name in ("sbsmi", "raw_byte", "entropy"):
        path = f"data/derived/{name}_dataset_v1/config.json"
        cfg = inputs.json(path, expected[path])
        require(cfg["manifest_sha256"] == expected["data/manifests/train_validation_manifest.csv"], "View source IDs changed")
        require(cfg["class_mapping_sha256"] == expected["docs/class_mapping.json"], "View class order changed")
        require(fr["source_sha256"][path] == expected[path], "G2/Phase1 view provenance differs")
    cases = {}
    for case in ("E1", "E2"):
        def read(name, csv_file=False):
            rel = case+"/"+name
            return (inputs.csv if csv_file else inputs.json)(H+"/"+rel, hc[rel])
        families = ordered_mapping(read("class_mapping.json"))
        spec = read("episode_spec.json")
        require((spec["train_end"], spec["A"], spec["B"]) == CASES[case], "Episode dates changed")
        support_rows = read("support.csv", True)
        require({r["family"] for r in support_rows} == set(families), "Episode support coverage")
        support = [c for c in families if next(r for r in support_rows if r["family"] == c)["eligible_S"] == "True"]
        stage_rows = {s: read(s+"_keep.csv", True) for s in ("train", "A", "B")}
        for stage, rows in stage_rows.items():
            checked_rows(rows, case, families, {stage})
            require(all(r["threshold"] == "0" for r in rows), "Primary retention changed")
            align = read(stage+"_view_alignment.json")
            require(align["views"] == list(VIEWS) and align["canonical_ids"] == [r["sha"] for r in rows], "G2 canonical order changed")
            require(align["source_manifest_hash"] == expected["data/manifests/train_validation_manifest.csv"], "G2 source binding")
        retained = retained_members(read("AB_10_members.csv", True), case=case, families=families)
        require({r["sha"] for r in retained} == {r["sha"] for s in ("A", "B") for r in stage_rows[s]}, "Membership retention differs")
        cases[case] = dict(families=families, support=support, train=stage_rows["train"], retained=retained,
                           support_rows=support_rows)
    fm = inputs.csv(F+"/members.csv", fc["members.csv"])
    for r in fm:
        require(r["month"] in ("2020-02", "2020-03"), "Nonhistorical member")
        r["stage"] = "A" if r["month"] == "2020-02" else "B"
    retained = retained_members(fm, case="Final", families=final_families)
    support_rows = inputs.csv(F+"/class_support.csv", fc["class_support.csv"])
    require({r["family"] for r in support_rows} == set(final_families), "Final support coverage")
    support = [c for c in final_families if next(r for r in support_rows if r["family"] == c)["candidate_S_star"] == "True"]
    require(fr["graph_threshold"] == 10 and fr["primary_retention_threshold"] == 0, "Phase1 policy changed")
    cases["Final"] = dict(families=final_families, support=support, train=[r for r in normalized if r["stage"] == "train"],
                          retained=retained, support_rows=support_rows)
    for case, bundle in cases.items():
        require(len(bundle["support"]) == {"E1": 17, "E2": 25, "Final": 25}[case], "Frozen S changed")
        require(len(bundle["train"]) == {"E1": 7034, "E2": 9803, "Final": 18061}[case], "TRAIN count changed")
        for r in bundle["train"] + bundle["retained"]:
            require(r["sha"] in canonical and all(r[k] == canonical[r["sha"]][k] for k in ("family", "timestamp")), "Manifest identity mismatch")
        for s in bundle["support_rows"]:
            for stage, month in zip(("A", "B"), CASES[case][1:]):
                rows = [r for r in bundle["retained"] if r["family"] == s["family"] and r["stage"] == stage]
                prefix = stage if case != "Final" else month
                group_key = prefix + ("_groups" if case != "Final" else "_groups_joint")
                require(len(rows) == int(s[prefix+"_samples"]) and len({r["group_id"] for r in rows}) == int(s[group_key]), "Frozen support counts disagree")
    return cases


def audit(inputs):
    bundles = load_cases(inputs)
    results = {}
    for case, b in bundles.items():
        ref = train_reference(b["train"], case=case, families=b["families"])
        pairs = reference_pairs(ref, case=case, families=b["families"])
        signal = [r for r in b["retained"] if r["family"] in b["support"]]
        selected, counts = cap_audit(signal, case=case, support=b["support"])
        # One canonical plan for all views, not evidence of actual embedding arrays.
        verify_alignment(selected, {view: selected for view in VIEWS})
        require([r["sha"] for r in train_reference(list(reversed(b["train"])), case=case, families=b["families"])] == [r["sha"] for r in ref], "Reference order dependence")
        repeated, _ = cap_audit(list(reversed(signal)), case=case, support=b["support"])
        require([r["sha"] for r in repeated] == [r["sha"] for r in selected], "Cap order dependence")
        below = [r for r in counts if r["post_cap_below5"]]
        lost = [r for r in counts if r["lost_at_least5"]]
        summary = dict(case=case, K=len(b["families"]), support=b["support"],
                       train_samples=len(b["train"]), train_reference_samples=len(ref), train_pairs=len(pairs),
                       selected_per_side=sum(r["samples_after"] for r in counts if r["stage"] == "A"),
                       post_cap_below5=[f'{r["family"]}/{r["stage"]}' for r in below],
                       lost_at_least5=[f'{r["family"]}/{r["stage"]}' for r in lost],
                       min_groups_after=min(r["groups_after"] for r in counts),
                       before=incidence(signal), after=incidence(selected),
                       group_support_decision="BLOCKED_OWNER_REVIEW" if below or lost else "PASS_BOTH_PREDICATES",
                       metadata_alignment="PASS_CANONICAL_PLAN_ONLY", actual_embedding_alignment="UNKNOWN")
        results[case] = dict(summary=summary, counts=counts, reference=ref, pairs=pairs, selected=selected,
                             mapping={c: i for i, c in enumerate(b["families"])})
    return results


def save_json(path, value):
    with path.open("x", encoding="utf-8") as f:
        f.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+"\n")


def save_csv(path, rows, fields):
    with path.open("x", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main():
    inputs = Inputs()
    results = audit(inputs)
    for path in sorted(CODE):
        inputs.read(path)
    # Verify immutability of exactly the inputs read; no traversal of data/.
    for path, digest in list(inputs.hashes.items()):
        inputs.read(path, digest)
    run = ROOT/"data/derived/g3b_measurement_validation"/datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    run.mkdir(parents=True, exist_ok=False)
    config = dict(namespace=NAMESPACE, seed=SEED, cap=CAP, mask_std_le=STD_MASK,
                  median="lower weighted median; exact rational family pair masses", cases=CASES,
                  views=VIEWS, observed_mmd=False, actual_embeddings=False, future_access=False)
    save_json(run/"config.json", config)
    for case, value in results.items():
        directory = run/case
        directory.mkdir()
        save_json(directory/"summary.json", value["summary"])
        save_json(directory/"class_mapping.json", value["mapping"])
        save_json(directory/"support.json", value["summary"]["support"])
        save_csv(directory/"train_reference.csv", value["reference"], ["sha", "family", "timestamp", "stage"])
        save_csv(directory/"train_pairs.csv", value["pairs"], ["family", "left", "right"])
        save_csv(directory/"matched_samples.csv", value["selected"], ["sha", "family", "timestamp", "stage", "group_id", "retained"])
        save_csv(directory/"cap_counts.csv", value["counts"], list(value["counts"][0]))
        identities = [[r["sha"], r["family"], r["stage"]] for r in value["selected"]]
        save_json(directory/"view_plan.json", dict(views=VIEWS, shared_manifest="matched_samples.csv",
                  ordered_ids_labels_stage_sha256=sha256(encoded(identities)),
                  reference_order_sha256=sha256(encoded([r["sha"] for r in value["reference"]])),
                  verification="source manifest/config and shared selection only; no image/embedding validation"))
    save_json(run/"report.json", dict(summaries={k: v["summary"] for k, v in results.items()},
              input_source_hashes=inputs.hashes, finite_allowlist=sorted(ALLOWLIST),
              head=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
              status="G3B_METADATA_COMPLETE; REAL_MEASUREMENT_BLOCKED_PENDING_REVIEW"))
    save_json(run/"checksums.json", {p.relative_to(run).as_posix(): sha256(p.read_bytes()) for p in sorted(run.rglob("*")) if p.is_file()})
    for case, v in results.items():
        s = v["summary"]
        print(f'{case}: reference={s["train_reference_samples"]}, per-side={s["selected_per_side"]}, below5={len(s["post_cap_below5"])}, lost>=5={len(s["lost_at_least5"])}, {s["group_support_decision"]}')
    print("Artifacts:", run)
    print("No embeddings/MMD/inference/graph reconstruction. STOP at G3-B.")


if __name__ == "__main__":
    main()
