import itertools
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.balanced_support_audit import (
    MONTHS, build_joint_graph, class_support, equal_side_capacity, allocation_witness,
)
from scripts.run_balanced_support_audit import check_prediction_ids, preprocessing_identity
from scripts.february_group_audit import save_csv


def row(i, m="02", family="a", content=None, null=False):
    return {"sha": f"{i:064x}", "split": "validation", "timestamp": f"2020-{m}-01 00:00:00+00:00",
            "family": family, "disarmed_content_sha256": f"{i if content is None else content:064x}",
            "tlsh": "TNULL" if null else f"T1{i:070X}", "tlsh_status": "TNULL" if null else "VALID"}


def distance(pairs):
    return lambda a, b: pairs.get(tuple(sorted((int(a[2:], 16), int(b[2:], 16)))), 100)


class TestBalancedSupportAudit(unittest.TestCase):
    def test_cross_month_excluded_bridge_merges_local_groups(self):
        rows = [row(1), row(2), row(3, "03", "b"), row(4, "03")]
        kept = {rows[i]["sha"] for i in (0, 1, 3)}
        members, groups, edges, stats = build_joint_graph(rows, kept, distance({(1, 3): 10, (2, 3): 9, (3, 4): 11}))
        self.assertEqual(len(groups), 2)
        self.assertEqual(stats["local_active_groups"], {"2020-02": 2, "2020-03": 1})
        self.assertEqual(stats["retained_only_active_groups"], 3)
        self.assertEqual(stats["cross_month_edges"], 2)
        self.assertEqual(sum(g["retained_samples"] for g in groups), 3)
        support = class_support(members, ["a", "b"])
        self.assertEqual(support[0]["2020-02_groups_joint"], 1)
        self.assertEqual(support[0]["2020-02_groups_local"], 2)
        self.assertEqual(support[1]["2020-03_samples"], 0)
        self.assertTrue(all(e["tlsh_distance"] <= 10 for e in edges))

    def test_exact_tnull_and_mixed_labels_are_not_split(self):
        rows = [row(1, null=True, content=10), row(2, "03", "b", content=10, null=True)]
        kept = {r["sha"] for r in rows}
        members, groups, edges, stats = build_joint_graph(rows, kept, distance({}))
        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0]["family_count_retained"], 2)
        self.assertEqual(stats["cross_month_exact_edges"], 1)
        self.assertIsNone(edges[0]["tlsh_distance"])
        self.assertEqual(groups[0]["months_retained"], "2020-02|2020-03")

    def test_graph_is_order_and_label_invariant(self):
        rows = [row(1), row(2, "03"), row(3, "03", "b")]
        kept = {r["sha"] for r in rows}
        first = build_joint_graph(rows, kept, distance({(1, 2): 10}))[0]
        other = build_joint_graph(list(reversed(rows)), kept, distance({(1, 2): 10}))[0]
        self.assertEqual(first, other)
        rows[0]["family"] = "different"
        changed = build_joint_graph(rows, kept, distance({(1, 2): 10}))[0]
        self.assertEqual([(r["sha"], r["group_id"]) for r in first], [(r["sha"], r["group_id"]) for r in changed])

    def test_future_duplicate_or_unknown_retained_rejected(self):
        cases = [([row(1), row(2, "04")], {row(1)["sha"]}),
                 ([row(1), row(1), row(2, "03")], {row(1)["sha"]}),
                 ([row(1), row(2, "03")], {row(3)["sha"]}),
                 ([row(1)], {row(1)["sha"]})]
        for rows, kept in cases:
            with self.assertRaises(ValueError):
                build_joint_graph(rows, kept, distance({}))

    def test_support_thresholds_and_absent_classes(self):
        members = []
        for m in MONTHS:
            for i in range(20):
                members.append({"sha": m + str(i), "month": m, "family": "a", "retained": True,
                                "group_id": "shared" + str(i // 4), "local_group_id": m + str(i // 4)})
        s = class_support(members, ["a", "absent"])
        self.assertTrue(s[0]["candidate_S_star"])
        self.assertEqual(s[0]["2020-02_groups_joint"], 5)
        self.assertEqual(s[0]["2020-02_cross_month_samples"], 20)
        self.assertEqual(s[0]["2020-02_equal_side_capacity"], 8)
        self.assertFalse(s[1]["candidate_S_star"])
        members[-1]["retained"] = False
        self.assertFalse(class_support(members, ["a"])[0]["candidate_S_star"])

    def test_subset_capacity_matches_brute_force_and_dominant_group(self):
        for sizes in ([], [20], [18, 1, 1], [4]*5, [1, 2, 7, 3], [10, 10, 10]):
            expected = max(min(sum(s for s, on in zip(sizes, bits) if on), sum(sizes)-sum(s for s, on in zip(sizes, bits) if on))
                           for bits in itertools.product((0, 1), repeat=len(sizes)))
            self.assertEqual(equal_side_capacity(sizes), expected)
        with self.assertRaises(ValueError):
            equal_side_capacity([-1])

    def test_joint_allocation_witness_preserves_mixed_group(self):
        members = []
        for m in MONTHS:
            for g in range(5):
                for c in ("a", "b"):
                    members.append({"sha": m+str(g)+c, "month": m, "family": c, "group_id": str(g), "retained": True})
        assignment, summary = allocation_witness(members, ["a", "b"])
        self.assertEqual(len(assignment), 10)
        self.assertEqual(len({(r["month"], r["group_id"]) for r in assignment}), 10)
        self.assertEqual(assignment, allocation_witness(list(reversed(members)), ["a", "b"])[0])
        self.assertTrue(all(r["side_0_samples"] + r["side_1_samples"] == 5 for r in summary))
        self.assertTrue(all(r["equal_budget"] == 2 for r in summary))

    def test_npz_identity_join_accepts_permutation_rejects_bad_mapping(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "predictions.npz"
            kept = [row(1), row(2, "03", "b")]
            np.savez(path, sample_identifiers=[kept[1]["sha"], kept[0]["sha"]], labels=[1, 0], class_names=["a", "b"])
            ids = check_prediction_ids(path, kept, ["a", "b"], {"a": 0, "b": 1})
            self.assertEqual(ids[0], kept[1]["sha"])
            with self.assertRaises(ValueError):
                check_prediction_ids(path, kept, ["b", "a"], {"a": 0, "b": 1})
            np.savez(path, sample_identifiers=[kept[0]["sha"]]*2, labels=[0, 0], class_names=["a", "b"])
            with self.assertRaises(ValueError):
                check_prediction_ids(path, kept, ["a", "b"], {"a": 0, "b": 1})

    def test_config_identity_matches_loader_semantics(self):
        import hashlib
        import json
        original = {"state_bits": 6, "image_size": [64, 64]}
        reordered = {"image_size": [64, 64], "state_bits": 6}
        expected = hashlib.sha256(json.dumps(original, sort_keys=True).encode("utf-8")).hexdigest()
        self.assertEqual(preprocessing_identity(reordered), expected)
        self.assertNotEqual(preprocessing_identity(original), preprocessing_identity(original | {"state_bits": 5}))
        self.assertNotEqual(expected, hashlib.sha256(json.dumps(original, indent=2).encode()).hexdigest())

    def test_graph_support_artifact_roundtrip(self):
        import csv
        rows = [row(1), row(2, "03")]
        members = build_joint_graph(rows, {r["sha"] for r in rows}, distance({}))[0]
        support = class_support(members, ["a"])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "support.csv"
            save_csv(path, support, list(support[0]))
            with path.open() as stream:
                saved = list(csv.DictReader(stream))
            self.assertEqual(saved[0]["2020-02_samples"], "1")
            self.assertEqual(saved[0]["candidate_S_star"], "False")
            self.assertIn("groups<5", saved[0]["failure_reasons"])


if __name__ == "__main__":
    unittest.main()
