import csv
import tempfile
import unittest
from pathlib import Path

from scripts.february_group_audit import (
    assess, assign_folds, build_groups, february_rows, save_csv,
)


def row(i, family="a", content=None, month="02", tnull=False):
    return {"sha": f"{i:064x}", "split": "validation",
            "timestamp": f"2020-{month}-01 00:00:00+00:00", "family": family,
            "disarmed_content_sha256": f"{content if content is not None else i:064x}",
            "tlsh": "TNULL" if tnull else f"T1{i:070X}",
            "tlsh_status": "TNULL" if tnull else "VALID"}


def distance(pairs):
    return lambda a, b: pairs.get(tuple(sorted((int(a[2:], 16), int(b[2:], 16)))), 100)


class TestFebruaryGroupAudit(unittest.TestCase):
    def test_excluded_bridge_transitivity_and_multiple_labels(self):
        rows = [row(1), row(2), row(3, "b")]
        members, groups, edges, direct = build_groups(rows, {rows[0]["sha"], rows[2]["sha"]},
                                                     distance({(1, 2): 10, (2, 3): 9}))
        self.assertEqual(len(groups), 1)
        self.assertEqual(direct, 2)
        self.assertEqual(groups[0]["family_count_retained"], 2)
        self.assertEqual(len(edges), 2)
        manifest = assign_folds(members, ["a", "b"])
        self.assertEqual(len(manifest), 2)
        self.assertNotIn(rows[1]["sha"], {r["sha"] for r in manifest})

    def test_exact_hash_handles_tnull_and_threshold_is_inclusive(self):
        rows = [row(1, content=99, tnull=True), row(2, content=99, tnull=True), row(3), row(4), row(5)]
        _, groups, edges, _ = build_groups(rows, {r["sha"] for r in rows}, distance({(3, 4): 10, (4, 5): 11}))
        self.assertEqual(sorted(g["all_samples"] for g in groups), [1, 2, 2])
        self.assertEqual(len(edges), 2)

    def test_rejects_march_duplicate_and_unknown_retained_id(self):
        for rows, kept in [([row(1, month="03")], {row(1)["sha"]}),
                           ([row(1), row(1)], {row(1)["sha"]}),
                           ([row(1)], {row(2)["sha"]})]:
            with self.assertRaises(ValueError):
                build_groups(rows, kept, distance({}))

    def test_support_and_deterministic_five_fold_separation(self):
        rows = [row(i, "a" if i <= 25 else "b") for i in range(1, 46)]
        members, groups, edges, _ = build_groups(rows, {r["sha"] for r in rows}, distance({}))
        manifest = assign_folds(members, ["a", "b", "absent"])
        other = build_groups(list(reversed(rows)), {r["sha"] for r in rows}, distance({}))[0]
        self.assertEqual(manifest, assign_folds(other, ["a", "b", "absent"]))
        support, folds, gates = assess(members, groups, manifest, ["a", "b", "absent"], edges)
        self.assertTrue(all(gates.values()))
        self.assertEqual([r["in_support"] for r in support], [True, True, False])
        self.assertEqual([f["score_samples"] for f in folds], [9] * 5)

    def test_twenty_samples_in_one_group_do_not_meet_support(self):
        rows = [row(i, content=999) for i in range(1, 21)]
        members, groups, edges, _ = build_groups(rows, {r["sha"] for r in rows}, distance({}))
        support, _, gates = assess(members, groups, assign_folds(members, ["a"]), ["a"], edges)
        self.assertFalse(support[0]["in_support"])
        self.assertFalse(gates["nonempty_approved_support"])
        self.assertFalse(gates["largest_component_within_ideal_fold_size"])

    def test_rejects_split_group_and_duplicate_fold_sample(self):
        rows = [row(1), row(2)]
        members, groups, edges, _ = build_groups(rows, {r["sha"] for r in rows}, distance({(1, 2): 0}))
        manifest = assign_folds(members, ["a"])
        for invalid in ([manifest[0], manifest[1] | {"fold": 1}], manifest + [manifest[0]]):
            with self.assertRaises(ValueError):
                assess(members, groups, invalid, ["a"], edges)

    def test_shared_csv_excludes_march_before_graph_and_manifest_roundtrip(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "historical.csv"
            rows = [row(i) for i in range(1, 26)] + [row(100, month="03")]
            save_csv(path, rows, list(rows[0]))
            selected = february_rows(path)
            self.assertEqual(len(selected), 25)
            members, _, _, _ = build_groups(selected, {r["sha"] for r in selected}, distance({}))
            manifest = assign_folds(members, ["a"])
            target = Path(directory) / "folds.csv"
            save_csv(target, manifest, list(manifest[0]))
            with target.open(newline="", encoding="utf-8") as stream:
                saved = list(csv.DictReader(stream))
            self.assertEqual([r["sha"] for r in saved], [r["sha"] for r in manifest])
            self.assertEqual({r["fold"] for r in saved}, set("01234"))


if __name__ == "__main__":
    unittest.main()
