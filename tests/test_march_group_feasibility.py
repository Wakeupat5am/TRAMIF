import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.february_group_audit import save_csv
from scripts.march_group_feasibility import march_components, march_rows, support_matrix, support_pilot


def row(i, family="a", month="03", content=None, tnull=False):
    return {"sha": f"{i:064x}", "split": "validation", "timestamp": f"2020-{month}-01 00:00:00+00:00",
            "family": family, "disarmed_content_sha256": f"{i if content is None else content:064x}",
            "tlsh": "TNULL" if tnull else f"T1{i:070X}", "tlsh_status": "TNULL" if tnull else "VALID"}


def distance(pairs):
    return lambda a, b: pairs.get(tuple(sorted((int(a[2:], 16), int(b[2:], 16)))), 100)


class TestMarchGroupingFeasibility(unittest.TestCase):
    def test_excluded_bridge_connects_retained_without_reentering_scoring(self):
        rows = [row(1), row(2), row(3, "b"), row(4)]
        kept = {r["sha"] for r in (rows[0], rows[2], rows[3])}
        members, groups, edges, direct = march_components(rows, kept, distance({(1, 2): 10, (2, 3): 9, (3, 4): 11}))
        self.assertEqual(len(groups), 2)
        self.assertEqual(direct, 3)
        self.assertEqual(groups[0]["family_count_retained"], 2)
        ids, matrix = support_matrix(members, ["a", "b", "absent"])
        self.assertEqual(matrix.sum(), 3)
        self.assertEqual(len(ids), 2)
        self.assertEqual(len(edges), 2)

    def test_exact_content_tnull_and_label_blind_deterministic_graph(self):
        rows = [row(1, tnull=True, content=99), row(2, "b", tnull=True, content=99), row(3)]
        kept = {r["sha"] for r in rows}
        members, groups, _, _ = march_components(rows, kept, distance({}))
        self.assertEqual(len(groups), 2)
        other = march_components(list(reversed(rows)), kept, distance({}))[0]
        self.assertEqual(members, other)
        rows[1]["family"] = "changed"
        changed = march_components(rows, kept, distance({}))[0]
        self.assertEqual([(r["sha"], r["group_id"]) for r in members], [(r["sha"], r["group_id"]) for r in changed])

    def test_other_month_duplicate_unknown_ids_rejected(self):
        cases = [([row(1, month="02")], {row(1)["sha"]}),
                 ([row(1, month="04")], {row(1)["sha"]}),
                 ([row(1), row(1)], {row(1)["sha"]}), ([row(1)], {row(2)["sha"]})]
        for rows, kept in cases:
            with self.assertRaises(ValueError):
                march_components(rows, kept, distance({}))

    def test_group_draws_preserve_mixed_classes_and_missing_replicates(self):
        matrix = np.array([[2, 3, 0, 0], [0, 0, 5, 0]])
        support, reps, draws, counts, missing = support_pilot(matrix, ["a", "b", "c", "absent"], 100, 42)
        np.testing.assert_array_equal(counts, draws @ matrix)
        np.testing.assert_array_equal(missing[:, 0], missing[:, 1])
        self.assertFalse(missing[:, 3].any())
        self.assertTrue(missing.any())
        self.assertEqual(len(reps), 100)  # No discard or redraw despite missing support.
        self.assertTrue(np.all(draws.sum(1) == 2))
        self.assertEqual(support[0]["analytic_missing_probability"], .25)
        self.assertEqual(support[0]["effective_groups_size_only"], 1)
        self.assertEqual(support[0]["pilot_missing_replicates"], int(missing[:, 0].sum()))

    def test_effective_group_count_and_seed_reproducibility(self):
        matrix = np.array([[1], [3]])
        first = support_pilot(matrix, ["a"], 20, 42)
        second = support_pilot(matrix, ["a"], 20, 42)
        self.assertEqual(first[0][0]["effective_groups_size_only"], 1.6)
        np.testing.assert_array_equal(first[2], second[2])
        self.assertFalse(first[4].any())
        self.assertGreater(len(set(first[3].sum(1))), 1)  # Cluster resampling varies N.

    def test_invalid_pilot_budget_and_counts_rejected(self):
        for count in (0, 501):
            with self.assertRaises(ValueError):
                support_pilot(np.ones((2, 2), dtype=int), ["a", "b"], count)
        for matrix in (np.zeros((2, 2), dtype=int), np.array([[-1, 2]]), np.ones((2, 2))):
            with self.assertRaises(ValueError):
                support_pilot(matrix, ["a", "b"])

    def test_csv_to_graph_to_pilot_file_integration(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            rows = [row(1, month="02"), row(2), row(3, "b")]
            path = folder / "historical.csv"
            save_csv(path, rows, list(rows[0]))
            selected = march_rows(path)
            self.assertEqual(len(selected), 2)
            members = march_components(selected, {r["sha"] for r in selected}, distance({}))[0]
            ids, matrix = support_matrix(members, ["a", "b"])
            _, reps, draws, counts, missing = support_pilot(matrix, ["a", "b"], 20)
            np.savez(folder / "pilot.npz", group_ids=ids, counts=counts, draws=draws, missing=missing)
            with np.load(folder / "pilot.npz", allow_pickle=False) as saved:
                np.testing.assert_array_equal(saved["counts"], saved["draws"] @ matrix)
                self.assertEqual(len(saved["missing"]), len(reps))


if __name__ == "__main__":
    unittest.main()
