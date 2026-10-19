import tempfile
import unittest
from pathlib import Path
import tlsh

from scripts.audit_historical_episodes import (
    Inputs, EPISODES, month, select_families, stage, pair_edges,
    retained_decisions, graph_members, support_table, audit_episode,
)
from scripts.near_duplicate_core import apply_historical_policy, time_key, validate_rows


def row(i, when, family="a", content=None, tnull=False):
    return {"sha": f"{i:064x}", "timestamp": when + " 00:00:00+00:00", "family": family,
            "disarmed_content_sha256": f"{content if content is not None else i+10000:064x}",
            "tlsh": "TNULL" if tnull else f"T1{i:070X}",
            "tlsh_status": "TNULL" if tnull else "VALID",
            "split": "train" if when < "2020-02-01" else "validation"}


def dist(mapping):
    return lambda a, b: mapping.get(tuple(sorted((int(a[2:], 16), int(b[2:], 16)))), 1000)


class TestHistoricalEpisodeAudit(unittest.TestCase):
    def test_prefix_eligibility_uses_unique_ids_and_three_months_only(self):
        rows = [row(i, ("2019-08-01", "2019-09-01", "2019-10-01")[i % 3]) for i in range(50)]
        rows += [row(i+100, "2019-11-01", "late") for i in range(100)]
        rows += [row(i+300, "2019-10-01", "one_month") for i in range(55)]
        rows += [dict(rows[0])]  # unique-file rule, not row count
        selected, counts = select_families(rows, "2019-11")
        self.assertEqual(selected, ["a"])
        self.assertEqual(next(r for r in counts if r["family"] == "a")["prefix_unique_sha"], 50)

    def test_49_unique_does_not_pass_even_with_repeated_row(self):
        rows = [row(i, ("2019-08-01", "2019-09-01", "2019-10-01")[i % 3]) for i in range(49)]
        self.assertEqual(select_families(rows + [dict(rows[0])], "2019-11")[0], [])

    def test_boundaries_and_utc_conversion(self):
        e1 = EPISODES["E1"]
        self.assertEqual([stage(row(i, d), e1) for i, d in enumerate(
            ("2019-10-31", "2019-11-01", "2019-12-01", "2020-01-01", "2020-02-01"))],
            ["train", "A", "B", "C", None])
        r = row(99, "2019-11-01")
        r["timestamp"] = "2019-11-01 00:00:00+07:00"
        self.assertEqual(month(r), "2019-10")

    def test_all_unordered_pairs_and_threshold_edges(self):
        rows = [row(1, "2019-09-01"), row(2, "2019-10-01"), row(3, "2019-11-01")]
        seen = []
        def compare(a, b):
            seen.append((a, b))
            return 10 if int(b[2:], 16) == 2 else 21
        self.assertEqual(pair_edges(rows, compare), [(0, 1, 10, False)])
        self.assertEqual(len(seen), 3)

    def test_native_object_distance_matches_public_function(self):
        data = bytes((i*i + i//7) % 256 for i in range(8192))
        h = tlsh.hash(data)
        self.assertNotEqual(h, "TNULL")
        rows = [row(1, "2019-09-01"), row(2, "2019-10-01")]
        for r in rows:
            r["tlsh"] = h
        self.assertEqual(pair_edges(rows), pair_edges(rows, tlsh.diff))
        self.assertEqual(pair_edges(rows), [(0, 1, 0, False)])

    def test_episode_integration_support_not_selected_by_outcome(self):
        rows = []
        for family in ("a", "b"):
            for i in range(50):
                rows.append(row(len(rows)+1, ("2019-08-01", "2019-09-01", "2019-10-01")[i%3], family))
            for date in ("2019-11-01", "2019-12-01", "2020-01-01"):
                for _ in range(20):
                    rows.append(row(len(rows)+1, date, family))
        rows.sort(key=time_key)
        result = audit_episode(rows, [], EPISODES["E1"])[0]
        self.assertEqual(result["computational_feasibility"], "PASS")
        self.assertEqual((result["K"], result["S_count"], result["S_eval_count"]), (2, 2, 2))
        self.assertEqual(result["threshold_counts"]["0"]["train"]["retained"], 100)
        for r in rows:
            if stage(r, EPISODES["E1"]) == "C":
                r["family"] = "a"
        changed = audit_episode(rows, [], EPISODES["E1"])[0]
        self.assertEqual(changed["S"], result["S"])
        self.assertEqual(changed["S_eval_count"], 1)

    def test_train_keeps_nonidentical_near_duplicates(self):
        rows = [row(1, "2019-08-01"), row(2, "2019-09-01")]
        ds = retained_decisions(rows, [(0, 1, 0, False)], EPISODES["E1"], 0)
        self.assertEqual([r["decision"] for r in ds], ["KEEP", "KEEP"])

    def test_excluded_reference_remains_active(self):
        rows = [row(1, "2019-09-01"), row(2, "2019-11-01"), row(3, "2019-12-01")]
        ds = retained_decisions(rows, [(0, 1, 0, False), (1, 2, 0, False)], EPISODES["E1"], 0)
        self.assertEqual([r["decision"] for r in ds], ["KEEP", "EXCLUDE", "EXCLUDE"])
        self.assertEqual(ds[2]["reference_sha"], rows[1]["sha"])

    def test_later_outcome_bridge_cannot_merge_signal_groups(self):
        rows = [row(1, "2019-11-01"), row(2, "2019-12-01"), row(3, "2020-01-01")]
        edges = [(0, 2, 10, False), (1, 2, 10, False)]
        kept = {r["sha"] for r in rows}
        ab = graph_members(rows, edges, [0, 1], kept, EPISODES["E1"], 10)
        whole = graph_members(rows, edges, range(3), kept, EPISODES["E1"], 10)
        self.assertEqual(len({r["group_id"] for r in ab}), 2)
        self.assertEqual(len({r["group_id"] for r in whole}), 1)

    def test_excluded_AB_bridge_is_preserved(self):
        rows = [row(1, "2019-11-01"), row(2, "2019-11-02"), row(3, "2019-12-01")]
        ab = graph_members(rows, [(0, 1, 5, False), (1, 2, 5, False)], range(3),
                           {rows[0]["sha"], rows[2]["sha"]}, EPISODES["E1"], 10)
        self.assertEqual(len({r["group_id"] for r in ab if r["retained"]}), 1)

    def test_labels_do_not_control_similarity(self):
        rows = [row(1, "2019-11-01", "a"), row(2, "2019-12-01", "b")]
        ds = retained_decisions(rows, [(0, 1, 0, False)], EPISODES["E1"], 0)
        self.assertEqual(ds[1]["decision"], "EXCLUDE")

    def test_tnull_exact_copy_still_excluded(self):
        rows = [row(1, "2019-09-01", content=1, tnull=True),
                row(2, "2019-11-01", content=1, tnull=True)]
        edges = pair_edges(rows)
        ds = retained_decisions(rows, edges, EPISODES["E1"], 0)
        self.assertEqual(ds[1]["reason"], "exact_content")
        self.assertIsNone(ds[1]["tlsh_distance"])

    def test_support_counts_joint_groups_and_outcome_cannot_select_S(self):
        members = [{"family": "a", "retained": True, "stage": s, "group_id": str(i % 5)}
                   for s in ("A", "B") for i in range(20)]
        a = support_table(members, ["a"], {})
        b = support_table(members, ["a"], {"a": 10})
        self.assertTrue(a[0]["eligible_S"])
        self.assertFalse(a[0]["S_eval"])
        self.assertTrue(b[0]["S_eval"])
        for r in members:
            r["group_id"] = "one"
        self.assertFalse(support_table(members, ["a"], {})[0]["eligible_S"])

    def test_primary_and_sensitivities_match_legacy_policy_on_shared_dates(self):
        train = [row(1, "2019-09-01"), row(2, "2019-10-01")]
        evaluation = [row(3, "2020-02-01"), row(4, "2020-02-02"), row(5, "2020-02-03")]
        compare = dist({(1, 2): 0, (1, 3): 10, (3, 4): 0, (4, 5): 20})
        legacy = apply_historical_policy(train, evaluation, compare)
        rows = sorted(train + evaluation, key=time_key)
        edges = pair_edges(rows, compare)
        for t in (0, 10, 20):
            ds = retained_decisions(rows, edges, EPISODES["E2"], t)
            self.assertEqual([r["decision"] for r in ds[2:]],
                             [r["decision"] for r in legacy["validation_decisions"][t]])
            self.assertEqual([r["reference_sha"] for r in ds[2:]],
                             [r["reference_sha"] for r in legacy["validation_decisions"][t]])

    def test_forbidden_paths_blocked_before_file_open(self):
        with tempfile.TemporaryDirectory() as tmp:
            reader = Inputs(Path(tmp))
            for path in ("data/raw/bodmas_metadata.csv", "data/predictions.npz", "../secret"):
                with self.assertRaisesRegex(ValueError, "allowlist"):
                    reader.read(path)

    def test_checksum_mismatch_stops_processing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "docs").mkdir()
            (root / "docs/RESEARCH_PROTOCOL_V1.md").write_text("changed")
            with self.assertRaisesRegex(ValueError, "checksum"):
                Inputs(root).read("docs/RESEARCH_PROTOCOL_V1.md")

    def test_synthetic_future_date_and_duplicate_id_rejected(self):
        with self.assertRaises(ValueError):
            validate_rows([row(1, "2020-04-01")], "validation")
        with self.assertRaises(ValueError):
            validate_rows([row(1, "2019-09-01"), row(1, "2019-09-01")], "train")


if __name__ == "__main__":
    unittest.main()
