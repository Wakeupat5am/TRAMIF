"""Read-only tests against the frozen historical allowlist, never real embeddings."""
from collections import Counter
import unittest

from scripts.audit_mmd_measurement_preparation import Inputs, audit
from scripts.mmd_measurement_preparation import CASES, CAP, ranking


class TestG3BHistoricalMetadata(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.inputs = Inputs()
        cls.results = audit(cls.inputs)

    def test_frozen_support_mapping_and_train_counts(self):
        for case, k, s, n in (("E1",32,17,7034),("E2",42,25,9803),("Final",51,25,18061)):
            value = self.results[case]
            self.assertEqual(value["summary"]["K"], k)
            self.assertEqual(len(value["summary"]["support"]), s)
            self.assertEqual(value["summary"]["train_samples"], n)
            self.assertEqual(sorted(value["mapping"].values()), list(range(k)))

    def test_train_reference_pairs_have_no_ab_or_future(self):
        for case, v in self.results.items():
            by_id = {r["sha"]:r for r in v["reference"]}
            self.assertEqual(len(by_id), len(v["reference"]))
            self.assertTrue(all("2019-08" <= r["timestamp"][:7] < CASES[case][0] for r in by_id.values()))
            self.assertTrue(all(n<=CAP for n in Counter(r["family"] for r in by_id.values()).values()))
            self.assertEqual(set(by_id)&{r["sha"] for r in v["selected"]}, set())
            for pair in v["pairs"]:
                self.assertLess(pair["left"], pair["right"])
                self.assertEqual(by_id[pair["left"]]["family"], pair["family"])
                self.assertEqual(by_id[pair["right"]]["family"], pair["family"])

    def test_cap_and_group_arithmetic_for_every_family_month(self):
        for case, v in self.results.items():
            for c in v["summary"]["support"]:
                a,b = [r for r in v["counts"] if r["family"]==c]
                n=min(a["samples_before"], b["samples_before"],128)
                for r in (a,b):
                    chosen = [s for s in v["selected"] if s["family"]==c and s["stage"]==r["stage"]]
                    self.assertEqual(len(chosen), n)
                    self.assertEqual(r["groups_after"], len({s["group_id"] for s in chosen}))
                    self.assertEqual(r["groups_lost"], r["groups_before"]-r["groups_after"])
                    self.assertEqual(r["post_cap_below5"], r["groups_after"]<5)
                    self.assertEqual(r["lost_at_least5"], r["groups_lost"]>=5)
                    self.assertEqual(chosen, sorted(chosen, key=lambda s:ranking("matched-"+r["stage"], case,c,s["sha"])))

    def test_unknown_real_alignment_not_promoted_to_pass(self):
        for v in self.results.values():
            self.assertEqual(v["summary"]["actual_embedding_alignment"], "UNKNOWN")
            self.assertEqual(v["summary"]["metadata_alignment"], "PASS_CANONICAL_PLAN_ONLY")

    def test_allowlist_blocks_predictions_raw_future_and_graphs(self):
        for path in ("data/raw/bodmas_metadata.csv", "data/future.csv", "data/predictions.npz",
                     "data/derived/historical_episode_feasibility/20261014_233013_930021/edges_le20.csv", "../outside"):
            with self.assertRaisesRegex(ValueError,"allowlist"):
                self.inputs.read(path)

    def test_checksums_fail_before_parsing(self):
        with self.assertRaisesRegex(ValueError, "Checksum mismatch"):
            self.inputs.read("docs/RESEARCH_PROTOCOL_V1.md", "0"*64)


if __name__ == "__main__":
    unittest.main()
