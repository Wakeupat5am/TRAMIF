import hashlib
import json
import math
from fractions import Fraction
import unittest

import numpy as np

from scripts.descriptive_mmd import biased_rbf_mmd2
from scripts.mmd_measurement_preparation import (
    CAP, NAMESPACE, VIEWS, ranking, train_reference, reference_pairs, checked_rows,
    cap_audit, verify_alignment, fit_train_normalizer, fit_train_bandwidth,
    weighted_median,
)


def row(i, family="a", stage="train", month=None, group=None):
    month = month or {"train": "2019-09", "A": "2019-11", "B": "2019-12"}[stage]
    return dict(sha=f"{i:064x}", family=family, stage=stage,
                timestamp=month+"-15 00:00:00+00:00", retained="True", group_id=group or f"g{i}")


def embeddings(rows, values):
    return {r["sha"]: value for r, value in zip(rows, values)}


class TestMeasurementPreparation(unittest.TestCase):
    def test_reference_cap_hash_contract_and_order(self):
        rows = [row(i) for i in range(1, 141)] + [row(200+i, "b") for i in range(3)]
        chosen = train_reference(rows, case="E1", families=["b", "a"])
        self.assertEqual(len(chosen), 131)
        self.assertEqual([r["family"] for r in chosen[:3]], ["b"]*3)
        def independent_key(r):
            payload = json.dumps(["TRAMIF-G3B-v1", 42, "train-reference", "E1", "a", r["sha"]], separators=(",", ":"), ensure_ascii=True).encode()
            return hashlib.sha256(payload).hexdigest(), r["sha"]
        self.assertEqual(chosen[3:], sorted(rows[:140], key=independent_key)[:128])
        self.assertEqual(chosen, train_reference(rows[::-1], case="E1", families=["b", "a"]))

    def test_reference_rejects_nontrain_future_duplicates_and_missing_family(self):
        for rows in ([row(1, stage="A")], [row(1, month="2019-11")], [row(1, month="2020-04")],
                     [row(1), row(1)], [dict(row(1), retained="False")], [dict(row(1), timestamp="2019-09-01")]):
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                train_reference(rows, case="E1", families=["a"])
        with self.assertRaises(ValueError):
            train_reference([row(1)], case="E1", families=["a", "b"])

    def test_utc_boundary_and_invalid_id(self):
        r = dict(row(1), timestamp="2019-11-01T01:00:00+02:00")
        self.assertEqual(len(train_reference([r], case="E1", families=["a"])), 1)
        for bad in (dict(row(1), sha="X"*64), dict(row(1), family="outside")):
            with self.assertRaises(ValueError):
                checked_rows([bad], "E1", ["a"], {"train"})

    def test_pairs_deterministic_no_self_duplicate_and_cap(self):
        rows = train_reference([row(i) for i in range(40)], case="E1", families=["a"])
        pairs = reference_pairs(rows, case="E1", families=["a"])
        self.assertEqual(len(pairs), CAP)
        self.assertEqual(len({(p["left"], p["right"]) for p in pairs}), CAP)
        self.assertTrue(all(p["left"] < p["right"] for p in pairs))
        self.assertEqual(pairs, reference_pairs(rows[::-1], case="E1", families=["a"]))
        expected = sorted(((a["sha"], b["sha"]) for a in sorted(rows, key=lambda r:r["sha"]) for b in sorted(rows, key=lambda r:r["sha"]) if a["sha"] < b["sha"]),
                          key=lambda p: ranking("train-pair", "E1", "a", *p))[:CAP]
        self.assertEqual([(p["left"], p["right"]) for p in pairs], expected)
        with self.assertRaises(ValueError):
            reference_pairs([row(1)], case="E1", families=["a"])

    def test_population_mixture_moments_hand_answer(self):
        rows = [row(1), row(2), row(3, "b")]
        data = embeddings(rows, [[0, 3], [2, 3], [10, 3]])
        norm = fit_train_normalizer(rows, data, case="E1", families=["a", "b"])
        # a mass1/2, b mass1/2: mu=5.5; variance=(30.25+12.25)/4+20.25/2=20.75.
        np.testing.assert_allclose(norm.mean, [5.5, 3], rtol=0, atol=1e-14)
        np.testing.assert_allclose(norm.std**2, [20.75, 0], rtol=0, atol=1e-14)
        self.assertEqual(norm.keep.tolist(), [True, False])
        self.assertEqual(norm.rank, 1)
        self.assertAlmostEqual(norm.condition_number, 1)
        self.assertFalse(norm.mean.flags.writeable)

    def test_reference_family_replication_not_file_weighting(self):
        rows = [row(1), row(2), row(3, "b"), row(4, "b")]
        data = embeddings(rows, [[0], [2], [10], [12]])
        base = fit_train_normalizer(rows, data, case="E1", families=["a", "b"])
        extra = [row(5), row(6)]
        other = fit_train_normalizer(rows+extra, data|embeddings(extra, [[0], [2]]), case="E1", families=["a", "b"])
        np.testing.assert_array_equal(base.mean, other.mean)
        np.testing.assert_allclose(base.std, other.std, rtol=0, atol=1e-14)

    def test_mask_boundary_all_masked_nonfinite_and_overflow(self):
        rows = [row(1), row(2)]
        data = embeddings(rows, [[-1e-8, -2e-8, -1], [1e-8, 2e-8, 1]])
        n = fit_train_normalizer(rows, data, case="E1", families=["a"])
        self.assertEqual(n.keep.tolist(), [False, True, True])
        self.assertEqual(n.rank, 1)
        self.assertIsNone(n.condition_number)
        for data in (embeddings(rows, [[1], [1]]), embeddings(rows, [[np.nan], [1]]), embeddings(rows, [[0], [np.inf]])):
            with self.assertRaises(ValueError):
                fit_train_normalizer(rows, data, case="E1", families=["a"])
        with self.assertRaises(FloatingPointError):
            fit_train_normalizer(rows, embeddings(rows, [[-1e308], [1e308]]), case="E1", families=["a"])

    def test_train_fit_invariant_to_ab_transform_and_input_order(self):
        rows = [row(1), row(2), row(3), row(4)]
        data = embeddings(rows, [[0, 0], [1, 2], [3, 1], [2, 4]])
        norm = fit_train_normalizer(rows, data, case="E1", families=["a"])
        h = fit_train_bandwidth(rows, data, normalizer=norm)
        old = norm.mean.copy(), norm.std.copy()
        for ab in ([[1000, -500]], [[-1000, 400], [0, 0]]):
            norm.transform(ab)
        np.testing.assert_array_equal(norm.mean, old[0])
        np.testing.assert_array_equal(norm.std, old[1])
        self.assertEqual(h, fit_train_bandwidth(rows[::-1], dict(reversed(list(data.items()))), normalizer=norm))
        other = fit_train_normalizer(rows[::-1], data, case="E1", families=["a"])
        np.testing.assert_array_equal(norm.mean, other.mean)
        self.assertEqual(norm.reference_sha256, other.reference_sha256)
        with self.assertRaises(ValueError):
            fit_train_normalizer(rows, data|{row(99)["sha"]: [99,99]}, case="E1", families=["a"])

    def test_bandwidth_lower_weighted_median_and_invalid_inputs(self):
        self.assertEqual(weighted_median([1, 3], [Fraction(1,2), Fraction(1,2)]), 1)
        self.assertEqual(weighted_median([0, 1, 2, 10], [Fraction(1,6)]*3+[Fraction(1,2)]), 2)
        self.assertEqual(weighted_median([0, 0, 4], [1,1,1]), 0)
        for values, weights in (([], []), ([1], [0]), ([1], [-1]), ([1], [float("nan")]), ([float("inf")], [1]), ([-1], [1]), ([1,2],[1])):
            with self.assertRaises(ValueError):
                weighted_median(values, weights)

    def test_bandwidth_hand_answer_equal_family_pair_mass(self):
        rows = [row(1), row(2), row(3), row(4, "b"), row(5, "b")]
        data = embeddings(rows, [[0], [1], [2], [0], [4]])
        norm = fit_train_normalizer(rows, data, case="E1", families=["a", "b"])
        result = fit_train_bandwidth(rows, data, normalizer=norm)
        # a distances 1,1,4 each mass1/6, b distance16 mass1/2; lower median 4/variance.
        self.assertAlmostEqual(result["bandwidth_h"], 4/(norm.std[0]**2), places=14)
        self.assertEqual(result["pair_mass_per_family"], {"a":"1/2", "b":"1/2"})

    def test_zero_pair_distances_retained_no_epsilon_rescue(self):
        rows = [row(1), row(2), row(3), row(4, "b"), row(5, "b")]
        data = embeddings(rows, [[0], [0], [0], [1], [2]])
        norm = fit_train_normalizer(rows, data, case="E1", families=["a", "b"])
        with self.assertRaisesRegex(ValueError, "Degenerate TRAIN bandwidth"):
            fit_train_bandwidth(rows, data, normalizer=norm)

    def test_provenance_rejects_changed_embedding_dates_and_case(self):
        rows = [row(1), row(2)]
        data = embeddings(rows, [[0], [2]])
        norm = fit_train_normalizer(rows, data, case="E1", families=["a"])
        with self.assertRaisesRegex(ValueError, "provenance"):
            fit_train_bandwidth(rows, embeddings(rows, [[0], [3]]), normalizer=norm)
        with self.assertRaisesRegex(ValueError, "provenance"):
            fit_train_bandwidth([dict(rows[0], timestamp="2019-10-01T00:00:00+00:00"), rows[1]], data, normalizer=norm)
        with self.assertRaises(ValueError):
            fit_train_normalizer([row(1, month="2020-01"), row(2)], data, case="E1", families=["a"])

    def test_transform_validation_and_mmd_integration_synthetic_only(self):
        rows = [row(1), row(2), row(3)]
        data = embeddings(rows, [[0, 1], [1, 1], [3, 1]])
        norm = fit_train_normalizer(rows, data, case="E1", families=["a"])
        h = fit_train_bandwidth(rows, data, normalizer=norm)["bandwidth_h"]
        a, b = norm.transform([[0,1],[1,1]]), norm.transform([[3,1],[4,1]])
        result = biased_rbf_mmd2(a, b, bandwidth_h=h)
        self.assertTrue(0 < result.mmd2 <= 2)
        for bad in ([[0]], [[np.nan,1]], [], [[1+1j,1]]):
            with self.assertRaises(ValueError):
                norm.transform(bad)

    def test_cap_balanced_counts_preserves_groups_and_reports_both_predicates(self):
        rows = [row(i+1, stage="A", group=f"g{i}") for i in range(140)]
        rows += [row(1000+i, stage="B", group=f"b{i}") for i in range(20)]
        selected, counts = cap_audit(rows, case="E1", support=["a"])
        self.assertEqual(len(selected), 40)
        self.assertEqual(counts[0]["groups_lost"], 120)
        self.assertTrue(counts[0]["lost_at_least5"])
        self.assertFalse(counts[0]["post_cap_below5"])
        self.assertEqual(selected, cap_audit(rows[::-1], case="E1", support=["a"])[0])
        self.assertTrue(all(r in rows for r in selected))

    def test_post_cap_below5_is_distinct_from_lost5(self):
        a = [row(i+1, stage="A") for i in range(30)]
        # Fixed synthetic fixture: ranked first20 in one group, remaining10 in4 groups.
        a = sorted(a, key=lambda r: ranking("matched-A", "E1", "a", r["sha"]))
        a = [dict(r, group_id="big" if i<20 else f"small{i%4}") for i,r in enumerate(a)]
        b = [row(100+i, stage="B") for i in range(20)]
        _, counts = cap_audit(a+b, case="E1", support=["a"])
        self.assertTrue(counts[0]["post_cap_below5"])
        self.assertFalse(counts[0]["lost_at_least5"])
        self.assertEqual(counts[0]["groups_lost"], 4)

    def test_cap_rejects_pre_support_excluded_and_wrong_month(self):
        for rows in ([row(1, stage="A")], [row(1, stage="A", month="2020-04")], [dict(row(1, stage="A"), retained="False")]):
            with self.assertRaises(ValueError):
                cap_audit(rows, case="E1", support=["a"])

    def test_alignment_rejects_permutation_missing_view_and_wrong_family(self):
        rows = [row(1, stage="A"), row(2, stage="B")]
        verify_alignment(rows, {v: rows.copy() for v in VIEWS})
        for replacement in (rows[::-1], rows[:1], [dict(rows[0], family="b"),rows[1]]):
            with self.assertRaises(ValueError):
                verify_alignment(rows, {"sbsmi":rows, "raw-byte":rows, "entropy":replacement})
        with self.assertRaises(ValueError):
            verify_alignment(rows, {"sbsmi":rows})


if __name__ == "__main__":
    unittest.main()
