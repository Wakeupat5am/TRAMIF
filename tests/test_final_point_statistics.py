import unittest
import numpy as np
from scripts import final_point_statistics as s
from scripts.multiseed_statistics import outcomes, fixed_support_f1
from scripts.uniform_fusion import classification_metrics


def aggregate(x, **overrides):
    config = dict(seed_ids=s.SEEDS, method_names=s.METHODS, month_names=s.MONTHS)
    config.update(overrides)
    return s.aggregate_monthly(x, **config)


class TestFinalPointStatistics(unittest.TestCase):
    def test_worst_before_seed_average_not_reverse(self):
        x = np.ones((5, 7, 6))
        for seed in range(5): x[seed, :, seed] = 0
        result = aggregate(x)["methods"]["sbsmi"]["outcomes"]
        self.assertAlmostEqual(result[s.OUTCOMES[0]]["mean"], 5/6)
        self.assertEqual(result[s.OUTCOMES[1]]["mean"], 0)
        self.assertAlmostEqual(x.mean(0).min(), .8)

    def test_unequal_month_sizes_do_not_weight_primary(self):
        sizes = [2, 4, 6, 8, 10, 100]
        y = np.concatenate([np.tile([0, 1], n//2) for n in sizes])
        months = np.repeat(np.arange(6), sizes)
        p = np.tile(y, (5, 7, 1)); p[:, :, months == 5] = 1-y[months == 5]
        expected, counts = outcomes(y, p, months, [[0, 1]]*6)
        x = np.ones((5, 7, 6)); x[:, :, 5] = 0
        actual = aggregate(x)["methods"]["sbsmi"]["outcomes"]
        self.assertEqual(counts, sizes)
        self.assertAlmostEqual(actual[s.OUTCOMES[0]]["mean"], expected[0, 0, 0])
        pooled = fixed_support_f1(y, p[0, 0], [0, 1])
        self.assertNotAlmostEqual(pooled, expected[0, 0, 0])

    def test_supported_vs_fixed51_and_absent_prediction(self):
        y = np.array([0, 1]); p = np.array([0, 50])
        self.assertEqual(fixed_support_f1(y, p, [0, 1]), .5)
        self.assertEqual(fixed_support_f1(y, p, list(range(51))), 1/51)

    def test_undefined_f1_zero_but_recall_absent_is_null(self):
        logp = np.log(np.full((2, 51), 1/51))
        r = classification_metrics(logp, np.array([0, 0]))
        self.assertEqual(r["per_class"][50]["f1"], 0)
        self.assertIsNone(r["per_class"][50]["recall"])

    def test_argmax_exact_tie_smallest_class(self):
        r = classification_metrics(np.log(np.full((2, 51), 1/51)), np.array([0, 1]))
        self.assertEqual(r["confusion_matrix"][0][0], 1)
        self.assertEqual(r["confusion_matrix"][1][0], 1)

    def test_macro_is_equal_class_not_support_weighted(self):
        y = np.array([0]*9+[1]); p = np.zeros(10, dtype=int)
        self.assertAlmostEqual(fixed_support_f1(y, p, [0, 1]), 9/19)

    def test_empty_month_or_support_mismatch_stops(self):
        y = np.tile([0, 1], 6); p = np.tile(y, (5, 7, 1)); m = np.repeat(np.arange(6), 2)
        with self.assertRaises(ValueError): outcomes(y[:-2], p[:, :, :-2], m[:-2], [[0, 1]]*6)
        with self.assertRaises(ValueError): outcomes(y, p, m, [[0]]*6)

    def test_exact_metadata_and_all_five_seeds_required(self):
        x = np.ones((5, 7, 6))
        for overrides in [dict(seed_ids=(42, 43, 44, 45)), dict(seed_ids=(43, 42, 44, 45, 46)),
                          dict(seed_ids=(42., 43., 44., 45., 46.)),
                          dict(method_names=tuple(reversed(s.METHODS))),
                          dict(month_names=tuple(reversed(s.MONTHS)))]:
            with self.subTest(overrides=overrides), self.assertRaises(ValueError): aggregate(x, **overrides)

    def test_nonfinite_out_of_range_and_wrong_shapes_stop(self):
        for v in [np.nan, np.inf, -.01, 1.01]:
            x = np.ones((5, 7, 6)); x[0, 0, 0] = v
            with self.subTest(value=v), self.assertRaises(ValueError): aggregate(x)
        for x in [np.ones((4, 7, 6)), np.ones((5, 6, 6)), np.ones((5, 7, 5)),
                  np.full((5, 7, 6), '0.5')]:
            with self.assertRaises(ValueError): aggregate(x)

    def test_sample_sd_ddof_one_and_all_seeds(self):
        x = np.broadcast_to(np.array([0, .25, .5, .75, 1])[:, None, None], (5, 7, 6))
        r = aggregate(x)["methods"]["sbsmi"]["outcomes"][s.OUTCOMES[0]]
        self.assertEqual(r["per_seed"], [0, .25, .5, .75, 1])
        self.assertEqual(r["mean"], .5)
        self.assertAlmostEqual(r["sample_sd"], (.625/4)**.5)
        self.assertEqual((r["min"], r["max"]), (0, 1))

    def test_paired_contrasts_orientation_eight_rows(self):
        x = np.full((5, 7, 6), .5); x[:, s.METHODS.index('quality-plus-drift')] = .6
        r = aggregate(x); self.assertEqual(len(r["contrasts"]), 8)
        self.assertEqual(sum(c['priority'] == 'primary' for c in r['contrasts']), 2)
        for c in r["contrasts"]:
            self.assertAlmostEqual(c["mean"], .1 if c["left"] == 'quality-plus-drift' else 0)
        self.assertIsNone(r["p_values"])
        self.assertEqual(r["interval_status"], 'NOT_ESTABLISHED_NOT_COMPUTED')

    def test_tied_worst_months_all_reported(self):
        r = aggregate(np.ones((5, 7, 6)))
        self.assertEqual(r["methods"]["sbsmi"]["worst_months_per_seed"], [list(s.MONTHS)]*5)

    def test_no_input_mutation_and_deterministic_result(self):
        x = np.arange(210, dtype=float).reshape(5, 7, 6)/209
        before = x.copy(); a = aggregate(x); b = aggregate(x)
        np.testing.assert_array_equal(x, before); self.assertEqual(a, b)


if __name__ == '__main__': unittest.main()
