import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.paired_group_bootstrap import (
    sample_metrics, aggregate_groups, draw_multiplicities, resampled_metrics,
    paired_differences, percentile_interval, distribution_diagnostic, influence_diagnostics,
)
from scripts.uniform_fusion import classification_metrics


class TestPairedGroupBootstrap(unittest.TestCase):
    def fixture(self):
        p = np.array([[[.8, .2], [.4, .6], [.7, .3], [.1, .9]],
                      [[.6, .4], [.3, .7], [.9, .1], [.4, .6]]])
        labels = np.array([0, 1, 1, 1])
        values = sample_metrics(np.log(p), labels)
        return values, np.array(["a", "a", "a", "b"]), labels, p

    def test_known_answer_metrics_and_multiclass_brier_sum(self):
        metrics = sample_metrics(np.log([[[.7, .2, .1]]]), np.array([0]))
        np.testing.assert_allclose(metrics[0, 0], [1, -np.log(.7), .14])

    def test_point_metrics_match_existing_evaluator(self):
        values, _, labels, probabilities = self.fixture()
        for m in range(2):
            expected = classification_metrics(np.log(probabilities[m]), labels)
            np.testing.assert_allclose(values[:, m].mean(0), [expected[k] for k in ("accuracy", "nll", "brier")], atol=1e-14)

    def test_deterministic_draws_and_saved_prefix(self):
        a = draw_multiplicities(17, 2000, 42)
        np.testing.assert_array_equal(a, draw_multiplicities(17, 2000, 42))
        np.testing.assert_array_equal(a[:500], draw_multiplicities(17, 500, 42))
        self.assertTrue(np.all(a.sum(1) == 17))
        self.assertFalse(np.array_equal(a, draw_multiplicities(17, 2000, 43)))

    def test_whole_group_pairing_matches_expanded_indices_and_variable_n(self):
        values, sample_groups, _, _ = self.fixture()
        _, inverse, sizes, sums = aggregate_groups(values, sample_groups)
        draws = np.array([[2, 0], [1, 1], [0, 2]])
        counts, boot = resampled_metrics(draws, sizes, sums)
        np.testing.assert_array_equal(counts, [6, 4, 2])
        for b in range(3):
            idx = np.repeat(np.arange(4), draws[b, inverse])
            np.testing.assert_allclose(boot[b], values[idx].mean(0), atol=1e-14)
            self.assertTrue(np.all(draws[b, inverse[:3]] == draws[b, 0]))

    def test_missing_class_replicates_are_retained_and_finite(self):
        logs = np.log(np.array([[[.8, .2], [.1, .9]], [[.6, .4], [.3, .7]]]))
        values = sample_metrics(logs, np.array([0, 1]))
        _, _, sizes, sums = aggregate_groups(values, np.array(["a", "b"]))
        _, metrics = resampled_metrics(np.array([[2, 0], [0, 2], [1, 1]]), sizes, sums)
        self.assertEqual(metrics.shape, (3, 2, 3))
        self.assertTrue(np.isfinite(metrics).all())
        self.assertTrue(np.isfinite(percentile_interval(paired_differences(metrics, 1, [0]))).all())

    def test_paired_delta_definition_and_identical_methods(self):
        values = np.array([[[.7, .5, .3], [.8, .4, .2]], [[.6, .6, .4], [.6, .5, .3]]])
        np.testing.assert_allclose(paired_differences(values, 1, [0])[:, 0], [[.1, -.1, -.1], [0, -.1, -.1]])
        values[:, 1] = values[:, 0]
        np.testing.assert_array_equal(percentile_interval(paired_differences(values, 1, [0])), np.zeros((2, 1, 3)))

    def test_percentile_linear_known_answer(self):
        values = np.arange(40, dtype=float)
        np.testing.assert_allclose(percentile_interval(values), [.975, 38.025])
        with self.assertRaises(ValueError):
            percentile_interval(np.array([0, np.nan, 1]))

    def test_invalid_nonfinite_metrics_and_draws_rejected(self):
        values, sg, _, _ = self.fixture()
        _, _, sizes, sums = aggregate_groups(values, sg)
        for draws in (np.array([[1, 0]]), np.array([[-1, 3]]), np.array([[1., 1.]])):
            with self.assertRaises(ValueError):
                resampled_metrics(draws, sizes, sums)
        with self.assertRaises(ValueError):
            sample_metrics(np.array([[[np.nan, 0]]]), np.array([0]))
        with self.assertRaises(ValueError):
            aggregate_groups(values, np.array(["a"]*4))

    def test_diagnostics_leave_one_out_match_explicit_deletion(self):
        values, sg, _, _ = self.fixture()
        groups, inverse, sizes, sums = aggregate_groups(values, sg)
        draws = draw_multiplicities(2, 20, 42)
        _, metrics = resampled_metrics(draws, sizes, sums)
        delta = paired_differences(metrics, 1, [0])[:, 0]
        rows, report = influence_diagnostics(sizes, sums, groups, draws, delta, 1, 0)
        for g, row in enumerate(rows):
            remaining = values[inverse != g].mean(0)
            np.testing.assert_allclose([row[f"loo_delta_{k}"] for k in ("accuracy", "nll", "brier")], remaining[1] - remaining[0])
        self.assertIn("most_influential", report)

    def test_distribution_diagnostic_finite_and_symmetric(self):
        d = distribution_diagnostic(np.linspace(-1, 1, 101))
        self.assertAlmostEqual(d["skewness_moment"], 0)
        self.assertEqual(sum(d["histogram_counts"]), 101)
        self.assertIsNone(distribution_diagnostic(np.ones(10))["excess_kurtosis_moment"])

    def test_npz_roundtrip_reproduces_all_metrics_and_ci(self):
        values, sg, _, _ = self.fixture()
        groups, _, sizes, sums = aggregate_groups(values, sg)
        draws = draw_multiplicities(2, 2000, 42)
        counts, metrics = resampled_metrics(draws, sizes, sums)
        ci = percentile_interval(paired_differences(metrics, 1, [0]))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "run.npz"
            np.savez_compressed(path, groups=groups, sizes=sizes, sums=sums, draws=draws)
            with np.load(path, allow_pickle=False) as saved:
                after_n, after_metrics = resampled_metrics(saved["draws"], saved["sizes"], saved["sums"])
            np.testing.assert_array_equal(counts, after_n)
            np.testing.assert_array_equal(metrics, after_metrics)
            np.testing.assert_array_equal(ci, percentile_interval(paired_differences(after_metrics, 1, [0])))


if __name__ == "__main__":
    unittest.main()
