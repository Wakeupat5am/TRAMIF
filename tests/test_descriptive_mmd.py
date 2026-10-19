import math
import unittest

import numpy as np

from scripts.descriptive_mmd import (
    ROUNDOFF_ATOL, _finish, biased_rbf_mmd2, class_conditional_mmd2,
)


def reference(x, y, h):
    # Independent scalar loops: no production helper, Gram matrix or block sum.
    def kernel(a, b):
        distance = math.fsum((float(u)-float(v))**2 for u, v in zip(a, b)) / len(a)
        return math.exp(-distance/(2*h))
    def mean(a, b):
        return math.fsum(kernel(u, v) for u in a for v in b)/(len(a)*len(b))
    return math.fsum((mean(x, x), mean(y, y), -2*mean(x, y)))


class TestDescriptiveMMD(unittest.TestCase):
    def test_identical_empirical_sets(self):
        x = np.random.default_rng(42).normal(size=(17, 7))
        for block in (1, 4, 32):
            self.assertAlmostEqual(biased_rbf_mmd2(x, x.copy(), bandwidth_h=0.7, block_size=block).mmd2, 0, delta=ROUNDOFF_ATOL)

    def test_known_answer_diagonal_bandwidth_and_dimension(self):
        # p=2, distance/p=2, h=2: cross kernel exp(-1/2).
        result = biased_rbf_mmd2([[0, 0]], [[2, 0]], bandwidth_h=2)
        self.assertEqual(result.mean_xx, 1)
        self.assertEqual(result.mean_yy, 1)
        self.assertAlmostEqual(result.mmd2, 2*(1-math.exp(-0.5)), places=14)

    def test_constructed_point_mass_shift(self):
        x = np.zeros((4, 3))
        values = [biased_rbf_mmd2(x, np.full((7, 3), d), bandwidth_h=1).mmd2 for d in (0, 0.5, 2)]
        self.assertEqual(values[0], 0)
        self.assertTrue(values[0] < values[1] < values[2])

    def test_symmetry_order_and_unequal_sizes(self):
        rng = np.random.default_rng(44)
        x, y = rng.normal(size=(9, 5)), rng.normal(size=(13, 5))
        a = biased_rbf_mmd2(x, y, bandwidth_h=0.8, block_size=4).mmd2
        b = biased_rbf_mmd2(y[::-1], x[::-1], bandwidth_h=0.8, block_size=3).mmd2
        self.assertAlmostEqual(a, b, delta=2e-14)

    def test_reference_and_block_sizes(self):
        rng = np.random.default_rng(42)
        for n, m, p in ((1, 3, 1), (7, 11, 4), (35, 33, 9), (3, 5, 2048)):
            x, y = rng.normal(size=(n, p)), rng.normal(size=(m, p))
            for h in (0.01, 0.7, 10):
                expected = reference(x, y, h)
                for block in (1, 7, 32, 128):
                    with self.subTest(n=n, m=m, p=p, h=h, block=block):
                        got = biased_rbf_mmd2(x, y, bandwidth_h=h, block_size=block)
                        self.assertAlmostEqual(got.mmd2, expected, delta=2e-14)
                        self.assertTrue(0 <= got.mmd2 <= 2)

    def test_duplicate_replication_preserves_empirical_measure(self):
        x, y = np.array([[0.], [1.]]), np.array([[2.], [3.], [2.]])
        expected = biased_rbf_mmd2(x, y, bandwidth_h=1).mmd2
        got = biased_rbf_mmd2(np.repeat(x, 3, axis=0), np.repeat(y, 2, axis=0), bandwidth_h=1).mmd2
        self.assertAlmostEqual(got, expected, places=14)

    def test_limits_and_underflow(self):
        result = biased_rbf_mmd2([[0.]], [[100.]], bandwidth_h=1)
        self.assertEqual(result.mmd2, 2)
        self.assertEqual(result.mean_xy, 0)
        self.assertEqual(biased_rbf_mmd2([[1.]], [[1.]], bandwidth_h=np.nextafter(0., 1.)).mmd2, 0)
        self.assertTrue(math.isfinite(biased_rbf_mmd2([[0.]], [[1.]], bandwidth_h=1e308).mmd2))

    def test_large_common_offset_avoids_gram_cancellation(self):
        a = biased_rbf_mmd2([[0., 0.]], [[1., 0.]], bandwidth_h=1).mmd2
        b = biased_rbf_mmd2([[1e12, 1e12]], [[1e12+1, 1e12]], bandwidth_h=1).mmd2
        self.assertEqual(a, b)

    def test_roundoff_only_clipping_and_diagnostics(self):
        result = _finish(0.5, 0.5, 0.5 + np.finfo(float).eps)
        self.assertLess(result.raw_mmd2, 0)
        self.assertEqual(result.mmd2, 0)
        self.assertTrue(result.negative_roundoff_clipped)
        with self.assertRaises(FloatingPointError):
            _finish(0., 0., 1e-6)
        with self.assertRaises(FloatingPointError):
            _finish(float('nan'), 1, 1)
        with self.assertRaises(FloatingPointError):
            _finish(1.1, 1, 0)

    def test_intermediate_overflow_fails(self):
        for x, y, h in (([[1e308]], [[-1e308]], 1), ([[0.]], [[1e200]], 1), ([[0.]], [[10.]], 1e-320)):
            with self.assertRaises(FloatingPointError):
                biased_rbf_mmd2(x, y, bandwidth_h=h)

    def test_invalid_embeddings(self):
        for x in ([], [1, 2], np.zeros((2, 0)), np.zeros((1, 2, 3)), [[float('nan')]], [[float('inf')]], [[1j]], [['1']], [[True]], [[None]], [[1], [1, 2]]):
            with self.subTest(x=str(x)), self.assertRaises(ValueError):
                biased_rbf_mmd2(x, [[0]], bandwidth_h=1)
        with self.assertRaises(ValueError):
            biased_rbf_mmd2([[1, 2]], [[1]], bandwidth_h=1)

    def test_invalid_parameters(self):
        for h in (0, -1, math.nan, math.inf, True, '1', [1], 1j):
            with self.assertRaises(ValueError):
                biased_rbf_mmd2([[0]], [[1]], bandwidth_h=h)
        for block in (0, -1, 1.5, True, '3'):
            with self.assertRaises(ValueError):
                biased_rbf_mmd2([[0]], [[1]], bandwidth_h=1, block_size=block)

    def test_cast_float64_and_no_input_mutation(self):
        x = np.array([[0, 1], [2, 3]], dtype=np.float32)
        original = x.copy()
        x.setflags(write=False)
        result = biased_rbf_mmd2(x, x+1, bandwidth_h=1)
        self.assertAlmostEqual(result.mmd2, reference(x, x+1, 1), places=14)
        np.testing.assert_array_equal(x, original)

    def test_equal_family_aggregation_not_sample_weighting(self):
        x = {'many': np.zeros((100, 1)), 'few': np.zeros((1, 1))}
        y = {'many': np.zeros((50, 1)), 'few': np.ones((2, 1))}
        r = class_conditional_mmd2(x, y, support=['few', 'many'], bandwidth_h=1)
        self.assertEqual(r.support, ('few', 'many'))
        self.assertAlmostEqual(r.mmd2, (2*(1-math.exp(-0.5)))/2, places=14)
        self.assertEqual(r.per_family[1].mmd2, 0)

    def test_missing_support_never_silently_dropped(self):
        for support in ([], ['a', 'a'], ['a', 'missing'], 'a', {'a'}, [1], ['']):
            with self.assertRaises(ValueError):
                class_conditional_mmd2({'a': [[0]]}, {'a': [[1]]}, support=support, bandwidth_h=1)
        with self.assertRaises(ValueError):
            class_conditional_mmd2({'a': [[0]], 'b': [[0, 1]]}, {'a': [[1]], 'b': [[1, 2]]}, support=['a','b'], bandwidth_h=1)

    def test_explicit_support_ignores_unselected_classes(self):
        r = class_conditional_mmd2({'a': [[0]], 'outside': [[math.nan]]}, {'a': [[1]]}, support=['a'], bandwidth_h=1)
        self.assertAlmostEqual(r.mmd2, 2*(1-math.exp(-0.5)), places=14)


if __name__ == '__main__':
    unittest.main()
