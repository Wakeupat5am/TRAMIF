"""CPU regression tests for the existing scalar calibration implementation."""
import math
import unittest

import torch
import torch.nn.functional as F

from scripts.calibrate_sbsmi import (
    T_MAX, T_MIN, fit_temperature, probability_metrics,
)


class TestCalibration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def test_known_optimum_and_metrics(self):
        logits = torch.tensor([[math.log(9), 0.0]] * 4, dtype=torch.float64)
        labels = torch.tensor([0, 0, 0, 1])
        temperature, boundary = fit_temperature(logits, labels)
        self.assertFalse(boundary)
        self.assertAlmostEqual(temperature, 2.0, places=8)
        result = probability_metrics(logits, labels, temperature)
        self.assertAlmostEqual(result['nll'], -(.75 * math.log(.75) + .25 * math.log(.25)))
        self.assertAlmostEqual(result['brier'], .375)
        self.assertAlmostEqual(result['ece_15'], 0.0)
        self.assertEqual(result['accuracy'], .75)
        self.assertEqual(sum(row['count'] for row in result['bins']), 4)

    def test_positive_temperature_preserves_predictions(self):
        logits = torch.tensor([[3., 2., -1.], [-2., 0., 4.], [1., 1., 0.]], dtype=torch.float64)
        expected = logits.argmax(1)
        for temperature in (T_MIN, 1., 2.5, T_MAX):
            self.assertTrue(torch.equal((logits / temperature).argmax(1), expected))

    def test_fit_does_not_increase_fit_nll(self):
        logits = torch.tensor([[9., 1., 0.], [1., 8., 0.], [0., 1., 7.], [6., 1., 0.]], dtype=torch.float64)
        labels = torch.tensor([0, 1, 2, 2])
        temperature, _ = fit_temperature(logits, labels)
        self.assertLessEqual(F.cross_entropy(logits / temperature, labels).item(),
                             F.cross_entropy(logits, labels).item() + 1e-12)

    def test_perfect_predictions_reach_lower_bound(self):
        logits = torch.tensor([[1., 0.], [0., 1.]], dtype=torch.float64)
        temperature, boundary = fit_temperature(logits, torch.tensor([0, 1]))
        self.assertTrue(boundary)
        self.assertEqual(temperature, T_MIN)

    def test_wrong_predictions_reach_upper_bound(self):
        logits = torch.tensor([[1., 0.], [0., 1.]], dtype=torch.float64)
        temperature, boundary = fit_temperature(logits, torch.tensor([1, 0]))
        self.assertTrue(boundary)
        self.assertEqual(temperature, T_MAX)

    def test_logit_translation_does_not_change_calibration(self):
        logits = torch.tensor([[2., 0.], [2., 0.], [2., 0.], [2., 0.]], dtype=torch.float64)
        labels = torch.tensor([0, 0, 0, 1])
        shifted = logits + torch.tensor([[1000.], [-1000.], [42.], [-42.]])
        original, _ = fit_temperature(logits, labels)
        translated, _ = fit_temperature(shifted, labels)
        self.assertAlmostEqual(original, translated, places=10)


if __name__ == '__main__':
    unittest.main()
