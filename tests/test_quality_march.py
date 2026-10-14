import unittest
import numpy as np

from scripts.evaluate_quality_march import compare


def batches():
    return [{"ids": np.array(["a", "b"]), "classes": np.array(["x", "y", "z"]),
             "labels": np.array([0, 1]), "temperature": 1.0,
             "logits": np.log(np.array(p))} for p in
            ([[.8, .1, .1], [.2, .7, .1]], [[.2, .7, .1], [.6, .3, .1]], [[.5, .4, .1], [.1, .8, .1]])]


class TestQualityMarch(unittest.TestCase):
    def test_probability_mixture_and_all_metrics_including_absent_class(self):
        b = batches()
        names, logs, metrics = compare(b, [.4, .3, .3])
        expected = sum(w * np.exp(v["logits"]) for w, v in zip([.4, .3, .3], b))
        np.testing.assert_allclose(np.exp(logs[-1]), expected, rtol=0, atol=1e-15)
        self.assertEqual(len(names), 5)
        self.assertAlmostEqual(metrics[names[-1]]["macro_f1_fixed"], 2 / 3)
        self.assertEqual(metrics[names[-1]]["macro_f1_supported"], 1)
        self.assertEqual(metrics[names[-1]]["supported_classes"], 2)

    def test_labels_change_scores_not_predictions_or_weights(self):
        b = batches()
        w = [.4017215600259309, .2866850774030112, .3115933625710579]
        _, logs, metrics = compare(b, w)
        for v in b:
            v["labels"] = np.array([1, 0])
        _, after, changed = compare(b, w)
        np.testing.assert_array_equal(logs, after)
        self.assertNotEqual(metrics["quality-only-calibrated"]["accuracy"], changed["quality-only-calibrated"]["accuracy"])
        self.assertEqual(w, [.4017215600259309, .2866850774030112, .3115933625710579])

    def test_alignment_failure_blocks_evaluation(self):
        for key in ("ids", "classes", "labels"):
            b = batches()
            b[1][key] = b[1][key][::-1]
            with self.assertRaises(ValueError):
                compare(b, [1/3]*3)


if __name__ == "__main__":
    unittest.main()
