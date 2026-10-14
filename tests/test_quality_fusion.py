import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from scripts.calibrate_sbsmi import fit_temperature
from scripts.february_group_audit import sha, save_csv
from scripts.quality_fusion import aligned_folds, cross_fit, estimate_quality, weighted_fusion
from scripts.uniform_fusion import calibrated_log_probabilities, load_bundle


def fixture():
    ids = np.array([f"{i:064x}" for i in range(10)])
    labels = np.arange(10) % 2
    logits = np.random.default_rng(42).normal(size=(10, 2))
    bundles = [{"ids": ids.copy(), "classes": np.array(["a", "b"]), "labels": labels.copy(),
                "logits": logits.copy() * (v + 1), "temperature": 1.0} for v in range(3)]
    manifest = [{"sha": ids[i], "family": "ab"[labels[i]], "fold": i % 5,
                 "timestamp": "2020-02-01 00:00:00+00:00", "retained": True,
                 "group_id": f"g{i}", "disarmed_content_sha256": ids[i]} for i in range(10)]
    return bundles, manifest


class TestQualityFusion(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_id_join_not_position_and_reject_mismatch(self):
        bundles, manifest = fixture()
        np.testing.assert_array_equal(aligned_folds(bundles, list(reversed(manifest))), np.arange(10) % 5)
        bundles[1]["ids"] = bundles[1]["ids"][::-1]
        with self.assertRaises(ValueError):
            aligned_folds(bundles, manifest)

    def test_reject_excluded_march_duplicate_group_content_and_label_mismatch(self):
        changes = [{"retained": False}, {"timestamp": "2020-03-01"}, {"group_id": "g1"},
                   {"disarmed_content_sha256": f"{1:064x}"}, {"family": "b"}, {"fold": 5}]
        for change in changes:
            bundles, manifest = fixture()
            manifest[0].update(change)
            with self.assertRaises(ValueError):
                aligned_folds(bundles, manifest)

    def test_fitter_receives_only_other_folds_and_exact_coverage(self):
        bundles, manifest = fixture()
        calls = []
        def fit(scores, labels):
            calls.append((scores.numpy().copy(), labels.numpy().copy()))
            return 2.0, False
        logs, folds, temperatures = cross_fit(bundles, manifest, fit)
        self.assertEqual(len(calls), 15)
        for v in range(3):
            for f in range(5):
                np.testing.assert_array_equal(calls[v * 5 + f][0], bundles[v]["logits"][folds != f])
                np.testing.assert_array_equal(calls[v * 5 + f][1], bundles[v]["labels"][folds != f])
            np.testing.assert_allclose(logs[v], calibrated_log_probabilities(bundles[v]["logits"], 2), atol=1e-14)
        self.assertTrue(all(t["fit_samples"] == 8 and t["score_samples"] == 2 for t in temperatures))

    def test_real_fitter_matches_manual_fold_fit(self):
        bundles, manifest = fixture()
        logs, folds, temperatures = cross_fit(bundles, manifest)
        manual, _ = fit_temperature(torch.tensor(bundles[0]["logits"][folds != 0]),
                                    torch.tensor(bundles[0]["labels"][folds != 0]))
        self.assertEqual(temperatures[0]["temperature"], manual)
        np.testing.assert_allclose(logs[0, folds == 0], calibrated_log_probabilities(bundles[0]["logits"][folds == 0], manual))

    def test_score_fold_labels_do_not_change_its_temperature(self):
        bundles, manifest = fixture()
        first = cross_fit(bundles, manifest)[2][0]["temperature"]
        for b in bundles:
            b["labels"][[0, 5]] = 1 - b["labels"][[0, 5]]
        for i in (0, 5):
            manifest[i]["family"] = "ab"[bundles[0]["labels"][i]]
        self.assertEqual(first, cross_fit(bundles, manifest)[2][0]["temperature"])

    def test_balanced_nll_known_answer_and_fixed_softmax(self):
        probabilities = np.array([[[.8, .2], [.6, .4], [.3, .7]],
                                  [[.5, .5], [.5, .5], [.5, .5]],
                                  [[.2, .8], [.4, .6], [.7, .3]]])
        result = estimate_quality(np.log(probabilities), np.array([0, 0, 1]), [0, 1])
        expected = np.array([(-np.log(.8) - np.log(.6)) / 4 - np.log(.7) / 2,
                             -np.log(.5), (-np.log(.2) - np.log(.4)) / 4 - np.log(.3) / 2])
        np.testing.assert_allclose(result["class_balanced_nll"], expected)
        np.testing.assert_allclose(result["weights"], np.exp(-expected) / np.exp(-expected).sum())
        self.assertNotEqual(result["standard_nll_all_february"][0], result["class_balanced_nll"][0])

    def test_fixed_support_missing_class_rejected(self):
        logs = np.full((3, 2, 2), -np.log(2))
        with self.assertRaises(ValueError):
            estimate_quality(logs, np.array([0, 0]), [0, 1])

    def test_weighted_probability_average_and_zero_weight(self):
        logs = np.log(np.array([[[.9, .1]], [[.2, .8]], [[.5, .5]]]))
        np.testing.assert_allclose(np.exp(weighted_fusion(logs, [0, .25, .75])), [[.425, .575]])
        for weights in ([1, 1, 1], [-1, 1, 1], [float("nan"), 0, 1]):
            with self.assertRaises(ValueError):
                weighted_fusion(logs, weights)

    def test_file_integration_and_march_labels_not_an_input_to_weights(self):
        bundles, manifest = fixture()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = []
            for v, b in enumerate(bundles):
                path = root / f"feb_{v}.npz"
                np.savez(path, sample_identifiers=b["ids"], class_names=b["classes"], labels=b["labels"], logits=b["logits"])
                paths.append(path)
            fp = root / "folds.csv"
            save_csv(fp, manifest, list(manifest[0]))
            with fp.open(newline="", encoding="utf-8") as stream:
                loaded_manifest = list(csv.DictReader(stream))
            march = root / "march_labels.npy"
            def run():
                loaded = [load_bundle(p, sha(p), 1.0) for p in paths]
                logs, _, _ = cross_fit(loaded, loaded_manifest)
                return estimate_quality(logs, loaded[0]["labels"], [0, 1])["weights"]
            np.save(march, [0, 0])
            before = run()
            np.save(march, [1, 1, 1, 1])
            self.assertEqual(before, run())


if __name__ == "__main__":
    unittest.main()
