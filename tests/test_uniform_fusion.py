import copy
import hashlib
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.uniform_fusion import load_bundle, fuse_uniform, classification_metrics


def bundles():
    return [{"ids": np.array(["a", "b"]), "classes": np.array(["one", "two"]),
             "labels": np.array([0, 1]), "logits": np.log(np.array([[p, 1-p], [1-p, p]])),
             "temperature": 1.0} for p in (.9, .6, .3)]


class TestUniformFusion(unittest.TestCase):
    def test_probability_mean_not_logit_mean(self):
        _, fused = fuse_uniform(bundles())
        np.testing.assert_allclose(np.exp(fused), [[.6, .4], [.4, .6]], atol=1e-14)

    def test_temperature_applied_before_average(self):
        inputs = bundles()
        for batch in inputs:
            batch['logits'] = np.log(np.array([[9., 1.], [1., 9.]]))
            batch['temperature'] = 2.0
        _, fused = fuse_uniform(inputs)
        np.testing.assert_allclose(np.exp(fused), [[.75, .25], [.25, .75]], atol=1e-14)

    def test_rejects_id_order_set_and_duplicates(self):
        for ids in (["b", "a"], ["a", "c"], ["a", "a"]):
            inputs = bundles()
            inputs[1]['ids'] = np.array(ids)
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                fuse_uniform(inputs)

    def test_rejects_class_and_label_mismatch(self):
        for key, value in [('classes', np.array(['two', 'one'])), ('labels', np.array([1, 0]))]:
            inputs = bundles()
            inputs[2][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                fuse_uniform(inputs)

    def test_labels_cannot_change_fusion(self):
        inputs = bundles()
        _, before = fuse_uniform(inputs)
        for batch in inputs:
            batch['labels'] = 1 - batch['labels']
        _, after = fuse_uniform(inputs)
        np.testing.assert_array_equal(before, after)

    def test_rejects_invalid_numeric_inputs(self):
        for key, value in [('temperature', 0.), ('temperature', np.nan),
                           ('temperature', True), ('logits', np.full((2, 2), np.inf)),
                           ('labels', np.array([-1, 1]))]:
            inputs = bundles()
            inputs[0][key] = value
            with self.subTest(key=key, value=str(value)), self.assertRaises(ValueError):
                fuse_uniform(inputs)
        with self.assertRaises(ValueError):
            fuse_uniform(bundles()[:2])

    def test_extreme_logits_have_finite_nll(self):
        inputs = bundles()
        for batch in inputs:
            batch['logits'] = np.array([[1000., -1000.], [1000., -1000.]])
        _, fused = fuse_uniform(inputs)
        result = classification_metrics(fused, inputs[0]['labels'])
        self.assertAlmostEqual(result['nll'], 1000.)
        self.assertAlmostEqual(result['accuracy'], .5)

    def test_known_metrics_include_absent_class(self):
        logp = np.log([[.75, .20, .05], [.75, .20, .05]])
        result = classification_metrics(logp, np.array([0, 1]))
        self.assertEqual(result['confusion_matrix'], [[1, 0, 0], [1, 0, 0], [0, 0, 0]])
        self.assertAlmostEqual(result['macro_f1_supported'], 1/3)
        self.assertAlmostEqual(result['macro_f1_fixed'], 2/9)
        self.assertAlmostEqual(result['balanced_accuracy'], .5)
        self.assertAlmostEqual(result['nll'], -(np.log(.75) + np.log(.2))/2)
        self.assertAlmostEqual(result['brier'], .655)
        self.assertAlmostEqual(result['ece_15'], .25)

    def test_file_loading_alignment_and_checksum_integration(self):
        with tempfile.TemporaryDirectory() as directory:
            loaded = []
            for index, batch in enumerate(bundles()):
                path = Path(directory) / f'{index}.npz'
                np.savez(path, sample_identifiers=batch['ids'], class_names=batch['classes'],
                         labels=batch['labels'], logits=batch['logits'])
                checksum = hashlib.sha256(path.read_bytes()).hexdigest()
                loaded.append(load_bundle(path, checksum, 1.0))
            _, fused = fuse_uniform(loaded)
            self.assertEqual(classification_metrics(fused, loaded[0]['labels'])['accuracy'], 1.)
            with self.assertRaises(ValueError):
                load_bundle(path, '0'*64, 1.)
            bad = copy.deepcopy(loaded)
            bad[1]['ids'] = bad[1]['ids'][::-1]
            with self.assertRaises(ValueError):
                fuse_uniform(bad)


if __name__ == '__main__':
    unittest.main()
