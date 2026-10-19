import unittest
import tempfile
from pathlib import Path
import numpy as np
from scripts.run_frozen_descriptive_drift import family_matrices, output_checksums
from scripts.episode_training_dataset import sha256


class TestFrozenDriftBinding(unittest.TestCase):
    def test_output_manifest_never_hashes_itself(self):
        with tempfile.TemporaryDirectory() as root:
            directory = Path(root)
            (directory/'report.json').write_bytes(b'result')
            (directory/'checksums.json').write_bytes(b'')
            self.assertEqual(output_checksums(directory), {'report.json': sha256(b'result')})

    def test_preserves_row_identity_and_support_order(self):
        rows = [dict(sha='a', family='b'), dict(sha='b', family='a'), dict(sha='c', family='b')]
        matrix = np.array([[1., 2.], [3., 4.], [5., 6.]])
        result = family_matrices(matrix, rows, ['a', 'b'])
        np.testing.assert_array_equal(result['a'], matrix[[1]])
        np.testing.assert_array_equal(result['b'], matrix[[0, 2]])

    def test_rejects_missing_family_duplicates_nonfinite(self):
        rows = [dict(sha='a', family='x')]
        with self.assertRaises(ValueError): family_matrices(np.ones((1, 2)), rows, ['x', 'y'])
        with self.assertRaises(ValueError): family_matrices(np.ones((2, 2)), rows*2, ['x'])
        with self.assertRaises(ValueError): family_matrices(np.full((1, 2), np.nan), rows, ['x'])


if __name__ == '__main__': unittest.main()
