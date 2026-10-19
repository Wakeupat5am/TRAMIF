import tempfile
import unittest
from pathlib import Path
import numpy as np
from scripts.mmd_measurement_preparation import fit_train_normalizer
from scripts.prepare_frozen_drift_measurements import parameter_arrays, parameter_hash, restore_normalizer


class TestFrozenPreparation(unittest.TestCase):
    def test_parameter_roundtrip_and_ab_does_not_refit(self):
        rows = [dict(sha=f'{i:064x}', family='a', stage='train', timestamp='2019-09-01T00:00:00+00:00') for i in range(3)]
        values = {r['sha']: np.array([float(i), 7.]) for i, r in enumerate(rows)}
        norm = fit_train_normalizer(rows, values, case='E1', families=['a'])
        metadata = dict(case='E1', families=['a'], reference_sha256=norm.reference_sha256,
                        embeddings_sha256=norm.embeddings_sha256, rank=norm.rank,
                        condition_number=norm.condition_number, parameters_payload_sha256=parameter_hash(norm))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'parameters.npz'
            np.savez(path, **parameter_arrays(norm))
            restored = restore_normalizer(path, metadata)
            before = parameter_hash(restored)
            for ab in ([[100., 7.]], [[-500., 7.], [200., 7.]]):
                np.testing.assert_array_equal(norm.transform(ab), restored.transform(ab))
            self.assertEqual(before, parameter_hash(restored))
            altered = dict(metadata, parameters_payload_sha256='0'*64)
            with self.assertRaises(ValueError): restore_normalizer(path, altered)


if __name__ == '__main__': unittest.main()
