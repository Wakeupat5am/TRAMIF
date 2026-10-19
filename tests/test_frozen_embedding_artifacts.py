import io
import unittest
import numpy as np
from scripts.verify_frozen_embedding_artifacts import validate_matrix, validate_identity, VIEWS
from scripts.episode_training_dataset import sha256


class TestArtifactVerification(unittest.TestCase):
    def encoded(self, array):
        stream = io.BytesIO()
        np.save(stream, array, allow_pickle=False)
        return stream.getvalue()

    def test_payload_hash_detects_changed_values(self):
        array = np.zeros((2, 512), dtype=np.float32)
        meta = dict(shape=[2, 512], output_sha256=sha256(array.tobytes()))
        validate_matrix(self.encoded(array), meta, 2, 512)
        array[0, 0] = 1
        with self.assertRaises(ValueError): validate_matrix(self.encoded(array), meta, 2, 512)

    def test_shape_dtype_and_nonfinite_fail_closed(self):
        for array in (np.zeros((1, 512), dtype=np.float32), np.zeros((2, 512), dtype=np.float64),
                      np.full((2, 512), np.nan, dtype=np.float32)):
            meta = dict(shape=list(array.shape), output_sha256=sha256(array.tobytes()))
            with self.assertRaises(ValueError): validate_matrix(self.encoded(array), meta, 2, 512)

    def test_identity_rejects_reordering_and_mapping(self):
        rows = [dict(sha='a'*64), dict(sha='b'*64)]
        mapping = {'family': 0}
        result = dict(samples=rows, class_to_idx=mapping, alignment='PASS',
                      views=dict.fromkeys(VIEWS), source_content_sha256=['c'*64, 'd'*64])
        validate_identity(result, rows, mapping)
        with self.assertRaises(ValueError): validate_identity(result, list(reversed(rows)), mapping)
        with self.assertRaises(ValueError): validate_identity(result, rows, {'other': 0})


if __name__ == '__main__': unittest.main()
