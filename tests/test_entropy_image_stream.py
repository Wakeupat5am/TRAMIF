import io
import unittest

import numpy as np

from scripts.entropy_core import local_entropy_values
from scripts.entropy_image_stream import stream_to_entropy_image
from scripts.masked_area import masked_area_resize
from scripts.raw_byte_core import bytes_to_layout


def reference_image(data):
    """Use the original, full-memory implementation on small test inputs."""
    entropy_values = local_entropy_values(data)
    layout, mask = bytes_to_layout(data, row_width=256)

    for offset, value in enumerate(entropy_values):
        row, column = divmod(offset, 256)
        layout[row][column] = value

    return np.asarray(
        masked_area_resize(layout, mask),
        dtype=np.float64,
    )


class ShortReadStream(io.BytesIO):
    def read(self, size=-1):
        if size < 0:
            size = 7
        return super().read(min(size, 7))


class TestEntropyImageStream(unittest.TestCase):
    def assert_matches_reference(self, data, chunk_size):
        actual = stream_to_entropy_image(
            io.BytesIO(data),
            file_size=len(data),
            chunk_size=chunk_size,
        )
        expected = reference_image(data)

        self.assertEqual(actual.shape, (64, 64))
        self.assertEqual(actual.dtype, np.dtype("float64"))
        self.assertTrue(np.isfinite(actual).all())
        self.assertTrue((actual >= 0).all())
        self.assertTrue((actual <= 1).all())

        np.testing.assert_allclose(
            actual,
            expected,
            rtol=0,
            atol=1e-12,
        )

    def test_matches_reference_across_sizes_and_chunks(self):
        for size in (1, 2, 127, 255, 256, 257, 383, 384, 385, 769, 16385):
            data = bytes(
                (index * 73 + index // 7) % 256
                for index in range(size)
            )

            for chunk_size in (1, 257, 65536):
                with self.subTest(size=size, chunk_size=chunk_size):
                    self.assert_matches_reference(data, chunk_size)

    def test_crosses_internal_buffer_boundary(self):
        # Internal buffer holds 65,536 entropy values.
        # Cover an exact flush and a flush followed by a padded final row.
        for size in (65536, 65539):
            data = bytes(
                (index * 29 + index // 13) % 256
                for index in range(size)
            )
            with self.subTest(size=size):
                self.assert_matches_reference(data, chunk_size=257)

    def test_padding_does_not_darken_valid_entropy(self):
        # Two equally frequent byte values: H = 1 bit, H/8 = 0.125.
        data = b"\x00\xff"
        actual = stream_to_entropy_image(
            io.BytesIO(data),
            file_size=len(data),
        )

        np.testing.assert_array_equal(
            actual[:, 0],
            np.full(64, 1 / 8),
        )
        np.testing.assert_array_equal(
            actual[:, 1:],
            np.zeros((64, 63)),
        )

    def test_uniform_byte_distribution_produces_one(self):
        # One complete window containing all 256 distinct byte values.
        data = bytes(range(256))
        actual = stream_to_entropy_image(
            io.BytesIO(data),
            file_size=len(data),
        )

        np.testing.assert_array_equal(
            actual,
            np.ones((64, 64)),
        )

    def test_constant_input_produces_zero(self):
        data = b"\x2a" * 513
        actual = stream_to_entropy_image(
            io.BytesIO(data),
            file_size=len(data),
        )

        np.testing.assert_array_equal(
            actual,
            np.zeros((64, 64)),
        )

    def test_short_reads_preserve_image(self):
        data = bytes(range(256)) * 3 + b"\x00\xff"
        actual = stream_to_entropy_image(
            ShortReadStream(data),
            file_size=len(data),
        )

        np.testing.assert_allclose(
            actual,
            reference_image(data),
            rtol=0,
            atol=1e-12,
        )

    def test_rejects_incorrect_declared_size(self):
        for declared_size in (2, 4):
            with self.subTest(declared_size=declared_size):
                with self.assertRaises(ValueError):
                    stream_to_entropy_image(
                        io.BytesIO(b"abc"),
                        file_size=declared_size,
                    )

    def test_rejects_invalid_parameters(self):
        for file_size in (0, -1, True, 1.5):
            with self.subTest(file_size=file_size):
                with self.assertRaises(ValueError):
                    stream_to_entropy_image(
                        io.BytesIO(b"abc"),
                        file_size=file_size,
                    )

        for chunk_size in (0, -1, True, 1.5):
            with self.subTest(chunk_size=chunk_size):
                with self.assertRaises(ValueError):
                    stream_to_entropy_image(
                        io.BytesIO(b"abc"),
                        file_size=3,
                        chunk_size=chunk_size,
                    )


if __name__ == "__main__":
    unittest.main()