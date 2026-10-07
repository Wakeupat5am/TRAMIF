import io
import unittest

import numpy as np

from scripts.masked_area import masked_area_resize
from scripts.raw_byte_core import bytes_to_layout
from scripts.raw_byte_stream import stream_to_raw_byte


def reference_image(data):
    values, mask = bytes_to_layout(data, row_width=256)
    resized = masked_area_resize(values, mask)

    return np.asarray(resized, dtype=np.float64) / 255.0


class ShortReadStream(io.BytesIO):
    """Simulate a stream that returns fewer bytes than requested."""

    def read(self, size=-1):
        if size < 0:
            size = 7
        return super().read(min(size, 7))


class TestRawByteStream(unittest.TestCase):
    def test_matches_reference_across_sizes_and_chunks(self):
        samples = [
            b"\x00",
            bytes([0, 64, 255, 32, 128]),
            bytes(range(255)),
            bytes(range(256)),
            bytes(range(256)) + b"\xff",
            bytes(
                (index * 73 + index // 7) % 256
                for index in range(769)
            ),
            bytes(
                (index * 29 + index // 13) % 256
                for index in range(16385)
            ),
        ]

        for data in samples:
            expected = reference_image(data)

            for chunk_size in (1, 255, 256, 257, 65536):
                with self.subTest(
                    input_bytes=len(data),
                    chunk_size=chunk_size,
                ):
                    actual = stream_to_raw_byte(
                        io.BytesIO(data),
                        file_size=len(data),
                        chunk_size=chunk_size,
                    )

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

    def test_padding_does_not_darken_valid_bytes(self):
        data = b"\xff\xff\xff"

        actual = stream_to_raw_byte(
            io.BytesIO(data),
            file_size=len(data),
        )

        # The first output column contains three real 255 bytes.
        np.testing.assert_array_equal(
            actual[:, 0],
            np.ones(64),
        )

        # All other columns have no real input bytes.
        np.testing.assert_array_equal(
            actual[:, 1:],
            np.zeros((64, 63)),
        )

    def test_real_zero_contributes_to_average(self):
        data = b"\x00\xff"

        actual = stream_to_raw_byte(
            io.BytesIO(data),
            file_size=len(data),
        )

        # The real zero must participate: (0 + 255) / 2 / 255 = 0.5.
        np.testing.assert_array_equal(
            actual[:, 0],
            np.full(64, 0.5),
        )

    def test_short_reads_preserve_byte_order(self):
        data = bytes(range(256)) * 3 + b"\x00\xff"

        actual = stream_to_raw_byte(
            ShortReadStream(data),
            file_size=len(data),
            chunk_size=65536,
        )

        np.testing.assert_allclose(
            actual,
            reference_image(data),
            rtol=0,
            atol=1e-12,
        )

    def test_rejects_invalid_parameters(self):
        for file_size in (0, -1, True, 1.5):
            with self.subTest(file_size=file_size):
                with self.assertRaises(ValueError):
                    stream_to_raw_byte(
                        io.BytesIO(b"abc"),
                        file_size=file_size,
                    )

        for chunk_size in (0, -1, True, 1.5):
            with self.subTest(chunk_size=chunk_size):
                with self.assertRaises(ValueError):
                    stream_to_raw_byte(
                        io.BytesIO(b"abc"),
                        file_size=3,
                        chunk_size=chunk_size,
                    )

    def test_rejects_incorrect_declared_size(self):
        for declared_size in (2, 4):
            with self.subTest(declared_size=declared_size):
                with self.assertRaises(ValueError):
                    stream_to_raw_byte(
                        io.BytesIO(b"abc"),
                        file_size=declared_size,
                    )


if __name__ == "__main__":
    unittest.main()