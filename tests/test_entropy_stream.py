import io
import unittest
from math import log2

import numpy as np

from scripts.entropy_core import local_entropy_values
from scripts.entropy_stream import iter_local_entropy


def collect_values(data, **kwargs):
    blocks = list(
        iter_local_entropy(
            io.BytesIO(data),
            file_size=len(data),
            **kwargs,
        )
    )
    return np.concatenate(blocks)


class ShortReadStream(io.BytesIO):
    """Return at most seven bytes, even when more were requested."""

    def read(self, size=-1):
        if size < 0:
            size = 7
        return super().read(min(size, 7))


class TestEntropyStream(unittest.TestCase):
    def test_matches_reference_across_sizes_and_chunks(self):
        sizes = (
            1, 2, 127, 128, 129,
            255, 256, 257,
            383, 384, 385,
            513, 769, 16385,
        )

        for size in sizes:
            data = bytes(
                (index * 73 + index // 7) % 256
                for index in range(size)
            )
            expected = np.asarray(
                local_entropy_values(data),
                dtype=np.float64,
            )

            for chunk_size in (1, 7, 127, 256, 257, 65536):
                with self.subTest(size=size, chunk_size=chunk_size):
                    actual = collect_values(data, chunk_size=chunk_size)

                    self.assertEqual(actual.shape, (size,))
                    self.assertEqual(actual.dtype, np.dtype("float64"))
                    self.assertTrue(np.isfinite(actual).all())
                    self.assertTrue((actual >= 0).all())
                    self.assertTrue((actual <= 1).all())

                    np.testing.assert_allclose(
                        actual, expected, rtol=0, atol=1e-12
                    )

    def test_overlapping_windows_are_averaged(self):
        data = bytes([0, 0, 0, 0, 1, 1])
        actual = collect_values(data, window_size=4, stride=2)

        np.testing.assert_allclose(
            actual,
            [0, 0, 1 / 16, 1 / 16, 1 / 8, 1 / 8],
            rtol=0,
            atol=1e-12,
        )

    def test_partial_final_window_has_no_padding(self):
        data = bytes([0, 0, 1, 1, 2])
        tail_entropy = (
            -(2 / 3) * log2(2 / 3)
            -(1 / 3) * log2(1 / 3)
        )

        actual = collect_values(data, window_size=4, stride=2)

        np.testing.assert_allclose(
            actual,
            [
                1 / 8,
                1 / 8,
                (1 + tail_entropy) / 16,
                (1 + tail_entropy) / 16,
                tail_entropy / 8,
            ],
            rtol=0,
            atol=1e-12,
        )

    def test_stops_at_first_window_reaching_end(self):
        actual = collect_values(
            bytes([0, 0, 1, 1]),
            window_size=4,
            stride=2,
        )

        np.testing.assert_array_equal(
            actual,
            np.full(4, 1 / 8),
        )

    def test_constant_input_has_zero_entropy(self):
        actual = collect_values(bytes([42]) * 513)

        np.testing.assert_array_equal(
            actual,
            np.zeros(513),
        )

    def test_short_reads_preserve_results(self):
        data = bytes(range(256)) * 3 + b"\x00\xff"
        blocks = list(
            iter_local_entropy(
                ShortReadStream(data),
                file_size=len(data),
                chunk_size=65536,
            )
        )

        np.testing.assert_allclose(
            np.concatenate(blocks),
            local_entropy_values(data),
            rtol=0,
            atol=1e-12,
        )

    def test_rejects_incorrect_declared_size(self):
        for declared_size in (2, 4):
            with self.subTest(declared_size=declared_size):
                with self.assertRaises(ValueError):
                    list(
                        iter_local_entropy(
                            io.BytesIO(b"abc"),
                            file_size=declared_size,
                        )
                    )

    def test_rejects_invalid_parameters(self):
        invalid_options = [
            {"file_size": 0},
            {"file_size": -1},
            {"file_size": True},
            {"file_size": 1.5},
            {"window_size": 0},
            {"window_size": True},
            {"window_size": 2.5},
            {"stride": 0},
            {"stride": True},
            {"stride": 1.5},
            {"window_size": 4, "stride": 5},
            {"chunk_size": 0},
            {"chunk_size": True},
            {"chunk_size": 1.5},
        ]

        for overrides in invalid_options:
            options = {"file_size": 3, **overrides}

            with self.subTest(options=options):
                with self.assertRaises(ValueError):
                    list(
                        iter_local_entropy(
                            io.BytesIO(b"abc"),
                            **options,
                        )
                    )


if __name__ == "__main__":
    unittest.main()