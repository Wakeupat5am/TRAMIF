import unittest
from math import log2

from scripts.entropy_core import local_entropy_values


class TestLocalEntropy(unittest.TestCase):
    def assert_values_close(self, actual, expected):
        self.assertEqual(len(actual), len(expected))

        for offset, (got, want) in enumerate(zip(actual, expected)):
            with self.subTest(offset=offset):
                self.assertAlmostEqual(got, want)

    def test_short_file_uses_only_existing_bytes(self):
        result = local_entropy_values(
            bytes([0, 255]),
            window_size=4,
            stride=2,
        )

        # One window with two equally frequent values: H = 1 bit.
        self.assert_values_close(result, [1 / 8, 1 / 8])

    def test_overlapping_windows_are_averaged(self):
        result = local_entropy_values(
            bytes([0, 0, 0, 0, 1, 1]),
            window_size=4,
            stride=2,
        )

        # Window [0:4]: H = 0.
        # Window [2:6]: H = 1.
        # Offsets 2 and 3 receive the average of both windows.
        self.assert_values_close(
            result,
            [0, 0, 1 / 16, 1 / 16, 1 / 8, 1 / 8],
        )

    def test_partial_final_window_has_no_padding(self):
        result = local_entropy_values(
            bytes([0, 0, 1, 1, 2]),
            window_size=4,
            stride=2,
        )

        # First window: [0, 0, 1, 1], H = 1.
        # Final window: [1, 1, 2], probabilities 2/3 and 1/3.
        tail_entropy = (
            -(2 / 3) * log2(2 / 3)
            -(1 / 3) * log2(1 / 3)
        )

        self.assert_values_close(
            result,
            [
                1 / 8,
                1 / 8,
                (1 + tail_entropy) / 16,
                (1 + tail_entropy) / 16,
                tail_entropy / 8,
            ],
        )

    def test_stops_when_first_window_reaches_end(self):
        result = local_entropy_values(
            bytes([0, 0, 1, 1]),
            window_size=4,
            stride=2,
        )

        # No extra [1, 1] window after the full-length window.
        self.assert_values_close(result, [1 / 8] * 4)

    def test_constant_file_has_zero_entropy_everywhere(self):
        data = bytes([42]) * 513

        result = local_entropy_values(data)

        self.assertEqual(len(result), len(data))
        self.assertTrue(all(value == 0.0 for value in result))

    def test_rejects_invalid_window_parameters(self):
        for window_size, stride in (
            (0, 1),
            (-1, 1),
            (4, 0),
            (4, 5),
            (4.5, 2),
            (4, 1.5),
        ):
            with self.subTest(window_size=window_size, stride=stride):
                with self.assertRaises(ValueError):
                    local_entropy_values(
                        bytes([0, 1]),
                        window_size=window_size,
                        stride=stride,
                    )

    def test_rejects_empty_input(self):
        with self.assertRaises(ValueError):
            local_entropy_values(b"")


if __name__ == "__main__":
    unittest.main()