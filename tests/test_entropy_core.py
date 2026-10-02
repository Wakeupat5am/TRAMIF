import unittest

from scripts.entropy_core import shannon_entropy


class TestEntropyCore(unittest.TestCase):
    def test_constant_bytes_have_zero_entropy(self):
        self.assertAlmostEqual(
            shannon_entropy(bytes([42]) * 256),
            0.0,
        )

    def test_two_equally_frequent_values_have_one_bit(self):
        data = bytes([0]) * 128 + bytes([255]) * 128

        self.assertAlmostEqual(shannon_entropy(data), 1.0)

    def test_four_equally_frequent_values_have_two_bits(self):
        data = bytes([0, 1, 2, 3]) * 64

        self.assertAlmostEqual(shannon_entropy(data), 2.0)

    def test_uniform_byte_distribution_has_eight_bits(self):
        data = bytes(range(256))

        self.assertAlmostEqual(shannon_entropy(data), 8.0)

    def test_order_does_not_change_entropy(self):
        grouped = bytes([0, 0, 0, 255, 255, 255])
        interleaved = bytes([0, 255, 0, 255, 0, 255])

        self.assertAlmostEqual(
            shannon_entropy(grouped),
            shannon_entropy(interleaved),
        )

    def test_empty_input_is_rejected(self):
        with self.assertRaises(ValueError):
            shannon_entropy(b"")


if __name__ == "__main__":
    unittest.main()