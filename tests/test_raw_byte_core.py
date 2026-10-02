import unittest

from scripts.raw_byte_core import bytes_to_layout


class TestRawByteCore(unittest.TestCase):
    def test_preserves_byte_order_and_marks_padding(self):
        data = bytes([0, 64, 128, 255, 32])

        values, mask = bytes_to_layout(data, row_width=4)

        self.assertEqual(
            values,
            [
                [0, 64, 128, 255],
                [32, 0, 0, 0],
            ],
        )
        self.assertEqual(
            mask,
            [
                [1, 1, 1, 1],
                [1, 0, 0, 0],
            ],
        )
        self.assertEqual(sum(sum(row) for row in mask), len(data))

    def test_real_zero_byte_is_valid(self):
        values, mask = bytes_to_layout(bytes([0]), row_width=2)

        self.assertEqual(values, [[0, 0]])
        self.assertEqual(mask, [[1, 0]])

    def test_complete_rows_need_no_padding(self):
        data = bytes([10, 20, 30, 40])

        values, mask = bytes_to_layout(data, row_width=2)

        self.assertEqual(values, [[10, 20], [30, 40]])
        self.assertEqual(mask, [[1, 1], [1, 1]])

    def test_rejects_empty_input(self):
        with self.assertRaises(ValueError):
            bytes_to_layout(b"", row_width=4)

    def test_rejects_invalid_row_width(self):
        for width in (0, -1, 2.5):
            with self.subTest(width=width):
                with self.assertRaises(ValueError):
                    bytes_to_layout(bytes([1]), row_width=width)


if __name__ == "__main__":
    unittest.main()