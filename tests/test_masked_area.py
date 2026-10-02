import unittest

from scripts.masked_area import masked_area_resize
from scripts.raw_byte_core import bytes_to_layout


class TestMaskedArea(unittest.TestCase):
    def test_padding_does_not_darken_average(self):
        result = masked_area_resize(
            values=[[100, 200, 0, 0]],
            mask=[[1, 1, 0, 0]],
            output_height=1,
            output_width=1,
        )

        self.assertEqual(result, [[150.0]])

    def test_real_zero_contributes_to_average(self):
        result = masked_area_resize(
            values=[[0, 200]],
            mask=[[1, 1]],
            output_height=1,
            output_width=1,
        )

        self.assertEqual(result, [[100.0]])

    def test_fractional_overlap_uses_area_weights(self):
        result = masked_area_resize(
            values=[[0, 120, 240]],
            mask=[[1, 1, 1]],
            output_height=1,
            output_width=2,
        )

        # Left: (0 * 1 + 120 * 0.5) / 1.5 = 40
        # Right: (120 * 0.5 + 240 * 1) / 1.5 = 200
        self.assertEqual(result, [[40.0, 200.0]])

    def test_vertical_overlap_uses_area_weights(self):
        result = masked_area_resize(
            values=[[0], [120], [240]],
            mask=[[1], [1], [1]],
            output_height=2,
            output_width=1,
        )

        self.assertEqual(result, [[40.0], [200.0]])

    def test_region_without_valid_data_becomes_zero(self):
        result = masked_area_resize(
            values=[[100, 200]],
            mask=[[1, 0]],
            output_height=1,
            output_width=2,
        )

        self.assertEqual(result, [[100.0, 0.0]])

    def test_default_size_preserves_constant_valid_input(self):
        # One full row; resizing also expands its height to 64.
        values, mask = bytes_to_layout(
            bytes([100]) * 256,
            row_width=256,
        )

        result = masked_area_resize(values, mask)

        self.assertEqual(len(result), 64)
        self.assertTrue(all(len(row) == 64 for row in result))
        self.assertTrue(
            all(value == 100.0 for row in result for value in row)
        )

    def test_rejects_invalid_inputs(self):
        invalid_inputs = [
            ([], [], 1, 1),
            ([[1, 2], [3]], [[1, 1], [1]], 1, 1),
            ([[1, 2]], [[1]], 1, 1),
            ([[1]], [[2]], 1, 1),
            ([[float("nan")]], [[1]], 1, 1),
            ([[1]], [[1]], 0, 1),
            ([[1]], [[1]], 1, -1),
        ]

        for values, mask, height, width in invalid_inputs:
            with self.subTest(
                values=values,
                mask=mask,
                height=height,
                width=width,
            ):
                with self.assertRaises(ValueError):
                    masked_area_resize(values, mask, height, width)


if __name__ == "__main__":
    unittest.main()