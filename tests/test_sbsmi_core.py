import unittest

from scripts.sbsmi_core import bytes_to_states, states_to_sbsmi


class TestSBSMICore(unittest.TestCase):
    def test_states_cross_byte_boundaries(self):
        # 00000000 00010000 10000011
        # 000000 000001 000010 000011
        data = bytes([0x00, 0x10, 0x83])

        self.assertEqual(
            bytes_to_states(data, l=6),
            [0, 1, 2, 3],
        )

    def test_most_significant_bits_are_read_first(self):
        # 1010 1011 0001 0000 -> 10, 11, 1, 0
        data = bytes([0xAB, 0x10])

        self.assertEqual(
            bytes_to_states(data, l=4),
            [10, 11, 1, 0],
        )

    def test_transition_direction_and_normalization(self):
        states = [0, 1, 0, 1, 0, 2]

        expected = [
            [0, 170, 85, 0],
            [255, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ]

        self.assertEqual(states_to_sbsmi(states, l=2), expected)

    def test_grayscale_uses_floor(self):
        # From state 0: half the transitions go to 1, half to 2.
        # floor(255 * 0.5) = 127.
        states = [0, 1, 0, 2]

        image = states_to_sbsmi(states, l=2)

        self.assertEqual(image[0], [0, 127, 127, 0])

    def test_incomplete_final_state_is_rejected(self):
        # One byte contains 8 bits: one 6-bit state and 2 leftover bits.
        with self.assertRaises(ValueError):
            bytes_to_states(bytes([0xFF]), l=6)


if __name__ == "__main__":
    unittest.main()