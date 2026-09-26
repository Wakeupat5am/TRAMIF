import tempfile
import unittest
from pathlib import Path

from scripts.sbsmi_core import bytes_to_states, states_to_sbsmi
from scripts.sbsmi_stream import file_to_sbsmi


class TestSBSMIStream(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.sample_path = Path(self.temp_dir.name) / "sample.bin"

    def test_matches_reference_across_chunk_sizes(self):
        samples = [
            bytes([0xFF]),
            bytes([0xFF, 0xFF]),
            bytes([0x00, 0x10, 0x83]),
            bytes(range(31)),
        ]

        for data in samples:
            self.sample_path.write_bytes(data)

            expected = states_to_sbsmi(
                bytes_to_states(data, l=6),
                l=6,
            )

            for chunk_size in (1, 2, 3, 7, 65536):
                with self.subTest(
                    sample_length=len(data),
                    chunk_size=chunk_size,
                ):
                    actual = file_to_sbsmi(
                        self.sample_path,
                        l=6,
                        chunk_size=chunk_size,
                    )
                    self.assertEqual(actual, expected)

    def test_matches_reference_for_supported_bit_lengths(self):
        data = bytes(range(17))
        self.sample_path.write_bytes(data)

        for l in range(1, 9):
            with self.subTest(l=l):
                expected = states_to_sbsmi(
                    bytes_to_states(data, l=l),
                    l=l,
                )

                actual = file_to_sbsmi(
                    self.sample_path,
                    l=l,
                    chunk_size=1,
                )

                self.assertEqual(actual, expected)

    def test_partial_block_matches_hand_calculation(self):
        self.sample_path.write_bytes(bytes([0xFF, 0xFF]))

        expected = [[0 for _ in range(64)] for _ in range(64)]
        expected[63][63] = 127
        expected[63][15] = 127

        actual = file_to_sbsmi(
            self.sample_path,
            l=6,
            chunk_size=1,
        )

        self.assertEqual(actual, expected)

    def test_rejects_input_without_a_transition(self):
        for data, l in ((b"", 6), (bytes([0xFF]), 8)):
            with self.subTest(data=data, l=l):
                self.sample_path.write_bytes(data)

                with self.assertRaises(ValueError):
                    file_to_sbsmi(self.sample_path, l=l)

    def test_rejects_zero_chunk_size(self):
        self.sample_path.write_bytes(bytes([0xFF, 0xFF]))

        with self.assertRaises(ValueError):
            file_to_sbsmi(self.sample_path, chunk_size=0)


if __name__ == "__main__":
    unittest.main()