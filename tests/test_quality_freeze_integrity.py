import tempfile
import unittest
from pathlib import Path

from scripts.check_quality_freeze import verify_checksums
from scripts.february_group_audit import sha


class TestQualityFreezeIntegrity(unittest.TestCase):
    def test_detects_changed_weights_or_manifest_after_freeze(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            weights, folds = folder / "weights.json", folder / "fold_manifest.csv"
            weights.write_text('{"weights": [0.4, 0.3, 0.3]}', encoding="utf-8")
            folds.write_text("sha,fold\na,0\n", encoding="utf-8")
            hashes = {p.name: sha(p) for p in (weights, folds)}
            verify_checksums(folder, hashes)
            original = weights.read_bytes()
            weights.write_text('{"weights": [0.8, 0.1, 0.1]}', encoding="utf-8")
            with self.assertRaises(ValueError):
                verify_checksums(folder, hashes)
            weights.write_bytes(original)
            folds.write_text("sha,fold\na,1\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                verify_checksums(folder, hashes)

    def test_rejects_checksum_path_escape(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                verify_checksums(Path(directory), {"../weights.json": "bad"})


if __name__ == "__main__":
    unittest.main()
