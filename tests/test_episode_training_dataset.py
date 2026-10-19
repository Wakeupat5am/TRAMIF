"""Synthetic fixtures only, except an explicitly metadata-only binding test."""
import copy
import csv
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image

from scripts import episode_training_dataset as d


def csv_data(rows):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode()


def image_data(view, matrix=None):
    out = io.BytesIO()
    if view == "sbsmi":
        pixels = np.arange(4096, dtype=np.uint16).reshape(64, 64).astype(np.uint8)
        if matrix is not None:
            pixels = matrix
        Image.fromarray(pixels).save(out, format="PNG")
        blob = out.getvalue()
        return blob, {"png_sha256": d.sha256(blob), "pixel_sha256": d.sha256(pixels.tobytes())}
    matrix = np.full((64, 64), 0.5, dtype="<f4") if matrix is None else matrix
    np.save(out, matrix, allow_pickle=False)
    blob = out.getvalue()
    return blob, {"npy_sha256": d.sha256(blob)}


class Fixture:
    """Small isolated synthetic trust root; never modifies production anchors."""
    def __enter__(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.rows, self.members = [], []
        self.mapping = {"a": 0, "b": 1}
        self.source = {f"f{i}": i for i in range(51)}
        self.source["a"] = self.source.pop("f3")
        self.source["b"] = self.source.pop("f7")
        for i in range(4):
            sha = f"{i + 1:064x}"
            family = "a" if i % 2 == 0 else "b"
            stamp = f"2019-09-0{i + 1} 00:00:00+00:00"
            self.rows.append(dict(sha=sha, timestamp=stamp, family=family, stage="train",
                threshold="0", decision="KEEP", reason="retained", reference_sha="",
                reference_stage="", tlsh_distance=""))
            self.members.append(dict(sha=sha, family=family, timestamp=stamp, stage="train",
                content_sha256=f"{i + 100:064x}", retained="True", group_id=sha))
        source_hash = self.put("docs/class_mapping.json", d.json_bytes({
            "num_classes": 51, "class_to_idx": self.source}))
        blobs = [csv_data(self.rows), d.json_bytes(self.mapping),
                 d.json_bytes({"train_end": "2019-11"}), csv_data(self.members)]
        hashes = tuple(self.put(f"{d.G2}/E1/{f}", b) for f, b in zip(d.FILES, blobs))
        storage = {}
        for view in d.VIEWS:
            folder = d.STORAGE[view][0]
            config = {"class_mapping_sha256": source_hash,
                      "manifest_sha256": d.HISTORICAL_MANIFEST_HASH,
                      "representation": view}
            config_hash = self.put(f"data/derived/{folder}/config.json", d.json_bytes(config))
            storage[view] = (folder, config_hash)
            config_id = d.sha256(json.dumps(config, sort_keys=True).encode())
            for row, member in zip(self.rows, self.members):
                blob, image_hash = image_data(view)
                report = {"sample_identifier": row["sha"], "split": "train",
                    "timestamp": row["timestamp"], "family": row["family"],
                    "class_index": self.source[row["family"]], "config_sha256": config_id,
                    "status": "PASS", "disarmed_content_sha256": member["content_sha256"], **image_hash}
                if view == "sbsmi":
                    report["pixel_round_trip"] = "PASS"
                else:
                    report.update(representation="raw-byte" if view == "raw-byte" else "local-entropy",
                                  image_shape=[64, 64], storage_dtype="<f4", array_round_trip="PASS")
                base = f"data/derived/{folder}/train/{row['sha']}"
                self.put(base + (".png" if view == "sbsmi" else ".npy"), blob)
                self.put(base + ".json", d.json_bytes(report))
        self.patches = [patch.object(d, "EPISODES", {"E1": (4, 2, "2019-11", hashes)}),
                        patch.object(d, "SOURCE_MAPPING_HASH", source_hash),
                        patch.object(d, "STORAGE", storage)]
        for p in self.patches:
            p.start()
        return self

    def put(self, path, blob):
        path = self.root / path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(blob)
        return d.sha256(blob)

    def dataset(self, view="raw-byte", receipts=None):
        return d.EpisodeTrainingDataset("E1", view, self.root, receipts)

    def __exit__(self, *args):
        for p in reversed(self.patches):
            p.stop()
        self.temp.cleanup()


class TestEpisodeTrainingDataset(unittest.TestCase):
    def test_frozen_real_metadata_only_no_image_reads(self):
        original = d.read_bound
        reads = []
        def metadata_only(root, path, expected=None):
            path = str(path).replace("\\", "/")
            self.assertNotIn("/train/", path)
            self.assertNotIn("_keep.csv", path.replace("train_keep.csv", ""))
            self.assertNotIn("data/raw", path)
            reads.append(path)
            return original(root, path, expected)
        with patch.object(d, "read_bound", side_effect=metadata_only):
            for case, n, k in (("E1", 7034, 32), ("E2", 9803, 42)):
                datasets = [d.EpisodeTrainingDataset(case, v) for v in d.VIEWS]
                self.assertTrue(all(len(x) == n and x.num_classes == k for x in datasets))
                self.assertEqual(datasets[0].samples, datasets[1].samples)
                self.assertEqual(datasets[1].samples, datasets[2].samples)
        self.assertTrue(reads)

    def test_episode_labels_are_not_final_indices_and_match_legacy_pixels(self):
        from scripts.sbsmi_dataset import SBSMIDataset
        from scripts.float_image_dataset import FloatImageDataset
        with Fixture() as f:
            for view in d.VIEWS:
                dataset = f.dataset(view)
                tensor, target, sha = dataset[0]
                self.assertEqual(target, 0)
                self.assertEqual(dataset.samples[0].source_class_index, 3)
                self.assertEqual(sha, f.rows[0]["sha"])
                cls = SBSMIDataset if view == "sbsmi" else FloatImageDataset
                legacy = cls.__new__(cls)  # never invoke full historical constructor
                legacy.samples = f.rows
                legacy.class_to_idx = f.source
                legacy.split = "train"
                legacy.config_id = dataset.config_id
                legacy.dataset_dir = f.root / dataset.storage_dir
                legacy.representation = "raw-byte" if view == "raw-byte" else "local-entropy"
                old_tensor, old_label = legacy[0]
                self.assertTrue(torch.equal(tensor, old_tensor))
                self.assertEqual(old_label, 3)
                self.assertEqual(tensor.shape, (1, 64, 64))
                self.assertEqual(tensor.dtype, torch.float32)

    def test_wrong_case_view_and_manifest_hash_fail_before_images(self):
        with Fixture() as f:
            for case, view in (("Final", "sbsmi"), ("E1", "unknown")):
                with self.assertRaises(ValueError):
                    d.EpisodeTrainingDataset(case, view, f.root)
            f.put(f"{d.G2}/E1/train_keep.csv", csv_data(list(reversed(f.rows))))
            with self.assertRaisesRegex(ValueError, "Checksum"):
                f.dataset()

    def test_mapping_and_spec_binding_fail_closed(self):
        for filename in ("class_mapping.json", "episode_spec.json", "train_00_members.csv"):
            with self.subTest(filename=filename), Fixture() as f:
                f.put(f"{d.G2}/E1/{filename}", b"{}")
                with self.assertRaisesRegex(ValueError, "Checksum"):
                    f.dataset()

    def test_chronology_retention_and_identity_rejections(self):
        with Fixture() as f:
            mutations = (("timestamp", "2019-11-01T00:00:00+00:00"),
                ("timestamp", "2020-04-01T00:00:00+00:00"), ("timestamp", "2019-09-01"),
                ("timestamp", "2019-09-01T00:00:00+07:00"), ("stage", "A"),
                ("decision", "EXCLUDE"), ("threshold", "10"), ("family", "unknown"),
                ("sha", "../escape"), ("reference_stage", "C"))
            for key, value in mutations:
                with self.subTest(key=key, value=value):
                    rows = copy.deepcopy(f.rows)
                    rows[0][key] = value
                    with self.assertRaises((ValueError, KeyError)):
                        d.validate_train_rows(rows, f.members, f.mapping, f.source, 4, "2019-11")
            members = copy.deepcopy(f.members)
            members[0]["retained"] = "False"
            with self.assertRaises(ValueError):
                d.validate_train_rows(f.rows, members, f.mapping, f.source, 4, "2019-11")
            with self.assertRaises(ValueError):
                d.validate_train_rows(f.rows + [f.rows[0]], f.members, f.mapping, f.source, 4, "2019-11")

    def test_shapes_scaling_and_no_second_normalization(self):
        for view in d.VIEWS:
            blob, report = image_data(view)
            tensor = d.decode_image(blob, report, view)
            if view == "sbsmi":
                self.assertEqual(tensor[0, 0, 0].item(), 0)
                self.assertEqual(tensor[0, 3, 63].item(), 1)
                self.assertAlmostEqual(tensor[0, 0, 1].item(), 1 / 255)
            else:
                self.assertTrue(torch.equal(tensor, torch.full((1, 64, 64), .5)))

    def test_corrupt_hash_dtype_shape_range_finite_and_pixels(self):
        for matrix in (np.zeros((32, 64), dtype="<f4"), np.zeros((64, 64), dtype="<f8"),
                       np.full((64, 64), np.nan, dtype="<f4"),
                       np.full((64, 64), np.inf, dtype="<f4"),
                       np.full((64, 64), -0.1, dtype="<f4"),
                       np.full((64, 64), 1.1, dtype="<f4")):
            blob, report = image_data("entropy", matrix)
            with self.assertRaises(ValueError):
                d.decode_image(blob, report, "entropy")
        blob, report = image_data("sbsmi")
        with self.assertRaises(ValueError):
            d.decode_image(blob + b"corrupt", report, "sbsmi")
        report["pixel_sha256"] = "0" * 64
        with self.assertRaises(ValueError):
            d.decode_image(blob, report, "sbsmi")
        with self.assertRaises(Exception):
            d.decode_image(b"truncated", {"npy_sha256": d.sha256(b"truncated")}, "raw-byte")

    def test_missing_and_wrong_report_or_image_rejected(self):
        for field, value in (("class_index", 0), ("disarmed_content_sha256", "0" * 64),
                             ("status", "FAIL"), ("split", "validation"), ("config_sha256", "0" * 64)):
            with Fixture() as f:
                dataset = f.dataset()
                path = f.root / dataset.storage_dir / "train" / (f.rows[0]["sha"] + ".json")
                report = json.loads(path.read_bytes())
                report[field] = value
                path.write_bytes(d.json_bytes(report))
                with self.assertRaises(ValueError):
                    dataset[0]
        with Fixture() as f:
            dataset = f.dataset()
            path = f.root / dataset.storage_dir / "train" / (f.rows[0]["sha"] + ".npy")
            path.unlink()  # synthetic temp file only
            with self.assertRaises(FileNotFoundError):
                dataset[0]

    def test_preflight_synthetic_only_and_mutation_detection(self):
        with Fixture() as f, patch("scripts.train_prefix_view.source_hashes", return_value={"fixture": "hash"}):
            receipt = d.preflight("E1", f.root)
            self.assertEqual(receipt["status"], "PASS")
            for view in d.VIEWS:
                images = receipt["views"][view]["images"]
                self.assertEqual([x["sha"] for x in images], [r["sha"] for r in f.rows])
                dataset = f.dataset(view, images)
                dataset[0]
                with self.assertRaises(ValueError):
                    f.dataset(view, list(reversed(images)))
            dataset = f.dataset("raw-byte", receipt["views"]["raw-byte"]["images"])
            path = f.root / dataset.storage_dir / "train" / (f.rows[0]["sha"] + ".json")
            path.write_bytes(path.read_bytes() + b"\n")
            with self.assertRaisesRegex(ValueError, "changed since preflight"):
                dataset[0]

    def test_path_escape_duplicate_json_and_mapping_types(self):
        with Fixture() as f:
            with self.assertRaises((ValueError, FileNotFoundError)):
                d.read_bound(f.root, "../outside.json")
        with self.assertRaises(ValueError):
            d.parse_json(b'{"a": 0, "a": 1}')
        for mapping in ({"a": True, "b": 0}, {"a": 0, "b": 2}, {"a": 0}):
            with self.assertRaises(ValueError):
                d.validate_mapping(mapping, 2)


if __name__ == "__main__":
    unittest.main()
