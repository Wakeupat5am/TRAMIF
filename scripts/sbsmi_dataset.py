import csv
import hashlib
import io
import json
import re
from datetime import date, datetime
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


class SBSMIDataset(Dataset):
    """Load frozen SBSMI images with their original train-only class mapping."""

    def __init__(self, project_root, split, month=None):
        if split not in {"train", "validation"}:
            raise ValueError("split must be 'train' or 'validation'.")

        if month is not None:
            if not isinstance(month, str):
                raise ValueError("month must be a YYYY-MM string.")

            if re.fullmatch(r"\d{4}-\d{2}", month) is None:
                raise ValueError("month must have YYYY-MM format.")

            date.fromisoformat(month + "-01")

        self.project_root = Path(project_root).resolve()
        self.split = split
        self.month = month
        self.dataset_dir = (
            self.project_root / "data" / "derived" / "sbsmi_dataset_v1"
        )

        manifest_path = (
            self.project_root
            / "data"
            / "manifests"
            / "train_validation_manifest.csv"
        )
        mapping_path = self.project_root / "docs" / "class_mapping.json"

        config = json.loads(
            (self.dataset_dir / "config.json").read_text(encoding="utf-8")
        )

        required_config = {
            "schema_version": 1,
            "representation": "SBSMI",
            "state_bits": 6,
            "image_size": [64, 64],
            "image_mode": "L",
            "model_normalization": "divide integer pixels by 255",
        }

        for key, expected in required_config.items():
            if config.get(key) != expected:
                raise ValueError(f"Unsupported dataset configuration: {key}")

        self.config_id = sha256_bytes(
            json.dumps(config, sort_keys=True).encode("utf-8")
        )

        manifest_bytes = manifest_path.read_bytes()
        mapping_bytes = mapping_path.read_bytes()

        if sha256_bytes(manifest_bytes) != config["manifest_sha256"]:
            raise ValueError("Manifest differs from the preprocessing version.")

        if sha256_bytes(mapping_bytes) != config["class_mapping_sha256"]:
            raise ValueError("Class mapping differs from preprocessing.")

        mapping = json.loads(mapping_bytes.decode("utf-8"))
        self.class_to_idx = mapping["class_to_idx"]
        self.num_classes = mapping["num_classes"]

        indices = list(self.class_to_idx.values())

        if self.num_classes != 51 or len(indices) != 51:
            raise ValueError("Expected 51 frozen classes.")

        if any(type(index) is not int for index in indices):
            raise ValueError("Class indices must be integers.")

        if sorted(indices) != list(range(51)):
            raise ValueError("Class indices must cover 0 to 50.")

        with io.StringIO(
            manifest_bytes.decode("utf-8-sig"), newline=""
        ) as file:
            rows = list(csv.DictReader(file))

        periods = {
            "train": (date(2019, 8, 1), date(2020, 2, 1)),
            "validation": (date(2020, 2, 1), date(2020, 4, 1)),
        }

        seen = set()
        self.samples = []

        for row in rows:
            sha = row["sha"]
            row_split = row["split"]

            if re.fullmatch(r"[0-9a-f]{64}", sha) is None:
                raise ValueError("Invalid sample identifier.")

            if sha in seen:
                raise ValueError(f"Duplicate sample identifier: {sha}")
            seen.add(sha)

            if row_split not in periods:
                raise ValueError(f"Unexpected split: {row_split}")

            if row["family"] not in self.class_to_idx:
                raise ValueError(f"Unknown family: {row['family']}")

            observed_date = datetime.fromisoformat(row["timestamp"]).date()
            start, stop = periods[row_split]

            if not start <= observed_date < stop:
                raise ValueError(f"Timestamp outside assigned split: {sha}")

            if row_split != split:
                continue

            if month is not None:
                if observed_date.strftime("%Y-%m") != month:
                    continue

            self.samples.append(row)

        if not self.samples:
            raise ValueError("No samples match the requested split/month.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        sha = sample["sha"]
        label = self.class_to_idx[sample["family"]]

        split_dir = self.dataset_dir / self.split
        image_path = split_dir / f"{sha}.png"
        report_path = split_dir / f"{sha}.json"

        report = json.loads(report_path.read_text(encoding="utf-8"))

        expected_fields = {
            "sample_identifier": sha,
            "split": self.split,
            "timestamp": sample["timestamp"],
            "family": sample["family"],
            "class_index": label,
            "config_sha256": self.config_id,
            "status": "PASS",
            "pixel_round_trip": "PASS",
        }

        for key, expected in expected_fields.items():
            if report.get(key) != expected:
                raise ValueError(f"Report mismatch for {sha}: {key}")

        # Hash and decode the same bytes, without reopening the PNG.
        image_bytes = image_path.read_bytes()

        if sha256_bytes(image_bytes) != report["png_sha256"]:
            raise ValueError(f"PNG checksum mismatch: {sha}")

        with Image.open(io.BytesIO(image_bytes)) as image:
            if image.format != "PNG":
                raise ValueError(f"Expected PNG: {sha}")

            if image.mode != "L" or image.size != (64, 64):
                raise ValueError(f"Expected grayscale 64 x 64 image: {sha}")

            pixels = np.array(image, dtype=np.uint8, copy=True)

        if sha256_bytes(pixels.tobytes()) != report["pixel_sha256"]:
            raise ValueError(f"Pixel checksum mismatch: {sha}")

        normalized = pixels.astype(np.float32) / np.float32(255.0)
        tensor = torch.from_numpy(normalized).unsqueeze(0)

        return tensor, label