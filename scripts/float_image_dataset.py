import csv
import hashlib
import io
import json
import re
from datetime import date, datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


class FloatImageDataset(Dataset):
    """Load normalized raw-byte or entropy NPY images."""

    def __init__(self, project_root, view, split, month=None):
        view_settings = {
            "raw-byte": ("raw_byte_dataset_v1", "raw-byte"),
            "entropy": ("entropy_dataset_v1", "local-entropy"),
        }

        if view not in view_settings:
            raise ValueError("view must be 'raw-byte' or 'entropy'.")
        if split not in {"train", "validation"}:
            raise ValueError("split must be 'train' or 'validation'.")

        if month is not None:
            if (
                not isinstance(month, str)
                or re.fullmatch(r"\d{4}-\d{2}", month) is None
            ):
                raise ValueError("month must have YYYY-MM format.")
            date.fromisoformat(month + "-01")

        self.project_root = Path(project_root).resolve()
        self.view = view
        self.split = split
        self.month = month

        folder, self.representation = view_settings[view]
        self.dataset_dir = self.project_root / "data/derived" / folder

        config = json.loads(
            (self.dataset_dir / "config.json").read_text(encoding="utf-8")
        )
        self.config = config

        required = {
            "schema_version": 1,
            "representation": self.representation,
            "row_width": 256,
            "image_shape": [64, 64],
            "resize": "mask-aware area average",
            "storage": "NPY",
            "storage_dtype": "<f4",
            "empty_output_region": 0,
        }

        if view == "raw-byte":
            required["normalization"] = (
                "divide area-averaged byte values by 255"
            )
        else:
            required.update({
                "window_size": 256,
                "stride": 128,
                "window_end_rule": "stop_at_first_window_reaching_eof",
                "partial_window_rule": "existing_bytes_only",
                "overlap_rule": "mean_of_covering_window_entropies",
                "normalization": (
                    "divide entropy in bits by 8; no further scaling"
                ),
            })

        for key, expected in required.items():
            if config.get(key) != expected:
                raise ValueError(f"Unsupported configuration: {key}")

        self.config_id = sha256_bytes(
            json.dumps(config, sort_keys=True).encode("utf-8")
        )

        manifest_path = (
            self.project_root
            / "data/manifests/train_validation_manifest.csv"
        )
        mapping_path = self.project_root / "docs/class_mapping.json"

        manifest_bytes = manifest_path.read_bytes()
        mapping_bytes = mapping_path.read_bytes()

        if sha256_bytes(manifest_bytes) != config["manifest_sha256"]:
            raise ValueError("Manifest differs from preprocessing.")
        if sha256_bytes(mapping_bytes) != config["class_mapping_sha256"]:
            raise ValueError("Class mapping differs from preprocessing.")

        mapping = json.loads(mapping_bytes.decode("utf-8"))
        self.class_to_idx = mapping["class_to_idx"]
        self.num_classes = mapping["num_classes"]

        labels = list(self.class_to_idx.values())
        if (
            self.num_classes != 51
            or len(labels) != 51
            or any(type(label) is not int for label in labels)
            or sorted(labels) != list(range(51))
        ):
            raise ValueError("Expected the frozen mapping of 51 classes.")

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

            observed = datetime.fromisoformat(row["timestamp"]).date()
            start, stop = periods[row_split]

            if not start <= observed < stop:
                raise ValueError(f"Timestamp outside assigned split: {sha}")

            if row_split != split:
                continue
            if month is not None and observed.strftime("%Y-%m") != month:
                continue

            self.samples.append(row)

        if not self.samples:
            raise ValueError("No samples match the requested split/month.")

    def __len__(self):
        return len(self.samples)

    def read_report(self, index):
        sample = self.samples[index]
        sha = sample["sha"]
        label = self.class_to_idx[sample["family"]]

        path = self.dataset_dir / self.split / f"{sha}.json"
        report = json.loads(path.read_text(encoding="utf-8"))

        expected_fields = {
            "sample_identifier": sha,
            "split": self.split,
            "timestamp": sample["timestamp"],
            "family": sample["family"],
            "class_index": label,
            "representation": self.representation,
            "config_sha256": self.config_id,
            "status": "PASS",
            "image_shape": [64, 64],
            "storage_dtype": "<f4",
            "array_round_trip": "PASS",
        }

        for key, expected in expected_fields.items():
            if report.get(key) != expected:
                raise ValueError(f"Report mismatch for {sha}: {key}")

        for key in ("disarmed_content_sha256", "npy_sha256"):
            value = report.get(key, "")
            if (
                not isinstance(value, str)
                or re.fullmatch(r"[0-9a-f]{64}", value) is None
            ):
                raise ValueError(f"Invalid report hash for {sha}: {key}")

        return report

    def __getitem__(self, index):
        sample = self.samples[index]
        sha = sample["sha"]
        label = self.class_to_idx[sample["family"]]
        report = self.read_report(index)

        path = self.dataset_dir / self.split / f"{sha}.npy"

        # Verify and decode the same bytes.
        image_bytes = path.read_bytes()
        if sha256_bytes(image_bytes) != report["npy_sha256"]:
            raise ValueError(f"NPY checksum mismatch: {sha}")

        matrix = np.load(io.BytesIO(image_bytes), allow_pickle=False)

        if matrix.shape != (64, 64):
            raise ValueError(f"Unexpected matrix shape: {sha}")
        if matrix.dtype != np.dtype("<f4"):
            raise ValueError(f"Expected float32 matrix: {sha}")
        if not np.isfinite(matrix).all():
            raise ValueError(f"Non-finite matrix values: {sha}")
        if (matrix < 0).any() or (matrix > 1).any():
            raise ValueError(f"Matrix outside [0, 1]: {sha}")

        # Values are already normalized. Do not divide by 255.
        matrix = np.array(matrix, dtype=np.float32, order="C", copy=True)
        tensor = torch.from_numpy(matrix).unsqueeze(0)

        return tensor, label