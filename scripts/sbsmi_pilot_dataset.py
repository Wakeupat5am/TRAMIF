import csv
import json
import re
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class SBSMIPilotDataset(Dataset):
    """Load verified SBSMI images from one completed training pilot."""

    def __init__(self, run_dir, mapping_path):
        self.run_dir = Path(run_dir)

        mapping = json.loads(
            Path(mapping_path).read_text(encoding="utf-8")
        )
        self.class_to_idx = mapping["class_to_idx"]
        self.num_classes = mapping["num_classes"]

        indices = list(self.class_to_idx.values())

        if self.num_classes != 51:
            raise ValueError("Expected 51 frozen classes.")

        if any(type(index) is not int for index in indices):
            raise ValueError("Class indices must be integers.")

        if sorted(indices) != list(range(self.num_classes)):
            raise ValueError("Class indices must cover 0 to 50 exactly.")

        with (self.run_dir / "manifest.csv").open(
            encoding="utf-8", newline=""
        ) as file:
            self.samples = list(csv.DictReader(file))

        with (self.run_dir / "summary.csv").open(
            encoding="utf-8", newline=""
        ) as file:
            summary_rows = list(csv.DictReader(file))

        if not self.samples:
            raise ValueError("Pilot manifest is empty.")

        sample_shas = [row["sha"] for row in self.samples]
        summary_shas = [row["sha"] for row in summary_rows]

        if len(sample_shas) != len(set(sample_shas)):
            raise ValueError("Duplicate SHA in pilot manifest.")

        if len(summary_shas) != len(set(summary_shas)):
            raise ValueError("Duplicate SHA in pilot summary.")

        if set(sample_shas) != set(summary_shas):
            raise ValueError("Pilot manifest and summary do not match.")

        status_by_sha = {
            row["sha"]: row["status"]
            for row in summary_rows
        }

        for sample in self.samples:
            sha = sample["sha"]

            if re.fullmatch(r"[0-9a-f]{64}", sha) is None:
                raise ValueError("Invalid sample identifier.")

            if sample["split"] != "train":
                raise ValueError("This smoke check accepts train only.")

            if sample["family"] not in self.class_to_idx:
                raise ValueError("Family is absent from the frozen mapping.")

            if status_by_sha[sha] != "PASS":
                raise ValueError(f"Preprocessing did not pass for {sha}.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        sha = sample["sha"]

        image_path = self.run_dir / f"{sha}_sbsmi.png"
        report_path = self.run_dir / f"{sha}.json"

        report = json.loads(report_path.read_text(encoding="utf-8"))

        if report["sample_identifier"] != sha:
            raise ValueError("Report sample identifier mismatch.")

        if report["family"] != sample["family"]:
            raise ValueError("Report family mismatch.")

        if report["split"] != "train":
            raise ValueError("Report split mismatch.")

        if report["all_png_round_trips"] != "PASS":
            raise ValueError("Report does not confirm PNG verification.")

        with Image.open(image_path) as image:
            if image.format != "PNG":
                raise ValueError("Expected a PNG image.")

            if image.mode != "L" or image.size != (64, 64):
                raise ValueError("Expected grayscale 64 x 64 image.")

            pixels = np.array(image, dtype=np.uint8, copy=True)

        normalized = pixels.astype(np.float32) / np.float32(255.0)

        # Compare the PNG with the saved preprocessing matrix.
        reference = np.asarray(
            report["normalized_matrices"]["sbsmi"],
            dtype=np.float32,
        )

        if reference.shape != (64, 64):
            raise ValueError("Invalid reference matrix dimensions.")

        if not np.isfinite(reference).all():
            raise ValueError("Reference matrix contains invalid values.")

        np.testing.assert_allclose(
            normalized,
            reference,
            rtol=0,
            atol=1e-7,
        )

        # PyTorch expects channels first: [1, 64, 64].
        tensor = torch.from_numpy(normalized).unsqueeze(0)
        label = self.class_to_idx[sample["family"]]

        return tensor, label