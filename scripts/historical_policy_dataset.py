import argparse
import csv
import hashlib
import io
import json
from datetime import datetime
from pathlib import Path

import torch
from torch.utils.data import Dataset

from scripts.float_image_dataset import FloatImageDataset
from scripts.sbsmi_dataset import SBSMIDataset


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


class HistoricalPolicyDataset(Dataset):
    """Select retained historical samples using an audited policy manifest."""

    def __init__(
        self,
        project_root,
        policy_dir,
        view,
        split,
        month=None,
        threshold=0,
    ):
        if view not in {"sbsmi", "raw-byte", "entropy"}:
            raise ValueError("Unknown image view.")
        if split not in {"train", "validation"}:
            raise ValueError("Only train and validation are supported.")
        if type(threshold) is not int or threshold not in {0, 10, 20}:
            raise ValueError("threshold must be 0, 10, or 20.")
        if month is not None:
            datetime.strptime(month, "%Y-%m")
            if len(month) != 7:
                raise ValueError("month must have YYYY-MM format.")

        self.project_root = Path(project_root).resolve()
        self.policy_dir = Path(policy_dir)
        if not self.policy_dir.is_absolute():
            self.policy_dir = self.project_root / self.policy_dir
        self.policy_dir = self.policy_dir.resolve()

        self.view = view
        self.split = split
        self.month = month
        self.threshold = threshold

        config_bytes = (self.policy_dir / "config.json").read_bytes()
        report_bytes = (self.policy_dir / "report.json").read_bytes()
        config = json.loads(config_bytes)
        report = json.loads(report_bytes)

        if report.get("status") != "HISTORICAL_POLICY_APPLIED":
            raise ValueError("Historical policy application is incomplete.")

        for key in (
            "decision_coverage_checks",
            "exclusion_witness_checks",
            "nested_retention_checks",
        ):
            if report.get(key) != "PASS":
                raise ValueError(f"Policy check did not pass: {key}")

        if config.get("primary_threshold") != 0:
            raise ValueError("Unexpected primary threshold.")

        input_files = {
            "manifest_sha256":
                "data/manifests/train_validation_manifest.csv",
            "class_mapping_sha256":
                "docs/class_mapping.json",
        }

        for key, relative_path in input_files.items():
            content = (self.project_root / relative_path).read_bytes()
            if sha256_bytes(content) != config[key]:
                raise ValueError(f"Policy input has changed: {relative_path}")

        policy_bytes = (
            self.policy_dir / "policy_snapshot.md"
        ).read_bytes()
        if sha256_bytes(policy_bytes) != config["policy_sha256"]:
            raise ValueError("Policy snapshot checksum mismatch.")

        relative_keep = f"threshold_{threshold:02d}/{split}_keep.csv"
        keep_bytes = (self.policy_dir / relative_keep).read_bytes()
        keep_hash = sha256_bytes(keep_bytes)

        expected_hash = report["output_csv_sha256"][relative_keep]
        if keep_hash != expected_hash:
            raise ValueError("Retained manifest checksum mismatch.")

        with io.StringIO(
            keep_bytes.decode("utf-8-sig"), newline=""
        ) as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames != ["sha", "split", "timestamp", "family"]:
                raise ValueError("Unexpected retained manifest columns.")
            retained = list(reader)

        expected_count = report["thresholds"][str(threshold)][
            f"{split}_kept"
        ]
        if len(retained) != expected_count:
            raise ValueError("Retained manifest count differs from report.")

        # Load the full split so every retained row can be validated,
        # including rows outside the requested month.
        if view == "sbsmi":
            self.base = SBSMIDataset(self.project_root, split)
        else:
            self.base = FloatImageDataset(
                self.project_root, view, split
            )

        self.class_to_idx = dict(self.base.class_to_idx)
        self.num_classes = self.base.num_classes
        self.preprocessing_config_id = self.base.config_id

        original_indices = {
            row["sha"]: index
            for index, row in enumerate(self.base.samples)
        }

        self.indices = []
        self.samples = []
        seen = set()
        previous_index = -1

        for row in retained:
            sha = row["sha"]
            if sha in seen:
                raise ValueError(f"Repeated retained identifier: {sha}")
            seen.add(sha)

            if sha not in original_indices:
                raise ValueError(f"Unknown retained identifier: {sha}")

            index = original_indices[sha]
            original = self.base.samples[index]

            for key in ("sha", "split", "timestamp", "family"):
                if row[key] != original[key]:
                    raise ValueError(f"Retained metadata differs: {sha}")

            if index <= previous_index:
                raise ValueError("Retained manifest changed sample order.")
            previous_index = index

            sample_month = datetime.fromisoformat(
                row["timestamp"]
            ).strftime("%Y-%m")

            if month is not None and sample_month != month:
                continue

            self.indices.append(index)
            self.samples.append(dict(original))

        if not self.samples:
            raise ValueError("No retained samples for this split/month.")

        # Record both preprocessing identity and subset identity.
        # They describe different parts of the dataset.
        self.selection = {
            "view": view,
            "split": split,
            "month": month,
            "threshold": threshold,
            "preprocessing_config_id": self.preprocessing_config_id,
            "policy_config_sha256": sha256_bytes(config_bytes),
            "policy_report_sha256": sha256_bytes(report_bytes),
            "retained_manifest_sha256": keep_hash,
            "selected_samples": len(self.samples),
        }
        self.selection_id = sha256_bytes(
            json.dumps(self.selection, sort_keys=True).encode("utf-8")
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.base[self.indices[index]]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy-dir", required=True)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    cases = [
        ("train", None, 18061),
        ("validation", "2020-02", 3799),
        ("validation", "2020-03", 4332),
    ]

    print("Purpose: check primary-policy dataset selection")
    print("Threshold: 0")

    for split, month, expected_count in cases:
        reference_samples = None
        reference_mapping = None

        for view in ("sbsmi", "raw-byte", "entropy"):
            dataset = HistoricalPolicyDataset(
                project_root=project_root,
                policy_dir=args.policy_dir,
                view=view,
                split=split,
                month=month,
                threshold=0,
            )

            if len(dataset) != expected_count:
                raise RuntimeError(
                    f"Unexpected sample count: {view}, {split}, {month}"
                )

            if reference_samples is None:
                reference_samples = dataset.samples
                reference_mapping = dataset.class_to_idx
            else:
                if dataset.samples != reference_samples:
                    raise RuntimeError("Views select different samples.")
                if dataset.class_to_idx != reference_mapping:
                    raise RuntimeError("Views use different class mappings.")

            # Read the first and last retained image to check that
            # subset indices actually reach the correct image and label.
            for index in sorted({0, len(dataset) - 1}):
                image, label = dataset[index]
                sample = dataset.samples[index]
                expected_label = dataset.class_to_idx[sample["family"]]

                if tuple(image.shape) != (1, 64, 64):
                    raise RuntimeError("Unexpected image shape.")
                if image.dtype != torch.float32:
                    raise RuntimeError("Unexpected image dtype.")
                if not torch.isfinite(image).all().item():
                    raise RuntimeError("Non-finite image values.")
                if ((image < 0) | (image > 1)).any().item():
                    raise RuntimeError("Image values outside [0, 1].")
                if label != expected_label:
                    raise RuntimeError("Image label mismatch.")

            period = month if month is not None else split
            print(f"{view} | {period} | samples={len(dataset)} | PASS")

    print("Sample order and class mapping agree across all three views.")
    print("Historical policy dataset smoke check: PASS")
    print("No training or model evaluation performed.")
    print("No dataset files changed.")


if __name__ == "__main__":
    main()