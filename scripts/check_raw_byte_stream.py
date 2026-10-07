import argparse
import csv
import hashlib
import io
import json
import os
import re
import time
import zlib
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from scripts.measure_preprocessing_memory import read_process_memory
from scripts.raw_byte_stream import stream_to_raw_byte


ROOT = Path(__file__).resolve().parents[1]
BINARY_FOLDER = ROOT / "data/raw/altered"
INDEX_PATH = ROOT / "data/manifests/train_validation_zip_index.csv"
OUTPUT_ROOT = ROOT / "data/derived/raw_byte_stream_checks"

TOLERANCE = 1e-12
MEMORY_BUDGET_MIB = 256
CHUNK_SIZE = 65536


def read_json(path):
    with path.open(encoding="utf-8") as file:
        return json.load(file)


class DigestReader:
    """Calculate input fingerprints on the same bytes used for the image."""

    def __init__(self, file):
        self.file = file
        self.sha256 = hashlib.sha256()
        self.crc32 = 0
        self.byte_count = 0

    def read(self, size):
        data = self.file.read(size)
        self.sha256.update(data)
        self.crc32 = zlib.crc32(data, self.crc32)
        self.byte_count += len(data)
        return data


def verify_matrix(matrix):
    if matrix.shape != (64, 64):
        raise ValueError("Expected a 64 x 64 matrix.")

    if not np.isfinite(matrix).all():
        raise ValueError("Matrix contains non-finite values.")

    if (matrix < 0).any() or (matrix > 1).any():
        raise ValueError("Matrix values must be in [0, 1].")


def convert_verified_sample(row, source_report):
    sample_id = row["sha"]

    if re.fullmatch(r"[0-9a-f]{64}", sample_id) is None:
        raise ValueError("Invalid sample identifier.")

    if row["split"] != "train":
        raise ValueError("This check accepts training samples only.")

    expected_size = int(row["file_size"])

    if (
        source_report.get("sample_identifier") != sample_id
        or source_report.get("split") != "train"
        or source_report.get("file_size") != expected_size
    ):
        raise ValueError("Source report does not match the selected sample.")

    expected_sha = source_report.get("disarmed_content_sha256", "")

    if re.fullmatch(r"[0-9a-f]{64}", expected_sha) is None:
        raise ValueError("Source report has no valid content SHA-256.")

    path = BINARY_FOLDER / f"{sample_id}.exe"
    started = time.perf_counter()

    with path.open("rb") as file:
        before = os.fstat(file.fileno())

        if before.st_size != expected_size:
            raise ValueError("Binary size differs from the manifest.")

        reader = DigestReader(file)

        matrix = stream_to_raw_byte(
            reader,
            file_size=expected_size,
            chunk_size=CHUNK_SIZE,
        )

        after = os.fstat(file.fileno())

        if (
            before.st_size != after.st_size
            or before.st_mtime_ns != after.st_mtime_ns
        ):
            raise RuntimeError("Binary changed during processing.")

    elapsed = time.perf_counter() - started

    if reader.byte_count != expected_size:
        raise ValueError("Unexpected input byte count.")

    actual_sha = reader.sha256.hexdigest()
    actual_crc = f"{reader.crc32 & 0xFFFFFFFF:08x}"

    if actual_sha != expected_sha:
        raise ValueError("Input content differs from the saved report.")

    if actual_crc != row["zip_crc32"].strip().lower():
        raise ValueError("Input CRC differs from the saved index.")

    verify_matrix(matrix)

    details = {
        "sha": sample_id,
        "family": row["family"],
        "file_size": expected_size,
        "disarmed_content_sha256": actual_sha,
        "crc32": actual_crc,
        "input_checks": "PASS",
        "matrix_checks": "PASS",
        "read_hash_and_transform_seconds": elapsed,
    }

    return matrix, details


def load_pilot(run_dir):
    config = read_json(run_dir / "config.json")
    manifest_bytes = (run_dir / "manifest.csv").read_bytes()

    if hashlib.sha256(manifest_bytes).hexdigest() != config["manifest_sha256"]:
        raise ValueError("Saved pilot manifest checksum differs.")

    expected_config = {
        "sample_count": 10,
        "row_width": 256,
        "image_shape": [64, 64],
        "resize": "mask-aware area average",
    }

    for key, expected in expected_config.items():
        if config.get(key) != expected:
            raise ValueError(f"Unexpected pilot configuration: {key}")

    rows = list(csv.DictReader(
        io.StringIO(manifest_bytes.decode("utf-8-sig"))
    ))

    if len(rows) != 10 or len({row["sha"] for row in rows}) != 10:
        raise ValueError("Expected ten distinct pilot samples.")

    return rows


def largest_train_sample():
    with INDEX_PATH.open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)

        # Keep only the current largest row in memory.
        return max(
            (row for row in reader if row["split"] == "train"),
            key=lambda row: (int(row["file_size"]), row["sha"]),
        )


def main():
    parser = argparse.ArgumentParser(
        description="Check streamed raw-byte images against saved BODMAS pilots."
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    samples = load_pilot(run_dir)
    memory_before = read_process_memory()

    print("Purpose: validate streamed raw-byte preprocessing")
    print("Reference: saved raw-byte pilot matrices")
    print("Absolute tolerance:", TOLERANCE)
    print("Chunk size:", CHUNK_SIZE)

    results = []

    for number, row in enumerate(samples, start=1):
        source_report = read_json(run_dir / f"{row['sha']}.json")

        expected = np.asarray(
            source_report["normalized_matrices"]["raw_byte"],
            dtype=np.float64,
        )
        verify_matrix(expected)

        actual, details = convert_verified_sample(row, source_report)
        maximum_error = float(np.max(np.abs(actual - expected)))

        if maximum_error > TOLERANCE:
            raise RuntimeError(
                f"Raw-byte matrix mismatch for {row['sha']}: "
                f"maximum error={maximum_error}"
            )

        details["maximum_absolute_error"] = maximum_error
        details["reference_comparison"] = "PASS"
        results.append(details)

        print(
            f"[{number}/{len(samples)}] {row['family']} | "
            f"PASS | max_error={maximum_error:.3e}"
        )

    print()
    print("Processing the largest training sample...")

    large_row = largest_train_sample()
    large_source_report = read_json(
        ROOT
        / "data/derived/sbsmi_dataset_v1/train"
        / f"{large_row['sha']}.json"
    )

    large_matrix, large_details = convert_verified_sample(
        large_row,
        large_source_report,
    )

    # The SBSMI report verifies the input bytes only.
    # Raw-byte values are not compared with SBSMI values.
    large_details["matrix_shape"] = list(large_matrix.shape)
    large_details["matrix_min"] = float(large_matrix.min())
    large_details["matrix_max"] = float(large_matrix.max())

    memory_after = read_process_memory()
    mib = 1024 ** 2

    peak_working_mib = memory_after["peak_working_set_bytes"] / mib
    peak_commit_mib = memory_after["peak_commit_bytes"] / mib
    memory_pass = (
        max(peak_working_mib, peak_commit_mib) <= MEMORY_BUDGET_MIB
    )

    report = {
        "purpose": "raw-byte streaming correctness and memory check",
        "pilot_run": str(run_dir),
        "pilot_samples": len(results),
        "pilot_comparisons_passed": len(results),
        "absolute_tolerance": TOLERANCE,
        "chunk_size": CHUNK_SIZE,
        "pilot_results": results,
        "large_sample": large_details,
        "memory_before": memory_before,
        "memory_after": memory_after,
        "memory_budget_mib": MEMORY_BUDGET_MIB,
        "memory_budget_pass": memory_pass,
        "memory_scope": (
            "Windows process peaks up to the measurement point, "
            "including imports, metadata loading, the ten pilot checks "
            "and the large raw-byte conversion. Not per-function peaks."
        ),
        "raw_byte_stream_sha256": hashlib.sha256(
            (ROOT / "scripts/raw_byte_stream.py").read_bytes()
        ).hexdigest(),
        "check_script_sha256": hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
        "status": "PASS" if memory_pass else "MEMORY_BUDGET_EXCEEDED",
    }

    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    output_folder = OUTPUT_ROOT / run_id
    output_folder.mkdir(parents=True, exist_ok=False)
    report_path = output_folder / "report.json"

    with report_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, allow_nan=False)
        file.write("\n")

    print("Pilot comparisons: 10/10 PASS")
    print("Large sample family:", large_row["family"])
    print("Large sample SHA:", large_row["sha"])
    print("Large sample MiB:", round(int(large_row["file_size"]) / mib, 3))
    print("Large sample input and matrix checks: PASS")
    print(
        "Large sample read/hash/transform seconds:",
        round(large_details["read_hash_and_transform_seconds"], 3),
    )
    print("Peak working set MiB:", round(peak_working_mib, 2))
    print("Peak commit MiB:", round(peak_commit_mib, 2))
    print("Memory budget MiB:", MEMORY_BUDGET_MIB)
    print("Memory budget:", "PASS" if memory_pass else "EXCEEDED")
    print("Report:", report_path)

    if not memory_pass:
        raise RuntimeError(
            "Inspect memory use before starting dataset generation."
        )


if __name__ == "__main__":
    main()