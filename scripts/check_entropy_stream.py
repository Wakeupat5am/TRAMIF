import argparse
import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import numpy as np

from scripts.check_raw_byte_stream import (
    DigestReader,
    largest_train_sample,
    load_pilot,
    read_json,
    verify_matrix,
)
from scripts.entropy_image_stream import stream_to_entropy_image
from scripts.measure_preprocessing_memory import read_process_memory


ROOT = Path(__file__).resolve().parents[1]
BINARY_DIR = ROOT / "data/raw/altered"
OUTPUT_ROOT = ROOT / "data/derived/entropy_stream_checks"

CHUNK_SIZE = 65536
TOLERANCE = 1e-12
MIB = 1024 ** 2


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def verify_entropy_config(run_dir):
    config = read_json(run_dir / "config.json")

    required = {
        "entropy_window_size": 256,
        "entropy_stride": 128,
        "entropy_tail_rule": "stop_at_first_window_reaching_eof",
        "entropy_partial_window": "existing_bytes_only",
        "entropy_overlap": "mean_of_covering_window_entropies",
        "empty_region_value": 0,
    }

    for key, expected in required.items():
        if config.get(key) != expected:
            raise ValueError(f"Unexpected entropy pilot configuration: {key}")


def convert_verified_sample(row, source_report):
    sample_id = row["sha"]

    if re.fullmatch(r"[0-9a-f]{64}", sample_id) is None:
        raise ValueError("Invalid sample identifier.")
    if row["split"] != "train":
        raise ValueError("This check accepts training samples only.")

    expected_size = int(row["file_size"])
    expected_crc = row["zip_crc32"].strip().lower()

    identity = {
        "sample_identifier": sample_id,
        "split": "train",
        "family": row["family"],
        "timestamp": row["timestamp"],
        "file_size": expected_size,
        "zip_crc32": expected_crc,
    }

    for key, value in identity.items():
        if source_report.get(key) != value:
            raise ValueError(f"Source report mismatch: {key}")

    expected_sha = source_report.get("disarmed_content_sha256", "")
    if re.fullmatch(r"[0-9a-f]{64}", expected_sha) is None:
        raise ValueError("Missing valid input content SHA-256.")

    path = BINARY_DIR / f"{sample_id}.exe"
    started = perf_counter()

    with path.open("rb") as file:
        before = os.fstat(file.fileno())
        if before.st_size != expected_size:
            raise ValueError("Binary size differs from the index.")

        reader = DigestReader(file)
        matrix = stream_to_entropy_image(
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

    elapsed = perf_counter() - started

    if reader.byte_count != expected_size:
        raise ValueError("Unexpected number of input bytes.")

    actual_sha = reader.sha256.hexdigest()
    actual_crc = f"{reader.crc32 & 0xFFFFFFFF:08x}"

    if actual_sha != expected_sha:
        raise ValueError("Input content differs from the saved report.")
    if actual_crc != expected_crc:
        raise ValueError("Input CRC differs from the saved index.")

    verify_matrix(matrix)
    if matrix.dtype != np.dtype("float64"):
        raise ValueError("Expected float64 calculation output.")

    details = {
        **identity,
        "disarmed_content_sha256": actual_sha,
        "input_checks": "PASS",
        "matrix_checks": "PASS",
        "matrix_shape": list(matrix.shape),
        "matrix_min": float(matrix.min()),
        "matrix_max": float(matrix.max()),
        "read_hash_and_transform_seconds": elapsed,
    }
    return matrix, details


def main():
    parser = argparse.ArgumentParser(
        description="Compare streamed entropy images with saved BODMAS pilots."
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    samples = load_pilot(run_dir)
    verify_entropy_config(run_dir)

    memory_before_checks = read_process_memory()

    print("Purpose: validate streamed entropy preprocessing")
    print("Reference: saved entropy pilot matrices")
    print("Split: train only")
    print("Window size: 256")
    print("Stride: 128")
    print("Chunk size:", CHUNK_SIZE)
    print("Absolute tolerance:", TOLERANCE)

    results = []
    reference_hashes = {}

    for number, row in enumerate(samples, start=1):
        report_path = run_dir / f"{row['sha']}.json"
        reference_hashes[report_path.name] = sha256_file(report_path)
        source_report = read_json(report_path)

        expected = np.asarray(
            source_report["normalized_matrices"]["entropy"],
            dtype=np.float64,
        )
        verify_matrix(expected)

        actual, details = convert_verified_sample(row, source_report)
        maximum_error = float(np.max(np.abs(actual - expected)))

        if maximum_error > TOLERANCE:
            raise RuntimeError(
                f"Entropy image mismatch: {row['sha']}, "
                f"maximum error={maximum_error:.6e}"
            )

        details["maximum_absolute_error"] = maximum_error
        details["reference_comparison"] = "PASS"
        results.append(details)

        print(
            f"[{number}/{len(samples)}] {row['family']} | "
            f"PASS | max_error={maximum_error:.3e}",
            flush=True,
        )

    print("\nProcessing the largest training sample...", flush=True)

    large_row = largest_train_sample()
    large_report_path = (
        ROOT / "data/derived/sbsmi_dataset_v1/train"
        / f"{large_row['sha']}.json"
    )
    large_source_report = read_json(large_report_path)

    memory_before_large = read_process_memory()

    large_matrix, large_details = convert_verified_sample(
        large_row,
        large_source_report,
    )

    memory_after_large = read_process_memory()

    # The SBSMI report verifies the input bytes only.
    # There is no comparison between entropy and SBSMI image values.
    large_details["reference_comparison"] = "NOT_PERFORMED"
    large_details["source_report_sha256"] = sha256_file(large_report_path)

    source_paths = [
        "scripts/entropy_core.py",
        "scripts/entropy_stream.py",
        "scripts/entropy_image_stream.py",
        "scripts/check_raw_byte_stream.py",
        "scripts/measure_preprocessing_memory.py",
        "scripts/check_entropy_stream.py",
    ]

    report = {
        "purpose": "entropy streaming correctness and resource observation",
        "status": "CORRECTNESS_PASS",
        "pilot_run": str(run_dir),
        "pilot_samples": len(results),
        "pilot_comparisons_passed": len(results),
        "absolute_tolerance": TOLERANCE,
        "window_size": 256,
        "stride": 128,
        "chunk_size": CHUNK_SIZE,
        "numpy_version": np.__version__,
        "pilot_config_sha256": sha256_file(run_dir / "config.json"),
        "pilot_reference_sha256": reference_hashes,
        "pilot_results": results,
        "large_sample": large_details,
        "memory_before_checks": memory_before_checks,
        "memory_before_large": memory_before_large,
        "memory_after_large": memory_after_large,
        "memory_assessment": "Measurements recorded for review; no automatic cap.",
        "memory_scope": (
            "Windows process measurements, including imported libraries. "
            "Peak counters cover the process lifetime. "
            "They are not isolated per-function allocation measurements."
        ),
        "source_sha256": {
            name: sha256_file(ROOT / name)
            for name in source_paths
        },
    }

    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    output_dir = OUTPUT_ROOT / run_id
    output_dir.mkdir(parents=True, exist_ok=False)
    report_path = output_dir / "report.json"

    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )

    print()
    print(f"Pilot comparisons: {len(results)}/{len(samples)} PASS")
    print("Large sample family:", large_row["family"])
    print("Large sample SHA:", large_row["sha"])
    print("Large sample MiB:", round(int(large_row["file_size"]) / MIB, 3))
    print("Large sample input and matrix checks: PASS")
    print(
        "Large sample read/hash/transform seconds:",
        round(large_details["read_hash_and_transform_seconds"], 3),
    )

    for label, memory in (
        ("Before all checks", memory_before_checks),
        ("Before large sample", memory_before_large),
        ("After large sample", memory_after_large),
    ):
        print(
            f"{label}: "
            f"working_set={memory['working_set_bytes'] / MIB:.2f} MiB | "
            f"commit={memory['commit_bytes'] / MIB:.2f} MiB | "
            f"peak_working_set={memory['peak_working_set_bytes'] / MIB:.2f} MiB | "
            f"peak_commit={memory['peak_commit_bytes'] / MIB:.2f} MiB"
        )

    print("Memory measurements recorded for review.")
    print("Correctness checks: PASS")
    print("Report:", report_path)


if __name__ == "__main__":
    main()