import argparse
import csv
import os
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import numpy as np

from scripts.build_raw_byte_dataset import (
    DigestReader,
    cached_output_is_valid,
    config_sha256,
    file_sha256,
    load_inputs,
    read_json,
    verify_matrix,
    write_json_atomic,
)
from scripts.entropy_image_stream import stream_to_entropy_image


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "data/manifests/train_validation_manifest.csv"
INDEX_PATH = ROOT / "data/manifests/train_validation_zip_index.csv"
MAPPING_PATH = ROOT / "docs/class_mapping.json"

BINARY_DIR = ROOT / "data/raw/altered"
SBSMI_DIR = ROOT / "data/derived/sbsmi_dataset_v1"
RAW_DIR = ROOT / "data/derived/raw_byte_dataset_v1"
OUTPUT_DIR = ROOT / "data/derived/entropy_dataset_v1"

CHUNK_SIZE = 65536
IMAGE_DTYPE = np.dtype("<f4")


def load_raw_config():
    config = read_json(RAW_DIR / "config.json")

    required = {
        "representation": "raw-byte",
        "image_shape": [64, 64],
        "storage_dtype": "<f4",
    }
    for key, expected in required.items():
        if config.get(key) != expected:
            raise ValueError(f"Unexpected raw-byte configuration: {key}")

    for key, path in (
        ("manifest_sha256", MANIFEST_PATH),
        ("zip_index_sha256", INDEX_PATH),
        ("class_mapping_sha256", MAPPING_PATH),
    ):
        if config.get(key) != file_sha256(path):
            raise ValueError(f"Raw-byte input provenance differs: {key}")

    return config_sha256(config)


def prepare_output():
    source_files = [
        "scripts/entropy_core.py",
        "scripts/entropy_stream.py",
        "scripts/entropy_image_stream.py",
        "scripts/build_raw_byte_dataset.py",
        "scripts/build_entropy_dataset.py",
    ]

    config = {
        "schema_version": 1,
        "representation": "local-entropy",
        "input_bytes": "entire disarmed binary",
        "window_size": 256,
        "stride": 128,
        "window_end_rule": "stop_at_first_window_reaching_eof",
        "partial_window_rule": "existing_bytes_only",
        "overlap_rule": "mean_of_covering_window_entropies",
        "normalization": "divide entropy in bits by 8; no further scaling",
        "row_width": 256,
        "image_shape": [64, 64],
        "resize": "mask-aware area average",
        "padding": "zero fill; exclude padded positions from averages",
        "empty_output_region": 0,
        "storage": "NPY",
        "storage_dtype": "<f4",
        "calculation_dtype": "float64",
        "chunk_size": CHUNK_SIZE,
        "numpy_version": np.__version__,
        "manifest_sha256": file_sha256(MANIFEST_PATH),
        "zip_index_sha256": file_sha256(INDEX_PATH),
        "class_mapping_sha256": file_sha256(MAPPING_PATH),
        "sbsmi_config_file_sha256": file_sha256(
            SBSMI_DIR / "config.json"
        ),
        "raw_byte_config_file_sha256": file_sha256(
            RAW_DIR / "config.json"
        ),
        "source_sha256": {
            name: file_sha256(ROOT / name)
            for name in source_files
        },
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    config_path = OUTPUT_DIR / "config.json"

    if config_path.exists():
        if read_json(config_path) != config:
            raise ValueError(
                "Configuration or source code changed. "
                "Review before reusing this dataset folder."
            )
    else:
        if any(OUTPUT_DIR.iterdir()):
            raise ValueError("Output folder is nonempty without config.json.")
        write_json_atomic(config_path, config)

    return config_sha256(config)


def source_content_sha(row, class_to_idx, sbsmi_config_id, raw_config_id):
    sha = row["sha"]
    split = row["split"]

    identity = {
        "sample_identifier": sha,
        "split": split,
        "timestamp": row["timestamp"],
        "family": row["family"],
        "class_index": class_to_idx[row["family"]],
        "file_size": int(row["file_size"]),
        "zip_crc32": row["zip_crc32"].lower(),
    }

    content_hashes = []

    for name, directory, expected_config_id in (
        ("SBSMI", SBSMI_DIR, sbsmi_config_id),
        ("raw-byte", RAW_DIR, raw_config_id),
    ):
        report = read_json(directory / split / f"{sha}.json")

        for key, value in identity.items():
            if report.get(key) != value:
                raise ValueError(f"{name} source report mismatch: {key}")

        if (
            report.get("status") != "PASS"
            or report.get("config_sha256") != expected_config_id
        ):
            raise ValueError(f"Invalid {name} source report provenance.")

        content_sha = report.get("disarmed_content_sha256", "")
        if re.fullmatch(r"[0-9a-f]{64}", content_sha) is None:
            raise ValueError(f"Missing valid content SHA-256: {name}")

        content_hashes.append(content_sha)

    if content_hashes[0] != content_hashes[1]:
        raise ValueError("SBSMI and raw-byte reports identify different bytes.")

    return identity, content_hashes[0]


def process_one(
    row,
    class_to_idx,
    config_id,
    sbsmi_config_id,
    raw_config_id,
):
    identity, content_sha = source_content_sha(
        row, class_to_idx, sbsmi_config_id, raw_config_id
    )

    sha = row["sha"]
    expected_size = identity["file_size"]

    expected = {
        **identity,
        "representation": "local-entropy",
        "disarmed_content_sha256": content_sha,
        "config_sha256": config_id,
    }

    split_dir = OUTPUT_DIR / row["split"]
    split_dir.mkdir(parents=True, exist_ok=True)

    matrix_path = split_dir / f"{sha}.npy"
    report_path = split_dir / f"{sha}.json"
    cached = cached_output_is_valid(matrix_path, report_path, expected)

    binary_path = BINARY_DIR / f"{sha}.exe"

    with binary_path.open("rb") as file:
        before = os.fstat(file.fileno())
        if before.st_size != expected_size:
            raise ValueError("Binary size differs from the index.")

        reader = DigestReader(file)

        if cached:
            # Rehash the source even when reusing an existing output.
            while reader.read(CHUNK_SIZE):
                pass
        else:
            matrix64 = stream_to_entropy_image(
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

    if reader.byte_count != expected_size:
        raise ValueError("Unexpected number of input bytes.")
    if reader.sha256.hexdigest() != content_sha:
        raise ValueError("Input differs from the raw-byte/SBSMI source.")
    if f"{reader.crc32 & 0xFFFFFFFF:08x}" != identity["zip_crc32"]:
        raise ValueError("Input CRC32 differs from the index.")

    if cached:
        return "VERIFIED"

    matrix = matrix64.astype(IMAGE_DTYPE)
    verify_matrix(matrix)

    cast_error = float(
        np.max(np.abs(matrix64 - matrix.astype(np.float64)))
    )
    if cast_error > 3e-8:
        raise RuntimeError("Unexpected float32 conversion error.")

    temporary = matrix_path.with_name(matrix_path.name + ".tmp")
    with temporary.open("wb") as file:
        np.save(file, matrix, allow_pickle=False)

    restored = np.load(temporary, allow_pickle=False)
    verify_matrix(restored)
    if not np.array_equal(matrix, restored):
        raise RuntimeError("NPY save/load changed the matrix.")

    npy_sha = file_sha256(temporary)
    temporary.replace(matrix_path)

    write_json_atomic(report_path, {
        **expected,
        "status": "PASS",
        "input_matches_raw_byte_and_sbsmi": True,
        "image_shape": [64, 64],
        "storage_dtype": "<f4",
        "array_round_trip": "PASS",
        "maximum_float32_cast_error": cast_error,
        "npy_sha256": npy_sha,
    })

    return "CREATED"


def main():
    parser = argparse.ArgumentParser(
        description="Build streamed entropy matrices for train/validation."
    )
    parser.add_argument("--limit", type=int, default=20)
    args = parser.parse_args()

    if args.limit < 1:
        raise ValueError("--limit must be positive.")

    samples, class_to_idx, sbsmi_config_id = load_inputs()
    raw_config_id = load_raw_config()
    config_id = prepare_output()

    selected = samples[:args.limit]
    split_counts = Counter(row["split"] for row in selected)

    print("Purpose: build entropy training matrices")
    print("Total manifest samples:", len(samples))
    print("Requested samples:", len(selected))
    print("Selected train:", split_counts["train"])
    print("Selected validation:", split_counts["validation"])
    print("Window size: 256")
    print("Stride: 128")
    print("Storage: float32 NPY, shape (64, 64), range [0, 1]")
    print("Input checked against raw-byte and SBSMI reports.")
    print("Output folder:", OUTPUT_DIR)
    print("Processing sequentially...", flush=True)

    log_dir = OUTPUT_DIR / "logs"
    log_dir.mkdir(exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    log_path = log_dir / f"{run_id}.csv"
    counts = Counter()

    with log_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["sha", "split", "family", "status", "seconds", "error"],
        )
        writer.writeheader()
        file.flush()

        for number, row in enumerate(selected, start=1):
            started = perf_counter()
            error_text = ""

            try:
                status = process_one(
                    row,
                    class_to_idx,
                    config_id,
                    sbsmi_config_id,
                    raw_config_id,
                )
            except Exception as error:
                status = "ERROR"
                error_text = f"{type(error).__name__}: {error}"

            seconds = perf_counter() - started
            counts[status] += 1

            writer.writerow({
                "sha": row["sha"],
                "split": row["split"],
                "family": row["family"],
                "status": status,
                "seconds": seconds,
                "error": error_text,
            })
            file.flush()

            if (
                len(selected) <= 20
                or number % 100 == 0
                or number == len(selected)
                or status == "ERROR"
            ):
                print(
                    f"[{number}/{len(selected)}] {status} | "
                    f"{row['split']} | {row['family']} | {seconds:.3f}s",
                    flush=True,
                )
                if error_text:
                    print(" ", error_text, flush=True)

            if status == "ERROR":
                break

    print()
    print("Created:", counts["CREATED"])
    print("Verified existing:", counts["VERIFIED"])
    print("Errors:", counts["ERROR"])
    print("Run log:", log_path)

    if counts["ERROR"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()