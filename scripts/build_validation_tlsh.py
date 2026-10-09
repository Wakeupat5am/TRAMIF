import argparse
import csv
import hashlib
import json
import os
import re
from collections import Counter
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

from scripts.build_raw_byte_dataset import load_inputs
from scripts.build_train_tlsh import (
    DigestReader,
    file_sha256,
    read_json,
    save_json,
    validate_record,
)
from scripts.check_tlsh import hash_stream


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "data/derived/validation_tlsh_v1"
TRAIN_CONFIG_PATH = ROOT / "data/derived/train_tlsh_v1/config.json"
SBSMI_DIR = ROOT / "data/derived/sbsmi_dataset_v1/validation"
BINARY_DIR = ROOT / "data/raw/altered"

CHUNK_SIZE = 65536
EXPECTED_VALIDATION_SAMPLES = 8263

CSV_FIELDS = [
    "sha", "split", "timestamp", "family", "file_size",
    "crc32", "disarmed_content_sha256", "tlsh",
    "tlsh_status", "config_sha256", "input_mtime_ns", "seconds",
]


def load_validation_inputs():
    samples, class_to_idx, sbsmi_config_id = load_inputs()
    rows = [row for row in samples if row["split"] == "validation"]

    if len(rows) != EXPECTED_VALIDATION_SAMPLES:
        raise ValueError("Expected 8,263 validation samples.")

    start = datetime(2020, 2, 1, tzinfo=timezone.utc)
    stop = datetime(2020, 4, 1, tzinfo=timezone.utc)

    for row in rows:
        observed = datetime.fromisoformat(row["timestamp"])

        if observed.tzinfo is None:
            raise ValueError("Timestamp must include a timezone.")

        observed = observed.astimezone(timezone.utc)
        if not start <= observed < stop:
            raise ValueError(f"Sample outside validation period: {row['sha']}")

    return rows, class_to_idx, sbsmi_config_id


def prepare_output(sample_count, sbsmi_config_id):
    installed_version = version("py-tlsh")
    if installed_version != "5.0.0":
        raise RuntimeError("Expected py-tlsh 5.0.0.")

    helper_path = ROOT / "scripts/check_tlsh.py"
    helper_sha = file_sha256(helper_path)
    index_path = ROOT / "data/manifests/train_validation_zip_index.csv"
    index_sha = file_sha256(index_path)

    train_config = read_json(TRAIN_CONFIG_PATH)
    required_train_config = {
        "py_tlsh": installed_version,
        "chunk_size": CHUNK_SIZE,
        "initial_buffer_bytes": 5,
        "hash_helper_sha256": helper_sha,
        "zip_index_sha256": index_sha,
    }

    for key, expected in required_train_config.items():
        if train_config.get(key) != expected:
            raise ValueError(f"Train/validation hashing setup differs: {key}")

    config = {
        "schema_version": 1,
        "purpose": "validation TLSH inventory",
        "py_tlsh": installed_version,
        "chunk_size": CHUNK_SIZE,
        "initial_buffer_bytes": 5,
        "validation_samples": sample_count,
        "zip_index_sha256": index_sha,
        "manifest_sha256": file_sha256(
            ROOT / "data/manifests/train_validation_manifest.csv"
        ),
        "class_mapping_sha256": file_sha256(
            ROOT / "docs/class_mapping.json"
        ),
        "sbsmi_config_sha256": sbsmi_config_id,
        "train_config_file_sha256": file_sha256(TRAIN_CONFIG_PATH),
        "hash_helper_sha256": helper_sha,
        "builder_sha256": file_sha256(Path(__file__)),
        "shared_helpers_sha256": {
            name: file_sha256(ROOT / name)
            for name in (
                "scripts/build_train_tlsh.py",
                "scripts/build_raw_byte_dataset.py",
            )
        },
    }

    config_id = hashlib.sha256(
        json.dumps(config, sort_keys=True).encode("utf-8")
    ).hexdigest()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    config_path = OUTPUT_DIR / "config.json"

    if config_path.exists():
        if read_json(config_path) != config:
            raise RuntimeError(
                "Configuration or source code changed. "
                "Review before reusing this output folder."
            )
    else:
        if any(OUTPUT_DIR.iterdir()):
            raise RuntimeError("Output folder is nonempty without config.json.")
        save_json(config_path, config)

    return config_id


def expected_identity(row, class_to_idx, config_id, sbsmi_config_id):
    report = read_json(SBSMI_DIR / f"{row['sha']}.json")

    required = {
        "sample_identifier": row["sha"],
        "split": "validation",
        "timestamp": row["timestamp"],
        "family": row["family"],
        "class_index": class_to_idx[row["family"]],
        "file_size": int(row["file_size"]),
        "zip_crc32": row["zip_crc32"].lower(),
        "config_sha256": sbsmi_config_id,
        "status": "PASS",
    }

    for key, expected in required.items():
        if report.get(key) != expected:
            raise ValueError(f"SBSMI source report mismatch: {key}")

    content_sha = report.get("disarmed_content_sha256", "")
    if re.fullmatch(r"[0-9a-f]{64}", content_sha) is None:
        raise ValueError("Missing valid disarmed content SHA-256.")

    return {
        "sha": row["sha"],
        "split": "validation",
        "timestamp": row["timestamp"],
        "family": row["family"],
        "file_size": int(row["file_size"]),
        "crc32": row["zip_crc32"].lower(),
        "disarmed_content_sha256": content_sha,
        "config_sha256": config_id,
    }


def process_sample(
    row, class_to_idx, config_id, sbsmi_config_id, records_dir
):
    expected = expected_identity(
        row, class_to_idx, config_id, sbsmi_config_id
    )
    binary_path = BINARY_DIR / f"{row['sha']}.exe"
    record_path = records_dir / f"{row['sha']}.json"

    current = binary_path.stat()
    if current.st_size != expected["file_size"]:
        raise ValueError("Binary size differs from the index.")

    if record_path.exists():
        record = read_json(record_path)
        validate_record(record, expected)

        if record.get("input_mtime_ns") == current.st_mtime_ns:
            # Same cache policy as the train inventory.
            # This does not rehash the binary.
            return record, "CACHED"

    started = perf_counter()

    with binary_path.open("rb") as file:
        before = os.fstat(file.fileno())
        if before.st_size != expected["file_size"]:
            raise ValueError("Binary size changed before reading.")

        reader = DigestReader(file)
        digest = hash_stream(reader, chunk_size=CHUNK_SIZE)
        after = os.fstat(file.fileno())

    if (
        before.st_size != after.st_size
        or before.st_mtime_ns != after.st_mtime_ns
    ):
        raise RuntimeError("Binary changed while being read.")

    if reader.byte_count != expected["file_size"]:
        raise ValueError("Read byte count differs from the index.")
    if f"{reader.crc32 & 0xFFFFFFFF:08x}" != expected["crc32"]:
        raise ValueError("Input CRC32 differs from the index.")
    if reader.sha256.hexdigest() != expected["disarmed_content_sha256"]:
        raise ValueError("Input bytes differ from the SBSMI source.")

    record = {
        **expected,
        "tlsh": digest,
        "tlsh_status": "TNULL" if digest == "TNULL" else "VALID",
        "input_mtime_ns": after.st_mtime_ns,
        "seconds": perf_counter() - started,
    }

    validate_record(record, expected)
    save_json(record_path, record)
    return record, "CREATED"


def main():
    parser = argparse.ArgumentParser(
        description="Build resumable TLSH records for validation samples."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Number of samples to visit; 0 means all validation samples.",
    )
    args = parser.parse_args()
    if args.limit < 0:
        parser.error("--limit must be zero or positive.")

    rows, class_to_idx, sbsmi_config_id = load_validation_inputs()
    config_id = prepare_output(len(rows), sbsmi_config_id)

    records_dir = OUTPUT_DIR / "records"
    records_dir.mkdir(exist_ok=True)

    selected = rows if args.limit == 0 else rows[:args.limit]
    results = []
    counts = Counter()

    print("Purpose: validation TLSH inventory")
    print("Total validation samples:", len(rows))
    print("Requested samples:", len(selected))
    print("Chunk size:", CHUNK_SIZE)
    print("Streaming policy: buffer the first 5 bytes")
    print("CACHED means reused; binary content is not rehashed.")
    print("Processing sequentially...", flush=True)

    for number, row in enumerate(selected, start=1):
        try:
            record, action = process_sample(
                row,
                class_to_idx,
                config_id,
                sbsmi_config_id,
                records_dir,
            )
        except Exception as error:
            raise RuntimeError(
                f"Stopped at sample {number}: {row['sha']}. "
                "Completed records are retained. "
                f"Cause: {error}"
            ) from error

        results.append(record)
        counts[action] += 1

        if (
            len(selected) <= 20
            or number == 1
            or number % 100 == 0
            or number == len(selected)
        ):
            print(
                f"[{number}/{len(selected)}] {action} | "
                f"{row['family']} | TLSH={record['tlsh_status']}",
                flush=True,
            )

    complete = len(results) == len(rows)
    suffix = "" if complete else "_preview"
    csv_path = OUTPUT_DIR / f"validation_tlsh{suffix}.csv"
    temporary_csv = csv_path.with_suffix(".csv.tmp")

    with temporary_csv.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(results)

    temporary_csv.replace(csv_path)

    valid = sum(row["tlsh_status"] == "VALID" for row in results)
    null_count = sum(row["tlsh_status"] == "TNULL" for row in results)

    summary = {
        "config_sha256": config_id,
        "complete_validation_inventory": complete,
        "total_validation_samples": len(rows),
        "processed_samples": len(results),
        "created_this_run": counts["CREATED"],
        "cached_this_run": counts["CACHED"],
        "valid_tlsh_hashes": valid,
        "tnull_samples": null_count,
        "errors": 0,
        "csv_sha256": file_sha256(csv_path),
        "near_duplicate_policy_applied": False,
        "samples_removed": 0,
    }
    save_json(OUTPUT_DIR / f"summary{suffix}.json", summary)

    print()
    print("Created:", counts["CREATED"])
    print("Cached:", counts["CACHED"])
    print("Valid TLSH hashes:", valid)
    print("TNULL samples:", null_count)
    print("Errors: 0")
    print("Complete validation inventory:", complete)
    print("Saved table:", csv_path)
    print("Inventory only. No grouping or sample removal performed.")


if __name__ == "__main__":
    main()