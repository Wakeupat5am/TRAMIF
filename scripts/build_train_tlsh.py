import argparse
import csv
import hashlib
import io
import json
import re
import time
import zlib
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

from scripts.check_tlsh import hash_stream


ROOT = Path(__file__).resolve().parents[1]
INDEX_PATH = ROOT / "data/manifests/train_validation_zip_index.csv"
BINARY_FOLDER = ROOT / "data/raw/altered"
SBSMI_REPORT_FOLDER = ROOT / "data/derived/sbsmi_dataset_v1/train"
OUTPUT_FOLDER = ROOT / "data/derived/train_tlsh_v1"

CHUNK_SIZE = 65536
EXPECTED_TRAIN_SAMPLES = 18061
TRAIN_START = datetime(2019, 8, 1, tzinfo=timezone.utc)
TRAIN_END = datetime(2020, 2, 1, tzinfo=timezone.utc)


def save_json(path, value):
    """Replace the destination only after the temporary file is complete."""
    temporary_path = path.with_suffix(path.suffix + ".tmp")

    with temporary_path.open("w", encoding="utf-8") as file:
        json.dump(value, file, indent=2, ensure_ascii=False)
        file.write("\n")

    temporary_path.replace(path)


def read_json(path):
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def file_sha256(path):
    # Used only for small source files.
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_train_rows():
    index_bytes = INDEX_PATH.read_bytes()
    text = index_bytes.decode("utf-8-sig")

    reader = csv.DictReader(io.StringIO(text))
    required = {
        "sha", "split", "timestamp",
        "family", "file_size", "zip_crc32",
    }

    if not required.issubset(reader.fieldnames or []):
        raise ValueError("ZIP index is missing required columns.")

    rows = [row for row in reader if row["split"] == "train"]

    if len(rows) != EXPECTED_TRAIN_SAMPLES:
        raise ValueError(
            f"Expected {EXPECTED_TRAIN_SAMPLES} train samples, "
            f"found {len(rows)}."
        )

    seen = set()

    for row in rows:
        sample_id = row["sha"].strip().lower()

        if re.fullmatch(r"[0-9a-f]{64}", sample_id) is None:
            raise ValueError(f"Invalid SHA identifier: {sample_id}")

        if sample_id in seen:
            raise ValueError(f"Duplicate SHA identifier: {sample_id}")

        seen.add(sample_id)
        row["sha"] = sample_id

        observed = datetime.fromisoformat(row["timestamp"])

        if observed.tzinfo is None:
            raise ValueError("Timestamp must include a timezone.")

        observed = observed.astimezone(timezone.utc)

        if not TRAIN_START <= observed < TRAIN_END:
            raise ValueError(f"Sample outside training period: {sample_id}")

        if not row["family"].strip():
            raise ValueError(f"Missing family: {sample_id}")

        row["file_size"] = int(row["file_size"])

        if row["file_size"] <= 0:
            raise ValueError(f"Invalid file size: {sample_id}")

        row["zip_crc32"] = row["zip_crc32"].strip().lower()

        if re.fullmatch(r"[0-9a-f]{8}", row["zip_crc32"]) is None:
            raise ValueError(f"Invalid CRC: {sample_id}")

    return rows, hashlib.sha256(index_bytes).hexdigest()


class DigestReader:
    """Track SHA-256, CRC and byte count while TLSH reads the stream."""

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


def expected_identity(row, config_id):
    report_path = SBSMI_REPORT_FOLDER / f"{row['sha']}.json"
    report = read_json(report_path)

    if report.get("sample_identifier") != row["sha"]:
        raise ValueError("SBSMI report identifies another sample.")

    if report.get("file_size") != row["file_size"]:
        raise ValueError("SBSMI report and ZIP index sizes differ.")

    content_sha = report.get("disarmed_content_sha256", "")

    if re.fullmatch(r"[0-9a-f]{64}", content_sha) is None:
        raise ValueError("Missing or invalid SBSMI content SHA-256.")

    return {
        "sha": row["sha"],
        "split": "train",
        "timestamp": row["timestamp"],
        "family": row["family"],
        "file_size": row["file_size"],
        "crc32": row["zip_crc32"],
        "disarmed_content_sha256": content_sha,
        "config_sha256": config_id,
    }


def validate_record(record, expected):
    for key, value in expected.items():
        if record.get(key) != value:
            raise ValueError(f"Saved record differs at field: {key}")

    digest = record.get("tlsh", "")

    if digest == "TNULL":
        expected_status = "TNULL"
    elif re.fullmatch(r"T1[0-9A-F]{70}", digest):
        expected_status = "VALID"
    else:
        raise ValueError("Saved record contains an invalid TLSH digest.")

    if record.get("tlsh_status") != expected_status:
        raise ValueError("Saved TLSH status is inconsistent.")


def process_sample(row, config_id, records_folder):
    expected = expected_identity(row, config_id)
    binary_path = BINARY_FOLDER / f"{row['sha']}.exe"
    record_path = records_folder / f"{row['sha']}.json"

    before = binary_path.stat()

    if before.st_size != row["file_size"]:
        raise ValueError("Binary size differs from the ZIP index.")

    if record_path.exists():
        record = read_json(record_path)
        validate_record(record, expected)

        if record.get("input_mtime_ns") == before.st_mtime_ns:
            # Reuse the previously checked result.
            # This does not rehash the binary.
            return record, "CACHED"

    started = time.perf_counter()

    with binary_path.open("rb") as file:
        reader = DigestReader(file)
        digest = hash_stream(reader, chunk_size=CHUNK_SIZE)

    after = binary_path.stat()

    if (
        before.st_size != after.st_size
        or before.st_mtime_ns != after.st_mtime_ns
    ):
        raise RuntimeError("Binary changed while being read.")

    if reader.byte_count != expected["file_size"]:
        raise ValueError("Read byte count differs from the ZIP index.")

    actual_crc = f"{reader.crc32 & 0xFFFFFFFF:08x}"

    if actual_crc != expected["crc32"]:
        raise ValueError("Binary CRC differs from the ZIP index.")

    actual_sha = reader.sha256.hexdigest()

    if actual_sha != expected["disarmed_content_sha256"]:
        raise ValueError("Binary content differs from the SBSMI input.")

    record = {
        **expected,
        "tlsh": digest,
        "tlsh_status": "TNULL" if digest == "TNULL" else "VALID",
        "input_mtime_ns": after.st_mtime_ns,
        "seconds": time.perf_counter() - started,
    }

    validate_record(record, expected)
    save_json(record_path, record)

    return record, "CREATED"


def main():
    parser = argparse.ArgumentParser(
        description="Build resumable TLSH records for training samples."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Number of train samples to process; 0 means all.",
    )
    args = parser.parse_args()

    if args.limit < 0:
        parser.error("--limit must be zero or positive.")

    if version("py-tlsh") != "5.0.0":
        raise RuntimeError("Expected py-tlsh 5.0.0.")

    rows, index_sha = load_train_rows()

    config = {
        "schema_version": 1,
        "purpose": "train-only TLSH inventory",
        "py_tlsh": "5.0.0",
        "chunk_size": CHUNK_SIZE,
        "initial_buffer_bytes": 5,
        "train_samples": len(rows),
        "zip_index_sha256": index_sha,
        "builder_sha256": file_sha256(Path(__file__)),
        "hash_helper_sha256": file_sha256(
            ROOT / "scripts/check_tlsh.py"
        ),
    }

    config_id = hashlib.sha256(
        json.dumps(config, sort_keys=True).encode("utf-8")
    ).hexdigest()

    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
    config_path = OUTPUT_FOLDER / "config.json"

    if config_path.exists():
        if read_json(config_path) != config:
            raise RuntimeError(
                "Configuration, source code or index has changed. "
                "Stop and review before reusing this output folder."
            )
    else:
        save_json(config_path, config)

    records_folder = OUTPUT_FOLDER / "records"
    records_folder.mkdir(exist_ok=True)

    selected = rows if args.limit == 0 else rows[:args.limit]
    results = []
    created = 0
    cached = 0

    print("Purpose: train-only TLSH inventory")
    print("Total train samples:", len(rows))
    print("Requested samples:", len(selected))
    print("Chunk size:", CHUNK_SIZE)
    print("Processing sequentially...")
    print("CACHED means reused; binary content is not rehashed.")

    for number, row in enumerate(selected, start=1):
        try:
            record, action = process_sample(
                row, config_id, records_folder
            )
        except Exception as error:
            raise RuntimeError(
                f"Stopped at sample {number}: {row['sha']}. "
                "Previously completed records are retained. "
                f"Cause: {error}"
            ) from error

        results.append(record)

        if action == "CREATED":
            created += 1
        else:
            cached += 1

        if (
            len(selected) <= 20
            or number == 1
            or number % 100 == 0
            or number == len(selected)
        ):
            print(
                f"[{number}/{len(selected)}] {action} | "
                f"{row['family']} | TLSH={record['tlsh_status']}"
            )

    complete = len(selected) == len(rows)
    suffix = "" if complete else "_preview"

    csv_path = OUTPUT_FOLDER / f"train_tlsh{suffix}.csv"
    temporary_csv = csv_path.with_suffix(".csv.tmp")

    fields = [
        "sha", "split", "timestamp", "family", "file_size",
        "crc32", "disarmed_content_sha256", "tlsh",
        "tlsh_status", "config_sha256", "input_mtime_ns", "seconds",
    ]

    with temporary_csv.open(
        "w", encoding="utf-8", newline=""
    ) as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)

    temporary_csv.replace(csv_path)

    valid = sum(r["tlsh_status"] == "VALID" for r in results)
    null_count = sum(r["tlsh_status"] == "TNULL" for r in results)

    summary = {
        "config_sha256": config_id,
        "complete_train_inventory": complete,
        "total_train_samples": len(rows),
        "processed_samples": len(results),
        "created_this_run": created,
        "cached_this_run": cached,
        "valid_tlsh_hashes": valid,
        "tnull_samples": null_count,
        "errors": 0,
        "csv_sha256": file_sha256(csv_path),
        "similarity_threshold": None,
        "samples_removed": 0,
    }

    save_json(
        OUTPUT_FOLDER / f"summary{suffix}.json",
        summary,
    )

    print()
    print("Created:", created)
    print("Cached:", cached)
    print("Valid TLSH hashes:", valid)
    print("TNULL samples:", null_count)
    print("Errors: 0")
    print("Complete train inventory:", complete)
    print("Saved table:", csv_path)
    print("No similarity threshold selected. No samples removed.")


if __name__ == "__main__":
    main()