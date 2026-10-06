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

import tlsh

from scripts.check_tlsh import hash_stream


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = (
    PROJECT_ROOT / "data/manifests/preprocessing_pilot.csv"
)
BINARY_FOLDER = PROJECT_ROOT / "data/raw/altered"
OUTPUT_ROOT = PROJECT_ROOT / "data/derived/tlsh_pilot"

MAX_FILE_BYTES = 5 * 1024 * 1024
CHUNK_SIZE = 65536

TRAIN_START = datetime(2019, 8, 1, tzinfo=timezone.utc)
TRAIN_END = datetime(2020, 2, 1, tzinfo=timezone.utc)


def load_pilot_manifest():
    required_columns = {
        "sha",
        "split",
        "timestamp",
        "family",
        "file_size",
        "zip_crc32",
    }

    with MANIFEST_PATH.open(
        encoding="utf-8-sig", newline=""
    ) as file:
        reader = csv.DictReader(file)

        if not required_columns.issubset(reader.fieldnames or []):
            raise ValueError("Pilot manifest is missing required columns.")

        rows = list(reader)

    if len(rows) != 10:
        raise ValueError(
            f"Expected 10 pilot samples, found {len(rows)}."
        )

    seen = set()

    for row in rows:
        sample_id = row["sha"].strip().lower()

        if re.fullmatch(r"[0-9a-f]{64}", sample_id) is None:
            raise ValueError(f"Invalid sample identifier: {sample_id}")

        if sample_id in seen:
            raise ValueError(f"Duplicate pilot sample: {sample_id}")

        seen.add(sample_id)
        row["sha"] = sample_id

        if row["split"].strip() != "train":
            raise ValueError("This pilot accepts training samples only.")

        observed_time = datetime.fromisoformat(row["timestamp"])

        if observed_time.tzinfo is None:
            raise ValueError("Timestamp must include a timezone.")

        observed_time = observed_time.astimezone(timezone.utc)

        if not TRAIN_START <= observed_time < TRAIN_END:
            raise ValueError(
                f"Sample is outside the training period: {sample_id}"
            )

        if not row["family"].strip():
            raise ValueError(f"Missing family label: {sample_id}")

        expected_size = int(row["file_size"])

        if not 0 < expected_size <= MAX_FILE_BYTES:
            raise ValueError(
                f"Sample exceeds the pilot size limit: {sample_id}"
            )

        crc_text = row["zip_crc32"].strip()

        if re.fullmatch(r"[0-9a-fA-F]{8}", crc_text) is None:
            raise ValueError(f"Invalid CRC value: {sample_id}")

    return rows


def process_sample(row):
    started = time.perf_counter()
    sample_id = row["sha"]
    binary_path = BINARY_FOLDER / f"{sample_id}.exe"
    expected_size = int(row["file_size"])

    if binary_path.stat().st_size != expected_size:
        raise ValueError("File size differs from the saved ZIP index.")

    # Read bytes only. The binary is never executed.
    # The bounded read also protects against an unexpectedly large file.
    read_started = time.perf_counter()

    with binary_path.open("rb") as file:
        data = file.read(MAX_FILE_BYTES + 1)

    read_seconds = time.perf_counter() - read_started

    if len(data) != expected_size:
        raise ValueError("Read byte count differs from the expected size.")

    actual_crc = zlib.crc32(data) & 0xFFFFFFFF
    expected_crc = int(row["zip_crc32"], 16)

    if actual_crc != expected_crc:
        raise ValueError("CRC differs from the saved ZIP index.")

    # This identifies the disarmed bytes, not the original sample name.
    content_sha256 = hashlib.sha256(data).hexdigest()

    one_shot_started = time.perf_counter()
    one_shot_hash = tlsh.hash(data)
    one_shot_seconds = time.perf_counter() - one_shot_started

    streaming_started = time.perf_counter()

    # Both methods receive exactly the same bytes.
    with io.BytesIO(data) as stream:
        streaming_hash = hash_stream(
            stream,
            chunk_size=CHUNK_SIZE,
        )

    streaming_seconds = time.perf_counter() - streaming_started

    if streaming_hash != one_shot_hash:
        raise RuntimeError(
            "Streaming and one-shot TLSH hashes differ: "
            f"{streaming_hash!r} != {one_shot_hash!r}"
        )

    if one_shot_hash == "TNULL":
        hash_status = "TNULL"
    else:
        if re.fullmatch(r"T1[0-9A-F]{70}", one_shot_hash) is None:
            raise RuntimeError("Unexpected TLSH digest format.")

        hash_status = "VALID"

    return {
        "sha": sample_id,
        "split": "train",
        "family": row["family"],
        "file_size": len(data),
        "disarmed_content_sha256": content_sha256,
        "crc32": f"{actual_crc:08x}",
        "crc_check": "PASS",
        "tlsh": one_shot_hash,
        "tlsh_status": hash_status,
        "exact_hash_match": True,
        "read_seconds": read_seconds,
        "one_shot_seconds": one_shot_seconds,
        "streaming_seconds": streaming_seconds,
        "total_seconds": time.perf_counter() - started,
        "status": "PASS",
        "error": "",
    }


def main():
    installed_version = version("py-tlsh")

    if installed_version != "5.0.0":
        raise RuntimeError("Expected py-tlsh 5.0.0.")

    rows = load_pilot_manifest()

    run_id = datetime.now(timezone.utc).strftime(
        "%Y%m%d_%H%M%S_%f"
    )
    output_folder = OUTPUT_ROOT / run_id
    output_folder.mkdir(parents=True, exist_ok=False)

    print("Purpose: TLSH engineering pilot")
    print("Split: train only")
    print("Samples:", len(rows))
    print("py-tlsh:", installed_version)
    print("Chunk size:", CHUNK_SIZE)
    print("Streaming policy: buffer the first 5 bytes")
    print("Output folder:", output_folder)

    results = []

    for index, row in enumerate(rows, start=1):
        try:
            result = process_sample(row)

            print(
                f"[{index}/{len(rows)}] "
                f"{row['family']} | "
                f"{result['file_size'] / (1024 * 1024):.3f} MiB | "
                f"CRC=PASS | exact_match=PASS | "
                f"TLSH={result['tlsh_status']} | "
                f"{result['total_seconds']:.4f}s"
            )

        except Exception as error:
            result = {
                "sha": row["sha"],
                "split": row["split"],
                "family": row["family"],
                "file_size": int(row["file_size"]),
                "status": "ERROR",
                "error": f"{type(error).__name__}: {error}",
            }

            print(
                f"[{index}/{len(rows)}] "
                f"{row['family']} | ERROR | {result['error']}"
            )

        results.append(result)

    fields = [
        "sha",
        "split",
        "family",
        "file_size",
        "disarmed_content_sha256",
        "crc32",
        "crc_check",
        "tlsh",
        "tlsh_status",
        "exact_hash_match",
        "read_seconds",
        "one_shot_seconds",
        "streaming_seconds",
        "total_seconds",
        "status",
        "error",
    ]

    summary_path = output_folder / "summary.csv"

    with summary_path.open(
        "w", encoding="utf-8", newline=""
    ) as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)

    successful = sum(r["status"] == "PASS" for r in results)
    valid_hashes = sum(
        r.get("tlsh_status") == "VALID" for r in results
    )
    null_hashes = sum(
        r.get("tlsh_status") == "TNULL" for r in results
    )
    errors = len(results) - successful

    report = {
        "purpose": "train-only TLSH engineering pilot",
        "py_tlsh_version": installed_version,
        "manifest": str(MANIFEST_PATH.relative_to(PROJECT_ROOT)),
        "manifest_sha256": hashlib.sha256(
            MANIFEST_PATH.read_bytes()
        ).hexdigest(),
        "pilot_script_sha256": hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
        "hash_helper_sha256": hashlib.sha256(
            (PROJECT_ROOT / "scripts/check_tlsh.py").read_bytes()
        ).hexdigest(),
        "chunk_size": CHUNK_SIZE,
        "initial_buffer_bytes": 5,
        "maximum_file_bytes": MAX_FILE_BYTES,
        "samples": len(results),
        "successful_checks": successful,
        "valid_tlsh_hashes": valid_hashes,
        "tnull_samples": null_hashes,
        "errors": errors,
        "similarity_threshold": None,
        "samples_removed": 0,
        "results": results,
    }

    report_path = output_folder / "report.json"

    with report_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False)
        file.write("\n")

    print()
    print("Successful checks:", successful)
    print("Valid TLSH hashes:", valid_hashes)
    print("TNULL samples:", null_hashes)
    print("Errors:", errors)
    print("Summary:", summary_path)
    print("Report:", report_path)
    print("No similarity threshold selected. No samples removed.")

    if errors:
        raise RuntimeError(
            "Pilot contains errors. Inspect the saved report."
        )


if __name__ == "__main__":
    main()