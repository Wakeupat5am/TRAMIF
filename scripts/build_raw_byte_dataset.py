import argparse
import csv
import hashlib
import json
import os
import re
import zlib
from collections import Counter
from datetime import date, datetime, timezone
from pathlib import Path
from time import perf_counter

import numpy as np

from scripts.raw_byte_stream import stream_to_raw_byte


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "data/manifests/train_validation_manifest.csv"
INDEX_PATH = ROOT / "data/manifests/train_validation_zip_index.csv"
MAPPING_PATH = ROOT / "docs/class_mapping.json"
BINARY_DIR = ROOT / "data/raw/altered"
SBSMI_DIR = ROOT / "data/derived/sbsmi_dataset_v1"
OUTPUT_DIR = ROOT / "data/derived/raw_byte_dataset_v1"

CHUNK_SIZE = 65536
IMAGE_DTYPE = np.dtype("<f4")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def config_sha256(config):
    encoded = json.dumps(config, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def write_json_atomic(path, value):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


class DigestReader:
    """Fingerprint the same bytes that are read by the transformation."""

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
    if matrix.dtype != IMAGE_DTYPE:
        raise ValueError("Expected float32 storage.")
    if not np.isfinite(matrix).all():
        raise ValueError("Matrix contains non-finite values.")
    if (matrix < 0).any() or (matrix > 1).any():
        raise ValueError("Matrix values must be in [0, 1].")


def cached_output_is_valid(matrix_path, report_path, expected):
    if not matrix_path.is_file() or not report_path.is_file():
        return False

    try:
        report = read_json(report_path)
        if not isinstance(report, dict):
            return False
        if any(report.get(key) != value for key, value in expected.items()):
            return False
        if report.get("status") != "PASS":
            return False
        if report.get("array_round_trip") != "PASS":
            return False
        if file_sha256(matrix_path) != report["npy_sha256"]:
            return False

        matrix = np.load(matrix_path, allow_pickle=False)
        verify_matrix(matrix)
        return True
    except (OSError, ValueError, KeyError, TypeError, EOFError):
        return False


def load_inputs():
    manifest = read_csv(MANIFEST_PATH)
    indexed = read_csv(INDEX_PATH)
    mapping = read_json(MAPPING_PATH)
    class_to_idx = mapping["class_to_idx"]

    labels = list(class_to_idx.values())
    if (
        mapping["num_classes"] != 51
        or len(labels) != 51
        or any(type(label) is not int for label in labels)
        or sorted(labels) != list(range(51))
    ):
        raise ValueError("Expected the frozen mapping of 51 classes.")

    if not manifest:
        raise ValueError("Manifest is empty.")
    if len({row["sha"] for row in manifest}) != len(manifest):
        raise ValueError("Duplicate SHA in manifest.")
    if len({row["sha"] for row in indexed}) != len(indexed):
        raise ValueError("Duplicate SHA in ZIP index.")

    index_by_sha = {row["sha"]: row for row in indexed}
    if {row["sha"] for row in manifest} != set(index_by_sha):
        raise ValueError("Manifest and ZIP index contain different samples.")

    periods = {
        "train": (date(2019, 8, 1), date(2020, 2, 1)),
        "validation": (date(2020, 2, 1), date(2020, 4, 1)),
    }
    samples = []

    for row in manifest:
        sha = row["sha"]
        split = row["split"]
        if re.fullmatch(r"[0-9a-f]{64}", sha) is None:
            raise ValueError("Invalid sample identifier.")
        if split not in periods:
            raise ValueError("Only train and validation are allowed.")
        if row["family"] not in class_to_idx:
            raise ValueError("Family is absent from the frozen mapping.")

        observed = datetime.fromisoformat(row["timestamp"]).date()
        start, stop = periods[split]
        if not start <= observed < stop:
            raise ValueError(f"Timestamp outside its split: {sha}")

        source = index_by_sha[sha]
        for key in ("sha", "split", "timestamp", "family"):
            if source[key] != row[key]:
                raise ValueError(f"Manifest/index mismatch: {sha}, {key}")
        if int(source["file_size"]) <= 0:
            raise ValueError("Invalid indexed file size.")
        if re.fullmatch(r"[0-9a-fA-F]{8}", source["zip_crc32"]) is None:
            raise ValueError("Invalid indexed CRC32.")

        samples.append(source)

    source_config = read_json(SBSMI_DIR / "config.json")
    for key, path in (
        ("manifest_sha256", MANIFEST_PATH),
        ("zip_index_sha256", INDEX_PATH),
        ("class_mapping_sha256", MAPPING_PATH),
    ):
        if source_config.get(key) != file_sha256(path):
            raise ValueError(f"SBSMI input provenance differs: {key}")

    return samples, class_to_idx, config_sha256(source_config)


def prepare_output():
    config = {
        "schema_version": 1,
        "representation": "raw-byte",
        "row_width": 256,
        "image_shape": [64, 64],
        "resize": "mask-aware area average",
        "padding": "zero fill; exclude padded positions from averages",
        "empty_output_region": 0,
        "input_bytes": "entire disarmed binary",
        "normalization": "divide area-averaged byte values by 255",
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
        "raw_byte_stream_source_sha256": file_sha256(
            ROOT / "scripts/raw_byte_stream.py"
        ),
        "builder_source_sha256": file_sha256(Path(__file__)),
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


def process_one(row, class_to_idx, config_id, sbsmi_config_id):
    sha = row["sha"]
    split = row["split"]
    expected_size = int(row["file_size"])

    # Reuse the recorded input fingerprint, not the SBSMI image.
    source_report = read_json(SBSMI_DIR / split / f"{sha}.json")
    identity = {
        "sample_identifier": sha,
        "split": split,
        "timestamp": row["timestamp"],
        "family": row["family"],
        "class_index": class_to_idx[row["family"]],
        "file_size": expected_size,
        "zip_crc32": row["zip_crc32"].lower(),
    }

    for key, value in identity.items():
        if source_report.get(key) != value:
            raise ValueError(f"SBSMI source report mismatch: {key}")
    if (
        source_report.get("status") != "PASS"
        or source_report.get("config_sha256") != sbsmi_config_id
    ):
        raise ValueError("Invalid SBSMI source report provenance.")

    content_sha = source_report.get("disarmed_content_sha256", "")
    if re.fullmatch(r"[0-9a-f]{64}", content_sha) is None:
        raise ValueError("Missing valid disarmed content SHA-256.")

    expected = {
        **identity,
        "representation": "raw-byte",
        "disarmed_content_sha256": content_sha,
        "config_sha256": config_id,
    }

    split_dir = OUTPUT_DIR / split
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
            # VERIFIED includes rehashing the source binary.
            while reader.read(CHUNK_SIZE):
                pass
        else:
            matrix64 = stream_to_raw_byte(
                reader, file_size=expected_size, chunk_size=CHUNK_SIZE
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
        raise ValueError("Input SHA-256 differs from the SBSMI source.")
    if f"{reader.crc32 & 0xFFFFFFFF:08x}" != identity["zip_crc32"]:
        raise ValueError("Input CRC32 differs from the index.")

    if cached:
        return "VERIFIED"

    # Store the precision used by the neural network, without 8-bit rounding.
    matrix = matrix64.astype(IMAGE_DTYPE)
    verify_matrix(matrix)
    cast_error = float(np.max(np.abs(matrix64 - matrix.astype(np.float64))))
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
        "image_shape": [64, 64],
        "storage_dtype": "<f4",
        "array_round_trip": "PASS",
        "maximum_float32_cast_error": cast_error,
        "npy_sha256": npy_sha,
    })
    return "CREATED"


def main():
    parser = argparse.ArgumentParser(
        description="Build streamed raw-byte matrices for train/validation."
    )
    parser.add_argument("--limit", type=int, default=20)
    args = parser.parse_args()
    if args.limit < 1:
        raise ValueError("--limit must be positive.")

    samples, class_to_idx, sbsmi_config_id = load_inputs()
    config_id = prepare_output()
    selected = samples[:args.limit]
    split_counts = Counter(row["split"] for row in selected)

    print("Purpose: build raw-byte training matrices")
    print("Total manifest samples:", len(samples))
    print("Requested samples:", len(selected))
    print("Selected train:", split_counts["train"])
    print("Selected validation:", split_counts["validation"])
    print("Storage: float32 NPY, shape (64, 64), range [0, 1]")
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
                    row, class_to_idx, config_id, sbsmi_config_id
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

            # Stop at the first error so a systematic issue is not repeated.
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