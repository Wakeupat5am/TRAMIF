import argparse
import csv
import hashlib
import json
import re
import zlib
from collections import Counter
from datetime import datetime
from pathlib import Path
from time import perf_counter

import PIL
from PIL import Image

from scripts.sbsmi_stream import file_to_sbsmi


CHUNK_SIZE = 65536


def fingerprint(path):
    """Read a file in chunks and compute size, SHA-256 and CRC32."""
    digest = hashlib.sha256()
    crc = 0
    size = 0

    with path.open("rb") as file:
        while chunk := file.read(CHUNK_SIZE):
            digest.update(chunk)
            crc = zlib.crc32(chunk, crc)
            size += len(chunk)

    return {
        "sha256": digest.hexdigest(),
        "bytes": size,
        "crc32": f"{crc & 0xFFFFFFFF:08x}",
    }


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def write_json_atomic(path, value):
    """Replace the final JSON only after writing a complete temp file."""
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def read_png_pixels(path):
    with Image.open(path) as image:
        if image.format != "PNG":
            raise ValueError("Expected PNG format.")

        if image.mode != "L" or image.size != (64, 64):
            raise ValueError("Expected grayscale 64 x 64 image.")

        return image.tobytes()


def cached_output_is_valid(image_path, report_path, expected):
    if not image_path.is_file() or not report_path.is_file():
        return False

    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))

        if not isinstance(report, dict):
            return False

        for key, value in expected.items():
            if report.get(key) != value:
                return False

        if report.get("status") != "PASS":
            return False

        if fingerprint(image_path)["sha256"] != report["png_sha256"]:
            return False

        pixels = read_png_pixels(image_path)
        return (
            hashlib.sha256(pixels).hexdigest()
            == report["pixel_sha256"]
        )
    except (OSError, ValueError, KeyError, TypeError):
        return False


def process_one(sample, project_root, output_dir, class_to_idx, config_id):
    sha = sample["sha"]
    binary_path = project_root / "data" / "raw" / "altered" / f"{sha}.exe"
    expected_size = int(sample["file_size"])

    if binary_path.stat().st_size != expected_size:
        raise ValueError("Binary size differs from the saved index.")

    # Verify source contents even when a cached output already exists.
    source = fingerprint(binary_path)

    if source["bytes"] != expected_size:
        raise ValueError("Read byte count differs from the saved index.")

    if source["crc32"] != sample["zip_crc32"].lower():
        raise ValueError("Binary CRC differs from the saved index.")

    split_dir = output_dir / sample["split"]
    split_dir.mkdir(parents=True, exist_ok=True)

    image_path = split_dir / f"{sha}.png"
    report_path = split_dir / f"{sha}.json"

    expected = {
        "sample_identifier": sha,
        "split": sample["split"],
        "timestamp": sample["timestamp"],
        "family": sample["family"],
        "class_index": class_to_idx[sample["family"]],
        "disarmed_content_sha256": source["sha256"],
        "file_size": source["bytes"],
        "zip_crc32": source["crc32"],
        "config_sha256": config_id,
    }

    if cached_output_is_valid(image_path, report_path, expected):
        return "VERIFIED"

    matrix = file_to_sbsmi(binary_path, l=6, chunk_size=CHUNK_SIZE)

    if len(matrix) != 64 or any(len(row) != 64 for row in matrix):
        raise RuntimeError("Unexpected SBSMI dimensions.")

    pixels = bytes(value for row in matrix for value in row)

    # Detect content changes between verification and image construction.
    if fingerprint(binary_path) != source:
        raise RuntimeError("Binary changed during processing.")

    temporary_image = image_path.with_name(image_path.name + ".tmp")

    with Image.frombytes("L", (64, 64), pixels) as image:
        image.save(temporary_image, format="PNG")

    if read_png_pixels(temporary_image) != pixels:
        raise RuntimeError("PNG pixel round-trip failed.")

    png_hash = fingerprint(temporary_image)["sha256"]
    temporary_image.replace(image_path)

    report = {
        **expected,
        "status": "PASS",
        "pixel_round_trip": "PASS",
        "pixel_sha256": hashlib.sha256(pixels).hexdigest(),
        "png_sha256": png_hash,
    }
    write_json_atomic(report_path, report)

    return "CREATED"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Maximum number of manifest samples to visit; default: 20.",
    )
    args = parser.parse_args()

    if args.limit < 1:
        raise ValueError("--limit must be positive.")

    project_root = Path(__file__).resolve().parents[1]

    manifest_path = (
        project_root / "data" / "manifests" / "train_validation_manifest.csv"
    )
    index_path = (
        project_root / "data" / "manifests" / "train_validation_zip_index.csv"
    )
    mapping_path = project_root / "docs" / "class_mapping.json"
    stream_path = project_root / "scripts" / "sbsmi_stream.py"
    output_dir = project_root / "data" / "derived" / "sbsmi_dataset_v1"

    manifest = read_csv(manifest_path)
    indexed_samples = read_csv(index_path)
    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    class_to_idx = mapping["class_to_idx"]

    if mapping["num_classes"] != 51 or len(class_to_idx) != 51:
        raise ValueError("Expected the frozen mapping of 51 families.")

    indices = list(class_to_idx.values())
    if any(type(index) is not int for index in indices):
        raise ValueError("Class indices must be integers.")

    if sorted(indices) != list(range(51)):
        raise ValueError("Class indices must cover 0 to 50.")

    if not manifest:
        raise ValueError("Manifest is empty.")

    if len({row["sha"] for row in manifest}) != len(manifest):
        raise ValueError("Duplicate SHA in manifest.")

    if len({row["sha"] for row in indexed_samples}) != len(indexed_samples):
        raise ValueError("Duplicate SHA in ZIP index.")

    index_by_sha = {row["sha"]: row for row in indexed_samples}

    if {row["sha"] for row in manifest} != set(index_by_sha):
        raise ValueError("Manifest and ZIP index contain different samples.")

    # Preserve manifest ordering and verify identity fields.
    samples = []

    for row in manifest:
        sha = row["sha"]

        if re.fullmatch(r"[0-9a-f]{64}", sha) is None:
            raise ValueError("Invalid sample identifier.")

        if row["split"] not in {"train", "validation"}:
            raise ValueError("Only train and validation are allowed.")

        if row["family"] not in class_to_idx:
            raise ValueError("Family is absent from the frozen cohort.")

        indexed = index_by_sha[sha]

        for key in ("sha", "split", "timestamp", "family"):
            if indexed[key] != row[key]:
                raise ValueError(f"Manifest/index mismatch: {sha}, {key}")

        if int(indexed["file_size"]) <= 0:
            raise ValueError("Invalid indexed file size.")

        if re.fullmatch(r"[0-9a-fA-F]{8}", indexed["zip_crc32"]) is None:
            raise ValueError("Invalid indexed CRC32.")

        samples.append(indexed)

    config = {
        "schema_version": 1,
        "representation": "SBSMI",
        "state_bits": 6,
        "bit_order": "MSB-first",
        "tail_rule": "retain_integer_zero_extend_high_bits",
        "zero_transition_row": 0,
        "quantization": "floor probability * 255",
        "image_size": [64, 64],
        "image_mode": "L",
        "model_normalization": "divide integer pixels by 255",
        "chunk_size": CHUNK_SIZE,
        "pillow_version": PIL.__version__,
        "manifest_sha256": fingerprint(manifest_path)["sha256"],
        "zip_index_sha256": fingerprint(index_path)["sha256"],
        "class_mapping_sha256": fingerprint(mapping_path)["sha256"],
        "sbsmi_stream_source_sha256": fingerprint(stream_path)["sha256"],
        "builder_source_sha256": fingerprint(Path(__file__))["sha256"],
    }

    config_id = hashlib.sha256(
        json.dumps(config, sort_keys=True).encode("utf-8")
    ).hexdigest()

    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = output_dir / "config.json"

    if config_path.exists():
        saved_config = json.loads(config_path.read_text(encoding="utf-8"))

        if saved_config != config:
            raise ValueError(
                "Configuration or source files changed. "
                "Review the change before reusing this output folder."
            )
    else:
        if any(output_dir.iterdir()):
            raise ValueError("Output folder is nonempty but has no config.")

        write_json_atomic(config_path, config)

    selected = samples[:args.limit]
    split_counts = Counter(row["split"] for row in selected)

    print("Total manifest samples:", len(samples))
    print("Samples in this run:", len(selected))
    print("Selected train:", split_counts["train"])
    print("Selected validation:", split_counts["validation"])
    print("Output folder:", output_dir)
    print("Processing sequentially...", flush=True)

    log_dir = output_dir / "logs"
    log_dir.mkdir(exist_ok=True)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    log_path = log_dir / f"{run_id}.csv"
    counts = Counter()

    with log_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "sha", "split", "family", "status", "seconds", "error"
            ],
        )
        writer.writeheader()
        file.flush()

        for number, sample in enumerate(selected, start=1):
            start = perf_counter()
            status = "ERROR"
            error_text = ""

            try:
                status = process_one(
                    sample,
                    project_root,
                    output_dir,
                    class_to_idx,
                    config_id,
                )
            except Exception as error:
                error_text = f"{type(error).__name__}: {error}"

            seconds = perf_counter() - start
            counts[status] += 1

            writer.writerow({
                "sha": sample["sha"],
                "split": sample["split"],
                "family": sample["family"],
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
                    f"{sample['split']} | {sample['family']} | "
                    f"{seconds:.3f}s",
                    flush=True,
                )

                if error_text:
                    print(" ", error_text, flush=True)

    print()
    print("Created:", counts["CREATED"])
    print("Verified existing:", counts["VERIFIED"])
    print("Errors:", counts["ERROR"])
    print("Run log:", log_path)

    if counts["ERROR"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()