import csv
import hashlib
import json
import math
import platform
import re
import sys
import zlib
from datetime import datetime
from pathlib import Path
from time import perf_counter

import PIL
from PIL import Image

from scripts.entropy_core import local_entropy_values
from scripts.masked_area import masked_area_resize
from scripts.raw_byte_core import bytes_to_layout
from scripts.sbsmi_stream import file_to_sbsmi


def make_raw_byte(data):
    values, mask = bytes_to_layout(data, row_width=256)
    resized = masked_area_resize(values, mask)

    return [
        [value / 255.0 for value in row]
        for row in resized
    ]


def make_entropy(data):
    local_values = local_entropy_values(
        data,
        window_size=256,
        stride=128,
    )
    layout, mask = bytes_to_layout(data, row_width=256)

    for offset, value in enumerate(local_values):
        row, column = divmod(offset, 256)
        layout[row][column] = value

    return masked_area_resize(layout, mask)


def verify_matrix(matrix):
    if len(matrix) != 64 or any(len(row) != 64 for row in matrix):
        raise ValueError("Expected a 64 x 64 matrix.")

    if any(
        not math.isfinite(value) or not 0 <= value <= 1
        for row in matrix
        for value in row
    ):
        raise ValueError("Normalized matrix has invalid values.")


def save_verified_png(path, pixel_bytes):
    with Image.frombytes("L", (64, 64), pixel_bytes) as image:
        image.save(path, format="PNG")

    with Image.open(path) as saved:
        if saved.format != "PNG":
            raise RuntimeError("Unexpected image format.")

        if saved.mode != "L" or saved.size != (64, 64):
            raise RuntimeError("Unexpected image mode or dimensions.")

        if saved.tobytes() != pixel_bytes:
            raise RuntimeError("PNG pixel round-trip failed.")


def hash_file(path):
    digest = hashlib.sha256()

    with path.open("rb") as file:
        while chunk := file.read(65536):
            digest.update(chunk)

    return digest.hexdigest()


def process_sample(
    sample,
    project_root,
    run_dir,
    result,
    max_file_bytes=5 * 1024 * 1024,
):
    """Process one sample using a caller-specified file size limit."""

    if type(max_file_bytes) is not int or max_file_bytes <= 0:
        raise ValueError("max_file_bytes must be a positive integer.")

    sha = sample["sha"]

    if re.fullmatch(r"[0-9a-f]{64}", sha) is None:
        raise ValueError("Invalid sample identifier.")

    if sample["split"] != "train":
        raise ValueError("This pilot accepts training samples only.")

    expected_size = int(sample["file_size"])

    if not 0 < expected_size <= max_file_bytes:
        raise ValueError("Sample exceeds the configured pilot size limit.")

    binary_path = (
        project_root / "data" / "raw" / "altered" / f"{sha}.exe"
    )

    if binary_path.stat().st_size != expected_size:
        raise ValueError("File size differs from the saved index.")

    start = perf_counter()
    data = binary_path.read_bytes()

    if len(data) != expected_size:
        raise ValueError("Read byte count differs from the saved index.")

    crc = f"{zlib.crc32(data) & 0xFFFFFFFF:08x}"

    if crc != sample["zip_crc32"].strip().lower():
        raise ValueError("CRC differs from the saved index.")

    content_sha = hashlib.sha256(data).hexdigest()
    result["read_verify_seconds"] = perf_counter() - start

    start = perf_counter()
    raw_matrix = make_raw_byte(data)
    result["raw_byte_seconds"] = perf_counter() - start

    start = perf_counter()
    entropy_matrix = make_entropy(data)
    result["entropy_seconds"] = perf_counter() - start

    # SBSMI timing includes a separate chunked read of the input file.
    start = perf_counter()
    sbsmi_pixels = file_to_sbsmi(
        binary_path,
        l=6,
        chunk_size=65536,
    )
    sbsmi_matrix = [
        [value / 255.0 for value in row]
        for row in sbsmi_pixels
    ]
    result["sbsmi_seconds"] = perf_counter() - start

    start = perf_counter()

    if hash_file(binary_path) != content_sha:
        raise RuntimeError("Input file changed during processing.")

    result["postcheck_seconds"] = perf_counter() - start

    matrices = {
        "raw_byte": raw_matrix,
        "entropy": entropy_matrix,
        "sbsmi": sbsmi_matrix,
    }

    for matrix in matrices.values():
        verify_matrix(matrix)

    start = perf_counter()

    for view, matrix in matrices.items():
        if view == "sbsmi":
            pixels = bytes(
                value
                for row in sbsmi_pixels
                for value in row
            )
        else:
            pixels = bytes(
                math.floor(value * 255)
                for row in matrix
                for value in row
            )

        image_path = run_dir / f"{sha}_{view}.png"
        save_verified_png(image_path, pixels)

    report = {
        "sample_identifier": sha,
        "family": sample["family"],
        "split": sample["split"],
        "timestamp": sample["timestamp"],
        "file_size": len(data),
        "maximum_file_bytes": max_file_bytes,
        "disarmed_content_sha256": content_sha,
        "zip_crc32": crc,
        "crc_matches_saved_index": True,
        "input_hash_unchanged_after_processing": True,
        "normalized_matrices": matrices,
        "all_png_round_trips": "PASS",
        "timings_seconds": {
            key: value
            for key, value in result.items()
            if key.endswith("_seconds") and isinstance(value, (int, float))
        },
    }

    (run_dir / f"{sha}.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    result["export_seconds"] = perf_counter() - start


def main():
    project_root = Path(__file__).resolve().parents[1]
    manifest_path = (
        project_root / "data" / "manifests" / "preprocessing_pilot.csv"
    )

    with manifest_path.open(encoding="utf-8", newline="") as file:
        samples = list(csv.DictReader(file))

    if len(samples) != 10:
        raise ValueError("Expected exactly 10 pilot samples.")

    if len({row["sha"] for row in samples}) != len(samples):
        raise ValueError("Duplicate sample identifiers.")

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir = (
        project_root / "data" / "derived" / "preprocessing_pilot" / run_id
    )
    run_dir.mkdir(parents=True, exist_ok=False)

    manifest_bytes = manifest_path.read_bytes()
    (run_dir / "manifest.csv").write_bytes(manifest_bytes)

    config = {
        "purpose": "engineering pilot, not model evaluation",
        "sample_count": len(samples),
        "maximum_file_bytes": 5 * 1024 * 1024,
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "python": sys.version,
        "platform": platform.platform(),
        "pillow": PIL.__version__,
        "row_width": 256,
        "image_shape": [64, 64],
        "resize": "mask-aware area average",
        "empty_region_value": 0,
        "entropy_window_size": 256,
        "entropy_stride": 128,
        "entropy_tail_rule": "stop_at_first_window_reaching_eof",
        "entropy_partial_window": "existing_bytes_only",
        "entropy_overlap": "mean_of_covering_window_entropies",
        "sbsmi_state_bits": 6,
        "sbsmi_bit_order": "MSB-first",
        "sbsmi_tail_rule": "retain_integer_zero_extend_high_bits",
        "sbsmi_chunk_size": 65536,
        "raw_entropy_png_rule": "floor normalized_value * 255",
        "sbsmi_png_rule": "preserve integer SBSMI pixels",
        "timing_note": (
            "Raw-byte and entropy use already-loaded bytes; "
            "SBSMI timing includes a separate chunked file read. "
            "Total includes reading, checks, construction and export."
        ),
        "peak_memory_measured": False,
    }

    (run_dir / "config.json").write_text(
        json.dumps(config, indent=2) + "\n",
        encoding="utf-8",
    )

    fields = [
        "sha",
        "family",
        "file_size",
        "status",
        "read_verify_seconds",
        "raw_byte_seconds",
        "entropy_seconds",
        "sbsmi_seconds",
        "postcheck_seconds",
        "export_seconds",
        "total_seconds",
        "error",
    ]

    success_count = 0
    summary_path = run_dir / "summary.csv"

    with summary_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        file.flush()

        for number, sample in enumerate(samples, start=1):
            result = {field: "" for field in fields}
            result.update({
                "sha": sample["sha"],
                "family": sample["family"],
                "file_size": sample["file_size"],
                "status": "ERROR",
            })

            print(
                f"[{number}/{len(samples)}] "
                f"{sample['family']} | {sample['sha']}",
                flush=True,
            )
            start = perf_counter()

            try:
                process_sample(
                    sample,
                    project_root,
                    run_dir,
                    result,
                    max_file_bytes=5 * 1024 * 1024,
                )
                result["status"] = "PASS"
                success_count += 1
            except Exception as error:
                result["error"] = f"{type(error).__name__}: {error}"

            result["total_seconds"] = perf_counter() - start
            writer.writerow(result)
            file.flush()

            print(
                f"  {result['status']} | "
                f"total={result['total_seconds']:.3f}s",
                flush=True,
            )

            if result["error"]:
                print(" ", result["error"], flush=True)

    print()
    print("Successful samples:", success_count)
    print("Failed samples:", len(samples) - success_count)
    print("Summary:", summary_path)
    print("Run folder:", run_dir)

    if success_count != len(samples):
        raise SystemExit(1)


if __name__ == "__main__":
    main()