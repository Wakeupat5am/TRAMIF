import csv
import hashlib
import json
import math
import re
import zlib
from pathlib import Path

from PIL import Image

from scripts.masked_area import masked_area_resize
from scripts.raw_byte_core import bytes_to_layout


def main():
    project_root = Path(__file__).resolve().parents[1]
    index_path = (
        project_root
        / "data"
        / "manifests"
        / "train_validation_zip_index.csv"
    )
    binary_dir = project_root / "data" / "raw" / "altered"
    output_dir = project_root / "data" / "derived" / "raw_byte_pilot"

    # Use the same selection rule as the SBSMI pilot.
    with index_path.open(encoding="utf-8", newline="") as file:
        sample = next(
            (
                row
                for row in csv.DictReader(file)
                if row["split"] == "train"
            ),
            None,
        )

    if sample is None:
        raise ValueError("No training sample found.")

    sha = sample["sha"]
    if re.fullmatch(r"[0-9a-f]{64}", sha) is None:
        raise ValueError("Invalid sample identifier.")

    binary_path = binary_dir / f"{sha}.exe"

    # Read bytes as data; do not execute the binary.
    data = binary_path.read_bytes()

    if len(data) != int(sample["file_size"]):
        raise ValueError("Binary size differs from the saved index.")

    crc = f"{zlib.crc32(data) & 0xFFFFFFFF:08x}"
    if crc != sample["zip_crc32"].strip().lower():
        raise ValueError("Binary CRC differs from the saved index.")

    content_sha256 = hashlib.sha256(data).hexdigest()

    # Arrange bytes and mark real data versus padding.
    values, mask = bytes_to_layout(data, row_width=256)
    valid_pixels = sum(sum(row) for row in mask)

    if valid_pixels != len(data):
        raise RuntimeError("Validity mask does not match byte count.")

    # Preserve floating-point averages during resizing.
    resized = masked_area_resize(
        values,
        mask,
        output_height=64,
        output_width=64,
    )

    if len(resized) != 64 or any(len(row) != 64 for row in resized):
        raise RuntimeError("Unexpected resized matrix dimensions.")

    if any(
        not math.isfinite(value) or not 0 <= value <= 255
        for row in resized
        for value in row
    ):
        raise RuntimeError("Invalid resized intensity.")

    # Fixed-range normalization, independent of dataset statistics.
    normalized = [
        [value / 255.0 for value in row]
        for row in resized
    ]

    # Quantize only the PNG preview.
    pixel_bytes = bytes(
        math.floor(value)
        for row in resized
        for value in row
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    image_path = output_dir / f"{sha}.png"
    report_path = output_dir / f"{sha}.json"

    with Image.frombytes("L", (64, 64), pixel_bytes) as image:
        image.save(image_path, format="PNG")

    with Image.open(image_path) as saved:
        if saved.format != "PNG":
            raise RuntimeError("Unexpected image format.")
        if saved.mode != "L" or saved.size != (64, 64):
            raise RuntimeError("Unexpected image mode or size.")
        if saved.tobytes() != pixel_bytes:
            raise RuntimeError("PNG pixel verification failed.")

    report = {
        "sample_identifier": sha,
        "split": sample["split"],
        "family": sample["family"],
        "timestamp": sample["timestamp"],
        "input_file": binary_path.relative_to(project_root).as_posix(),
        "file_size": len(data),
        "disarmed_content_sha256": content_sha256,
        "zip_crc32": crc,
        "crc_matches_saved_index": True,
        "representation": "raw-byte",
        "row_width": 256,
        "layout_height": len(values),
        "valid_pixels": valid_pixels,
        "padding_pixels": len(values) * 256 - valid_pixels,
        "resize": "mask-aware area average",
        "empty_region_value": 0,
        "image_size": [64, 64],
        "normalization": "divide floating-point intensity by 255",
        "normalized_matrix": normalized,
        "png_role": "preview",
        "png_quantization": "floor intensity in [0, 255]",
        "pixel_round_trip": "PASS",
    }

    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )

    print("Selected split:", sample["split"])
    print("Family:", sample["family"])
    print("Sample identifier:", sha)
    print("Read bytes:", len(data))
    print("CRC matches saved ZIP index: PASS")
    print(f"Layout shape: {len(values)} x 256")
    print("Valid pixels:", valid_pixels)
    print("Padding pixels:", len(values) * 256 - valid_pixels)
    print("Image size: (64, 64)")
    print("Image mode: L")
    print("Pixel round-trip: PASS")
    print("Image:", image_path.relative_to(project_root))
    print("Report:", report_path.relative_to(project_root))


if __name__ == "__main__":
    main()