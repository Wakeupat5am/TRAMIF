import csv
import hashlib
import json
import math
import re
import zlib
from pathlib import Path

from PIL import Image

from scripts.entropy_core import local_entropy_values
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
    output_dir = project_root / "data" / "derived" / "entropy_pilot"

    # Select the same sample as the earlier pilots.
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

    # Read bytes only; do not execute the binary.
    data = binary_path.read_bytes()

    if len(data) != int(sample["file_size"]):
        raise ValueError("Binary size differs from the saved index.")

    crc = f"{zlib.crc32(data) & 0xFFFFFFFF:08x}"
    if crc != sample["zip_crc32"].strip().lower():
        raise ValueError("Binary CRC differs from the saved index.")

    content_sha256 = hashlib.sha256(data).hexdigest()

    # Confirm that the raw-byte pilot used the same disarmed bytes.
    raw_report_path = (
        project_root
        / "data"
        / "derived"
        / "raw_byte_pilot"
        / f"{sha}.json"
    )
    raw_report = json.loads(raw_report_path.read_text(encoding="utf-8"))

    if raw_report["sample_identifier"] != sha:
        raise ValueError("Raw-byte report identifies a different sample.")

    if raw_report["disarmed_content_sha256"] != content_sha256:
        raise ValueError("Input bytes differ from the raw-byte pilot.")

    window_size = 256
    stride = 128
    row_width = 256

    local_values = local_entropy_values(
        data,
        window_size=window_size,
        stride=stride,
    )

    if len(local_values) != len(data):
        raise RuntimeError("Expected one entropy value per byte.")

    # Reuse the raw-byte layout and mask.
    # Replace each real byte intensity with its local entropy.
    layout, mask = bytes_to_layout(data, row_width=row_width)

    for offset, entropy_value in enumerate(local_values):
        row, column = divmod(offset, row_width)
        layout[row][column] = entropy_value

    valid_pixels = sum(sum(row) for row in mask)
    if valid_pixels != len(data):
        raise RuntimeError("Validity mask does not match byte count.")

    normalized = masked_area_resize(
        layout,
        mask,
        output_height=64,
        output_width=64,
    )

    if len(normalized) != 64 or any(
        len(row) != 64 for row in normalized
    ):
        raise RuntimeError("Unexpected resized matrix dimensions.")

    if any(
        not math.isfinite(value) or not 0 <= value <= 1
        for row in normalized
        for value in row
    ):
        raise RuntimeError("Entropy image must contain values in [0, 1].")

    # Keep floats in the report; quantize only the PNG preview.
    pixel_bytes = bytes(
        math.floor(value * 255)
        for row in normalized
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
        "input_matches_raw_byte_pilot": True,
        "representation": "local-entropy",
        "window_size": window_size,
        "stride": stride,
        "window_end_rule": "stop_at_first_window_reaching_eof",
        "partial_window_rule": "use_existing_bytes_without_padding",
        "overlap_rule": "mean_of_covering_window_entropies",
        "normalization": "divide_entropy_in_bits_by_8",
        "row_width": row_width,
        "layout_height": len(layout),
        "valid_pixels": valid_pixels,
        "padding_pixels": len(layout) * row_width - valid_pixels,
        "resize": "mask-aware area average",
        "empty_region_value": 0,
        "image_size": [64, 64],
        "normalized_matrix": normalized,
        "png_role": "preview",
        "png_quantization": "floor normalized_value * 255",
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
    print("Input matches raw-byte pilot: PASS")
    print("Window size:", window_size)
    print("Stride:", stride)
    print("Local entropy values:", len(local_values))
    print(f"Layout shape: {len(layout)} x {row_width}")
    print("Padding pixels:", len(layout) * row_width - valid_pixels)
    print("Image size: (64, 64)")
    print("Image mode: L")
    print("Pixel round-trip: PASS")
    print("Image:", image_path.relative_to(project_root))
    print("Report:", report_path.relative_to(project_root))


if __name__ == "__main__":
    main()