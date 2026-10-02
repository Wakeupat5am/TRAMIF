import csv
import hashlib
import json
import re
import zlib
from pathlib import Path

from PIL import Image

from scripts.sbsmi_stream import file_to_sbsmi


def main():
    project_root = Path(__file__).resolve().parents[1]
    binary_dir = project_root / "data" / "raw" / "altered"
    index_path = (
        project_root
        / "data"
        / "manifests"
        / "train_validation_zip_index.csv"
    )
    output_dir = project_root / "data" / "derived" / "sbsmi_pilot"

    # Choose one training sample from the saved index.
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
        raise ValueError("No training sample found in the index.")

    sha = sample["sha"]
    if re.fullmatch(r"[0-9a-f]{64}", sha) is None:
        raise ValueError("Invalid sample identifier.")

    binary_path = binary_dir / f"{sha}.exe"
    if not binary_path.is_file():
        raise FileNotFoundError(f"Binary not found: {binary_path}")

    print("Selected split:", sample["split"])
    print("Family:", sample["family"])
    print("Sample identifier:", sha)

    # Read as data only; compute size, CRC and content hash.
    digest = hashlib.sha256()
    crc = 0
    bytes_read = 0

    with binary_path.open("rb") as file:
        while True:
            chunk = file.read(65536)
            if not chunk:
                break

            digest.update(chunk)
            crc = zlib.crc32(chunk, crc)
            bytes_read += len(chunk)

    if bytes_read != int(sample["file_size"]):
        raise ValueError("Binary size differs from the saved index.")

    actual_crc = f"{crc & 0xFFFFFFFF:08x}"
    if actual_crc != sample["zip_crc32"].strip().lower():
        raise ValueError("Binary CRC differs from the saved ZIP index.")

    # Build the image using the existing streaming implementation.
    matrix = file_to_sbsmi(binary_path, l=6, chunk_size=65536)
    pixel_bytes = bytes(value for row in matrix for value in row)

    output_dir.mkdir(parents=True, exist_ok=True)
    image_path = output_dir / f"{sha}.png"
    report_path = output_dir / f"{sha}.json"

    with Image.frombytes("L", (64, 64), pixel_bytes) as image:
        image.save(image_path, format="PNG")

    # Confirm that saving and reopening preserves every pixel.
    with Image.open(image_path) as saved:
        if saved.format != "PNG":
            raise RuntimeError("Unexpected image format.")

        if saved.mode != "L" or saved.size != (64, 64):
            raise RuntimeError("Unexpected image mode or dimensions.")

        if saved.tobytes() != pixel_bytes:
            raise RuntimeError("PNG pixel verification failed.")

    report = {
        "sample_identifier": sha,
        "split": sample["split"],
        "family": sample["family"],
        "timestamp": sample["timestamp"],
        "input_method": "extracted_disarmed_file",
        "input_file": binary_path.relative_to(project_root).as_posix(),
        "file_size": bytes_read,
        "zip_crc32": actual_crc,
        "crc_matches_saved_index": True,
        "disarmed_content_sha256": digest.hexdigest(),
        "representation": "SBSMI",
        "state_bits": 6,
        "bit_order": "MSB-first",
        "tail_rule": "retain_integer_zero_extend_high_bits",
        "image_size": [64, 64],
        "image_mode": "L",
        "pixel_round_trip": "PASS",
    }

    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print("Read bytes:", bytes_read)
    print("CRC matches saved ZIP index: PASS")
    print("Image size: (64, 64)")
    print("Image mode: L")
    print("Nonzero pixels:", sum(value > 0 for value in pixel_bytes))
    print("Pixel round-trip: PASS")
    print("Image:", image_path.relative_to(project_root))
    print("Report:", report_path.relative_to(project_root))


if __name__ == "__main__":
    main()