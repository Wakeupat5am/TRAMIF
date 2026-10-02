import csv
import re
from collections import Counter
from pathlib import Path
from zipfile import ZipFile


def main():
    project_root = Path(__file__).resolve().parents[1]

    zip_path = (
        project_root
        / "data"
        / "raw"
        / "bodmas_disarmed_malware_binaries.zip"
    )
    manifest_path = (
        project_root
        / "data"
        / "manifests"
        / "train_validation_manifest.csv"
    )
    output_path = (
        project_root
        / "data"
        / "manifests"
        / "train_validation_zip_index.csv"
    )

    with manifest_path.open(encoding="utf-8", newline="") as file:
        samples = list(csv.DictReader(file))

    if not samples:
        raise ValueError("The manifest is empty.")

    seen_shas = set()

    for sample in samples:
        sha = sample["sha"]

        if re.fullmatch(r"[0-9a-f]{64}", sha) is None:
            raise ValueError(f"Invalid SHA: {sha}")

        if sha in seen_shas:
            raise ValueError(f"Duplicate manifest SHA: {sha}")

        if sample["split"] not in {"train", "validation"}:
            raise ValueError(f"Unexpected split: {sample['split']}")

        seen_shas.add(sha)

    indexed_rows = []
    missing_shas = []

    with ZipFile(zip_path, "r") as archive:
        members = {}

        for entry in archive.infolist():
            if entry.is_dir():
                continue

            if entry.filename in members:
                raise ValueError(
                    f"Duplicate ZIP member name: {entry.filename}"
                )

            members[entry.filename] = entry

        for sample in samples:
            member_name = f"altered/{sample['sha']}.exe"
            entry = members.get(member_name)

            if entry is None:
                missing_shas.append(sample["sha"])
                continue

            indexed_rows.append({
                "sha": sample["sha"],
                "split": sample["split"],
                "timestamp": sample["timestamp"],
                "family": sample["family"],
                "zip_member": entry.filename,
                "file_size": entry.file_size,
                "compressed_size": entry.compress_size,
                "zip_crc32": f"{entry.CRC:08x}",
            })

    counts = Counter(row["split"] for row in indexed_rows)
    total_bytes = sum(row["file_size"] for row in indexed_rows)

    print("Manifest samples:", len(samples))
    print("Matched train:", counts["train"])
    print("Matched validation:", counts["validation"])
    print("Missing samples:", len(missing_shas))
    print(f"Matched uncompressed size: {total_bytes / (1024 ** 3):.2f} GiB")

    if missing_shas:
        print("First missing SHA:", missing_shas[:5])
        raise ValueError("Incomplete ZIP coverage; index was not written.")

    fieldnames = [
        "sha",
        "split",
        "timestamp",
        "family",
        "zip_member",
        "file_size",
        "compressed_size",
        "zip_crc32",
    ]

    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(indexed_rows)

    print("Saved index to:", output_path.relative_to(project_root))


if __name__ == "__main__":
    main()