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

    if not zip_path.is_file():
        raise FileNotFoundError(f"ZIP not found: {zip_path}")

    print("ZIP:", zip_path.name)
    print(f"Archive size: {zip_path.stat().st_size / (1024 ** 3):.2f} GiB")

    with ZipFile(zip_path, "r") as archive:
        entries = archive.infolist()
        files = [entry for entry in entries if not entry.is_dir()]

        compressed_bytes = sum(entry.compress_size for entry in files)
        uncompressed_bytes = sum(entry.file_size for entry in files)

        print("Total entries:", len(entries))
        print("File entries:", len(files))
        print(f"Compressed content: {compressed_bytes / (1024 ** 3):.2f} GiB")
        print(f"Uncompressed content: {uncompressed_bytes / (1024 ** 3):.2f} GiB")

        print("\nFirst 10 file entries:")
        for entry in files[:10]:
            print(
                repr(entry.filename),
                "| bytes:",
                entry.file_size,
                "| compression:",
                entry.compress_type,
            )


if __name__ == "__main__":
    main()