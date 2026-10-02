import csv
from pathlib import Path


PILOT_COUNT = 10
MAX_FILE_BYTES = 5 * 1024 * 1024


def main():
    project_root = Path(__file__).resolve().parents[1]
    index_path = (
        project_root
        / "data"
        / "manifests"
        / "train_validation_zip_index.csv"
    )
    output_path = (
        project_root
        / "data"
        / "manifests"
        / "preprocessing_pilot.csv"
    )

    with index_path.open(encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        fieldnames = reader.fieldnames

        candidates = [
            row
            for row in reader
            if row["split"] == "train"
            and 0 < int(row["file_size"]) <= MAX_FILE_BYTES
        ]

    if len(candidates) < PILOT_COUNT:
        raise ValueError("Not enough training samples within the size limit.")

    if len({row["sha"] for row in candidates}) != len(candidates):
        raise ValueError("Duplicate SHA in candidate samples.")

    # Sort by size; SHA provides a deterministic tie-break.
    candidates.sort(
        key=lambda row: (int(row["file_size"]), row["sha"])
    )

    # Choose evenly spaced ranks, including the smallest and largest
    # candidates. This spreads the pilot across the size distribution.
    positions = [
        i * (len(candidates) - 1) // (PILOT_COUNT - 1)
        for i in range(PILOT_COUNT)
    ]
    selected = [candidates[position] for position in positions]

    if len({row["sha"] for row in selected}) != PILOT_COUNT:
        raise RuntimeError("Pilot selection contains duplicate samples.")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(selected)

    print("Selection purpose: preprocessing engineering pilot")
    print("Split: train only")
    print("Maximum allowed file size: 5 MiB")
    print("Candidates within limit:", len(candidates))
    print("Selected samples:", len(selected))
    print("Distinct families:", len({row["family"] for row in selected}))
    print()

    for number, row in enumerate(selected, start=1):
        size_mib = int(row["file_size"]) / (1024 ** 2)
        print(
            f"{number:02d}. "
            f"family={row['family']} | "
            f"size={size_mib:.3f} MiB | "
            f"sha={row['sha']}"
        )

    print()
    print("Saved manifest:", output_path.relative_to(project_root))


if __name__ == "__main__":
    main()