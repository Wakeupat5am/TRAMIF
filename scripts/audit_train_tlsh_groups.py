import csv
import hashlib
import io
import json
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
INVENTORY_FOLDER = ROOT / "data/derived/train_tlsh_v1"
CSV_PATH = INVENTORY_FOLDER / "train_tlsh.csv"
SUMMARY_PATH = INVENTORY_FOLDER / "summary.json"
OUTPUT_ROOT = ROOT / "data/derived/train_tlsh_groups"

EXPECTED_SAMPLES = 18061
TRAIN_START = datetime(2019, 8, 1, tzinfo=timezone.utc)
TRAIN_END = datetime(2020, 2, 1, tzinfo=timezone.utc)


def load_verified_table():
    with SUMMARY_PATH.open(encoding="utf-8") as file:
        summary = json.load(file)

    if summary.get("complete_train_inventory") is not True:
        raise ValueError("The train inventory is incomplete.")

    if summary.get("errors") != 0:
        raise ValueError("The train inventory contains errors.")

    csv_bytes = CSV_PATH.read_bytes()
    csv_sha256 = hashlib.sha256(csv_bytes).hexdigest()

    if csv_sha256 != summary.get("csv_sha256"):
        raise ValueError("CSV checksum differs from the saved summary.")

    reader = csv.DictReader(
        io.StringIO(csv_bytes.decode("utf-8-sig"))
    )
    required = {
        "sha",
        "split",
        "timestamp",
        "family",
        "disarmed_content_sha256",
        "tlsh",
        "tlsh_status",
        "config_sha256",
    }

    if not required.issubset(reader.fieldnames or []):
        raise ValueError("Required CSV columns are missing.")

    rows = list(reader)

    if (
        len(rows) != EXPECTED_SAMPLES
        or len(rows) != summary.get("processed_samples")
        or len(rows) != summary.get("total_train_samples")
    ):
        raise ValueError("Unexpected number of train samples.")

    seen = set()

    for row in rows:
        sample_id = row["sha"]

        if re.fullmatch(r"[0-9a-f]{64}", sample_id) is None:
            raise ValueError("Invalid sample identifier.")

        if sample_id in seen:
            raise ValueError(f"Repeated sample identifier: {sample_id}")

        seen.add(sample_id)

        if row["split"] != "train":
            raise ValueError("Found a non-training sample.")

        observed = datetime.fromisoformat(row["timestamp"])

        if observed.tzinfo is None:
            raise ValueError("Timestamp must include a timezone.")

        observed = observed.astimezone(timezone.utc)

        if not TRAIN_START <= observed < TRAIN_END:
            raise ValueError("Found a sample outside the training period.")

        if not row["family"].strip():
            raise ValueError("Missing family label.")

        if re.fullmatch(
            r"[0-9a-f]{64}", row["disarmed_content_sha256"]
        ) is None:
            raise ValueError("Invalid disarmed content SHA-256.")

        if row["config_sha256"] != summary.get("config_sha256"):
            raise ValueError("Record and inventory configurations differ.")

        if row["tlsh_status"] != "VALID":
            raise ValueError(
                "This audit expects the completed all-valid inventory."
            )

        if re.fullmatch(r"T1[0-9A-F]{70}", row["tlsh"]) is None:
            raise ValueError("Invalid TLSH digest.")

    return rows, csv_sha256


def write_csv(path, fields, rows):
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    rows, input_sha256 = load_verified_table()

    by_tlsh = defaultdict(list)

    for row in rows:
        by_tlsh[row["tlsh"]].append(row)

    # Retain only digest groups containing at least two samples.
    repeated = [
        (digest, members)
        for digest, members in sorted(by_tlsh.items())
        if len(members) >= 2
    ]

    group_rows = []
    member_rows = []

    for number, (digest, members) in enumerate(repeated, start=1):
        group_id = f"G{number:06d}"

        content_hashes = {
            row["disarmed_content_sha256"] for row in members
        }
        families = sorted({row["family"] for row in members})
        months = sorted({
            datetime.fromisoformat(row["timestamp"])
            .astimezone(timezone.utc)
            .strftime("%Y-%m")
            for row in members
        })

        group_rows.append({
            "group_id": group_id,
            "tlsh": digest,
            "samples": len(members),
            "unique_content_sha256": len(content_hashes),
            "family_count": len(families),
            "families": "|".join(families),
            "month_count": len(months),
            "months": "|".join(months),
            "pairs": len(members) * (len(members) - 1) // 2,
        })

        for row in sorted(members, key=lambda item: item["sha"]):
            member_rows.append({
                "group_id": group_id,
                "sha": row["sha"],
                "family": row["family"],
                "timestamp": row["timestamp"],
                "disarmed_content_sha256": (
                    row["disarmed_content_sha256"]
                ),
                "tlsh": digest,
            })

    run_id = datetime.now(timezone.utc).strftime(
        "%Y%m%d_%H%M%S_%f"
    )
    output_folder = OUTPUT_ROOT / run_id
    output_folder.mkdir(parents=True, exist_ok=False)

    write_csv(
        output_folder / "groups.csv",
        [
            "group_id", "tlsh", "samples", "unique_content_sha256",
            "family_count", "families", "month_count", "months", "pairs",
        ],
        group_rows,
    )

    write_csv(
        output_folder / "members.csv",
        [
            "group_id", "sha", "family", "timestamp",
            "disarmed_content_sha256", "tlsh",
        ],
        member_rows,
    )

    multiple_content_groups = sum(
        group["unique_content_sha256"] > 1 for group in group_rows
    )
    multiple_family_groups = sum(
        group["family_count"] > 1 for group in group_rows
    )

    report = {
        "purpose": "train-only identical TLSH digest audit",
        "input_csv_sha256": input_sha256,
        "script_sha256": hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
        "train_samples": len(rows),
        "unique_content_sha256": len({
            row["disarmed_content_sha256"] for row in rows
        }),
        "unique_tlsh_digests": len(by_tlsh),
        "identical_tlsh_groups": len(group_rows),
        "samples_in_groups": len(member_rows),
        "pairs_with_identical_tlsh": sum(
            group["pairs"] for group in group_rows
        ),
        "groups_with_multiple_content_hashes": multiple_content_groups,
        "groups_with_multiple_family_labels": multiple_family_groups,
        "largest_group_samples": max(
            (group["samples"] for group in group_rows),
            default=0,
        ),
        "grouping_rule": "exact equality of the full TLSH digest",
        "near_duplicate_policy_selected": False,
        "samples_removed": 0,
    }

    report_path = output_folder / "report.json"

    with report_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False)
        file.write("\n")

    print("Purpose: identical TLSH digest audit")
    print("Split: train only")
    print("CSV checksum: PASS")
    print("Train samples:", report["train_samples"])
    print("Unique content SHA-256:", report["unique_content_sha256"])
    print("Unique TLSH digests:", report["unique_tlsh_digests"])
    print("Identical TLSH groups:", report["identical_tlsh_groups"])
    print("Samples in groups:", report["samples_in_groups"])
    print(
        "Pairs with identical TLSH:",
        report["pairs_with_identical_tlsh"],
    )
    print(
        "Groups with multiple content hashes:",
        multiple_content_groups,
    )
    print(
        "Groups with multiple family labels:",
        multiple_family_groups,
    )
    print("Largest group:", report["largest_group_samples"])
    print("Report:", report_path)
    print("Groups:", output_folder / "groups.csv")
    print("Members:", output_folder / "members.csv")
    print("No near-duplicate policy selected. No samples removed.")


if __name__ == "__main__":
    main()