import csv
import hashlib
import io
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import tlsh

from scripts.build_raw_byte_dataset import load_inputs
from scripts.near_duplicate_core import (
    THRESHOLDS,
    apply_historical_policy,
    identity_fields,
    time_key,
    validate_rows,
)


ROOT = Path(__file__).resolve().parents[1]
POLICY_PATH = ROOT / "docs/near_duplicate_protocol.md"
OUTPUT_ROOT = ROOT / "data/derived/historical_near_duplicates_v1"
MANIFEST_FIELDS = ["sha", "split", "timestamp", "family"]


def sha256_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path, value):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_csv(path, fields, rows):
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def load_inventory(split, expected_count, manifest_by_sha):
    folder = ROOT / "data/derived" / f"{split}_tlsh_v1"
    table_path = folder / f"{split}_tlsh.csv"
    config_path = folder / "config.json"
    summary_path = folder / "summary.json"

    config = read_json(config_path)
    summary = read_json(summary_path)
    config_id = hashlib.sha256(
        json.dumps(config, sort_keys=True).encode("utf-8")
    ).hexdigest()

    if summary.get(f"complete_{split}_inventory") is not True:
        raise ValueError(f"Incomplete {split} inventory.")
    if summary.get("errors") != 0:
        raise ValueError(f"{split} inventory contains errors.")
    if summary.get("config_sha256") != config_id:
        raise ValueError(f"{split} configuration checksum mismatch.")

    table_bytes = table_path.read_bytes()
    if hashlib.sha256(table_bytes).hexdigest() != summary.get("csv_sha256"):
        raise ValueError(f"{split} table checksum mismatch.")

    with io.StringIO(table_bytes.decode("utf-8-sig"), newline="") as file:
        rows = list(csv.DictReader(file))

    if (
        len(rows) != expected_count
        or summary.get("processed_samples") != expected_count
        or summary.get(f"total_{split}_samples") != expected_count
    ):
        raise ValueError(f"Unexpected {split} sample count.")

    rows = validate_rows(rows, split)
    expected_ids = {
        sha
        for sha, row in manifest_by_sha.items()
        if row["split"] == split
    }

    if {row["sha"] for row in rows} != expected_ids:
        raise ValueError(f"{split} inventory differs from the manifest.")

    for row in rows:
        source = manifest_by_sha[row["sha"]]

        for key in MANIFEST_FIELDS:
            if row[key] != source[key]:
                raise ValueError(f"Inventory/manifest mismatch: {key}")

        if int(row["file_size"]) != int(source["file_size"]):
            raise ValueError("Inventory/index file size mismatch.")
        if row["crc32"].lower() != source["zip_crc32"].lower():
            raise ValueError("Inventory/index CRC mismatch.")
        if row["config_sha256"] != config_id:
            raise ValueError("Record configuration mismatch.")

    valid_count = sum(row["tlsh_status"] == "VALID" for row in rows)
    null_count = sum(row["tlsh_status"] == "TNULL" for row in rows)

    if (
        valid_count != summary.get("valid_tlsh_hashes")
        or null_count != summary.get("tnull_samples")
    ):
        raise ValueError("Inventory status counts differ from summary.")

    source_hashes = {
        path.relative_to(ROOT).as_posix(): sha256_file(path)
        for path in (table_path, config_path, summary_path)
    }
    return rows, config, source_hashes


def verify_output_rows(output_rows, input_rows, all_by_sha, threshold):
    expected_ids = {row["sha"] for row in input_rows}

    if (
        len(output_rows) != len(input_rows)
        or {row["sha"] for row in output_rows} != expected_ids
    ):
        raise RuntimeError("Decision coverage is incomplete or duplicated.")

    first_train_content = {}
    for row in sorted(input_rows, key=time_key):
        if row["split"] == "train":
            first_train_content.setdefault(
                row["disarmed_content_sha256"], row["sha"]
            )

    for decision in output_rows:
        source = all_by_sha[decision["sha"]]

        for key, expected in identity_fields(source).items():
            if decision.get(key) != expected:
                raise RuntimeError(f"Decision identity mismatch: {key}")
        if decision["threshold"] != threshold:
            raise RuntimeError("Decision threshold mismatch.")

        if source["split"] == "train":
            first_sha = first_train_content[
                source["disarmed_content_sha256"]
            ]
            expected_action = "KEEP" if source["sha"] == first_sha else "EXCLUDE"
            if decision["decision"] != expected_action:
                raise RuntimeError("Incorrect exact-duplicate train decision.")
            if expected_action == "EXCLUDE":
                if decision["reference_sha"] != first_sha:
                    raise RuntimeError("Train reference is not the first copy.")

        if decision["decision"] == "KEEP":
            if decision["reference_sha"]:
                raise RuntimeError("Retained sample has an exclusion witness.")
            continue

        if decision["decision"] != "EXCLUDE":
            raise RuntimeError("Unknown decision.")

        reference = all_by_sha[decision["reference_sha"]]
        if time_key(reference) >= time_key(source):
            raise RuntimeError("Exclusion reference is not earlier.")

        if source["split"] == "validation":
            if (
                decision["reference_split"] != reference["split"]
                or decision["reference_timestamp"] != reference["timestamp"]
            ):
                raise RuntimeError("Exclusion reference metadata mismatch.")

        if decision["reason"] == "exact_sha256":
            if (
                source["disarmed_content_sha256"]
                != reference["disarmed_content_sha256"]
            ):
                raise RuntimeError("Invalid exact-duplicate witness.")
        elif decision["reason"] == "tlsh":
            if source["split"] != "validation":
                raise RuntimeError("Near-duplicate train samples must be kept.")
            if (
                source["tlsh_status"] != "VALID"
                or reference["tlsh_status"] != "VALID"
            ):
                raise RuntimeError("Invalid TLSH exclusion witness.")

            # Recheck the saved witness with the string API.
            distance = tlsh.diff(source["tlsh"], reference["tlsh"])
            if (
                distance > threshold
                or distance != decision["tlsh_distance"]
            ):
                raise RuntimeError("TLSH exclusion witness does not match.")
        else:
            raise RuntimeError("Unknown exclusion reason.")


def progress(phase, completed, total):
    if completed == 1 or completed % 500 == 0 or completed == total:
        print(f"{phase}: {completed}/{total}", flush=True)


def export_threshold(output_dir, threshold, result, manifest, families):
    folder = output_dir / f"threshold_{threshold:02d}"
    folder.mkdir()

    members = result["train_members"][threshold]
    decisions = result["validation_decisions"][threshold]

    write_csv(folder / "train_members.csv", list(members[0]), members)
    write_csv(
        folder / "validation_decisions.csv",
        list(decisions[0]),
        decisions,
    )

    train_keep = {
        row["sha"] for row in members if row["decision"] == "KEEP"
    }
    validation_keep = {
        row["sha"] for row in decisions if row["decision"] == "KEEP"
    }

    # Keep the original manifest order in exported model input lists.
    for name, retained in (
        ("train_keep.csv", train_keep),
        ("validation_keep.csv", validation_keep),
    ):
        rows = [
            {key: row[key] for key in MANIFEST_FIELDS}
            for row in manifest
            if row["sha"] in retained
        ]
        write_csv(folder / name, MANIFEST_FIELDS, rows)

    by_group = defaultdict(list)
    for row in members:
        by_group[row["group_id"]].append(row)

    group_summary = []
    for group_id, group in sorted(by_group.items()):
        group_families = sorted({row["family"] for row in group})
        group_summary.append({
            "group_id": group_id,
            "samples": len(group),
            "retained_samples": sum(
                row["decision"] == "KEEP" for row in group
            ),
            "family_count": len(group_families),
            "families": "|".join(group_families),
        })

    write_csv(
        folder / "train_group_summary.csv",
        list(group_summary[0]),
        group_summary,
    )

    counts = Counter(
        (
            time_key(row)[0].strftime("%Y-%m"),
            row["family"],
            row["decision"],
        )
        for row in decisions
    )

    count_rows = []
    monthly = {}

    for month in ("2020-02", "2020-03"):
        month_keep = 0
        month_exclude = 0

        for family in sorted(families):
            kept = counts[(month, family, "KEEP")]
            excluded = counts[(month, family, "EXCLUDE")]
            count_rows.append({
                "month": month,
                "family": family,
                "input_samples": kept + excluded,
                "kept": kept,
                "excluded": excluded,
            })
            month_keep += kept
            month_exclude += excluded

        monthly[month] = {
            "kept": month_keep,
            "excluded": month_exclude,
        }

    write_csv(
        folder / "validation_counts.csv",
        ["month", "family", "input_samples", "kept", "excluded"],
        count_rows,
    )

    return {
        "train_kept": len(train_keep),
        "train_exact_excluded": len(members) - len(train_keep),
        "train_groups_including_singletons": len(group_summary),
        "train_groups_with_multiple_samples": sum(
            row["samples"] > 1 for row in group_summary
        ),
        "train_groups_with_multiple_families": sum(
            row["family_count"] > 1 for row in group_summary
        ),
        "largest_train_group": max(row["samples"] for row in group_summary),
        "validation_kept": len(validation_keep),
        "validation_excluded": len(decisions) - len(validation_keep),
        "validation_exclusion_reasons": dict(Counter(
            row["reason"]
            for row in decisions
            if row["decision"] == "EXCLUDE"
        )),
        "validation_by_month": monthly,
    }


def main():
    if version("py-tlsh") != "5.0.0":
        raise RuntimeError("Expected py-tlsh 5.0.0.")
    if tuple(THRESHOLDS) != (0, 10, 20):
        raise RuntimeError("Unexpected policy thresholds.")

    manifest, class_to_idx, _ = load_inputs()
    manifest_by_sha = {row["sha"]: row for row in manifest}

    train, train_config, train_sources = load_inventory(
        "train", 18061, manifest_by_sha
    )
    validation, validation_config, validation_sources = load_inventory(
        "validation", 8263, manifest_by_sha
    )

    for key in (
        "py_tlsh",
        "chunk_size",
        "initial_buffer_bytes",
        "hash_helper_sha256",
        "zip_index_sha256",
    ):
        if train_config.get(key) != validation_config.get(key):
            raise ValueError(f"Inventory hashing configurations differ: {key}")

    if train_config["zip_index_sha256"] != sha256_file(
        ROOT / "data/manifests/train_validation_zip_index.csv"
    ):
        raise ValueError("Inventory index differs from the current index.")

    if validation_config["train_config_file_sha256"] != sha256_file(
        ROOT / "data/derived/train_tlsh_v1/config.json"
    ):
        raise ValueError("Validation references another train configuration.")

    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    output_dir = OUTPUT_ROOT / run_id
    output_dir.mkdir(parents=True, exist_ok=False)

    policy_bytes = POLICY_PATH.read_bytes()
    (output_dir / "policy_snapshot.md").write_bytes(policy_bytes)

    config = {
        "purpose": "apply fixed historical near-duplicate policy",
        "primary_threshold": 0,
        "sensitivity_thresholds": [10, 20],
        "train_samples": len(train),
        "validation_samples": len(validation),
        "py_tlsh": version("py-tlsh"),
        "distance": "standard TLSH diff including length",
        "validation_references": "all earlier train/validation records",
        "ordering": "UTC timestamp, then SHA",
        "policy_sha256": hashlib.sha256(policy_bytes).hexdigest(),
        "manifest_sha256": sha256_file(
            ROOT / "data/manifests/train_validation_manifest.csv"
        ),
        "class_mapping_sha256": sha256_file(
            ROOT / "docs/class_mapping.json"
        ),
        "inventory_sources": {**train_sources, **validation_sources},
        "source_sha256": {
            name: sha256_file(ROOT / name)
            for name in (
                "scripts/near_duplicate_core.py",
                "scripts/apply_historical_near_duplicates.py",
                "scripts/build_raw_byte_dataset.py",
            )
        },
    }
    save_json(output_dir / "config.json", config)

    print("Input checksums and identities: PASS")
    print("Train samples:", len(train))
    print("Validation samples:", len(validation))
    print("Thresholds: 0 (primary), 10 and 20 (sensitivity)")
    print("Output folder:", output_dir)
    print("Comparing historical hashes...", flush=True)

    started = perf_counter()
    result = apply_historical_policy(
        train,
        validation,
        progress=progress,
    )
    elapsed = perf_counter() - started

    all_by_sha = {row["sha"]: row for row in train + validation}
    previous_keep = None
    previous_train_keep = None

    for threshold in THRESHOLDS:
        members = result["train_members"][threshold]
        decisions = result["validation_decisions"][threshold]

        verify_output_rows(members, train, all_by_sha, threshold)
        verify_output_rows(decisions, validation, all_by_sha, threshold)

        kept = {row["sha"] for row in decisions if row["decision"] == "KEEP"}
        train_kept = {row["sha"] for row in members if row["decision"] == "KEEP"}

        if previous_keep is not None and not kept.issubset(previous_keep):
            raise RuntimeError("Stronger filtering unexpectedly retains more samples.")
        if previous_train_keep is not None and train_kept != previous_train_keep:
            raise RuntimeError("Train retention changed with near-duplicate threshold.")

        previous_keep = kept
        previous_train_keep = train_kept

    summaries = {}
    for threshold in THRESHOLDS:
        summaries[str(threshold)] = export_threshold(
            output_dir, threshold, result, manifest, class_to_idx
        )

    report = {
        "status": "HISTORICAL_POLICY_APPLIED",
        "decision_coverage_checks": "PASS",
        "exclusion_witness_checks": "PASS",
        "nested_retention_checks": "PASS",
        "retained_validation_check": (
            "Earlier candidates were exhaustively considered by the policy core "
            "for retained samples. No independent second all-pairs pass."
        ),
        "train_tnull": sum(row["tlsh_status"] == "TNULL" for row in train),
        "validation_tnull": sum(
            row["tlsh_status"] == "TNULL" for row in validation
        ),
        "comparison_seconds": elapsed,
        "thresholds": summaries,
        "future_test_processed": False,
        "binary_or_image_files_deleted": 0,
        "output_csv_sha256": {
            path.relative_to(output_dir).as_posix(): sha256_file(path)
            for path in sorted(output_dir.glob("threshold_*/*.csv"))
        },
    }
    save_json(output_dir / "report.json", report)

    print("\nDecision coverage and exclusion witnesses: PASS")
    for threshold in THRESHOLDS:
        summary = summaries[str(threshold)]
        print(f"\nThreshold: {threshold}")
        print("Train kept:", summary["train_kept"])
        print("Train exact copies excluded:", summary["train_exact_excluded"])
        print("Train groups:", summary["train_groups_including_singletons"])
        print("Largest train group:", summary["largest_train_group"])
        print("Mixed-family train groups:", summary["train_groups_with_multiple_families"])
        print("Validation kept:", summary["validation_kept"])
        print("Validation excluded:", summary["validation_excluded"])
        for month, counts in summary["validation_by_month"].items():
            print(
                f"  {month}: kept={counts['kept']}, "
                f"excluded={counts['excluded']}"
            )

    print("\nHistorical policy application completed.")
    print("Report:", output_dir / "report.json")
    print("No binary/image files deleted. No labels changed.")
    print("No model training or future-test processing performed.")


if __name__ == "__main__":
    main()