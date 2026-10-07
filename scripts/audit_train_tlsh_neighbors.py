import argparse
import csv
import hashlib
import json
import statistics
import time
from collections import Counter
from importlib.metadata import version
from pathlib import Path

import tlsh

from scripts.audit_train_tlsh_groups import load_verified_table
from scripts.check_tlsh import (
    REFERENCE_A,
    REFERENCE_B,
    REFERENCE_DISTANCE,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_FOLDER = ROOT / "data/derived/train_tlsh_neighbors_v1"
SELECTION_SEED = 42

# Reporting cutoffs only; these are not sample-removal rules.
REPORT_CUTOFFS = (0, 10, 20, 30, 50)


def save_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")

    with temporary.open("w", encoding="utf-8") as file:
        json.dump(value, file, indent=2, ensure_ascii=False)
        file.write("\n")

    temporary.replace(path)


def read_json(path):
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def source_sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_digest(digest):
    obj = tlsh.Tlsh()
    obj.fromTlshStr(digest)
    return obj


def check_distance_api(rows, objects):
    first = parse_digest(REFERENCE_A)
    second = parse_digest(REFERENCE_B)

    if first.diff(second) != REFERENCE_DISTANCE:
        raise RuntimeError("Reference object distance mismatch.")

    if second.diff(first) != REFERENCE_DISTANCE:
        raise RuntimeError("Reverse object distance mismatch.")

    if first.diff(first) != 0:
        raise RuntimeError("Object self-distance must be zero.")

    # Check that the faster object API agrees with the string API.
    for index in range(min(8, len(rows) - 1)):
        object_distance = objects[index].diff(objects[index + 1])
        string_distance = tlsh.diff(
            rows[index]["tlsh"],
            rows[index + 1]["tlsh"],
        )

        if object_distance != string_distance:
            raise RuntimeError("Object and string distances differ.")

    print("Distance API checks: PASS")


def nearest_neighbor(query_index, rows, objects, config_id):
    started = time.perf_counter()
    query = rows[query_index]
    query_hash = objects[query_index]

    best_distance = None
    best_index = None
    tie_count = 0
    same_family_ties = 0
    different_family_ties = 0

    # Rows are sorted by SHA, making the representative deterministic.
    for candidate_index, candidate_hash in enumerate(objects):
        if candidate_index == query_index:
            continue

        distance = query_hash.diff(candidate_hash)

        if best_distance is None or distance < best_distance:
            best_distance = distance
            best_index = candidate_index
            tie_count = 0
            same_family_ties = 0
            different_family_ties = 0

        if distance == best_distance:
            tie_count += 1

            if rows[candidate_index]["family"] == query["family"]:
                same_family_ties += 1
            else:
                different_family_ties += 1

    neighbor = rows[best_index]

    return {
        "query_sha": query["sha"],
        "query_family": query["family"],
        "query_timestamp": query["timestamp"],
        "neighbor_sha": neighbor["sha"],
        "neighbor_family": neighbor["family"],
        "neighbor_timestamp": neighbor["timestamp"],
        "minimum_distance": best_distance,
        "nearest_tie_count": tie_count,
        "same_family_at_minimum": same_family_ties,
        "different_family_at_minimum": different_family_ties,
        "candidates_compared": len(rows) - 1,
        "comparison_seconds": time.perf_counter() - started,
        "config_sha256": config_id,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Exact nearest TLSH neighbors within training data."
    )
    parser.add_argument(
        "--queries",
        type=int,
        default=64,
        help="Number of query samples; 0 means every train sample.",
    )
    args = parser.parse_args()

    installed_version = version("py-tlsh")

    if installed_version != "5.0.0":
        raise RuntimeError("Expected py-tlsh 5.0.0.")

    rows, csv_sha256 = load_verified_table()
    rows = sorted(rows, key=lambda row: row["sha"])
    sample_count = len(rows)

    if args.queries < 0 or args.queries > sample_count:
        parser.error(f"--queries must be between 0 and {sample_count}.")

    objects = [parse_digest(row["tlsh"]) for row in rows]
    check_distance_api(rows, objects)

    config = {
        "schema_version": 1,
        "purpose": "train-only exact nearest TLSH neighbor audit",
        "input_csv_sha256": csv_sha256,
        "py_tlsh": installed_version,
        "distance": "Tlsh.diff with length contribution",
        "exclude_self": True,
        "train_samples": sample_count,
        "selection_seed": SELECTION_SEED,
        "selection": "ascending SHA256 of seed:sample_sha",
        "representative_tie_break": "smallest candidate SHA",
        "script_sha256": source_sha256(Path(__file__)),
        "loader_sha256": source_sha256(
            ROOT / "scripts/audit_train_tlsh_groups.py"
        ),
        "reference_module_sha256": source_sha256(
            ROOT / "scripts/check_tlsh.py"
        ),
    }

    config_id = hashlib.sha256(
        json.dumps(config, sort_keys=True).encode("utf-8")
    ).hexdigest()

    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
    config_path = OUTPUT_FOLDER / "config.json"

    if config_path.exists():
        if read_json(config_path) != config:
            raise RuntimeError(
                "Input, configuration or source code changed. "
                "Review before reusing this output folder."
            )
    else:
        save_json(config_path, config)

    records_folder = OUTPUT_FOLDER / "records"
    records_folder.mkdir(exist_ok=True)

    # Selection depends only on the fixed seed and sample identifier.
    query_order = sorted(
        range(sample_count),
        key=lambda index: hashlib.sha256(
            f"{SELECTION_SEED}:{rows[index]['sha']}".encode("utf-8")
        ).digest(),
    )

    query_count = args.queries or sample_count
    selected = query_order[:query_count]

    print("Purpose: nearest TLSH neighbor audit")
    print("Split: train only")
    print("Reference samples:", sample_count)
    print("Query samples:", query_count)
    print("Candidates per query:", sample_count - 1)
    print("Self-comparisons: excluded")
    print("Processing sequentially...")

    results = []
    created = 0
    cached = 0
    new_comparison_seconds = 0.0

    for number, query_index in enumerate(selected, start=1):
        query_sha = rows[query_index]["sha"]
        record_path = records_folder / f"{query_sha}.json"

        if record_path.exists():
            result = read_json(record_path)

            if (
                result.get("config_sha256") != config_id
                or result.get("query_sha") != query_sha
                or result.get("candidates_compared") != sample_count - 1
                or result.get("neighbor_sha") == query_sha
            ):
                raise RuntimeError("Cached neighbor record is inconsistent.")

            action = "CACHED"
            cached += 1
        else:
            result = nearest_neighbor(
                query_index, rows, objects, config_id
            )
            save_json(record_path, result)
            action = "CREATED"
            created += 1
            new_comparison_seconds += result["comparison_seconds"]

        results.append(result)

        if number == 1 or number % 8 == 0 or number == query_count:
            print(
                f"[{number}/{query_count}] {action} | "
                f"distance={result['minimum_distance']} | "
                f"ties={result['nearest_tie_count']}"
            )

    complete = query_count == sample_count
    suffix = "" if complete else f"_q{query_count}"
    csv_path = OUTPUT_FOLDER / f"nearest_neighbors{suffix}.csv"
    temporary_csv = csv_path.with_suffix(".csv.tmp")

    with temporary_csv.open(
        "w", encoding="utf-8", newline=""
    ) as file:
        writer = csv.DictWriter(file, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)

    temporary_csv.replace(csv_path)

    distances = [row["minimum_distance"] for row in results]
    histogram = Counter(distances)

    cutoff_counts = {
        str(cutoff): sum(distance <= cutoff for distance in distances)
        for cutoff in REPORT_CUTOFFS
    }

    queries_with_cross_family_ties = sum(
        row["different_family_at_minimum"] > 0
        for row in results
    )

    estimated_full_minutes = None

    if created and new_comparison_seconds > 0:
        estimated_full_minutes = (
            new_comparison_seconds / created * sample_count / 60
        )

    report = {
        "config_sha256": config_id,
        "complete_train_audit": complete,
        "train_samples": sample_count,
        "query_samples": query_count,
        "candidates_per_query": sample_count - 1,
        "created_this_run": created,
        "cached_this_run": cached,
        "minimum_nearest_distance": min(distances),
        "median_nearest_distance": statistics.median(distances),
        "maximum_nearest_distance": max(distances),
        "nearest_distance_histogram": {
            str(distance): count
            for distance, count in sorted(histogram.items())
        },
        "queries_with_nearest_distance_at_most": cutoff_counts,
        "queries_with_cross_family_neighbor_at_minimum": (
            queries_with_cross_family_ties
        ),
        "new_comparison_seconds": new_comparison_seconds,
        "estimated_full_comparison_minutes": estimated_full_minutes,
        "estimate_excludes_loading_and_record_writes": True,
        "csv_sha256": source_sha256(csv_path),
        "near_duplicate_policy_selected": False,
        "samples_removed": 0,
    }

    report_path = OUTPUT_FOLDER / f"summary{suffix}.json"
    save_json(report_path, report)

    print()
    print("Created:", created)
    print("Cached:", cached)
    print("Query samples:", query_count)
    print("Nearest distance min:", min(distances))
    print("Nearest distance median:", statistics.median(distances))
    print("Nearest distance max:", max(distances))

    for cutoff in REPORT_CUTOFFS:
        count = cutoff_counts[str(cutoff)]
        print(f"Queries with nearest distance <= {cutoff}: {count}")

    print(
        "Queries with a cross-family neighbor at minimum:",
        queries_with_cross_family_ties,
    )

    if estimated_full_minutes is not None:
        print(
            "Estimated full comparison minutes (excluding I/O):",
            round(estimated_full_minutes, 2),
        )

    print("Complete train audit:", complete)
    print("Table:", csv_path)
    print("Report:", report_path)
    print("No near-duplicate policy selected. No samples removed.")


if __name__ == "__main__":
    main()