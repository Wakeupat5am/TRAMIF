import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BINARY_FOLDER = ROOT / "data/raw/altered"

SAMPLES = [
    {
        "family": "small",
        "sha": (
            "46e7a9298222e91984ef4492a7b4e20aba2293d3eac766633903f597d52d88ed"
        ),
        "size": 9600947,
        "content_sha256": (
            "a88f694a4c470ac81cd75022abc1a7497da50343216347d25ac4eeac8344b791"
        ),
    },
    {
        "family": "sillyp2p",
        "sha": (
            "84f0d44c2820069c124b288b904147b2e58714318163f86368e2191cd1f4a8b1"
        ),
        "size": 10177127,
        "content_sha256": (
            "6693065658371d01c137310ca4b50dba4447f8e1bc2e8460e1655a69203d65af"
        ),
    },
]

BLOCK_SIZE = 4096
ANCHOR_COUNT = 64


def read_verified_sample(sample):
    path = BINARY_FOLDER / f"{sample['sha']}.exe"

    # Bounded read; the file is read as data only.
    with path.open("rb") as file:
        data = file.read(sample["size"] + 1)

    if len(data) != sample["size"]:
        raise ValueError(f"Size mismatch: {sample['family']}")

    actual_sha = hashlib.sha256(data).hexdigest()

    if actual_sha != sample["content_sha256"]:
        raise ValueError(f"Content SHA mismatch: {sample['family']}")

    return data


def inspect_anchor(shorter, longer, offset):
    block = shorter[offset:offset + BLOCK_SIZE]

    first_source = shorter.find(block)
    second_source = shorter.find(block, first_source + 1)

    result = {
        "source_offset": offset,
        "length": len(block),
        "block_sha256": hashlib.sha256(block).hexdigest(),
        "target_offset": None,
        "shift": None,
    }

    if second_source != -1:
        result["status"] = "AMBIGUOUS_SOURCE"
        return result

    first_target = longer.find(block)

    if first_target == -1:
        result["status"] = "NOT_FOUND"
        return result

    second_target = longer.find(block, first_target + 1)

    if second_target != -1:
        result["status"] = "AMBIGUOUS_TARGET"
        return result

    result["status"] = "UNIQUE_MATCH"
    result["target_offset"] = first_target
    result["shift"] = first_target - offset
    return result


def main():
    shorter = read_verified_sample(SAMPLES[0])
    longer = read_verified_sample(SAMPLES[1])

    if not BLOCK_SIZE <= len(shorter) < len(longer):
        raise ValueError("Unexpected input sizes.")

    prefix = 0

    while (
        prefix < len(shorter)
        and shorter[prefix] == longer[prefix]
    ):
        prefix += 1

    # Do not overlap the already matched prefix in the shorter file.
    suffix = 0

    while (
        suffix < len(shorter) - prefix
        and shorter[-1 - suffix] == longer[-1 - suffix]
    ):
        suffix += 1

    shorter_middle = len(shorter) - prefix - suffix
    longer_middle = len(longer) - prefix - suffix

    # If true, deleting one contiguous region from the longer file
    # reproduces the shorter file exactly. This describes bytes,
    # not the historical process that produced either file.
    single_insertion_compatible = shorter_middle == 0

    offsets = [
        (len(shorter) - BLOCK_SIZE) * index // (ANCHOR_COUNT - 1)
        for index in range(ANCHOR_COUNT)
    ]

    anchors = [
        inspect_anchor(shorter, longer, offset)
        for offset in offsets
    ]

    status_counts = Counter(item["status"] for item in anchors)
    shift_counts = Counter(
        item["shift"]
        for item in anchors
        if item["status"] == "UNIQUE_MATCH"
    )

    report = {
        "purpose": "inspect train mixed-label TLSH group G000038",
        "samples": SAMPLES,
        "content_sha256_checks": "PASS",
        "size_difference": len(longer) - len(shorter),
        "common_prefix_bytes": prefix,
        "common_suffix_bytes_nonoverlapping": suffix,
        "shorter_middle_bytes": shorter_middle,
        "longer_middle_bytes": longer_middle,
        "compatible_with_one_contiguous_insertion": (
            single_insertion_compatible
        ),
        "anchor_count": ANCHOR_COUNT,
        "anchor_block_size": BLOCK_SIZE,
        "anchor_status_counts": dict(status_counts),
        "unique_match_shift_counts": {
            str(shift): count
            for shift, count in sorted(shift_counts.items())
        },
        "anchors": anchors,
        "script_sha256": hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
        "samples_removed": 0,
        "labels_changed": 0,
    }

    run_id = datetime.now(timezone.utc).strftime(
        "%Y%m%d_%H%M%S_%f"
    )
    output_folder = (
        ROOT / "data/derived/mixed_tlsh_pair" / run_id
    )
    output_folder.mkdir(parents=True, exist_ok=False)
    report_path = output_folder / "report.json"

    with report_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False)
        file.write("\n")

    print("Pair: small / sillyp2p")
    print("Content SHA-256 checks: PASS")
    print("Shorter file bytes:", len(shorter))
    print("Longer file bytes:", len(longer))
    print("Size difference:", len(longer) - len(shorter))
    print("Common prefix bytes:", prefix)
    print("Common suffix bytes (nonoverlapping):", suffix)
    print("Shorter unmatched middle:", shorter_middle)
    print("Longer unmatched middle:", longer_middle)
    print(
        "Compatible with one contiguous insertion:",
        single_insertion_compatible,
    )

    print()
    print("Sampled anchors:", ANCHOR_COUNT)
    print("Bytes per anchor:", BLOCK_SIZE)

    for status in (
        "UNIQUE_MATCH",
        "NOT_FOUND",
        "AMBIGUOUS_SOURCE",
        "AMBIGUOUS_TARGET",
    ):
        print(f"{status}: {status_counts[status]}")

    print("Shifts among unique matches (target minus source):")

    for shift, count in sorted(shift_counts.items()):
        print(f"  shift={shift:+d} bytes | anchors={count}")

    print("Report:", report_path)
    print("Binary files and labels unchanged.")


if __name__ == "__main__":
    main()