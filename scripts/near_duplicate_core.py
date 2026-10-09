import re
from datetime import datetime, timezone

import tlsh


THRESHOLDS = (0, 10, 20)


def time_key(row):
    observed = datetime.fromisoformat(row["timestamp"])
    if observed.tzinfo is None:
        raise ValueError("Timestamp must include a timezone.")
    return observed.astimezone(timezone.utc), row["sha"]


def validate_rows(rows, split):
    periods = {
        "train": (
            datetime(2019, 8, 1, tzinfo=timezone.utc),
            datetime(2020, 2, 1, tzinfo=timezone.utc),
        ),
        "validation": (
            datetime(2020, 2, 1, tzinfo=timezone.utc),
            datetime(2020, 4, 1, tzinfo=timezone.utc),
        ),
    }
    start, stop = periods[split]
    seen = set()

    for row in rows:
        sha = row["sha"]
        if re.fullmatch(r"[0-9a-f]{64}", sha) is None:
            raise ValueError("Invalid sample identifier.")
        if sha in seen:
            raise ValueError(f"Duplicate sample identifier: {sha}")
        seen.add(sha)

        if row["split"] != split:
            raise ValueError("Unexpected split.")
        if not row["family"].strip():
            raise ValueError("Missing family.")

        observed, _ = time_key(row)
        if not start <= observed < stop:
            raise ValueError("Timestamp outside assigned split.")

        content_sha = row["disarmed_content_sha256"]
        if re.fullmatch(r"[0-9a-f]{64}", content_sha) is None:
            raise ValueError("Invalid content SHA-256.")

        status = row["tlsh_status"]
        digest = row["tlsh"]

        if status == "TNULL":
            if digest != "TNULL":
                raise ValueError("Inconsistent TNULL record.")
        elif status == "VALID":
            if re.fullmatch(r"T1[0-9A-F]{70}", digest) is None:
                raise ValueError("Invalid TLSH format.")
        else:
            raise ValueError("Unknown TLSH status.")

    return sorted(rows, key=time_key)


class UnionFind:
    """Maintain groups connected by exact or near-duplicate relations."""

    def __init__(self, count):
        self.parent = list(range(count))

    def find(self, index):
        while self.parent[index] != index:
            self.parent[index] = self.parent[self.parent[index]]
            index = self.parent[index]
        return index

    def union(self, first, second):
        first = self.find(first)
        second = self.find(second)
        if first != second:
            # Keep the earliest chronological index as representative.
            self.parent[max(first, second)] = min(first, second)


def identity_fields(row):
    return {
        "sha": row["sha"],
        "split": row["split"],
        "timestamp": row["timestamp"],
        "family": row["family"],
        "disarmed_content_sha256": row["disarmed_content_sha256"],
        "tlsh_status": row["tlsh_status"],
    }


def apply_historical_policy(
    train_rows,
    validation_rows,
    distance_function=None,
    progress=None,
):
    """Apply the fixed historical policy at thresholds 0, 10 and 20.

    Production uses TLSH object.diff(), including its length component.
    distance_function is an optional test substitute accepting two digests.

    All earlier validation records remain eligible reference records,
    regardless of whether they were kept or excluded.

    This function returns decisions; it does not change files.
    """
    train = validate_rows(train_rows, "train")
    validation = validate_rows(validation_rows, "validation")

    if not train or not validation:
        raise ValueError("Both historical splits must be nonempty.")

    if {row["sha"] for row in train} & {
        row["sha"] for row in validation
    }:
        raise ValueError("Sample identifiers overlap across splits.")

    rows = train + validation
    train_count = len(train)

    objects = []
    if distance_function is None:
        for row in rows:
            if row["tlsh_status"] == "TNULL":
                objects.append(None)
            else:
                obj = tlsh.Tlsh()
                obj.fromTlshStr(row["tlsh"])
                objects.append(obj)

    def distance(first, second):
        if (
            rows[first]["tlsh_status"] == "TNULL"
            or rows[second]["tlsh_status"] == "TNULL"
        ):
            return None
        if distance_function is not None:
            return distance_function(
                rows[first]["tlsh"], rows[second]["tlsh"]
            )
        return objects[first].diff(objects[second])

    groups = {
        threshold: UnionFind(train_count)
        for threshold in THRESHOLDS
    }

    # Maps each content SHA to its earliest occurrence.
    first_content = {}
    train_exact_references = []

    for current, row in enumerate(train):
        content = row["disarmed_content_sha256"]
        exact_reference = first_content.get(content)
        train_exact_references.append(exact_reference)

        if exact_reference is not None:
            for grouping in groups.values():
                grouping.union(current, exact_reference)

        for previous in range(current):
            # Exact copies have already been connected above.
            if rows[previous]["disarmed_content_sha256"] == content:
                continue

            value = distance(current, previous)
            if value is None or value > THRESHOLDS[-1]:
                continue

            for threshold in THRESHOLDS:
                if value <= threshold:
                    groups[threshold].union(current, previous)

        first_content.setdefault(content, current)

        if progress is not None:
            progress("train", current + 1, train_count)

    train_members = {threshold: [] for threshold in THRESHOLDS}

    for threshold in THRESHOLDS:
        grouping = groups[threshold]

        for index, row in enumerate(train):
            root = grouping.find(index)
            exact_reference = train_exact_references[index]
            excluded = exact_reference is not None

            train_members[threshold].append({
                **identity_fields(row),
                "threshold": threshold,
                "group_id": f"T{threshold}_{train[root]['sha']}",
                "decision": "EXCLUDE" if excluded else "KEEP",
                "reason": "exact_sha256" if excluded else "retained_train",
                "reference_sha": (
                    train[exact_reference]["sha"] if excluded else ""
                ),
            })

    decisions = {threshold: [] for threshold in THRESHOLDS}

    for number, row in enumerate(validation, start=1):
        current = train_count + number - 1
        content = row["disarmed_content_sha256"]
        exact_reference = first_content.get(content)

        witnesses = {threshold: None for threshold in THRESHOLDS}

        if exact_reference is not None:
            for threshold in THRESHOLDS:
                witnesses[threshold] = (
                    exact_reference, "exact_sha256", None
                )
        else:
            # Scan ALL earlier records, including excluded validation rows.
            for previous in range(current):
                value = distance(current, previous)
                if value is None or value > THRESHOLDS[-1]:
                    continue

                for threshold in THRESHOLDS:
                    if (
                        value <= threshold
                        and witnesses[threshold] is None
                    ):
                        witnesses[threshold] = (
                            previous, "tlsh", value
                        )

                # Every threshold has a valid earlier witness.
                if all(item is not None for item in witnesses.values()):
                    break

        for threshold in THRESHOLDS:
            witness = witnesses[threshold]

            if witness is None:
                decision = {
                    "decision": "KEEP",
                    "reason": "no_earlier_match",
                    "reference_sha": "",
                    "reference_split": "",
                    "reference_timestamp": "",
                    "tlsh_distance": None,
                }
            else:
                previous, reason, value = witness
                reference = rows[previous]

                if time_key(reference) >= time_key(row):
                    raise RuntimeError("Witness is not earlier in ordering.")

                decision = {
                    "decision": "EXCLUDE",
                    "reason": reason,
                    "reference_sha": reference["sha"],
                    "reference_split": reference["split"],
                    "reference_timestamp": reference["timestamp"],
                    "tlsh_distance": value,
                }

            decisions[threshold].append({
                **identity_fields(row),
                "threshold": threshold,
                **decision,
            })

        # Do not restrict this dictionary to retained samples.
        first_content.setdefault(content, current)

        if progress is not None:
            progress("validation", number, len(validation))

    return {
        "thresholds": list(THRESHOLDS),
        "train_members": train_members,
        "validation_decisions": decisions,
    }