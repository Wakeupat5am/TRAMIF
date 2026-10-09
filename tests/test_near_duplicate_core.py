import unittest

from scripts.near_duplicate_core import apply_historical_policy


def make_row(
    identifier,
    split,
    family="family_a",
    content=None,
    timestamp=None,
    tnull=False,
):
    if timestamp is None:
        timestamp = (
            "2019-09-01 00:00:00+00:00"
            if split == "train"
            else "2020-02-01 00:00:00+00:00"
        )

    return {
        "sha": f"{identifier:064x}",
        "split": split,
        "timestamp": timestamp,
        "family": family,
        "disarmed_content_sha256": (
            f"{content if content is not None else identifier + 1000:064x}"
        ),
        "tlsh": "TNULL" if tnull else f"T1{identifier:070X}",
        "tlsh_status": "TNULL" if tnull else "VALID",
    }


def distances(pair_values):
    def lookup(first, second):
        first_id = int(first[2:], 16)
        second_id = int(second[2:], 16)
        key = tuple(sorted((first_id, second_id)))
        return pair_values.get(key, 1000)
    return lookup


class TestNearDuplicateCore(unittest.TestCase):
    def test_thresholds_apply_without_family_filter(self):
        train = [make_row(1, "train", family="family_a")]
        validation = [make_row(2, "validation", family="family_b")]

        result = apply_historical_policy(
            train, validation, distances({(1, 2): 10})
        )

        decisions = result["validation_decisions"]
        self.assertEqual(decisions[0][0]["decision"], "KEEP")
        self.assertEqual(decisions[10][0]["decision"], "EXCLUDE")
        self.assertEqual(decisions[20][0]["decision"], "EXCLUDE")
        self.assertEqual(decisions[10][0]["tlsh_distance"], 10)

    def test_excluded_validation_still_serves_as_reference(self):
        train = [make_row(1, "train")]
        validation = [
            make_row(2, "validation"),
            make_row(
                3, "validation",
                timestamp="2020-03-01 00:00:00+00:00",
            ),
        ]

        result = apply_historical_policy(
            train,
            validation,
            distances({(1, 2): 0, (2, 3): 0}),
        )

        for threshold in (0, 10, 20):
            decisions = result["validation_decisions"][threshold]
            self.assertEqual(decisions[0]["decision"], "EXCLUDE")
            self.assertEqual(decisions[1]["decision"], "EXCLUDE")
            self.assertEqual(
                decisions[1]["reference_sha"],
                validation[0]["sha"],
            )

    def test_later_bridge_does_not_change_earlier_decision(self):
        train = [make_row(1, "train")]
        validation = [
            make_row(2, "validation"),
            make_row(
                3, "validation",
                timestamp="2020-03-01 00:00:00+00:00",
            ),
        ]

        result = apply_historical_policy(
            train,
            validation,
            distances({(1, 3): 0, (2, 3): 0}),
        )

        for threshold in (0, 10, 20):
            decisions = result["validation_decisions"][threshold]
            self.assertEqual(decisions[0]["decision"], "KEEP")
            self.assertEqual(decisions[1]["decision"], "EXCLUDE")

    def test_timestamp_tie_is_resolved_by_sha(self):
        first = make_row(2, "validation")
        second = make_row(3, "validation")

        result = apply_historical_policy(
            [make_row(1, "train")],
            [second, first],
            distances({(2, 3): 0}),
        )

        decisions = result["validation_decisions"][0]
        self.assertEqual(decisions[0]["sha"], first["sha"])
        self.assertEqual(decisions[0]["decision"], "KEEP")
        self.assertEqual(decisions[1]["reference_sha"], first["sha"])

    def test_tnull_still_allows_exact_duplicate_detection(self):
        train = [make_row(1, "train", content=99, tnull=True)]
        validation = [
            make_row(2, "validation", content=99, tnull=True),
            make_row(3, "validation", content=100, tnull=True),
        ]

        def forbidden_distance(first, second):
            raise AssertionError("Must not calculate distance for TNULL.")

        result = apply_historical_policy(
            train, validation, forbidden_distance
        )

        for threshold in (0, 10, 20):
            decisions = result["validation_decisions"][threshold]
            self.assertEqual(decisions[0]["reason"], "exact_sha256")
            self.assertEqual(decisions[0]["decision"], "EXCLUDE")
            self.assertEqual(decisions[1]["decision"], "KEEP")
            self.assertEqual(decisions[1]["tlsh_status"], "TNULL")

    def test_train_groups_are_transitive_without_removing_near_copies(self):
        train = [
            make_row(1, "train"),
            make_row(2, "train"),
            make_row(3, "train"),
            make_row(4, "train", content=1001),
        ]

        result = apply_historical_policy(
            train,
            [make_row(5, "validation")],
            distances({(1, 2): 10, (2, 3): 10}),
        )

        members = result["train_members"][10]
        self.assertEqual(len({row["group_id"] for row in members}), 1)
        self.assertEqual(
            [row["decision"] for row in members],
            ["KEEP", "KEEP", "KEEP", "EXCLUDE"],
        )
        self.assertEqual(members[3]["reference_sha"], train[0]["sha"])

        # At threshold zero, only the exact duplicate joins sample 1.
        strict = result["train_members"][0]
        self.assertEqual(len({row["group_id"] for row in strict}), 3)

    def test_rejects_timestamp_outside_split(self):
        wrong = make_row(
            2, "validation",
            timestamp="2020-04-01 00:00:00+00:00",
        )
        with self.assertRaises(ValueError):
            apply_historical_policy(
                [make_row(1, "train")],
                [wrong],
                distances({}),
            )

    def test_rejects_duplicate_sample_identifiers(self):
        repeated = make_row(2, "validation")
        with self.assertRaises(ValueError):
            apply_historical_policy(
                [make_row(1, "train")],
                [repeated, repeated],
                distances({}),
            )


if __name__ == "__main__":
    unittest.main()