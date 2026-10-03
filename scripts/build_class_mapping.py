import csv
import json
from pathlib import Path


def main():
    project_root = Path(__file__).resolve().parents[1]
    cohort_path = project_root / "docs" / "train_family_cohort.csv"
    output_path = project_root / "docs" / "class_mapping.json"

    # Read the cohort already selected using training data only.
    with cohort_path.open(encoding="utf-8-sig", newline="") as file:
        families = [
            row["family"].strip()
            for row in csv.DictReader(file)
        ]

    if not families or any(not family for family in families):
        raise ValueError("Cohort contains no families or an empty name.")

    if len(families) != len(set(families)):
        raise ValueError("Cohort contains duplicate family names.")

    if len(families) != 51:
        raise ValueError("Expected the frozen cohort of 51 families.")

    # Alphabetical ordering makes label indices reproducible.
    families = sorted(families)
    class_to_idx = {
        family: index
        for index, family in enumerate(families)
    }

    mapping = {
        "schema_version": 1,
        "source": "docs/train_family_cohort.csv",
        "num_classes": len(families),
        "class_to_idx": class_to_idx,
    }

    # Do not silently replace a different existing label mapping.
    if output_path.exists():
        existing = json.loads(output_path.read_text(encoding="utf-8"))

        if existing != mapping:
            raise ValueError(
                "Existing class mapping differs. "
                "Review the difference before changing label indices."
            )

        print("Existing mapping matches: PASS")
    else:
        output_path.write_text(
            json.dumps(mapping, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print("Created class mapping.")

    print("Number of classes:", len(families))
    print("Label range: 0 to", len(families) - 1)
    print("First 10 labels:")

    for family in families[:10]:
        print(f"  {class_to_idx[family]} -> {family}")

    print("Saved to:", output_path.relative_to(project_root))


if __name__ == "__main__":
    main()