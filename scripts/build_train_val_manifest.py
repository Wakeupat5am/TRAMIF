import csv
from datetime import date, datetime
from pathlib import Path

metadata_path = Path("data/raw/bodmas_metadata.csv")
cohort_path = Path("docs/train_family_cohort.csv")
manifest_path = Path("data/manifests/train_validation_manifest.csv")

with cohort_path.open(encoding="utf-8", newline="") as file:
    cohort = {row["family"] for row in csv.DictReader(file)}

samples = []
seen_shas = set()
split_counts = {"train": 0, "validation": 0}

with metadata_path.open(encoding="utf-8-sig", newline="") as file:
    for row in csv.DictReader(file):
        family = row["family"].strip()
        if family not in cohort:
            continue

        observed_date = datetime.fromisoformat(row["timestamp"]).date()

        if date(2019, 8, 1) <= observed_date < date(2020, 2, 1):
            split = "train"
        elif date(2020, 2, 1) <= observed_date < date(2020, 4, 1):
            split = "validation"
        else:
            continue

        sha = row["sha"].strip().lower()
        if sha in seen_shas:
            raise ValueError(f"SHA appears more than once: {sha}")

        seen_shas.add(sha)
        split_counts[split] += 1
        samples.append({
            "sha": sha,
            "split": split,
            "timestamp": row["timestamp"],
            "family": family,
        })

samples.sort(
    key=lambda item: (
        0 if item["split"] == "train" else 1,
        item["timestamp"],
        item["sha"],
    )
)

manifest_path.parent.mkdir(parents=True, exist_ok=True)

with manifest_path.open("w", encoding="utf-8", newline="") as file:
    writer = csv.DictWriter(
        file,
        fieldnames=["sha", "split", "timestamp", "family"],
    )
    writer.writeheader()
    writer.writerows(samples)

print("Train SHA:", split_counts["train"])
print("Validation SHA:", split_counts["validation"])
print("Total SHA:", len(samples))
print("Saved manifest to:", manifest_path)