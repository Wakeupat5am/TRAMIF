import csv
from datetime import date, datetime
from pathlib import Path

cohort_path = Path("docs/train_family_cohort.csv")
metadata_path = Path("data/raw/bodmas_metadata.csv")

with cohort_path.open(encoding="utf-8", newline="") as file:
    cohort = {row["family"] for row in csv.DictReader(file)}

train_shas = set()
validation_rows = 0
validation_labeled_rows = 0
validation_cohort_rows = 0
validation_cohort_shas = set()
validation_cohort_families = set()

with metadata_path.open(encoding="utf-8-sig", newline="") as file:
    for row in csv.DictReader(file):
        observed_date = datetime.fromisoformat(row["timestamp"]).date()
        sha = row["sha"].strip().lower()
        family = row["family"].strip()

        if date(2019, 8, 1) <= observed_date < date(2020, 2, 1):
            train_shas.add(sha)

        if date(2020, 2, 1) <= observed_date < date(2020, 4, 1):
            validation_rows += 1

            if family:
                validation_labeled_rows += 1

                if family in cohort:
                    validation_cohort_rows += 1
                    validation_cohort_shas.add(sha)
                    validation_cohort_families.add(family)

missing_families = sorted(cohort - validation_cohort_families)

print("Frozen train families:", len(cohort))
print("Validation rows:", validation_rows)
print("Validation rows with family:", validation_labeled_rows)
print("Validation rows in frozen cohort:", validation_cohort_rows)
print("Unique validation SHA in cohort:", len(validation_cohort_shas))
print("Cohort families absent from validation:", missing_families)
print(
    "Train/validation cohort SHA overlap:",
    len(train_shas & validation_cohort_shas),
)