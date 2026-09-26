import csv
from datetime import date, datetime
from pathlib import Path
from collections import Counter , defaultdict


data_path = Path("data/raw/bodmas_metadata.csv")
total_rows = 0
train_rows = 0
train_rows_with_family = 0

train_labeled_shas = set()
train_families = set()
family_counts = Counter()
family_months = defaultdict(set)
family_shas = defaultdict(set)


with data_path.open(encoding="utf-8-sig", newline="") as file:
    reader = csv.DictReader(file)

    for row in reader:
        total_rows += 1
        observed_date = datetime.fromisoformat(row["timestamp"]).date()

        if date(2019, 8, 1) <= observed_date < date(2020, 2, 1):
            train_rows += 1
            if row["family"].strip():
                train_rows_with_family += 1
                train_labeled_shas.add(row['sha'].strip().lower())
                train_families.add(row['family'].strip())
                family_counts[row['family'].strip()] += 1
                family_months[row["family"].strip()].add(observed_date.strftime("%Y-%m"))
                family_shas[row["family"].strip()].add(row['sha'].strip().lower())

print("Total rows:", total_rows)
print("Train rows:", train_rows)
print("Train rows with family:", train_rows_with_family)
print('Unique labeled train SHA:', len(train_labeled_shas))
print('Distinct train families:', len(train_families))

for family, count in family_counts.most_common(10):
    print(f"{family}: {count} files, {len(family_months[family])} months")



eligible_families = [
    family
    for family in family_counts
    if len(family_shas[family]) >= 50 and len(family_months[family]) >= 3
]

eligible_files = sum(len(family_shas[family]) for family in eligible_families)

print("Eligible families:", len(eligible_families))
print("Train files in eligible families:", eligible_files)
print("Train files in excluded families:", train_rows_with_family - eligible_files)

cohort_path = Path("docs/train_family_cohort.csv")

with cohort_path.open("w", encoding="utf-8", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["family", "unique_train_sha", "train_months"])

    for family in sorted(eligible_families):
        writer.writerow([
            family,
            len(family_shas[family]),
            len(family_months[family]),
        ])

print("Saved cohort to:", cohort_path)
