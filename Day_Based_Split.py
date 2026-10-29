from pathlib import Path
import pandas as pd

DATASET = Path(
    "/Users/almaleki/Desktop/Research/Research_OCT2025/Codes/"
    "PhD_Research_1_CIC2018Dataset/.venv/bin/multiclass_balanced.csv"
)

df = pd.read_csv(
    DATASET,
    usecols=["Timestamp", "AttackCategory"],
    low_memory=False,
)

timestamps = df["Timestamp"].astype("string").str.strip()

print("\nTIMESTAMP EXAMPLES")
print(timestamps.drop_duplicates().head(10).to_string(index=False))

# Compare both interpretations before deciding the date convention.
day_first = pd.to_datetime(
    timestamps, format="mixed", dayfirst=True, errors="coerce"
)
month_first = pd.to_datetime(
    timestamps, format="mixed", dayfirst=False, errors="coerce"
)

both_valid = day_first.notna() & month_first.notna()
different_dates = (
    both_valid
    & (day_first.dt.normalize() != month_first.dt.normalize())
)

print("\nUnparsed timestamps (day-first):", day_first.isna().sum())
print("Unparsed timestamps (month-first):", month_first.isna().sum())
print("Rows with different date interpretations:", different_dates.sum())

for name, parsed in [
    ("DAY-FIRST INTERPRETATION", day_first),
    ("MONTH-FIRST INTERPRETATION", month_first),
]:
    counts = pd.crosstab(
        parsed.dt.strftime("%Y-%m-%d").fillna("UNPARSED"),
        df["AttackCategory"],
    )
    print(f"\n{name}: RECORDS BY DATE AND CLASS")
    print(counts.to_string())

    valid_counts = counts.drop(index="UNPARSED", errors="ignore")
    print("\nNumber of dates containing each class:")
    print((valid_counts > 0).sum().to_string())

print("\nDiagnostic only: no dataset or partition was changed.")