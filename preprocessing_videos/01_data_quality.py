# ==========================================
# DATA QUALITY ASSESSMENT
# ==========================================

import pandas as pd
import os

# ==========================================
# SELECT CATEGORY
# ==========================================

CATEGORY = input(
    "Enter category (infrastructure, controversies, faculty_research, rankings, administration): "
).strip().lower()

DATA_PATH = f"data/{CATEGORY}/videos.csv"

# ==========================================
# LOAD DATASET
# ==========================================

if not os.path.exists(DATA_PATH):
    print(f"\n❌ File not found: {DATA_PATH}")
    exit()

df = pd.read_csv(DATA_PATH)

print(f"\n✅ {CATEGORY.upper()} dataset loaded successfully.\n")

# ==========================================
# BASIC INFORMATION
# ==========================================

print("=" * 70)
print(f"DATA QUALITY REPORT : {CATEGORY.upper()}")
print("=" * 70)

print(f"\nDataset Shape : {df.shape}")
print(f"Rows          : {df.shape[0]}")
print(f"Columns       : {df.shape[1]}")

# ==========================================
# COLUMN NAMES
# ==========================================

print("\n" + "=" * 70)
print("COLUMN NAMES")
print("=" * 70)

for column in df.columns:
    print(column)

# ==========================================
# DATA TYPES
# ==========================================

print("\n" + "=" * 70)
print("DATA TYPES")
print("=" * 70)

print(df.dtypes)

# ==========================================
# MISSING VALUES
# ==========================================

print("\n" + "=" * 70)
print("MISSING VALUES")
print("=" * 70)

missing = df.isnull().sum()

missing = missing[missing > 0]

if missing.empty:
    print("No missing values found.")
else:
    print(missing)

# ==========================================
# MISSING VALUE PERCENTAGE
# ==========================================

print("\n" + "=" * 70)
print("MISSING VALUE PERCENTAGE")
print("=" * 70)

missing_percent = (
    df.isnull().sum() /
    len(df)
) * 100

missing_percent = missing_percent[
    missing_percent > 0
].sort_values(ascending=False)

if missing_percent.empty:
    print("No missing values.")
else:
    print(missing_percent.round(2))

# ==========================================
# DUPLICATE ROWS
# ==========================================

print("\n" + "=" * 70)
print("DUPLICATE ROWS")
print("=" * 70)

duplicate_rows = df.duplicated().sum()

print(f"Duplicate Rows : {duplicate_rows}")

# ==========================================
# DUPLICATE VIDEO IDs
# ==========================================

print("\n" + "=" * 70)
print("DUPLICATE VIDEO IDs")
print("=" * 70)

duplicate_ids = df["video_id"].duplicated().sum()

print(f"Duplicate Video IDs : {duplicate_ids}")

# ==========================================
# UNIQUE UNIVERSITIES
# ==========================================

print("\n" + "=" * 70)
print("UNIQUE UNIVERSITIES")
print("=" * 70)

print(df["university_name"].nunique())

# ==========================================
# UNIQUE CHANNELS
# ==========================================

print("\n" + "=" * 70)
print("UNIQUE CHANNELS")
print("=" * 70)

print(df["channel_name"].nunique())

# ==========================================
# MEMORY USAGE
# ==========================================

print("\n" + "=" * 70)
print("MEMORY USAGE")
print("=" * 70)

memory = df.memory_usage(
    deep=True
).sum() / (1024 * 1024)

print(f"{memory:.2f} MB")

# ==========================================
# NUMERICAL SUMMARY
# ==========================================

print("\n" + "=" * 70)
print("NUMERICAL SUMMARY")
print("=" * 70)

print(df.describe())

# ==========================================
# CATEGORICAL SUMMARY
# ==========================================

print("\n" + "=" * 70)
print("TOP 10 UNIVERSITIES")
print("=" * 70)

print(
    df["university_name"]
    .value_counts()
    .head(10)
)

print("\n" + "=" * 70)
print("TOP 10 CHANNELS")
print("=" * 70)

print(
    df["channel_name"]
    .value_counts()
    .head(10)
)

# ==========================================
# SAMPLE DATA
# ==========================================

print("\n" + "=" * 70)
print("FIRST 5 ROWS")
print("=" * 70)

print(df.head())

print("\n" + "=" * 70)
print("LAST 5 ROWS")
print("=" * 70)

print(df.tail())

print(f"\n✅ Data Quality Assessment for {CATEGORY.upper()} completed successfully.")