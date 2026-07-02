# ==========================================
# HANDLE MISSING VALUES
# ==========================================

import pandas as pd
import os

# ==========================================
# SELECT CATEGORY
# ==========================================

CATEGORY = input(
    "Enter category (infrastructure, controversies, faculty_research, rankings): "
).strip().lower()

INPUT_FILE = f"data/{CATEGORY}/videos.csv"
OUTPUT_DIR = "cleaned_data"

os.makedirs(OUTPUT_DIR, exist_ok=True)

OUTPUT_FILE = f"{OUTPUT_DIR}/{CATEGORY}_clean.csv"

# ==========================================
# LOAD DATASET
# ==========================================

if not os.path.exists(INPUT_FILE):
    print(f"\n❌ File not found: {INPUT_FILE}")
    exit()

df = pd.read_csv(INPUT_FILE)

print(f"\n✅ {CATEGORY.upper()} dataset loaded successfully.")

# ==========================================
# MISSING VALUE SUMMARY (BEFORE)
# ==========================================

print("\n" + "=" * 70)
print("MISSING VALUES BEFORE CLEANING")
print("=" * 70)

print(df.isnull().sum())

# ==========================================
# HANDLE DESCRIPTION
# ==========================================

if "description" in df.columns:

    missing = df["description"].isnull().sum()

    df["description"] = df["description"].fillna("")

    print(f"\n✔ Filled {missing} missing descriptions with empty strings.")

# ==========================================
# HANDLE TITLE
# ==========================================

if "title" in df.columns:

    missing = df["title"].isnull().sum()

    if missing > 0:

        df = df.dropna(subset=["title"])

        print(f"✔ Removed {missing} rows with missing titles.")

# ==========================================
# HANDLE CHANNEL NAME
# ==========================================

if "channel_name" in df.columns:

    missing = df["channel_name"].isnull().sum()

    df["channel_name"] = df["channel_name"].fillna("Unknown Channel")

    print(f"✔ Filled {missing} missing channel names.")

# ==========================================
# HANDLE LANGUAGE
# ==========================================

if "language" in df.columns:

    missing = df["language"].isnull().sum()

    df["language"] = df["language"].fillna("unknown")

    print(f"✔ Filled {missing} missing language values.")

# ==========================================
# HANDLE NUMERIC COLUMNS
# ==========================================

numeric_columns = [

    "views",
    "likes",
    "comment_count"

]

for column in numeric_columns:

    if column in df.columns:

        missing = df[column].isnull().sum()

        df[column] = df[column].fillna(0)

        if missing > 0:
            print(f"✔ Filled {missing} missing values in {column}.")

# ==========================================
# VERIFY
# ==========================================

print("\n" + "=" * 70)
print("MISSING VALUES AFTER CLEANING")
print("=" * 70)

print(df.isnull().sum())

# ==========================================
# SAVE CLEAN DATA
# ==========================================

df.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8"
)

print(f"\n✅ Clean dataset saved to:\n{OUTPUT_FILE}")