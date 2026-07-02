"""
===========================================================
06_validation.py

Validation Script for YouTube University Dataset

Checks

1. Dataset Loading
2. Missing Values (Important Columns Only)
3. Duplicate Rows
4. Duplicate Video IDs
5. Negative Numerical Values
6. Required Columns
7. Engineered Features
8. Empty Combined Text

Author : Ananya Upadhyay
===========================================================
"""

# ==========================================================
# IMPORT LIBRARIES
# ==========================================================

import os
import pandas as pd

# ==========================================================
# REQUIRED COLUMNS
# ==========================================================

REQUIRED_COLUMNS = [

    "video_id",
    "university_name",

    "title",
    "description",

    "clean_title",
    "clean_description",
    "combined_text",

    "views",
    "likes",
    "comment_count",

    "upload_date"

]

# ==========================================================
# IMPORTANT COLUMNS
# (Only these must never be missing)
# ==========================================================

IMPORTANT_COLUMNS = [

    "video_id",

    "title",

    "clean_title",

    "combined_text",

    "views",
    "likes",
    "comment_count",

    "upload_date"

]

# ==========================================================
# ENGINEERED FEATURES
# ==========================================================

ENGINEERED_COLUMNS = [

    "title_word_count",
    "title_char_count",

    "description_word_count",
    "description_char_count",

    "combined_word_count",
    "combined_char_count",

    "emoji_count",
    "hashtag_count",
    "mention_count",

    "exclamation_count",
    "question_count",

    "engagement_rate",

    "log_views",
    "log_likes",
    "log_comments",

    "video_age_days"

]

# ==========================================================
# CHECK REQUIRED COLUMNS
# ==========================================================

def check_columns(df):

    missing = []

    for column in REQUIRED_COLUMNS:

        if column not in df.columns:
            missing.append(column)

    return missing


# ==========================================================
# CHECK ENGINEERED FEATURES
# ==========================================================

def check_engineered(df):

    missing = []

    for column in ENGINEERED_COLUMNS:

        if column not in df.columns:
            missing.append(column)

    return missing

# ==========================================================
# MAIN PROGRAM
# ==========================================================

def main():

    print("=" * 60)
    print("DATASET VALIDATION")
    print("=" * 60)

    category = input(
        "\nEnter category "
        "(infrastructure, controversies, faculty_research, rankings): "
    ).strip().lower()

    INPUT_FILE = f"cleaned_data/{category}_feature_engineered.csv"

    if not os.path.exists(INPUT_FILE):

        print("\n❌ Input file not found.")
        return

    print("\nLoading dataset...")

    df = pd.read_csv(INPUT_FILE)

    print(f"✓ Loaded {len(df)} rows.")

    # ======================================================
    # REQUIRED COLUMNS
    # ======================================================

    missing_columns = check_columns(df)

    # ======================================================
    # ENGINEERED FEATURES
    # ======================================================

    missing_features = check_engineered(df)

    # ======================================================
    # IMPORTANT MISSING VALUES
    # ======================================================

    important_missing = (
        df[IMPORTANT_COLUMNS]
        .isnull()
        .sum()
    )

    important_missing = important_missing[
        important_missing > 0
    ]

    total_missing = important_missing.sum()

    # ======================================================
    # OPTIONAL MISSING VALUES
    # ======================================================

    optional_columns = [

        column for column in df.columns

        if column not in IMPORTANT_COLUMNS

    ]

    optional_missing = (
        df[optional_columns]
        .isnull()
        .sum()
    )

    optional_missing = optional_missing[
        optional_missing > 0
    ]

    # ======================================================
    # DUPLICATES
    # ======================================================

    duplicate_rows = df.duplicated().sum()

    duplicate_ids = df["video_id"].duplicated().sum()

    # ======================================================
    # NEGATIVE VALUES
    # ======================================================

    negative_views = (df["views"] < 0).sum()

    negative_likes = (df["likes"] < 0).sum()

    negative_comments = (df["comment_count"] < 0).sum()

    negative_age = (df["video_age_days"] < 0).sum()

    negative_engagement = (
        df["engagement_rate"] < 0
    ).sum()

    # ======================================================
    # EMPTY TEXT
    # ======================================================

    empty_text = (
        df["combined_text"]
        .fillna("")
        .str.strip()
        .eq("")
        .sum()
    )

    # ======================================================
    # REPORT
    # ======================================================

    print("\n" + "=" * 60)
    print("VALIDATION REPORT")
    print("=" * 60)

    print(f"Category                : {category}")
    print(f"Rows                    : {len(df)}")
    print(f"Columns                 : {len(df.columns)}")

    print("\nMissing Values (Important) :", total_missing)
    print("Duplicate Rows             :", duplicate_rows)
    print("Duplicate Video IDs        :", duplicate_ids)

    print("\nNegative Views             :", negative_views)
    print("Negative Likes             :", negative_likes)
    print("Negative Comments          :", negative_comments)
    print("Negative Video Age         :", negative_age)
    print("Negative Engagement        :", negative_engagement)

    print("\nEmpty Combined Text        :", empty_text)

    # ======================================================
    # IMPORTANT MISSING DETAILS
    # ======================================================

    if len(important_missing) > 0:

        print("\nImportant Columns with Missing Values:")

        print(important_missing)

    else:

        print("\n✓ No missing values in important columns.")

    # ======================================================
    # OPTIONAL MISSING DETAILS
    # ======================================================

    if len(optional_missing) > 0:

        print("\nOptional Columns with Missing Values:")

        print(optional_missing)

    else:

        print("\n✓ No missing values in optional columns.")

    # ======================================================
    # REQUIRED COLUMNS
    # ======================================================

    print("\nRequired Columns")

    if len(missing_columns) == 0:

        print("✓ All required columns present.")

    else:

        print("❌ Missing Required Columns:")

        for column in missing_columns:

            print("   -", column)

    # ======================================================
    # ENGINEERED FEATURES
    # ======================================================

    print("\nEngineered Features")

    if len(missing_features) == 0:

        print("✓ All engineered features present.")

    else:

        print("❌ Missing Engineered Features:")

        for feature in missing_features:

            print("   -", feature)

    # ======================================================
    # FINAL STATUS
    # ======================================================

    passed = (

        total_missing == 0 and
        duplicate_rows == 0 and
        duplicate_ids == 0 and
        negative_views == 0 and
        negative_likes == 0 and
        negative_comments == 0 and
        negative_age == 0 and
        negative_engagement == 0 and
        empty_text == 0 and
        len(missing_columns) == 0 and
        len(missing_features) == 0

    )

    print("\n" + "=" * 60)

    if passed:

        print("✅ DATASET VALIDATED SUCCESSFULLY")

    else:

        print("⚠ DATASET HAS VALIDATION ISSUES")

    print("=" * 60)


# ==========================================================
# DRIVER CODE
# ==========================================================

if __name__ == "__main__":
    main()