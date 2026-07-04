"""
============================================================
05_comment_validation.py

Final Validation for Comment Dataset

Input:
    analysis_comments/<category>_comment_feature_engineered.csv

Output:
    analysis_comments/reports/<category>_validation_report.csv

Author : Ananya Upadhyay
============================================================
"""

# ==========================================================
# IMPORT LIBRARIES
# ==========================================================

import os
import pandas as pd

# ==========================================================
# FOLDERS
# ==========================================================

INPUT_FOLDER = "analysis_comments"

REPORT_FOLDER = os.path.join(
    INPUT_FOLDER,
    "reports"
)

os.makedirs(
    REPORT_FOLDER,
    exist_ok=True
)

# ==========================================================
# CATEGORY
# ==========================================================

def get_category():

    print("=" * 60)
    print("COMMENT DATASET VALIDATION")
    print("=" * 60)

    category = input(
        "\nEnter category "
        "(infrastructure, controversies, faculty_research, rankings): "
    ).strip().lower()

    return category


# ==========================================================
# LOAD DATASET
# ==========================================================

def load_dataset(category):

    input_file = os.path.join(

        INPUT_FOLDER,

        f"{category}_comment_feature_engineered.csv"

    )

    if not os.path.exists(input_file):

        print("\nDataset not found.")

        print(input_file)

        exit()

    print("\nLoading dataset...")

    df = pd.read_csv(

        input_file,

        low_memory=False

    )

    print(f"Loaded {len(df)} rows.")

    return df


# ==========================================================
# DATASET INFORMATION
# ==========================================================

def dataset_information(df):

    print("\n" + "=" * 60)
    print("DATASET INFORMATION")
    print("=" * 60)

    print(f"Rows           : {len(df)}")

    print(f"Columns        : {len(df.columns)}")

    memory = (

        df.memory_usage(deep=True)

        .sum()

        / (1024 ** 2)

    )

    print(f"Memory (MB)    : {memory:.2f}")

# ==========================================================
# MISSING VALUE VALIDATION
# ==========================================================

def missing_value_validation(df):

    print("\n" + "=" * 60)
    print("MISSING VALUE VALIDATION")
    print("=" * 60)

    ignore_columns = [

        "parent_comment_id"

    ]

    missing = (

        df.drop(

            columns=ignore_columns,

            errors="ignore"

        )

        .isnull()

        .sum()

    )

    missing = missing[missing > 0]

    if len(missing) == 0:

        print("✓ No Important Missing Values")

    else:

        print(missing)


# ==========================================================
# DUPLICATE VALIDATION
# ==========================================================

def duplicate_validation(df):

    print("\n" + "=" * 60)
    print("DUPLICATE VALIDATION")
    print("=" * 60)

    duplicate_rows = df.duplicated().sum()

    print(f"Duplicate Rows : {duplicate_rows}")

    if "comment_id" in df.columns:

        duplicate_ids = (

            df["comment_id"]

            .duplicated()

            .sum()

        )

        print(f"Duplicate Comment IDs : {duplicate_ids}")


# ==========================================================
# EMPTY COMMENT VALIDATION
# ==========================================================

def empty_comment_validation(df):

    print("\n" + "=" * 60)
    print("EMPTY COMMENT VALIDATION")
    print("=" * 60)

    clean_empty = (

        df["clean_comment"]

        .fillna("")

        .astype(str)

        .str.strip()

        .eq("")

        .sum()

    )

    combined_empty = (

        df["combined_comment"]

        .fillna("")

        .astype(str)

        .str.strip()

        .eq("")

        .sum()

    )

    print(f"Empty Clean Comments    : {clean_empty}")

    print(f"Empty Combined Comments : {combined_empty}")


# ==========================================================
# NUMERIC VALIDATION
# ==========================================================

def numeric_validation(df):

    print("\n" + "=" * 60)
    print("NUMERIC VALIDATION")
    print("=" * 60)

    numeric_columns = [

        "likes",

        "reply_count",

        "comment_length",

        "word_count"

    ]

    for column in numeric_columns:

        if column in df.columns:

            negatives = (

                df[column] < 0

            ).sum()

            print(

                f"Negative {column} : {negatives}"

            )


# ==========================================================
# FEATURE VALIDATION
# ==========================================================

def feature_validation(df):

    print("\n" + "=" * 60)
    print("FEATURE VALIDATION")
    print("=" * 60)

    required_features = [

        "comment_length",

        "word_count",

        "avg_word_length",

        "sentence_count",

        "hashtag_count",

        "mention_count",

        "emoji_count",

        "uppercase_word_count",

        "exclamation_count",

        "question_count",

        "comment_year",

        "comment_month",

        "comment_day",

        "comment_weekday",

        "comment_hour"

    ]

    missing_features = [

        feature

        for feature in required_features

        if feature not in df.columns

    ]

    if len(missing_features) == 0:

        print("✓ All Engineered Features Present")

    else:

        print("Missing Features")

        print(missing_features)


# ==========================================================
# DATE VALIDATION
# ==========================================================

def date_validation(df):

    print("\n" + "=" * 60)
    print("DATE VALIDATION")
    print("=" * 60)

    if "published_at" not in df.columns:

        print("published_at column not found.")

        return

    dates = pd.to_datetime(

        df["published_at"],

        errors="coerce"

    )

    invalid = dates.isnull().sum()

    print(f"Invalid Dates : {invalid}")

    if invalid != len(df):

        print(f"Earliest Date : {dates.min()}")

        print(f"Latest Date   : {dates.max()}")

# ==========================================================
# FINAL STATUS
# ==========================================================

def final_status(df):

    print("\n" + "=" * 60)
    print("FINAL STATUS")
    print("=" * 60)

    passed = True

    # Duplicate Rows
    if df.duplicated().sum() > 0:

        passed = False

    # Duplicate Comment IDs
    if "comment_id" in df.columns:

        if df["comment_id"].duplicated().sum() > 0:

            passed = False

    # Empty Clean Comments
    if "clean_comment" in df.columns:

        empty = (

            df["clean_comment"]

            .fillna("")

            .astype(str)

            .str.strip()

            .eq("")

            .sum()

        )

        if empty > 0:

            passed = False

    if passed:

        print("✅ Dataset Validation PASSED")

    else:

        print("⚠ Dataset Validation FAILED")

    return passed


# ==========================================================
# SAVE REPORT
# ==========================================================

def save_report(df, category, status):

    report = {

        "Rows": len(df),

        "Columns": len(df.columns),

        "Duplicate Rows": df.duplicated().sum(),

        "Duplicate Comment IDs": (
            df["comment_id"].duplicated().sum()
            if "comment_id" in df.columns
            else 0
        ),

        "Empty Clean Comments": (
            df["clean_comment"]
            .fillna("")
            .astype(str)
            .str.strip()
            .eq("")
            .sum()
            if "clean_comment" in df.columns
            else 0
        ),

        "Validation Status": (
            "PASS"
            if status
            else "FAIL"
        )

    }

    report_df = pd.DataFrame(

        report.items(),

        columns=[

            "Metric",

            "Value"

        ]

    )

    report_file = os.path.join(

        REPORT_FOLDER,

        f"{category}_validation_report.csv"

    )

    report_df.to_csv(

        report_file,

        index=False

    )

    print("\nValidation Report Saved")

    print(report_file)


# ==========================================================
# MAIN
# ==========================================================

def main():

    category = get_category()

    df = load_dataset(category)

    dataset_information(df)

    missing_value_validation(df)

    duplicate_validation(df)

    empty_comment_validation(df)

    numeric_validation(df)

    feature_validation(df)

    date_validation(df)

    status = final_status(df)

    save_report(

        df,

        category,

        status

    )

    print("\n" + "=" * 60)
    print("COMMENT DATASET VALIDATION COMPLETED")
    print("=" * 60)


# ==========================================================
# DRIVER
# ==========================================================

if __name__ == "__main__":

    main()

#.  python preprocessing_comment/05_comment_validation.py