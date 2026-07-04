"""
============================================================
02_comment_missing_values.py

Handle Missing Values for YouTube Comments Dataset

Input:
    analysis_comments/<category>_quality_checked.csv

Output:
    analysis_comments/<category>_comment_clean.csv

Report:
    analysis_comments/reports/<category>_missing_value_report.csv

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

OUTPUT_FOLDER = "analysis_comments"

REPORT_FOLDER = os.path.join(
    OUTPUT_FOLDER,
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
    print("COMMENT MISSING VALUE HANDLING")
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

        f"{category}_quality_checked.csv"

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
# MISSING VALUE SUMMARY
# ==========================================================

def missing_summary(df, title):

    print("\n" + "=" * 60)

    print(title)

    print("=" * 60)

    missing = df.isnull().sum()

    missing = missing[missing > 0]

    if len(missing) == 0:

        print("✓ No Missing Values")

    else:

        print(missing)

# ==========================================================
# HANDLE MISSING VALUES
# ==========================================================

def handle_missing_values(df):

    print("\n" + "=" * 60)
    print("HANDLING MISSING VALUES")
    print("=" * 60)

    report = {}

    # ------------------------------------------------------
    # Remove Duplicate Rows
    # ------------------------------------------------------

    duplicate_rows = df.duplicated().sum()

    df.drop_duplicates(
        inplace=True
    )

    report["Duplicate Rows Removed"] = duplicate_rows

    print(f"✓ Removed {duplicate_rows} duplicate rows.")

    # ------------------------------------------------------
    # Remove Duplicate Comment IDs
    # ------------------------------------------------------

    duplicate_comment_ids = 0

    if "comment_id" in df.columns:

        duplicate_comment_ids = (

            df["comment_id"]

            .duplicated()

            .sum()

        )

        df.drop_duplicates(

            subset="comment_id",

            keep="first",

            inplace=True

        )

    report["Duplicate Comment IDs Removed"] = duplicate_comment_ids

    print(
        f"✓ Removed {duplicate_comment_ids} duplicate comment IDs."
    )

    # ------------------------------------------------------
    # Remove Empty Comments
    # ------------------------------------------------------

    empty_comments = 0

    if "comment_text" in df.columns:

        empty_comments = (

            df["comment_text"]

            .fillna("")

            .astype(str)

            .str.strip()

            .eq("")

            .sum()

        )

        df = df[

            df["comment_text"]

            .fillna("")

            .astype(str)

            .str.strip()

            != ""

        ]

    report["Empty Comments Removed"] = empty_comments

    print(f"✓ Removed {empty_comments} empty comments.")

    # ------------------------------------------------------
    # Remove Moderation Status
    # ------------------------------------------------------

    if "moderation_status" in df.columns:

        df.drop(

            columns=["moderation_status"],

            inplace=True

        )

        print("✓ Removed moderation_status column.")

    # ------------------------------------------------------
    # Fill Author Name
    # ------------------------------------------------------

    if "author_name" in df.columns:

        missing = df["author_name"].isnull().sum()

        df["author_name"] = df["author_name"].fillna(
            "Unknown Author"
        )

        report["author_name Filled"] = missing

        print(
            f"✓ Filled {missing} missing author names."
        )

    # ------------------------------------------------------
    # Fill Author Channel ID
    # ------------------------------------------------------

    if "author_channel_id" in df.columns:

        missing = df["author_channel_id"].isnull().sum()

        df["author_channel_id"] = df["author_channel_id"].fillna(
            "Unknown"
        )

        report["author_channel_id Filled"] = missing

        print(
            f"✓ Filled {missing} missing author channel IDs."
        )

    # ------------------------------------------------------
    # Fill Author Profile URL
    # ------------------------------------------------------

    if "author_profile_url" in df.columns:

        missing = df["author_profile_url"].isnull().sum()

        df["author_profile_url"] = df["author_profile_url"].fillna(
            "Not Available"
        )

        report["author_profile_url Filled"] = missing

        print(
            f"✓ Filled {missing} missing profile URLs."
        )

    # ------------------------------------------------------
    # Fill Author Thumbnail
    # ------------------------------------------------------

    if "author_thumbnail" in df.columns:

        missing = df["author_thumbnail"].isnull().sum()

        df["author_thumbnail"] = df["author_thumbnail"].fillna(
            "Not Available"
        )

        report["author_thumbnail Filled"] = missing

        print(
            f"✓ Filled {missing} missing thumbnails."
        )

    # ------------------------------------------------------
    # Fill Likes
    # ------------------------------------------------------

    if "likes" in df.columns:

        missing = df["likes"].isnull().sum()

        df["likes"] = df["likes"].fillna(0)

        report["likes Filled"] = missing

        print(
            f"✓ Filled {missing} missing likes."
        )

    # ------------------------------------------------------
    # Fill Reply Count
    # ------------------------------------------------------

    if "reply_count" in df.columns:

        missing = df["reply_count"].isnull().sum()

        df["reply_count"] = df["reply_count"].fillna(0)

        report["reply_count Filled"] = missing

        print(
            f"✓ Filled {missing} missing reply counts."
        )

    # ------------------------------------------------------
    # Fill Boolean
    # ------------------------------------------------------

    if "is_reply" in df.columns:

        missing = df["is_reply"].isnull().sum()

        df["is_reply"] = df["is_reply"].fillna(False)

        report["is_reply Filled"] = missing

        print(
            f"✓ Filled {missing} missing boolean values."
        )

    return df, report

# ==========================================================
# SAVE REPORT
# ==========================================================

def save_report(report, category):

    report_df = pd.DataFrame({

        "Metric": list(report.keys()),

        "Value": list(report.values())

    })

    report_file = os.path.join(

        REPORT_FOLDER,

        f"{category}_missing_value_report.csv"

    )

    report_df.to_csv(

        report_file,

        index=False

    )

    print("\nReport Saved")

    print(report_file)


# ==========================================================
# SAVE DATASET
# ==========================================================

def save_dataset(df, category):

    output_file = os.path.join(

        OUTPUT_FOLDER,

        f"{category}_comment_clean.csv"

    )

    df.to_csv(

        output_file,

        index=False,

        encoding="utf-8"

    )

    print("\nDataset Saved")

    print(output_file)


# ==========================================================
# FINAL VALIDATION
# ==========================================================

def final_validation(df):

    print("\n" + "=" * 60)
    print("FINAL VALIDATION")
    print("=" * 60)

    ignore_columns = [

        "parent_comment_id"

    ]

    validation_df = df.drop(

        columns=ignore_columns,

        errors="ignore"

    )

    missing = validation_df.isnull().sum()

    missing = missing[missing > 0]

    if len(missing) == 0:

        print("✓ No Important Missing Values Remaining")

    else:

        print(missing)

    print(f"\nFinal Rows    : {len(df)}")

    print(f"Final Columns : {len(df.columns)}")


# ==========================================================
# MAIN
# ==========================================================

def main():

    category = get_category()

    df = load_dataset(category)

    missing_summary(

        df,

        "MISSING VALUES BEFORE CLEANING"

    )

    df, report = handle_missing_values(df)

    missing_summary(

        df,

        "MISSING VALUES AFTER CLEANING"

    )

    final_validation(df)

    save_dataset(

        df,

        category

    )

    save_report(

        report,

        category

    )

    print("\n" + "=" * 60)
    print("COMMENT MISSING VALUE HANDLING COMPLETED")
    print("=" * 60)


# ==========================================================
# DRIVER
# ==========================================================

if __name__ == "__main__":

    main()