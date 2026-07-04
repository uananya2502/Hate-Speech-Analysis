"""
============================================================
01_comment_data_quality.py

Comment Dataset Quality Assessment

Input:
    comments/<category>_comments.csv

Output:
    analysis_comments/<category>_quality_checked.csv

Author : Ananya Upadhyay
============================================================
"""

# ==========================================================
# IMPORT LIBRARIES
# ==========================================================

import os
import pandas as pd

# ==========================================================
# INPUT / OUTPUT FOLDERS
# ==========================================================

INPUT_FOLDER = "comments"

OUTPUT_FOLDER = "analysis_comments"

os.makedirs(
    OUTPUT_FOLDER,
    exist_ok=True
)

# ==========================================================
# CATEGORY SELECTION
# ==========================================================

def get_category():

    print("=" * 60)
    print("COMMENT DATA QUALITY")
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
        f"{category}_comments.csv"
    )

    if not os.path.exists(input_file):

        print("\nInput file not found.")

        print(input_file)

        exit()

    print("\nLoading dataset...")

    df = pd.read_csv(
        input_file,
        low_memory=False
    )

    print(f"Loaded {len(df)} comments.")

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

    memory = round(

        df.memory_usage(
            deep=True
        ).sum() / (1024 * 1024),

        2

    )

    print(f"Memory (MB)    : {memory}")


# ==========================================================
# MISSING VALUE ANALYSIS
# ==========================================================

def missing_value_analysis(df):

    print("\n" + "=" * 60)
    print("MISSING VALUE ANALYSIS")
    print("=" * 60)

    IGNORE_COLUMNS = [
    "parent_comment_id",
    "moderation_status"
]

    missing = (
        df.drop(
            columns=IGNORE_COLUMNS,
            errors="ignore"
        )
        .isnull()
        .sum()
    )

    missing = missing[missing > 0]

    if len(missing) == 0:

        print("✓ No Missing Values")

    else:

        print(missing)
    
    print("\nIgnored Columns")
    print("----------------")
    print("parent_comment_id (expected for top-level comments)")
    print("moderation_status (not returned by YouTube API)")


# ==========================================================
# DUPLICATE ANALYSIS
# ==========================================================

def duplicate_analysis(df):

    print("\n" + "=" * 60)
    print("DUPLICATE ANALYSIS")
    print("=" * 60)

    duplicate_rows = df.duplicated().sum()

    print(f"Duplicate Rows : {duplicate_rows}")

    if "comment_id" in df.columns:

        duplicate_comment_ids = (

            df["comment_id"]

            .duplicated()

            .sum()

        )

        print(

            f"Duplicate Comment IDs : "

            f"{duplicate_comment_ids}"

        )

# ==========================================================
# EMPTY COMMENT ANALYSIS
# ==========================================================

def empty_comment_analysis(df):

    print("\n" + "=" * 60)
    print("EMPTY COMMENT ANALYSIS")
    print("=" * 60)

    if "comment_text" not in df.columns:

        print("comment_text column not found.")

        return

    empty_comments = (

        df["comment_text"]

        .fillna("")

        .astype(str)

        .str.strip()

        .eq("")

        .sum()

    )

    print(f"Empty Comments : {empty_comments}")


# ==========================================================
# NUMERIC VALIDATION
# ==========================================================

def numeric_validation(df):

    print("\n" + "=" * 60)
    print("NUMERIC VALIDATION")
    print("=" * 60)

    if "likes" in df.columns:

        negative_likes = (

            df["likes"] < 0

        ).sum()

        print(f"Negative Likes       : {negative_likes}")

    if "reply_count" in df.columns:

        negative_replies = (

            df["reply_count"] < 0

        ).sum()

        print(f"Negative Reply Count : {negative_replies}")


# ==========================================================
# REPLY ANALYSIS
# ==========================================================

def reply_analysis(df):

    print("\n" + "=" * 60)
    print("REPLY ANALYSIS")
    print("=" * 60)

    if "is_reply" not in df.columns:

        print("is_reply column not found.")

        return

    total_replies = df["is_reply"].sum()

    total_comments = len(df) - total_replies

    percentage = round(

        (total_replies / len(df)) * 100,

        2

    )

    print(f"Top Level Comments : {total_comments}")

    print(f"Replies            : {total_replies}")

    print(f"Reply Percentage   : {percentage}%")


# ==========================================================
# AUTHOR ANALYSIS
# ==========================================================

def author_analysis(df):

    print("\n" + "=" * 60)
    print("AUTHOR ANALYSIS")
    print("=" * 60)

    if "author_name" in df.columns:

        print(

            "Unique Authors :",

            df["author_name"].nunique()

        )

    if "author_channel_id" in df.columns:

        print(

            "Unique Author Channels :",

            df["author_channel_id"].nunique()

        )


# ==========================================================
# UNIVERSITY ANALYSIS
# ==========================================================

def university_analysis(df):

    print("\n" + "=" * 60)
    print("UNIVERSITY ANALYSIS")
    print("=" * 60)

    if "university_name" not in df.columns:

        return

    print(

        "Unique Universities :",

        df["university_name"].nunique()

    )

    print("\nTop 10 Universities\n")

    print(

        df["university_name"]

        .value_counts()

        .head(10)

    )


# ==========================================================
# DATE ANALYSIS
# ==========================================================

def date_analysis(df):

    print("\n" + "=" * 60)
    print("DATE ANALYSIS")
    print("=" * 60)

    if "published_at" not in df.columns:

        print("published_at column not found.")

        return

    dates = pd.to_datetime(

        df["published_at"],

        errors="coerce"

    )

    invalid_dates = dates.isna().sum()

    print(f"Invalid Dates : {invalid_dates}")

    print(f"Earliest Date : {dates.min()}")

    print(f"Latest Date   : {dates.max()}")


# ==========================================================
# VIDEO COVERAGE
# ==========================================================

def video_coverage(df):

    print("\n" + "=" * 60)
    print("VIDEO COVERAGE")
    print("=" * 60)

    if "video_id" not in df.columns:

        return

    unique_videos = df["video_id"].nunique()

    average = round(

        len(df) / unique_videos,

        2

    )

    print(f"Unique Videos            : {unique_videos}")

    print(f"Average Comments/Video   : {average}")
# ==========================================================
# QUALITY REPORT
# ==========================================================

def quality_report(df):

    print("\n" + "=" * 60)
    print("QUALITY REPORT")
    print("=" * 60)

    duplicate_rows = df.duplicated().sum()

    duplicate_comment_ids = 0

    if "comment_id" in df.columns:

        duplicate_comment_ids = df["comment_id"].duplicated().sum()

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

    negative_likes = 0

    if "likes" in df.columns:

        negative_likes = (df["likes"] < 0).sum()

    negative_replies = 0

    if "reply_count" in df.columns:

        negative_replies = (df["reply_count"] < 0).sum()

    issues = (
        duplicate_rows
        + duplicate_comment_ids
        + empty_comments
        + negative_likes
        + negative_replies
    )

    print(f"Total Quality Issues : {issues}")

    if issues == 0:

        print("\n✅ Dataset Passed Quality Check")

    else:

        print("\n⚠ Dataset Needs Cleaning")

# ==========================================================
# SAVE DATASET
# ==========================================================

def save_dataset(df, category):

    output_file = os.path.join(

        OUTPUT_FOLDER,

        f"{category}_quality_checked.csv"

    )

    df.to_csv(

        output_file,

        index=False,

        encoding="utf-8"

    )

    print("\nSaved File")

    print(output_file)


# ==========================================================
# MAIN
# ==========================================================

def main():

    category = get_category()

    df = load_dataset(category)

    dataset_information(df)

    missing_value_analysis(df)

    duplicate_analysis(df)

    empty_comment_analysis(df)

    numeric_validation(df)

    reply_analysis(df)

    author_analysis(df)

    university_analysis(df)

    date_analysis(df)

    video_coverage(df)

    quality_report(df)

    save_dataset(

        df,

        category

    )

    print("\n" + "=" * 60)

    print("COMMENT DATA QUALITY COMPLETED")

    print("=" * 60)


# ==========================================================
# DRIVER
# ==========================================================

if __name__ == "__main__":

    main()

#run
# python preprocessing_comment/01_comment_data_quality.py