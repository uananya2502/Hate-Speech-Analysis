"""
============================================================
03_comment_text_cleaning.py

Comment Text Cleaning

Input:
    analysis_comments/<category>_comment_clean.csv

Output:
    analysis_comments/<category>_comment_text_cleaned.csv

Report:
    analysis_comments/reports/<category>_text_cleaning_report.csv

Author : Ananya Upadhyay
============================================================
"""

# ==========================================================
# IMPORT LIBRARIES
# ==========================================================

import os
import re
import html
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
    print("COMMENT TEXT CLEANING")
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

        f"{category}_comment_clean.csv"

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
# REMOVE HTML TAGS
# ==========================================================

def remove_html_tags(text):

    if pd.isna(text):

        return ""

    return re.sub(

        r"<.*?>",

        " ",

        str(text)

    )


# ==========================================================
# REMOVE URLS
# ==========================================================

def remove_urls(text):

    return re.sub(
        r"http\S+|https\S+|www\.\S+",
        "",
        str(text)
)


# ==========================================================
# REMOVE EMAILS
# ==========================================================

def remove_emails(text):

    return re.sub(
        r"\S+@\S+",
        "",
        str(text)
)


# ==========================================================
# REMOVE NEWLINES
# ==========================================================

def remove_newlines(text):

    text = str(text)

    text = text.replace("\n", " ")

    text = text.replace("\r", " ")

    text = text.replace("\t", " ")

    return text


# ==========================================================
# DECODE HTML ENTITIES
# ==========================================================

def decode_html(text):

    return html.unescape(

        str(text)

    )


# ==========================================================
# REDUCE REPEATED PUNCTUATION
# ==========================================================

def reduce_punctuation(text):

    text = re.sub(
        r"!{2,}",
        "!",
        text
    )

    text = re.sub(
        r"\?{2,}",
        "?",
        text
    )

    text = re.sub(
        r"\.{2,}",
        ".",
        text
    )

    return text

# ==========================================================
# NORMALIZE SPACES
# ==========================================================

def normalize_spaces(text):

    return re.sub(

        r"\\s+",

        " ",

        str(text)

    ).strip()

# ==========================================================
# COMPLETE TEXT CLEANING PIPELINE
# ==========================================================

def clean_text(text):

    if pd.isna(text):

        return ""

    text = str(text)

    text = decode_html(text)

    text = remove_html_tags(text)

    text = remove_urls(text)

    text = remove_emails(text)

    text = remove_newlines(text)

    text = reduce_punctuation(text)

    text = normalize_spaces(text)

    return text


# ==========================================================
# CLEAN COMMENT DATASET
# ==========================================================

def clean_comments(df):

    print("\nCleaning comments...")

    report = {}

    # ------------------------------------------------------
    # Average Length Before Cleaning
    # ------------------------------------------------------

    before_length = (

        df["comment_text"]

        .fillna("")

        .astype(str)

        .str.len()

        .mean()

    )

    report["Average Length Before"] = round(
        before_length,
        2
    )

    # ------------------------------------------------------
    # Count URLs
    # ------------------------------------------------------

    url_count = (

        df["comment_text"]

        .fillna("")

        .astype(str)

        .str.count(
            r"http\S+|https\S+|www\.\S+"
        )

        .sum()

    )

    report["URLs Removed"] = int(url_count)

    # ------------------------------------------------------
    # Count Emails
    # ------------------------------------------------------

    email_count = (

        df["comment_text"]

        .fillna("")

        .astype(str)

        .str.count(
            r"\S+@\S+"
        )

        .sum()

    )

    report["Emails Removed"] = int(email_count)

    # ------------------------------------------------------
    # Create Clean Comment
    # ------------------------------------------------------

    print("Creating clean_comment...")

    df["clean_comment"] = (

        df["comment_text"]

        .fillna("")

        .apply(clean_text)

    )

    # ------------------------------------------------------
    # Remove Comments That Became Empty
    # ------------------------------------------------------

    before = len(df)

    df = df[
        df["clean_comment"]
        .str.strip()
        .ne("")
    ].copy()

    removed = before - len(df)

    report["Empty Comments Removed After Cleaning"] = removed

    print(
        f"✓ Removed {removed} comments that became empty after cleaning."
    )

    # ------------------------------------------------------
    # Create Combined Comment
    # ------------------------------------------------------

    print("Creating combined_comment...")

    df["combined_comment"] = (

        df["clean_comment"]

        .apply(normalize_spaces)

    )

    # ------------------------------------------------------
    # Average Length After Cleaning
    # ------------------------------------------------------

    after_length = (

        df["combined_comment"]

        .astype(str)

        .str.len()

        .mean()

    )

    report["Average Length After"] = round(
        after_length,
        2
    )

    # ------------------------------------------------------
    # Final Report
    # ------------------------------------------------------

    report["Comments Before Cleaning"] = before

    report["Comments After Cleaning"] = len(df)

    report["Comments Cleaned"] = len(df)

    report["Dataset Status"] = "PASS"

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

        f"{category}_text_cleaning_report.csv"

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

        f"{category}_comment_text_cleaned.csv"

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

    required_columns = [

        "clean_comment",

        "combined_comment"

    ]

    missing_columns = [

        col for col in required_columns

        if col not in df.columns

    ]

    if len(missing_columns) == 0:

        print("✓ All required columns created.")

    else:

        print("Missing Columns")

        print(missing_columns)

    empty_clean = (

        df["clean_comment"]

        .fillna("")

        .astype(str)

        .str.strip()

        .eq("")

        .sum()

    )

    print(f"\nEmpty Clean Comments : {empty_clean}")

    print(f"Rows                 : {len(df)}")

    print(f"Columns              : {len(df.columns)}")


# ==========================================================
# MAIN
# ==========================================================

def main():

    category = get_category()

    df = load_dataset(category)

    df, report = clean_comments(df)

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
    print("COMMENT TEXT CLEANING COMPLETED")
    print("=" * 60)


# ==========================================================
# DRIVER
# ==========================================================

if __name__ == "__main__":

    main()

# python preprocessing_comment/03_comment_text_cleaning.py