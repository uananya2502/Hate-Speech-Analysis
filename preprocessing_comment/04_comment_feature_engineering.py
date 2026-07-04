"""
============================================================
04_comment_feature_engineering.py

Comment Feature Engineering

Input:
    analysis_comments/<category>_comment_text_cleaned.csv

Output:
    analysis_comments/<category>_comment_feature_engineered.csv

Report:
    analysis_comments/reports/<category>_feature_engineering_report.csv

Author : Ananya Upadhyay
============================================================
"""

# ==========================================================
# IMPORT LIBRARIES
# ==========================================================

import os
import re
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
    print("COMMENT FEATURE ENGINEERING")
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

        f"{category}_comment_text_cleaned.csv"

    )

    if not os.path.exists(input_file):

        print("\nInput dataset not found.")

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
# TEXT FEATURE FUNCTIONS
# ==========================================================

def word_count(text):

    if pd.isna(text):

        return 0

    return len(str(text).split())


def comment_length(text):

    if pd.isna(text):

        return 0

    return len(str(text))


def average_word_length(text):

    if pd.isna(text):

        return 0

    words = str(text).split()

    if len(words) == 0:

        return 0

    return round(

        sum(len(word) for word in words) /

        len(words),

        2

    )


def sentence_count(text):

    if pd.isna(text):

        return 0

    sentences = re.split(

        r"[.!?]+",

        str(text)

    )

    sentences = [

        sentence

        for sentence in sentences

        if sentence.strip()

    ]

    return len(sentences)


# ==========================================================
# WRITING STYLE FEATURES
# ==========================================================

def hashtag_count(text):

    return len(

        re.findall(

            r"#\w+",

            str(text)

        )

    )


def mention_count(text):

    return len(

        re.findall(

            r"@\w+",

            str(text)

        )

    )


def exclamation_count(text):

    return str(text).count("!")


def question_count(text):

    return str(text).count("?")


# ==========================================================
# FEATURE ENGINEERING
# ==========================================================

def feature_engineering(df):

    print("\nCreating engineered features...")

    report = {}

    # ------------------------------------------------------
    # Basic Text Features
    # ------------------------------------------------------

    df["comment_length"] = (

        df["clean_comment"]

        .apply(comment_length)

    )

    df["word_count"] = (

        df["clean_comment"]

        .apply(word_count)

    )

    df["avg_word_length"] = (

        df["clean_comment"]

        .apply(average_word_length)

    )

    df["sentence_count"] = (

        df["clean_comment"]

        .apply(sentence_count)

    )

    # ------------------------------------------------------
    # Writing Style Features
    # ------------------------------------------------------

    df["hashtag_count"] = (

        df["clean_comment"]

        .apply(hashtag_count)

    )

    df["mention_count"] = (

        df["clean_comment"]

        .apply(mention_count)

    )

    df["exclamation_count"] = (

        df["clean_comment"]

        .apply(exclamation_count)

    )

    df["question_count"] = (

        df["clean_comment"]

        .apply(question_count)

    )

    # ------------------------------------------------------
    # Emoji Count
    # ------------------------------------------------------

    emoji_pattern = re.compile(

        "["
        "\U0001F600-\U0001F64F"
        "\U0001F300-\U0001F5FF"
        "\U0001F680-\U0001F6FF"
        "\U0001F1E0-\U0001F1FF"
        "\U00002700-\U000027BF"
        "\U000024C2-\U0001F251"
        "]+",

        flags=re.UNICODE

    )

    df["emoji_count"] = (

        df["clean_comment"]

        .apply(

            lambda x: len(

                emoji_pattern.findall(

                    str(x)

                )

            )

        )

    )

    # ------------------------------------------------------
    # Uppercase Word Count
    # ------------------------------------------------------

    df["uppercase_word_count"] = (

        df["clean_comment"]

        .apply(

            lambda x: sum(

                word.isupper()

                for word in str(x).split()

            )

        )

    )

    # ------------------------------------------------------
    # Time Features
    # ------------------------------------------------------

    df["published_at"] = pd.to_datetime(

        df["published_at"],

        errors="coerce"

    )

    df["comment_year"] = (

        df["published_at"]

        .dt.year

    )

    df["comment_month"] = (

        df["published_at"]

        .dt.month

    )

    df["comment_day"] = (

        df["published_at"]

        .dt.day

    )

    df["comment_weekday"] = (

        df["published_at"]

        .dt.day_name()

    )

    df["comment_hour"] = (

        df["published_at"]

        .dt.hour

    )

    # ------------------------------------------------------
    # Report
    # ------------------------------------------------------

    report["Rows"] = len(df)

    report["Columns"] = len(df.columns)

    report["New Features Created"] = 13

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

        f"{category}_feature_engineering_report.csv"

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

        f"{category}_comment_feature_engineered.csv"

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

    missing_columns = [

        col

        for col in required_columns

        if col not in df.columns

    ]

    if len(missing_columns) == 0:

        print("✓ All engineered features created.")

    else:

        print("\nMissing Columns")

        print(missing_columns)

    print(f"\nRows    : {len(df)}")

    print(f"Columns : {len(df.columns)}")


# ==========================================================
# MAIN
# ==========================================================

def main():

    category = get_category()

    df = load_dataset(category)

    df, report = feature_engineering(df)

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
    print("COMMENT FEATURE ENGINEERING COMPLETED")
    print("=" * 60)


# ==========================================================
# DRIVER
# ==========================================================

if __name__ == "__main__":

    main()

#   python preprocessing_comment/04_comment_feature_engineering.py