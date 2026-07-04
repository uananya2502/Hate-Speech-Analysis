"""
============================================================
01_comment_eda.py

Exploratory Data Analysis for YouTube Comments

Input:
    cleaned_data_comments/<category>_comment_feature_engineered.csv

Outputs:
    analysis_comments/reports/
    analysis_comments/tables/
    analysis_comments/figures/

Author : Ananya Upadhyay
============================================================
"""

# ==========================================================
# IMPORT LIBRARIES
# ==========================================================

import os

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt


# ==========================================================
# FOLDERS
# ==========================================================

INPUT_FOLDER = "cleaned_data_comments"

REPORT_FOLDER = os.path.join(
    "analysis_comments",
    "reports"
)

TABLE_FOLDER = os.path.join(
    "analysis_comments",
    "tables"
)

FIGURE_FOLDER = os.path.join(
    "analysis_comments",
    "figures"
)

os.makedirs(
    REPORT_FOLDER,
    exist_ok=True
)

os.makedirs(
    TABLE_FOLDER,
    exist_ok=True
)

os.makedirs(
    FIGURE_FOLDER,
    exist_ok=True
)


# ==========================================================
# CATEGORY
# ==========================================================

def get_category():

    print("=" * 60)
    print("COMMENT EXPLORATORY DATA ANALYSIS")
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

    print(f"Loaded {len(df)} comments.")

    return df


# ==========================================================
# DATASET OVERVIEW
# ==========================================================

def dataset_overview(df):

    print("\n" + "=" * 60)
    print("DATASET OVERVIEW")
    print("=" * 60)

    print(f"Rows        : {len(df)}")

    print(f"Columns     : {len(df.columns)}")

    memory = (

        df.memory_usage(

            deep=True

        ).sum()

        / (1024 ** 2)

    )

    print(f"Memory (MB) : {memory:.2f}")

    print("\nColumn Data Types\n")

    print(df.dtypes)

    overview = pd.DataFrame({

        "Metric": [

            "Rows",

            "Columns",

            "Memory (MB)"

        ],

        "Value": [

            len(df),

            len(df.columns),

            round(memory, 2)

        ]

    })

    overview.to_csv(

        os.path.join(

            TABLE_FOLDER,

            "dataset_overview.csv"

        ),

        index=False

    )


# ==========================================================
# MISSING VALUE ANALYSIS
# ==========================================================

def missing_value_analysis(df):

    print("\n" + "=" * 60)
    print("MISSING VALUE ANALYSIS")
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

        print("✓ No Missing Values")

    else:

        print(missing)

    missing.to_csv(

        os.path.join(

            TABLE_FOLDER,

            "missing_values.csv"

        )

    )

# ==========================================================
# UNIVERSITY ANALYSIS
# ==========================================================

def university_analysis(df):

    print("\n" + "=" * 60)
    print("UNIVERSITY ANALYSIS")
    print("=" * 60)

    university = (

        df["university_name"]

        .value_counts()

    )

    print(university.head(10))

    university.to_csv(

        os.path.join(

            TABLE_FOLDER,

            "comments_per_university.csv"

        )

    )


# ==========================================================
# VIDEO ANALYSIS
# ==========================================================

def video_analysis(df):

    print("\n" + "=" * 60)
    print("VIDEO ANALYSIS")
    print("=" * 60)

    video = (

        df.groupby("video_title")

        .size()

        .sort_values(

            ascending=False

        )

    )

    print(video.head(10))

    video.to_csv(

        os.path.join(

            TABLE_FOLDER,

            "comments_per_video.csv"

        )

    )


# ==========================================================
# LIKES ANALYSIS
# ==========================================================

def likes_analysis(df):

    print("\n" + "=" * 60)
    print("LIKES ANALYSIS")
    print("=" * 60)

    print(

        df["likes"]

        .describe()

    )

    likes_summary = (

        df["likes"]

        .describe()

    )

    likes_summary.to_csv(

        os.path.join(

            TABLE_FOLDER,

            "likes_summary.csv"

        )

    )


# ==========================================================
# REPLY ANALYSIS
# ==========================================================

def reply_analysis(df):

    print("\n" + "=" * 60)
    print("REPLY ANALYSIS")
    print("=" * 60)

    print(

        df["reply_count"]

        .describe()

    )

    reply_summary = (

        df["reply_count"]

        .describe()

    )

    reply_summary.to_csv(

        os.path.join(

            TABLE_FOLDER,

            "reply_summary.csv"

        )

    )


# ==========================================================
# COMMENT LENGTH ANALYSIS
# ==========================================================

def comment_length_analysis(df):

    print("\n" + "=" * 60)
    print("COMMENT LENGTH ANALYSIS")
    print("=" * 60)

    print(

        df["comment_length"]

        .describe()

    )

    length_summary = (

        df["comment_length"]

        .describe()

    )

    length_summary.to_csv(

        os.path.join(

            TABLE_FOLDER,

            "comment_length_summary.csv"

        )

    )


# ==========================================================
# WORD COUNT ANALYSIS
# ==========================================================

def word_count_analysis(df):

    print("\n" + "=" * 60)
    print("WORD COUNT ANALYSIS")
    print("=" * 60)

    print(

        df["word_count"]

        .describe()

    )

    word_summary = (

        df["word_count"]

        .describe()

    )

    word_summary.to_csv(

        os.path.join(

            TABLE_FOLDER,

            "word_count_summary.csv"

        )

    )


# ==========================================================
# TIME ANALYSIS
# ==========================================================

def time_analysis(df):

    print("\n" + "=" * 60)
    print("TIME ANALYSIS")
    print("=" * 60)

    year = (

        df["comment_year"]

        .value_counts()

        .sort_index()

    )

    month = (

        df["comment_month"]

        .value_counts()

        .sort_index()

    )

    weekday = (

        df["comment_weekday"]

        .value_counts()

    )

    hour = (

        df["comment_hour"]

        .value_counts()

        .sort_index()

    )

    print("\nComments by Year")

    print(year)

    year.to_csv(

        os.path.join(

            TABLE_FOLDER,

            "comments_by_year.csv"

        )

    )

    month.to_csv(

        os.path.join(

            TABLE_FOLDER,

            "comments_by_month.csv"

        )

    )

    weekday.to_csv(

        os.path.join(

            TABLE_FOLDER,

            "comments_by_weekday.csv"

        )

    )

    hour.to_csv(

        os.path.join(

            TABLE_FOLDER,

            "comments_by_hour.csv"

        )

    )

# ==========================================================
# COMMENTS PER UNIVERSITY
# ==========================================================

def plot_comments_per_university(df):

    university = (

        df["university_name"]

        .value_counts()

        .head(10)

    )

    plt.figure(figsize=(10,6))

    university.plot(kind="bar")

    plt.title("Top 10 Universities by Comments")

    plt.xlabel("University")

    plt.ylabel("Number of Comments")

    plt.xticks(rotation=45, ha="right")

    plt.tight_layout()

    plt.savefig(

        os.path.join(

            FIGURE_FOLDER,

            "comments_per_university.png"

        )

    )

    plt.close()


# ==========================================================
# COMMENT LENGTH HISTOGRAM
# ==========================================================

def plot_comment_length(df):

    plt.figure(figsize=(8,6))

    plt.hist(

        df["comment_length"],

        bins=30

    )

    plt.title("Comment Length Distribution")

    plt.xlabel("Characters")

    plt.ylabel("Frequency")

    plt.tight_layout()

    plt.savefig(

        os.path.join(

            FIGURE_FOLDER,

            "comment_length_histogram.png"

        )

    )

    plt.close()


# ==========================================================
# WORD COUNT HISTOGRAM
# ==========================================================

def plot_word_count(df):

    plt.figure(figsize=(8,6))

    plt.hist(

        df["word_count"],

        bins=30

    )

    plt.title("Word Count Distribution")

    plt.xlabel("Words")

    plt.ylabel("Frequency")

    plt.tight_layout()

    plt.savefig(

        os.path.join(

            FIGURE_FOLDER,

            "word_count_histogram.png"

        )

    )

    plt.close()


# ==========================================================
# COMMENTS BY HOUR
# ==========================================================

def plot_comments_by_hour(df):

    hour = (

        df["comment_hour"]

        .value_counts()

        .sort_index()

    )

    plt.figure(figsize=(10,5))

    plt.plot(

        hour.index,

        hour.values,

        marker="o"

    )

    plt.title("Comments by Hour")

    plt.xlabel("Hour")

    plt.ylabel("Comments")

    plt.tight_layout()

    plt.savefig(

        os.path.join(

            FIGURE_FOLDER,

            "comments_by_hour.png"

        )

    )

    plt.close()


# ==========================================================
# CORRELATION MATRIX
# ==========================================================

def correlation_analysis(df):

    numeric = [

        "likes",

        "reply_count",

        "comment_length",

        "word_count",

        "emoji_count",

        "hashtag_count",

        "mention_count"

    ]

    corr = (

        df[numeric]

        .corr(

            numeric_only=True

        )

    )

    corr.to_csv(

        os.path.join(

            TABLE_FOLDER,

            "correlation_matrix.csv"

        )

    )

    plt.figure(figsize=(8,6))

    plt.imshow(

        corr,

        aspect="auto"

    )

    plt.colorbar()

    plt.xticks(

        range(len(corr.columns)),

        corr.columns,

        rotation=90

    )

    plt.yticks(

        range(len(corr.columns)),

        corr.columns

    )

    plt.tight_layout()

    plt.savefig(

        os.path.join(

            FIGURE_FOLDER,

            "correlation_heatmap.png"

        )

    )

    plt.close()


# ==========================================================
# SAVE REPORT
# ==========================================================

def save_report(df, category):

    report_file = os.path.join(

        REPORT_FOLDER,

        f"{category}_comment_eda_report.txt"

    )

    with open(

        report_file,

        "w",

        encoding="utf-8"

    ) as file:

        file.write("COMMENT EDA REPORT\n")

        file.write("=" * 40)

        file.write("\n\n")

        file.write(f"Rows : {len(df)}\n")

        file.write(f"Columns : {len(df.columns)}\n")

        file.write(f"Universities : {df['university_name'].nunique()}\n")

        file.write(f"Videos : {df['video_id'].nunique()}\n")

        file.write(f"Comments : {len(df)}\n")

    print("\nEDA Report Saved")

    print(report_file)


# ==========================================================
# MAIN
# ==========================================================

def main():

    category = get_category()

    df = load_dataset(category)

    dataset_overview(df)

    missing_value_analysis(df)

    university_analysis(df)

    video_analysis(df)

    likes_analysis(df)

    reply_analysis(df)

    comment_length_analysis(df)

    word_count_analysis(df)

    time_analysis(df)

    plot_comments_per_university(df)

    plot_comment_length(df)

    plot_word_count(df)

    plot_comments_by_hour(df)

    correlation_analysis(df)

    save_report(

        df,

        category

    )

    print("\n" + "=" * 60)

    print("COMMENT EDA COMPLETED")

    print("=" * 60)


# ==========================================================
# DRIVER
# ==========================================================

if __name__ == "__main__":

    main()