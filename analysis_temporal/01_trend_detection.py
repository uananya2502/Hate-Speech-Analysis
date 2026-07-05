"""
============================================================
01_trend_detection.py

Temporal Trend Analysis

Input
------
cleaned_data_comments/<category>_comment_topics.csv

Outputs
-------
analysis_temporal/
    figures/
    tables/
    reports/

============================================================
"""

# ==========================================================
# ADD PROJECT ROOT
# ==========================================================

import os
import sys

PROJECT_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        ".."
    )
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ==========================================================
# IMPORT LIBRARIES
# ==========================================================

import pandas as pd
import matplotlib.pyplot as plt

# ==========================================================
# FOLDERS
# ==========================================================

INPUT_FOLDER = "cleaned_data_comments"

TABLE_FOLDER = os.path.join(
    "analysis_temporal",
    "tables"
)

FIGURE_FOLDER = os.path.join(
    "analysis_temporal",
    "figures"
)

REPORT_FOLDER = os.path.join(
    "analysis_temporal",
    "reports"
)

os.makedirs(TABLE_FOLDER, exist_ok=True)
os.makedirs(FIGURE_FOLDER, exist_ok=True)
os.makedirs(REPORT_FOLDER, exist_ok=True)

# ==========================================================
# CATEGORY
# ==========================================================

def get_category():

    print("=" * 60)
    print("TEMPORAL TREND ANALYSIS")
    print("=" * 60)

    return input(
        "\nEnter category (infrastructure, controversies, faculty_research, rankings): "
    ).strip().lower()


# ==========================================================
# LOAD DATASET
# ==========================================================

def load_dataset(category):

    input_file = os.path.join(

        INPUT_FOLDER,

        f"{category}_comment_topics.csv"

    )

    if not os.path.exists(input_file):

        print("\nDataset Not Found")

        exit()

    print("\nLoading dataset...")

    df = pd.read_csv(
        input_file,
        low_memory=False
    )

    df["published_at"] = pd.to_datetime(
        df["published_at"]
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

    print(f"Rows        : {len(df)}")
    print(f"Columns     : {len(df.columns)}")

    print(
        f"Memory (MB) : {df.memory_usage(deep=True).sum()/(1024**2):.2f}"
    )

    print("\nDate Range")

    print(
        df["published_at"].min()
    )

    print(
        df["published_at"].max()
    )

# ==========================================================
# MONTHLY COMMENT TREND
# ==========================================================

def monthly_comment_trend(df):

    print("\nGenerating Monthly Comment Trend...")

    monthly = df.copy()

    monthly["Year_Month"] = (

        monthly["published_at"]

        .dt.to_period("M")

        .astype(str)

    )

    monthly = (

        monthly

        .groupby("Year_Month")

        .size()

        .reset_index(name="Comments")

    )

    monthly.to_csv(

        os.path.join(

            TABLE_FOLDER,

            "monthly_comments.csv"

        ),

        index=False

    )

    plt.figure(figsize=(14,6))

    plt.plot(

        monthly["Year_Month"],

        monthly["Comments"],

        marker="o"

    )

    plt.xticks(rotation=90)

    plt.xlabel("Month")

    plt.ylabel("Comments")

    plt.title("Monthly Comment Trend")

    plt.tight_layout()

    plt.savefig(

        os.path.join(

            FIGURE_FOLDER,

            "monthly_comments.png"

        ),

        dpi=300

    )

    plt.close()

    print("✓ monthly_comments.csv")

    print("✓ monthly_comments.png")

    return monthly


# ==========================================================
# YEARLY COMMENT TREND
# ==========================================================

def yearly_comment_trend(df):

    print("\nGenerating Yearly Comment Trend...")

    yearly = (

        df

        .groupby(df["published_at"].dt.year)

        .size()

        .reset_index(name="Comments")

    )

    yearly.columns = [

        "Year",

        "Comments"

    ]

    yearly.to_csv(

        os.path.join(

            TABLE_FOLDER,

            "yearly_comments.csv"

        ),

        index=False

    )

    plt.figure(figsize=(8,5))

    plt.bar(

        yearly["Year"].astype(str),

        yearly["Comments"]

    )

    plt.xlabel("Year")

    plt.ylabel("Comments")

    plt.title("Year-wise Comment Trend")

    plt.tight_layout()

    plt.savefig(

        os.path.join(

            FIGURE_FOLDER,

            "yearly_comments.png"

        ),

        dpi=300

    )

    plt.close()

    print("✓ yearly_comments.csv")

    print("✓ yearly_comments.png")

    return yearly

# ==========================================================
# MONTHLY SENTIMENT TREND
# ==========================================================

def monthly_sentiment_trend(df):

    print("\nGenerating Monthly Sentiment Trend...")

    temp = df.copy()

    temp["Year_Month"] = (

        temp["published_at"]

        .dt.to_period("M")

        .astype(str)

    )

    sentiment = pd.crosstab(

        temp["Year_Month"],

        temp["sentiment"]

    )

    sentiment.to_csv(

        os.path.join(

            TABLE_FOLDER,

            "monthly_sentiment.csv"

        )

    )

    plt.figure(figsize=(14,6))

    for column in sentiment.columns:

        plt.plot(

            sentiment.index,

            sentiment[column],

            marker="o",

            label=column

        )

    plt.xticks(rotation=90)

    plt.xlabel("Month")

    plt.ylabel("Comments")

    plt.title("Monthly Sentiment Trend")

    plt.legend()

    plt.tight_layout()

    plt.savefig(

        os.path.join(

            FIGURE_FOLDER,

            "monthly_sentiment.png"

        ),

        dpi=300

    )

    plt.close()

    print("✓ monthly_sentiment.csv")

    print("✓ monthly_sentiment.png")

    return sentiment
# ==========================================================
# UNIVERSITY TREND
# ==========================================================

def university_trend(df):

    print("\nGenerating University Trend...")

    temp = df.copy()

    temp["Year_Month"] = (

        temp["published_at"]

        .dt.to_period("M")

        .astype(str)

    )

    top_universities = (

        temp["university_name"]

        .value_counts()

        .head(5)

        .index

    )

    subset = temp[

        temp["university_name"]

        .isin(top_universities)

    ]

    university = pd.crosstab(

        subset["Year_Month"],

        subset["university_name"]

    )

    university.to_csv(

        os.path.join(

            TABLE_FOLDER,

            "university_monthly.csv"

        )

    )

    plt.figure(figsize=(15,6))

    for column in university.columns:

        plt.plot(

            university.index,

            university[column],

            marker="o",

            label=column

        )

    plt.xticks(rotation=90)

    plt.xlabel("Month")

    plt.ylabel("Comments")

    plt.title("University Discussion Trend")

    plt.legend()

    plt.tight_layout()

    plt.savefig(

        os.path.join(

            FIGURE_FOLDER,

            "university_trend.png"

        ),

        dpi=300

    )

    plt.close()

    print("✓ university_monthly.csv")

    print("✓ university_trend.png")

    return university
# ==========================================================
# TOPIC TREND
# ==========================================================

def topic_trend(df):

    print("\nGenerating Topic Trend...")

    temp = df.copy()

    temp["Year_Month"] = (

        temp["published_at"]

        .dt.to_period("M")

        .astype(str)

    )

    topic = pd.crosstab(

        temp["Year_Month"],

        temp["dominant_topic"]

    )

    topic.to_csv(

        os.path.join(

            TABLE_FOLDER,

            "topic_monthly.csv"

        )

    )

    plt.figure(figsize=(15,6))

    for column in topic.columns:

        plt.plot(

            topic.index,

            topic[column],

            marker="o",

            label=f"Topic {column}"

        )

    plt.xticks(rotation=90)

    plt.xlabel("Month")

    plt.ylabel("Comments")

    plt.title("Topic Trend Over Time")

    plt.legend()

    plt.tight_layout()

    plt.savefig(

        os.path.join(

            FIGURE_FOLDER,

            "topic_trend.png"

        ),

        dpi=300

    )

    plt.close()

    print("✓ topic_monthly.csv")

    print("✓ topic_trend.png")

    return topic

# ==========================================================
# SAVE REPORT
# ==========================================================

def save_report(

    category,

    monthly,

    yearly,

    sentiment

):

    report = os.path.join(

        REPORT_FOLDER,

        f"{category}_trend_report.txt"

    )

    with open(

        report,

        "w",

        encoding="utf-8"

    ) as file:

        file.write("="*60+"\n")

        file.write("TEMPORAL TREND ANALYSIS REPORT\n")

        file.write("="*60+"\n\n")

        file.write("MONTHLY COMMENTS\n")

        file.write("-"*40+"\n")

        file.write(

            monthly.to_string(index=False)

        )

        file.write("\n\n")

        file.write("YEARLY COMMENTS\n")

        file.write("-"*40+"\n")

        file.write(

            yearly.to_string(index=False)

        )

        file.write("\n\n")

        file.write("MONTHLY SENTIMENT\n")

        file.write("-"*40+"\n")

        file.write(

            sentiment.to_string()

        )

    print("\nReport Saved")

    print(report)

# ==========================================================
# FINAL VALIDATION
# ==========================================================

def final_validation():

    print("\n"+"="*60)

    print("FINAL VALIDATION")

    print("="*60)

    print("✓ Monthly Trend")

    print("✓ Yearly Trend")

    print("✓ Sentiment Trend")

    print("✓ University Trend")

    print("✓ Topic Trend")

    print("✓ CSV Tables")

    print("✓ Figures")

    print("✓ Report")

# ==========================================================
# MAIN
# ==========================================================

def main():

    category = get_category()

    df = load_dataset(category)

    dataset_information(df)

    monthly = monthly_comment_trend(df)

    yearly = yearly_comment_trend(df)

    sentiment = monthly_sentiment_trend(df)

    university_trend(df)

    topic_trend(df)

    save_report(

        category,

        monthly,

        yearly,

        sentiment

    )

    final_validation()

    print("\n"+"="*60)

    print("TEMPORAL ANALYSIS COMPLETED")

    print("="*60)


# ==========================================================
# DRIVER
# ==========================================================

if __name__ == "__main__":

    main()

#.    python analysis_temporal/01_trend_detection.py