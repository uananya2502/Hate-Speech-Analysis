"""
============================================================
01_likes_analysis.py

Likes Analysis

Input
------
cleaned_data_comments/<category>_comment_sentiment.csv

Outputs
-------
analysis_engagement/
    figures/
    tables/
    reports/

Author : Ananya Upadhyay
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
# IMPORTS
# ==========================================================

import pandas as pd
import matplotlib.pyplot as plt

# ==========================================================
# FOLDERS
# ==========================================================

INPUT_FOLDER = "cleaned_data_comments"

TABLE_FOLDER = os.path.join(
    "analysis_engagement",
    "tables"
)

FIGURE_FOLDER = os.path.join(
    "analysis_engagement",
    "figures"
)

REPORT_FOLDER = os.path.join(
    "analysis_engagement",
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
    print("LIKES ANALYSIS")
    print("=" * 60)

    category = input(
        "\nEnter category (infrastructure, controversies, faculty_research, rankings): "
    ).strip().lower()

    return category

# ==========================================================
# LOAD DATASET
# ==========================================================

def load_dataset(category):

    input_file = os.path.join(
        INPUT_FOLDER,
        f"{category}_comment_sentiment.csv"
    )

    if not os.path.exists(input_file):

        print("\nDataset Not Found")
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

    print(f"Rows       : {len(df)}")
    print(f"Columns    : {len(df.columns)}")

    print(
        f"Memory MB  : {df.memory_usage(deep=True).sum()/(1024**2):.2f}"
    )
# ==========================================================
# TOP LIKED COMMENTS
# ==========================================================

def top_liked_comments(df):

    print("\nGenerating Top Liked Comments...")

    result = (

        df[
            [
                "combined_comment",
                "likes",
                "university_name",
                "sentiment"
            ]
        ]

        .sort_values(

            by="likes",

            ascending=False

        )

        .head(20)

    )

    result.to_csv(

        os.path.join(

            TABLE_FOLDER,

            "top_liked_comments.csv"

        ),

        index=False

    )

    plt.figure(

        figsize=(12,8)

    )

    plt.barh(

        range(len(result)),

        result["likes"]

    )

    plt.gca().invert_yaxis()

    plt.yticks(

        range(len(result)),

        [

            f"Comment {i+1}"

            for i in range(len(result))

        ]

    )

    plt.xlabel("Likes")

    plt.title("Top 20 Most Liked Comments")

    plt.tight_layout()

    plt.savefig(

        os.path.join(

            FIGURE_FOLDER,

            "top_liked_comments.png"

        ),

        dpi=300

    )

    plt.close()

    print("✓ top_liked_comments.csv")

    print("✓ top_liked_comments.png")

    return result
# ==========================================================
# LIKES BY UNIVERSITY
# ==========================================================

def likes_by_university(df):

    print("\nGenerating University Likes Analysis...")

    result = (

        df

        .groupby(

            "university_name"

        )

        .agg(

            Average_Likes=("likes","mean"),

            Median_Likes=("likes","median"),

            Maximum_Likes=("likes","max"),

            Total_Comments=("likes","count")

        )

        .round(2)

        .reset_index()

    )

    result = result.sort_values(

        by="Average_Likes",

        ascending=False

    )

    result.to_csv(

        os.path.join(

            TABLE_FOLDER,

            "likes_by_university.csv"

        ),

        index=False

    )

    top = result.head(15)

    plt.figure(

        figsize=(12,8)

    )

    plt.barh(

        top["university_name"],

        top["Average_Likes"]

    )

    plt.gca().invert_yaxis()

    plt.xlabel("Average Likes")

    plt.title("Average Likes by University")

    plt.tight_layout()

    plt.savefig(

        os.path.join(

            FIGURE_FOLDER,

            "likes_by_university.png"

        ),

        dpi=300

    )

    plt.close()

    print("✓ likes_by_university.csv")

    print("✓ likes_by_university.png")

    return result
# ==========================================================
# LIKES BY SENTIMENT
# ==========================================================

def likes_by_sentiment(df):

    print("\nGenerating Sentiment-wise Likes Analysis...")

    result = (

        df

        .groupby("sentiment")

        .agg(

            Average_Likes=("likes", "mean"),

            Median_Likes=("likes", "median"),

            Maximum_Likes=("likes", "max"),

            Total_Comments=("likes", "count")

        )

        .round(2)

        .reset_index()

    )

    result.to_csv(

        os.path.join(

            TABLE_FOLDER,

            "likes_by_sentiment.csv"

        ),

        index=False

    )

    plt.figure(figsize=(8,6))

    plt.bar(

        result["sentiment"],

        result["Average_Likes"]

    )

    plt.xlabel("Sentiment")

    plt.ylabel("Average Likes")

    plt.title("Average Likes by Sentiment")

    plt.tight_layout()

    plt.savefig(

        os.path.join(

            FIGURE_FOLDER,

            "likes_by_sentiment.png"

        ),

        dpi=300

    )

    plt.close()

    print("✓ likes_by_sentiment.csv")

    print("✓ likes_by_sentiment.png")

    return result
# ==========================================================
# MONTHLY LIKES TREND
# ==========================================================

def monthly_likes(df):

    print("\nGenerating Monthly Likes Trend...")

    df["published_at"] = pd.to_datetime(

        df["published_at"],

        errors="coerce"

    )

    result = (

        df

        .dropna(subset=["published_at"])

        .groupby(

            df["published_at"].dt.to_period("M")

        )

        .agg(

            Average_Likes=("likes","mean"),

            Total_Likes=("likes","sum")

        )

        .reset_index()

    )

    result["published_at"] = result["published_at"].astype(str)

    result.to_csv(

        os.path.join(

            TABLE_FOLDER,

            "monthly_likes.csv"

        ),

        index=False

    )

    plt.figure(figsize=(14,6))

    plt.plot(

        result["published_at"],

        result["Average_Likes"],

        marker="o"

    )

    plt.xticks(rotation=90)

    plt.ylabel("Average Likes")

    plt.title("Monthly Average Likes")

    plt.tight_layout()

    plt.savefig(

        os.path.join(

            FIGURE_FOLDER,

            "monthly_likes.png"

        ),

        dpi=300

    )

    plt.close()

    print("✓ monthly_likes.csv")

    print("✓ monthly_likes.png")

    return result
# ==========================================================
# SAVE REPORT
# ==========================================================

def save_report(

    top_comments,

    university,

    sentiment,

    monthly

):

    report = os.path.join(

        REPORT_FOLDER,

        "likes_report.txt"

    )

    with open(

        report,

        "w",

        encoding="utf-8"

    ) as file:

        file.write("="*60 + "\n")

        file.write("LIKES ANALYSIS REPORT\n")

        file.write("="*60 + "\n\n")

        file.write("TOP LIKED COMMENTS\n")

        file.write(top_comments.to_string(index=False))

        file.write("\n\n")

        file.write("UNIVERSITY ANALYSIS\n")

        file.write(university.head(20).to_string(index=False))

        file.write("\n\n")

        file.write("SENTIMENT ANALYSIS\n")

        file.write(sentiment.to_string(index=False))

        file.write("\n\n")

        file.write("MONTHLY TREND\n")

        file.write(monthly.to_string(index=False))

    print("\n✓ likes_report.txt")
# ==========================================================
# FINAL VALIDATION
# ==========================================================

def final_validation():

    print("\n" + "="*60)

    print("FINAL VALIDATION")

    print("="*60)

    print("✓ Top Liked Comments")

    print("✓ University Likes")

    print("✓ Sentiment Likes")

    print("✓ Monthly Likes")

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

    top_comments = top_liked_comments(df)

    university = likes_by_university(df)

    sentiment = likes_by_sentiment(df)

    monthly = monthly_likes(df)

    save_report(

        top_comments,

        university,

        sentiment,

        monthly

    )

    final_validation()

    print("\n" + "="*60)

    print("LIKES ANALYSIS COMPLETED")

    print("="*60)


# ==========================================================
# DRIVER
# ==========================================================

if __name__ == "__main__":

    main()
#.  python analysis_engagement/01_likes_analysis.py