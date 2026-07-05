"""
============================================================
03_university_comparison.py

University Comparison Dashboard

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
    print("UNIVERSITY COMPARISON")
    print("=" * 60)

    return input(
        "\nEnter category (infrastructure, controversies, faculty_research, rankings): "
    ).strip().lower()

# ==========================================================
# LOAD DATASET
# ==========================================================

def load_dataset(category):

    file = os.path.join(
        INPUT_FOLDER,
        f"{category}_comment_sentiment.csv"
    )

    if not os.path.exists(file):

        print("Dataset Not Found")
        exit()

    print("\nLoading dataset...")

    df = pd.read_csv(
        file,
        low_memory=False
    )

    print(f"Loaded {len(df)} comments.")

    return df
# ==========================================================
# UNIVERSITY COMPARISON
# ==========================================================

def university_comparison(df):

    print("\nGenerating University Comparison...")

    # ------------------------------------------
    # Top 15 Universities
    # ------------------------------------------

    top_universities = (

        df["university_name"]

        .value_counts()

        .head(15)

        .index

    )

    df = df[
        df["university_name"].isin(top_universities)
    ].copy()

    # ------------------------------------------
    # Basic Statistics
    # ------------------------------------------

    comparison = (

        df

        .groupby("university_name")

        .agg(

            Total_Comments=("combined_comment", "count"),

            Average_Likes=("likes", "mean"),

            Average_Replies=("reply_count", "mean")

        )

        .round(2)

    )

    # ------------------------------------------
    # Sentiment Percentages
    # ------------------------------------------

    sentiment = (

        pd.crosstab(

            df["university_name"],

            df["sentiment"],

            normalize="index"

        )

        * 100

    ).round(2)

    sentiment.columns = [

        f"{col}_Percentage"

        for col in sentiment.columns

    ]

    # ------------------------------------------
    # Merge
    # ------------------------------------------

    comparison = comparison.join(sentiment)

    comparison = comparison.reset_index()

    comparison.to_csv(

        os.path.join(

            TABLE_FOLDER,

            "university_comparison.csv"

        ),

        index=False

    )

    print("✓ university_comparison.csv")

    return comparison
# ==========================================================
# UNIVERSITY VISUALIZATIONS
# ==========================================================

def university_visualizations(comparison):

    print("\nGenerating Comparison Visualizations...")

    # ======================================================
    # Total Comments
    # ======================================================

    top = comparison.sort_values(
        by="Total_Comments",
        ascending=False
    )

    plt.figure(figsize=(12,8))

    plt.barh(
        top["university_name"],
        top["Total_Comments"]
    )

    plt.gca().invert_yaxis()

    plt.xlabel("Total Comments")

    plt.title("Comments by University")

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            FIGURE_FOLDER,
            "university_comparison.png"
        ),
        dpi=300
    )

    plt.close()

    print("✓ university_comparison.png")

    # ======================================================
    # Sentiment Comparison
    # ======================================================

    plt.figure(figsize=(12,8))

    x = range(len(comparison))

    plt.bar(
        x,
        comparison["Positive_Percentage"],
        label="Positive"
    )

    plt.bar(
        x,
        comparison["Negative_Percentage"],
        bottom=comparison["Positive_Percentage"],
        label="Negative"
    )

    plt.bar(
        x,
        comparison["Neutral_Percentage"],
        bottom=(
            comparison["Positive_Percentage"] +
            comparison["Negative_Percentage"]
        ),
        label="Neutral"
    )

    plt.xticks(
        x,
        comparison["university_name"],
        rotation=90
    )

    plt.ylabel("Percentage")

    plt.title("Sentiment Distribution by University")

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            FIGURE_FOLDER,
            "sentiment_comparison.png"
        ),
        dpi=300
    )

    plt.close()

    print("✓ sentiment_comparison.png")

    # ======================================================
    # Likes vs Replies
    # ======================================================

    plt.figure(figsize=(12,8))

    width = 0.35

    x = range(len(comparison))

    plt.bar(
        [i-width/2 for i in x],
        comparison["Average_Likes"],
        width=width,
        label="Likes"
    )

    plt.bar(
        [i+width/2 for i in x],
        comparison["Average_Replies"],
        width=width,
        label="Replies"
    )

    plt.xticks(
        x,
        comparison["university_name"],
        rotation=90
    )

    plt.ylabel("Average")

    plt.title("Average Likes vs Replies")

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            FIGURE_FOLDER,
            "engagement_comparison.png"
        ),
        dpi=300
    )

    plt.close()

    print("✓ engagement_comparison.png")
# ==========================================================
# SAVE REPORT
# ==========================================================

def save_report(comparison):

    report = os.path.join(
        REPORT_FOLDER,
        "university_comparison_report.txt"
    )

    with open(
        report,
        "w",
        encoding="utf-8"
    ) as file:

        file.write("=" * 60 + "\n")
        file.write("UNIVERSITY COMPARISON REPORT\n")
        file.write("=" * 60 + "\n\n")

        file.write(
            f"Universities Compared : {len(comparison)}\n\n"
        )

        file.write(
            comparison.to_string(index=False)
        )

    print("\n✓ university_comparison_report.txt")

# ==========================================================
# FINAL VALIDATION
# ==========================================================

def final_validation():

    print("\n" + "=" * 60)

    print("FINAL VALIDATION")

    print("=" * 60)

    print("✓ University Comparison")

    print("✓ Comparison Charts")

    print("✓ CSV Tables")

    print("✓ Figures")

    print("✓ Report")
# ==========================================================
# MAIN
# ==========================================================

def main():

    category = get_category()

    df = load_dataset(category)

    comparison = university_comparison(df)

    university_visualizations(comparison)

    save_report(comparison)

    final_validation()

    print("\n" + "=" * 60)

    print("UNIVERSITY COMPARISON COMPLETED")

    print("=" * 60)


# ==========================================================
# DRIVER
# ==========================================================

if __name__ == "__main__":

    main()
# python analysis_engagement/03_university_comparison.py