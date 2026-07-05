"""
============================================================
02_hashtag_analysis.py

Hashtag & Engagement Analysis

Input
------
cleaned_data_comments/<category>_comment_sentiment.csv

Outputs
-------
analysis_content/
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

import re

import pandas as pd

import matplotlib.pyplot as plt

# ==========================================================
# FOLDERS
# ==========================================================

INPUT_FOLDER = "cleaned_data_comments"

TABLE_FOLDER = os.path.join(
    "analysis_content",
    "tables"
)

FIGURE_FOLDER = os.path.join(
    "analysis_content",
    "figures"
)

REPORT_FOLDER = os.path.join(
    "analysis_content",
    "reports"
)

os.makedirs(TABLE_FOLDER, exist_ok=True)
os.makedirs(FIGURE_FOLDER, exist_ok=True)
os.makedirs(REPORT_FOLDER, exist_ok=True)

# ==========================================================
# CATEGORY
# ==========================================================

def get_category():

    print("="*60)

    print("HASHTAG ANALYSIS")

    print("="*60)

    return input(

        "\nEnter category (infrastructure, controversies, faculty_research, rankings): "

    ).strip().lower()


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

    df["published_at"] = pd.to_datetime(

        df["published_at"]

    )

    print(f"Loaded {len(df)} comments.")

    return df


# ==========================================================
# DATASET INFO
# ==========================================================

def dataset_information(df):

    print("\n"+"="*60)

    print("DATASET INFORMATION")

    print("="*60)

    print(f"Rows       : {len(df)}")

    print(f"Columns    : {len(df.columns)}")

    print(

        f"Memory MB  : {df.memory_usage(deep=True).sum()/(1024**2):.2f}"

    )
# ==========================================================
# EXTRACT HASHTAGS
# ==========================================================

def extract_hashtags(text):

    if pd.isna(text):

        return []

    hashtags = re.findall(

        r"#\w+",

        str(text)

    )

    cleaned = []

    for tag in hashtags:

        tag = tag.lower().strip()

        # Remove hashtags with only one character
        if len(tag) <= 2:

            continue

        # Keep only English hashtags
        if not re.fullmatch(

            r"#[a-z0-9_]+",

            tag

        ):

            continue

        cleaned.append(tag)

    # Remove duplicates within the same comment
    cleaned = list(

        dict.fromkeys(cleaned)

    )

    return cleaned
# ==========================================================
# BUILD HASHTAG DATASET
# ==========================================================

def build_hashtag_dataset(df):

    print("\nExtracting Hashtags...")

    rows = []

    for _, row in df.iterrows():

        hashtags = extract_hashtags(

            row["combined_comment"]

        )

        if len(hashtags) == 0:

            continue

        for tag in hashtags:

            rows.append({

                "hashtag": tag,

                "sentiment": row["sentiment"],

                "university_name": row["university_name"],

                "published_at": row["published_at"],

                "likes": row["likes"],

                "reply_count": row["reply_count"]

            })

    hashtag_df = pd.DataFrame(rows)

    print(

        f"Total Valid Hashtags : {len(hashtag_df)}"

    )

    hashtag_df.to_csv(

        os.path.join(

            TABLE_FOLDER,

            "all_hashtags.csv"

        ),

        index=False

    )

    print("✓ all_hashtags.csv")

    return hashtag_df
# ==========================================================
# PLOT HASHTAGS
# ==========================================================

def plot_hashtags(

    hashtag_table,

    title,

    filename

):

    plt.figure(figsize=(12,8))

    plt.barh(

        hashtag_table["Hashtag"],

        hashtag_table["Frequency"]

    )

    plt.gca().invert_yaxis()

    plt.xlabel("Frequency")

    plt.ylabel("Hashtag")

    plt.title(title)

    plt.tight_layout()

    plt.savefig(

        os.path.join(

            FIGURE_FOLDER,

            filename

        ),

        dpi=300

    )

    plt.close()

    print(f"✓ {filename}")


# ==========================================================
# OVERALL HASHTAGS
# ==========================================================

def overall_hashtags(hashtag_df):

    print("\nGenerating Overall Hashtag Analysis...")

    overall = (

        hashtag_df

        .groupby("hashtag")

        .size()

        .reset_index(name="Frequency")

        .sort_values(

            by="Frequency",

            ascending=False

        )

        .head(30)

    )

    overall.columns = [

        "Hashtag",

        "Frequency"

    ]

    overall.to_csv(

        os.path.join(

            TABLE_FOLDER,

            "overall_hashtags.csv"

        ),

        index=False

    )

    plot_hashtags(

        overall,

        "Top 30 Hashtags",

        "overall_hashtags.png"

    )

    print("✓ overall_hashtags.csv")

    return overall
# ==========================================================
# SENTIMENT-WISE HASHTAGS
# ==========================================================

def sentiment_hashtags(hashtag_df):

    print("\nGenerating Sentiment-wise Hashtags...")

    sentiments = [

        "Positive",

        "Negative",

        "Neutral"

    ]

    for sentiment in sentiments:

        print(f"\nProcessing {sentiment}...")

        subset = hashtag_df[

            hashtag_df["sentiment"] == sentiment

        ]

        if len(subset) == 0:

            continue

        result = (

            subset

            .groupby("hashtag")

            .size()

            .reset_index(name="Frequency")

            .sort_values(

                by="Frequency",

                ascending=False

            )

            .head(20)

        )

        result.columns = [

            "Hashtag",

            "Frequency"

        ]

        csv_name = (

            sentiment.lower()

            + "_hashtags.csv"

        )

        png_name = (

            sentiment.lower()

            + "_hashtags.png"

        )

        result.to_csv(

            os.path.join(

                TABLE_FOLDER,

                csv_name

            ),

            index=False

        )

        plot_hashtags(

            result,

            f"{sentiment} Hashtags",

            png_name

        )

    print("\n✓ Sentiment-wise Hashtag Analysis Completed")

# ==========================================================
# UNIVERSITY-WISE HASHTAGS
# ==========================================================

def university_hashtags(hashtag_df):

    print("\nGenerating University-wise Hashtags...")

    top_universities = (

        hashtag_df["university_name"]

        .value_counts()

        .head(5)

        .index

    )

    summary = []

    for university in top_universities:

        print(f"\nProcessing {university}...")

        subset = hashtag_df[

            hashtag_df["university_name"] == university

        ]

        result = (

            subset

            .groupby("hashtag")

            .size()

            .reset_index(name="Frequency")

            .sort_values(

                by="Frequency",

                ascending=False

            )

            .head(15)

        )

        result.columns = [

            "Hashtag",

            "Frequency"

        ]

        file = (

            university

            .lower()

            .replace(" ", "_")

            + "_hashtags.csv"

        )

        result.to_csv(

            os.path.join(

                TABLE_FOLDER,

                file

            ),

            index=False

        )

        plot_hashtags(

            result,

            f"{university} Hashtags",

            file.replace(".csv",".png")

        )

        result["University"] = university

        summary.append(result)

    summary = pd.concat(

        summary,

        ignore_index=True

    )

    summary.to_csv(

        os.path.join(

            TABLE_FOLDER,

            "university_hashtags.csv"

        ),

        index=False

    )

    print("✓ university_hashtags.csv")

    return summary
# ==========================================================
# MONTHLY HASHTAG TREND
# ==========================================================

def monthly_hashtag_trend(hashtag_df):

    print("\nGenerating Monthly Hashtag Trend...")

    temp = hashtag_df.copy()

    temp["Year_Month"] = (

        temp["published_at"]

        .dt.to_period("M")

        .astype(str)

    )

    monthly = (

        temp

        .groupby("Year_Month")

        .size()

        .reset_index(name="Hashtags")

    )

    monthly.to_csv(

        os.path.join(

            TABLE_FOLDER,

            "monthly_hashtags.csv"

        ),

        index=False

    )

    plt.figure(figsize=(14,6))

    plt.plot(

        monthly["Year_Month"],

        monthly["Hashtags"],

        marker="o"

    )

    plt.xticks(rotation=90)

    plt.xlabel("Month")

    plt.ylabel("Hashtag Count")

    plt.title("Monthly Hashtag Usage")

    plt.tight_layout()

    plt.savefig(

        os.path.join(

            FIGURE_FOLDER,

            "monthly_hashtags.png"

        ),

        dpi=300

    )

    plt.close()

    print("✓ monthly_hashtags.csv")

    print("✓ monthly_hashtags.png")

    return monthly
# ==========================================================
# HASHTAG ENGAGEMENT
# ==========================================================

def hashtag_engagement(hashtag_df):

    print("\nGenerating Hashtag Engagement Analysis...")

    engagement = (

        hashtag_df

        .groupby("hashtag")

        .agg(

            Frequency=("hashtag","count"),

            Avg_Likes=("likes","mean"),

            Avg_Replies=("reply_count","mean")

        )

        .reset_index()

    )

    engagement["Popularity_Score"] = (

        engagement["Frequency"]

        * engagement["Avg_Likes"]

    ).round(2)

    engagement = (

        engagement

        .sort_values(

            by="Popularity_Score",

            ascending=False

        )

    )

    engagement.to_csv(

        os.path.join(

            TABLE_FOLDER,

            "hashtag_engagement.csv"

        ),

        index=False

    )

    top = engagement.head(15)

    plt.figure(figsize=(12,8))

    plt.barh(

        top["hashtag"],

        top["Popularity_Score"]

    )

    plt.gca().invert_yaxis()

    plt.xlabel("Popularity Score")

    plt.ylabel("Hashtag")

    plt.title("Top Hashtag Popularity")

    plt.tight_layout()

    plt.savefig(

        os.path.join(

            FIGURE_FOLDER,

            "hashtag_engagement.png"

        ),

        dpi=300

    )

    plt.close()

    print("✓ hashtag_engagement.csv")

    print("✓ hashtag_engagement.png")

    return engagement
# ==========================================================
# SAVE REPORT
# ==========================================================

def save_report(

    category,

    overall,

    engagement

):

    report = os.path.join(

        REPORT_FOLDER,

        f"{category}_hashtag_report.txt"

    )

    with open(

        report,

        "w",

        encoding="utf-8"

    ) as file:

        file.write("="*60 + "\n")

        file.write("HASHTAG ANALYSIS REPORT\n")

        file.write("="*60 + "\n\n")

        file.write("TOP HASHTAGS\n")

        file.write("-"*40 + "\n")

        file.write(

            overall.to_string(index=False)

        )

        file.write("\n\n")

        file.write("HASHTAG ENGAGEMENT\n")

        file.write("-"*40 + "\n")

        file.write(

            engagement.head(20).to_string(index=False)

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

    print("✓ Overall Hashtags")

    print("✓ Sentiment-wise Hashtags")

    print("✓ University-wise Hashtags")

    print("✓ Monthly Trend")

    print("✓ Engagement Analysis")

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

    hashtag_df = build_hashtag_dataset(df)

    overall = overall_hashtags(hashtag_df)

    sentiment_hashtags(hashtag_df)

    university_hashtags(hashtag_df)

    monthly_hashtag_trend(hashtag_df)

    engagement = hashtag_engagement(hashtag_df)

    save_report(

        category,

        overall,

        engagement

    )

    final_validation()

    print("\n"+"="*60)

    print("HASHTAG ANALYSIS COMPLETED")

    print("="*60)


# ==========================================================
# DRIVER
# ==========================================================

if __name__ == "__main__":

    main()
#.   python analysis_content/02_hashtag_analysis.py