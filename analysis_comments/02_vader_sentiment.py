"""
============================================================
02_vader_sentiment.py

Hybrid Sentiment Analysis
(VADER + Hinglish Lexicon)

Input:
    cleaned_data_comments/<category>_comment_feature_engineered.csv

Output:
    cleaned_data_comments/<category>_comment_sentiment.csv

Reports:
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

import re

import pandas as pd

import matplotlib.pyplot as plt

import nltk

from nltk.sentiment import SentimentIntensityAnalyzer

# ==========================================================
# DOWNLOAD VADER
# ==========================================================

try:

    nltk.data.find("sentiment/vader_lexicon.zip")

except LookupError:

    nltk.download("vader_lexicon")

sia = SentimentIntensityAnalyzer()

# ==========================================================
# FOLDERS
# ==========================================================

INPUT_FOLDER = "cleaned_data_comments"

OUTPUT_FOLDER = "cleaned_data_comments"

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

os.makedirs(REPORT_FOLDER, exist_ok=True)
os.makedirs(TABLE_FOLDER, exist_ok=True)
os.makedirs(FIGURE_FOLDER, exist_ok=True)

# ==========================================================
# CATEGORY
# ==========================================================

def get_category():

    print("=" * 60)
    print("HYBRID SENTIMENT ANALYSIS")
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
# HINGLISH SENTIMENT LEXICON
# ==========================================================

HINGLISH_POSITIVE = {

    "acha": 1.0,
    "accha": 1.0,
    "achha": 1.0,
    "badhiya": 1.5,
    "badiya": 1.5,
    "mast": 1.8,
    "zabardast": 2.0,
    "sahi": 1.0,
    "awesome": 2.0,
    "excellent": 2.0,
    "best": 2.0,
    "beautiful": 1.8,
    "amazing": 2.0,
    "super": 1.8,
    "fantastic": 2.0,
    "love": 2.0,
    "great": 1.8,
    "nice": 1.2,
    "perfect": 2.0,
    "helpful": 1.5,
    "clean": 1.0,
    "good": 1.0,
    "top": 1.5,
    "excellent": 2.0,
    "peaceful": 1.5,
    "comfortable": 1.5,
    "recommended": 2.0

}

HINGLISH_NEGATIVE = {

    "bakwaas": -2.0,
    "bekar": -1.8,
    "bekaar": -1.8,
    "ghatiya": -2.2,
    "faltu": -1.5,
    "fraud": -2.0,
    "fake": -2.0,
    "scam": -2.2,
    "worst": -2.5,
    "waste": -2.0,
    "pathetic": -2.2,
    "poor": -1.5,
    "bad": -1.5,
    "useless": -2.0,
    "disappointed": -2.0,
    "disappointing": -2.0,
    "dirty": -1.5,
    "boring": -1.2,
    "expensive": -1.0,
    "overpriced": -1.5,
    "horrible": -2.5,
    "terrible": -2.5,
    "awful": -2.5

}

# ==========================================================
# SPELLING NORMALIZATION
# ==========================================================

NORMALIZATION = {

    "accha": "acha",
    "achha": "acha",
    "badiya": "badhiya",
    "bekaar": "bekar",
    "masttt": "mast",
    "mastt": "mast",
    "gud": "good",
    "gr8": "great",
    "awsm": "awesome",
    "osm": "awesome"

}

# ==========================================================
# PROFANITY LEXICON
# ==========================================================

# NOTE:
# Populate this set with the abusive/profane words you want
# to detect in your dataset. Keep this list private and
# separate from the dissertation text.

PROFANITY = {

    # Add your profanity terms here.

}

# ==========================================================
# NORMALIZE TEXT
# ==========================================================

def normalize_hinglish(text):

    words = str(text).lower().split()

    normalized = []

    for word in words:

        normalized.append(

            NORMALIZATION.get(

                word,

                word

            )

        )

    return normalized


# ==========================================================
# HINGLISH SCORE
# ==========================================================

def hinglish_score(words):

    score = 0

    positive = 0

    negative = 0

    for word in words:

        if word in HINGLISH_POSITIVE:

            score += HINGLISH_POSITIVE[word]

            positive += 1

        if word in HINGLISH_NEGATIVE:

            score += HINGLISH_NEGATIVE[word]

            negative += 1

    return score, positive, negative


# ==========================================================
# PROFANITY DETECTION
# ==========================================================

def profanity_detection(words):

    count = 0

    for word in words:

        if word in PROFANITY:

            count += 1

    return count

# ==========================================================
# FINAL SENTIMENT LABEL
# ==========================================================

def get_sentiment_label(score):

    if score >= 0.05:

        return "Positive"

    elif score <= -0.05:

        return "Negative"

    else:

        return "Neutral"


# ==========================================================
# HYBRID SENTIMENT ANALYSIS
# ==========================================================

def hybrid_sentiment_analysis(df):

    print("\nRunning Hybrid Sentiment Analysis...")

    # ---------------------------------------------
    # Normalize Hinglish
    # ---------------------------------------------

    normalized_words = (

        df["combined_comment"]

        .fillna("")

        .apply(normalize_hinglish)

    )

    # ---------------------------------------------
    # Hinglish Scores
    # ---------------------------------------------

    hinglish_result = normalized_words.apply(

        hinglish_score

    )

    df["hinglish_score"] = (

        hinglish_result

        .apply(lambda x: x[0])

    )

    df["positive_hinglish_words"] = (

        hinglish_result

        .apply(lambda x: x[1])

    )

    df["negative_hinglish_words"] = (

        hinglish_result

        .apply(lambda x: x[2])

    )

    # ---------------------------------------------
    # Profanity
    # ---------------------------------------------

    df["profanity_count"] = (

        normalized_words

        .apply(profanity_detection)

    )

    df["contains_profanity"] = (

        df["profanity_count"] > 0

    )

    # ---------------------------------------------
    # VADER Scores
    # ---------------------------------------------

    vader_scores = (

        df["combined_comment"]

        .fillna("")

        .astype(str)

        .apply(

            lambda text:

            sia.polarity_scores(text)

        )

    )

    df["negative"] = (

        vader_scores

        .apply(

            lambda x: x["neg"]

        )

    )

    df["neutral"] = (

        vader_scores

        .apply(

            lambda x: x["neu"]

        )

    )

    df["positive"] = (

        vader_scores

        .apply(

            lambda x: x["pos"]

        )

    )

    df["compound"] = (

        vader_scores

        .apply(

            lambda x: x["compound"]

        )

    )

    # ---------------------------------------------
    # Hybrid Adjustment
    # ---------------------------------------------

    df["adjusted_compound"] = (

        df["compound"]

        +

        (df["hinglish_score"] * 0.20)

    )

    df["adjusted_compound"] = (

        df["adjusted_compound"]

        .clip(-1, 1)

    )

    # ---------------------------------------------
    # Final Sentiment
    # ---------------------------------------------

    df["sentiment"] = (

        df["adjusted_compound"]

        .apply(get_sentiment_label)

    )

    print("✓ Hybrid sentiment completed.")

    print()

    print("New Columns Created")

    print("-------------------")

    print("negative")

    print("neutral")

    print("positive")

    print("compound")

    print("hinglish_score")

    print("adjusted_compound")

    print("positive_hinglish_words")

    print("negative_hinglish_words")

    print("profanity_count")

    print("contains_profanity")

    print("sentiment")

    return df

# ==========================================================
# SENTIMENT DISTRIBUTION
# ==========================================================

def sentiment_distribution(df):

    print("\n" + "=" * 60)
    print("SENTIMENT DISTRIBUTION")
    print("=" * 60)

    sentiment = df["sentiment"].value_counts()

    percentage = (

        sentiment / len(df) * 100

    ).round(2)

    result = pd.DataFrame({

        "Count": sentiment,

        "Percentage": percentage

    })

    print(result)

    result.to_csv(

        os.path.join(

            TABLE_FOLDER,

            "sentiment_distribution.csv"

        )

    )


# ==========================================================
# SENTIMENT BY UNIVERSITY
# ==========================================================

def sentiment_by_university(df):

    table = pd.crosstab(

        df["university_name"],

        df["sentiment"]

    )

    table["Total"] = table.sum(axis=1)

    table["Positive %"] = (

        table["Positive"]

        / table["Total"]

        * 100

    ).round(2)

    table["Negative %"] = (

        table["Negative"]

        / table["Total"]

        * 100

    ).round(2)

    table["Neutral %"] = (

        table["Neutral"]

        / table["Total"]

        * 100

    ).round(2)

    table = table.sort_values(

        "Total",

        ascending=False

    )

    print("\nTop Universities")

    print(table.head(10))

    table.to_csv(

        os.path.join(

            TABLE_FOLDER,

            "sentiment_by_university.csv"

        )

    )


# ==========================================================
# SENTIMENT BY YEAR
# ==========================================================

def sentiment_by_year(df):

    table = pd.crosstab(

        df["comment_year"],

        df["sentiment"]

    )

    print("\nSentiment by Year")

    print(table)

    table.to_csv(

        os.path.join(

            TABLE_FOLDER,

            "sentiment_by_year.csv"

        )

    )


# ==========================================================
# ENGAGEMENT ANALYSIS
# ==========================================================

def engagement_analysis(df):

    table = (

        df.groupby("sentiment")[

            [

                "likes",

                "reply_count"

            ]

        ]

        .mean()

        .round(2)

    )

    print("\nAverage Engagement")

    print(table)

    table.to_csv(

        os.path.join(

            TABLE_FOLDER,

            "engagement_by_sentiment.csv"

        )

    )


# ==========================================================
# PROFANITY ANALYSIS
# ==========================================================

def profanity_analysis(df):

    summary = (

        df["contains_profanity"]

        .value_counts()

    )

    summary.to_csv(

        os.path.join(

            TABLE_FOLDER,

            "profanity_distribution.csv"

        )

    )

    university = (

        df.groupby(

            "university_name"

        )["profanity_count"]

        .sum()

        .sort_values(

            ascending=False

        )

    )

    university.to_csv(

        os.path.join(

            TABLE_FOLDER,

            "profanity_by_university.csv"

        )

    )


# ==========================================================
# TOP POSITIVE COMMENTS
# ==========================================================

def top_positive_comments(df):

    table = (

        df

        .sort_values(

            "adjusted_compound",

            ascending=False

        )

        .head(20)

    )

    table.to_csv(

        os.path.join(

            TABLE_FOLDER,

            "top_positive_comments.csv"

        ),

        index=False

    )


# ==========================================================
# TOP NEGATIVE COMMENTS
# ==========================================================

def top_negative_comments(df):

    table = (

        df

        .sort_values(

            "adjusted_compound"

        )

        .head(20)

    )

    table.to_csv(

        os.path.join(

            TABLE_FOLDER,

            "top_negative_comments.csv"

        ),

        index=False

    )


# ==========================================================
# SENTIMENT BAR CHART
# ==========================================================

def plot_sentiment_distribution(df):

    plt.figure(figsize=(6,5))

    sentiment = df["sentiment"].value_counts()

    sentiment.plot(kind="bar")

    plt.title("Sentiment Distribution")

    plt.xlabel("Sentiment")

    plt.ylabel("Comments")

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            FIGURE_FOLDER,
            "sentiment_distribution.png"
        )
    )

    plt.close()

# ==========================================================
# PROFANITY BAR CHART
# ==========================================================

def plot_profanity(df):

    plt.figure(figsize=(5,5))

    profanity = df["contains_profanity"].value_counts()

    profanity.plot(kind="bar")

    plt.title("Comments Containing Profanity")

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            FIGURE_FOLDER,
            "profanity_distribution.png"
        )
    )

    plt.close()


# ==========================================================
# SAVE DATASET
# ==========================================================

def save_dataset(df, category):

    output_file = os.path.join(

        OUTPUT_FOLDER,

        f"{category}_comment_sentiment.csv"

    )

    df.to_csv(

        output_file,

        index=False,

        encoding="utf-8"

    )

    print("\nDataset Saved")

    print(output_file)


# ==========================================================
# SAVE REPORT
# ==========================================================

def save_report(df, category):

    report_file = os.path.join(

        REPORT_FOLDER,

        f"{category}_hybrid_sentiment_report.txt"

    )

    sentiment = (

        df["sentiment"]

        .value_counts()

    )

    profanity = (

        df["contains_profanity"]

        .sum()

    )

    with open(

        report_file,

        "w",

        encoding="utf-8"

    ) as file:

        file.write("HYBRID SENTIMENT ANALYSIS REPORT\n")

        file.write("=" * 50)

        file.write("\n\n")

        file.write(f"Total Comments : {len(df)}\n")

        file.write(f"Comments with Profanity : {profanity}\n\n")

        file.write("Sentiment Distribution\n")

        file.write("----------------------\n")

        for label, count in sentiment.items():

            percentage = (

                count / len(df)

            ) * 100

            file.write(

                f"{label}: {count} ({percentage:.2f}%)\n"

            )

        file.write("\n")

        file.write(

            f"Average VADER Compound : "

            f"{df['compound'].mean():.3f}\n"

        )

        file.write(

            f"Average Hybrid Compound : "

            f"{df['adjusted_compound'].mean():.3f}\n"

        )

        file.write(

            f"Average Hinglish Score : "

            f"{df['hinglish_score'].mean():.3f}\n"

        )

    print("\nReport Saved")

    print(report_file)


# ==========================================================
# FINAL VALIDATION
# ==========================================================

def final_validation(df):

    print("\n" + "=" * 60)

    print("FINAL VALIDATION")

    print("=" * 60)

    required_columns = [

        "negative",

        "neutral",

        "positive",

        "compound",

        "hinglish_score",

        "adjusted_compound",

        "positive_hinglish_words",

        "negative_hinglish_words",

        "profanity_count",

        "contains_profanity",

        "sentiment"

    ]

    missing = [

        col

        for col in required_columns

        if col not in df.columns

    ]

    if len(missing) == 0:

        print("✓ All sentiment columns created.")

    else:

        print("Missing Columns")

        print(missing)

    print(f"\nRows    : {len(df)}")

    print(f"Columns : {len(df.columns)}")


# ==========================================================
# MAIN
# ==========================================================

def main():

    category = get_category()

    df = load_dataset(category)

    df = hybrid_sentiment_analysis(df)

    sentiment_distribution(df)

    sentiment_by_university(df)

    sentiment_by_year(df)

    engagement_analysis(df)

    profanity_analysis(df)

    top_positive_comments(df)

    top_negative_comments(df)

    plot_sentiment_distribution(df)

    plot_profanity(df)

    final_validation(df)

    save_dataset(

        df,

        category

    )

    save_report(

        df,

        category

    )

    print("\n" + "=" * 60)

    print("HYBRID SENTIMENT ANALYSIS COMPLETED")

    print("=" * 60)


# ==========================================================
# DRIVER
# ==========================================================

if __name__ == "__main__":

    main()

## python analysis_comments/02_vader_sentiment.py