"""
===========================================================
05_feature_engineering.py

Feature Engineering for YouTube University Dataset

Creates numerical and textual features for
EDA, TF-IDF, LDA, VADER, Network Analysis
and statistical analysis.

Author : Ananya Upadhyay
===========================================================
"""

# ==========================================================
# IMPORT LIBRARIES
# ==========================================================

import os
import re
import math
import pandas as pd
from datetime import datetime

# ==========================================================
# WORD COUNT
# ==========================================================

def word_count(text):
    """
    Count total words.
    """

    if pd.isna(text):
        return 0

    return len(str(text).split())


# ==========================================================
# CHARACTER COUNT
# ==========================================================

def character_count(text):
    """
    Count total characters.
    """

    if pd.isna(text):
        return 0

    return len(str(text))


# ==========================================================
# EMOJI COUNT
# ==========================================================

def emoji_count(text):
    """
    Count emojis.
    """

    if pd.isna(text):
        return 0

    text = str(text)

    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"
        "\U0001F300-\U0001F5FF"
        "\U0001F680-\U0001F6FF"
        "\U0001F700-\U0001F77F"
        "\U0001F780-\U0001F7FF"
        "\U0001F800-\U0001F8FF"
        "\U0001F900-\U0001F9FF"
        "\U0001FA00-\U0001FAFF"
        "\U00002700-\U000027BF"
        "]+",
        flags=re.UNICODE
    )

    return len(emoji_pattern.findall(text))


# ==========================================================
# HASHTAG COUNT
# ==========================================================

def hashtag_count(text):
    """
    Count hashtags.
    """

    if pd.isna(text):
        return 0

    return len(re.findall(r"#\w+", str(text)))


# ==========================================================
# MENTION COUNT
# ==========================================================

def mention_count(text):
    """
    Count @mentions.
    """

    if pd.isna(text):
        return 0

    return len(re.findall(r"@\w+", str(text)))


# ==========================================================
# EXCLAMATION COUNT
# ==========================================================

def exclamation_count(text):
    """
    Count exclamation marks.
    """

    if pd.isna(text):
        return 0

    return str(text).count("!")


# ==========================================================
# QUESTION COUNT
# ==========================================================

def question_count(text):
    """
    Count question marks.
    """

    if pd.isna(text):
        return 0

    return str(text).count("?")


# ==========================================================
# LOG TRANSFORMATION
# ==========================================================

def safe_log(value):
    """
    log(1+x)
    Prevents log(0) error.
    """

    try:
        return math.log1p(float(value))
    except:
        return 0

# ==========================================================
# ENGAGEMENT RATE
# ==========================================================

def engagement_rate(views, likes, comments):
    """
    Calculate engagement rate.

    Formula:
    (Likes + Comments) / Views
    """

    try:

        views = float(views)
        likes = float(likes)
        comments = float(comments)

        if views == 0:
            return 0

        return round((likes + comments) / views, 6)

    except:
        return 0


# ==========================================================
# VIDEO AGE
# ==========================================================

def video_age_days(upload_date):
    """
    Calculate age of the video in days.
    """

    try:

        upload_date = str(upload_date)

        upload_date = upload_date[:10]

        upload_date = datetime.strptime(
            upload_date,
            "%Y-%m-%d"
        )

        today = datetime.today()

        return (today - upload_date).days

    except:

        return None


# ==========================================================
# MAIN PROGRAM
# ==========================================================

def main():

    print("=" * 60)
    print("FEATURE ENGINEERING")
    print("=" * 60)

    category = input(
        "\nEnter category "
        "(infrastructure, controversies, faculty_research, rankings): "
    ).strip().lower()

    INPUT_FILE = (
        f"cleaned_data/{category}_text_cleaned.csv"
    )

    OUTPUT_FILE = (
        f"cleaned_data/{category}_feature_engineered.csv"
    )

    if not os.path.exists(INPUT_FILE):

        print("\nInput file not found.")

        return

    print("\nLoading dataset...")

    df = pd.read_csv(INPUT_FILE)

    print(f"Loaded {len(df)} rows.")

    # ======================================================
    # TITLE FEATURES
    # ======================================================

    print("\nCreating title features...")

    df["title_word_count"] = df["clean_title"].apply(
        word_count
    )

    df["title_char_count"] = df["clean_title"].apply(
        character_count
    )

    # ======================================================
    # DESCRIPTION FEATURES
    # ======================================================

    print("Creating description features...")

    df["description_word_count"] = (
        df["clean_description"]
        .apply(word_count)
    )

    df["description_char_count"] = (
        df["clean_description"]
        .apply(character_count)
    )

    # ======================================================
    # COMBINED FEATURES
    # ======================================================

    print("Creating combined text features...")

    df["combined_word_count"] = (
        df["combined_text"]
        .apply(word_count)
    )

    df["combined_char_count"] = (
        df["combined_text"]
        .apply(character_count)
    )

    # ======================================================
    # SOCIAL MEDIA FEATURES
    # ======================================================

    print("Creating social media features...")

    df["emoji_count"] = (
        df["combined_text"]
        .apply(emoji_count)
    )

    df["hashtag_count"] = (
        df["combined_text"]
        .apply(hashtag_count)
    )

    df["mention_count"] = (
        df["combined_text"]
        .apply(mention_count)
    )

    df["exclamation_count"] = (
        df["combined_text"]
        .apply(exclamation_count)
    )

    df["question_count"] = (
        df["combined_text"]
        .apply(question_count)
    )

    # ======================================================
    # ENGAGEMENT FEATURES
    # ======================================================

    print("Creating engagement features...")

    df["engagement_rate"] = df.apply(
        lambda row: engagement_rate(
            row["views"],
            row["likes"],
            row["comment_count"]
        ),
        axis=1
    )

    df["log_views"] = (
        df["views"]
        .apply(safe_log)
    )

    df["log_likes"] = (
        df["likes"]
        .apply(safe_log)
    )

    df["log_comments"] = (
        df["comment_count"]
        .apply(safe_log)
    )

    # ======================================================
    # TIME FEATURES
    # ======================================================

    print("Creating time features...")

    df["video_age_days"] = (
        df["upload_date"]
        .apply(video_age_days)
    )

    # ======================================================
    # SAVE DATASET
    # ======================================================

    print("\nSaving dataset...")

    df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    # ======================================================
    # REPORT
    # ======================================================

    print("\n" + "=" * 60)
    print("FEATURE ENGINEERING REPORT")
    print("=" * 60)

    print(f"Category            : {category}")
    print(f"Rows                : {len(df)}")
    print(f"Columns             : {len(df.columns)}")

    print("\nNew Features Created")

    new_features = [

        "title_word_count",
        "title_char_count",

        "description_word_count",
        "description_char_count",

        "combined_word_count",
        "combined_char_count",

        "emoji_count",
        "hashtag_count",
        "mention_count",

        "exclamation_count",
        "question_count",

        "engagement_rate",

        "log_views",
        "log_likes",
        "log_comments",

        "video_age_days"

    ]

    for feature in new_features:
        print(f"✓ {feature}")

    print("\n" + "=" * 60)
    print("FEATURE SUMMARY")
    print("=" * 60)

    print(f"Average Words             : {df['combined_word_count'].mean():.2f}")
    print(f"Average Characters        : {df['combined_char_count'].mean():.2f}")
    print(f"Average Emojis            : {df['emoji_count'].mean():.2f}")
    print(f"Average Hashtags          : {df['hashtag_count'].mean():.2f}")
    print(f"Average Engagement Rate   : {df['engagement_rate'].mean():.6f}")

    print("\nSaved File")

    print(f"✓ {OUTPUT_FILE}")

    print("\nPreview")

    preview_columns = [

        "title",

        "combined_word_count",

        "emoji_count",

        "hashtag_count",

        "engagement_rate",

        "video_age_days"

    ]

    print(df[preview_columns].head())

    print("\n✅ Feature Engineering Completed Successfully.")


# ==========================================================
# DRIVER CODE
# ==========================================================

if __name__ == "__main__":
    main()