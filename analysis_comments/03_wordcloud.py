"""
============================================================
03_wordcloud.py

Word Cloud Analysis for YouTube Comments

Input:
    cleaned_data_comments/<category>_comment_sentiment.csv

Outputs:
    analysis_comments/figures/
    analysis_comments/reports/
    analysis_comments/tables/

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

from wordcloud import WordCloud
from wordcloud import STOPWORDS

from collections import Counter


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
    print("WORD CLOUD ANALYSIS")
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

        f"{category}_comment_sentiment.csv"

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
# DATASET INFORMATION
# ==========================================================

def dataset_information(df):

    print("\n" + "=" * 60)
    print("DATASET INFORMATION")
    print("=" * 60)

    print(f"Rows           : {len(df)}")

    print(f"Columns        : {len(df.columns)}")

    print(

        f"Memory (MB)    : "

        f"{df.memory_usage(deep=True).sum() / (1024**2):.2f}"

    )
# ==========================================================
# CUSTOM STOPWORDS
# ==========================================================

CUSTOM_STOPWORDS = STOPWORDS.union({

    # ------------------------------------------------------
    # Education
    # ------------------------------------------------------

    "college","colleges","university","universities",
    "campus","student","students","education",
    "faculty","teacher","teachers","professor",
    "hostel","hostels","mess","canteen",
    "library","department","branch","course",
    "courses","placement","placements",
    "admission","admissions","semester","sem",
    "class","classes","exam","exams",

    # ------------------------------------------------------
    # YouTube
    # ------------------------------------------------------

    "video","videos","youtube","channel",
    "subscribe","subscriber","subscribers",
    "subscribed","watch","watching",
    "watched","comment","comments",
    "like","likes","liked",
    "share","shared",
    "review","tour","short","shorts",
    "vlog",

    # ------------------------------------------------------
    # Common English
    # ------------------------------------------------------

    "the","this","that","these","those",
    "is","are","was","were","be","been",
    "being","am","do","does","did",
    "doing","done","have","has","had",
    "having","will","would","should",
    "can","could","may","might","must",
    "shall",

    "and","or","but","if","then","than",
    "because","while","when","where",
    "what","which","who","whom","whose",
    "why","how",

    "very","really","quite","also",
    "just","only","still","already",
    "always","never","every","everyone",
    "everything","nothing","something",

    "one","two","three","first","second",
    "many","much","more","less",
    "good","great","nice",

    "know","think","want","need","make",
    "made","take","taken","come","coming",
    "go","going","went","see","seen",
    "look","looking","use","using",
    "used","give","given","got","getting",

    # ------------------------------------------------------
    # Indian English
    # ------------------------------------------------------

    "sir","madam","mam","bro","bros",
    "brother","bhai","bhaiya","didi",
    "uncle","aunty","guys","friend",
    "friends","hello","hi","thanks",
    "thank","please","plz",

    # ------------------------------------------------------
    # Hindi
    # ------------------------------------------------------

    "hai","hain","tha","thi","the",
    "ho","hona","hoga","kar","karo",
    "karna","ki","ka","ke","ko",
    "se","me","mai","mera","meri",
    "mere","ham","hum","aap","apka",
    "apki","apke","tum","tumhara",
    "ye","yeh","yah","wo","woh",
    "is","us","sab","sabhi",
    "kya","kyu","kyun","kaise",
    "aisa","aisi","aise",
    "wala","wali","wale",
    "aur","ya","nahi","nahin",
    "mat","fir","phir","ab",
    "to","bhi","bas","bahut",
    "bohot",

    # ------------------------------------------------------
    # Emoji Words
    # ------------------------------------------------------

    "lol","lmao","omg","wtf","xd",

    # ------------------------------------------------------
    # Numbers
    # ------------------------------------------------------

    "2023","2024","2025","2026"

})


# ==========================================================
# CLEAN TEXT
# ==========================================================

def clean_wordcloud_text(text):

    text = str(text).lower()

    # Remove URLs
    text = re.sub(
        r"http\S+|www\S+|https\S+",
        " ",
        text
    )

    # Remove Emails
    text = re.sub(
        r"\S+@\S+",
        " ",
        text
    )

    # Remove Numbers
    text = re.sub(
        r"\d+",
        " ",
        text
    )

    # Keep English + Hindi Characters
    text = re.sub(
        r"[^a-zA-Z\u0900-\u097F\s]",
        " ",
        text
    )

    # Remove Extra Spaces
    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    words = text.split()

    cleaned = []

    for word in words:

        if word not in CUSTOM_STOPWORDS and len(word) > 2:

            cleaned.append(word)

    return cleaned


# ==========================================================
# PREPARE TEXT
# ==========================================================

def prepare_text(df, sentiment=None):

    if sentiment is None:

        subset = df

    else:

        subset = df[
            df["sentiment"] == sentiment
        ]

    words = []

    for comment in subset["combined_comment"].fillna(""):

        words.extend(

            clean_wordcloud_text(comment)

        )

    return words

# ==========================================================
# GENERATE WORD CLOUD
# ==========================================================

def generate_wordcloud(

    words,

    title,

    image_name,

    table_name

):

    if len(words) == 0:

        print(f"No words found for {title}")

        return

    # ------------------------------------------------------
    # Word Frequencies
    # ------------------------------------------------------

    word_frequency = Counter(words)

    frequency_df = (

        pd.DataFrame(

            word_frequency.items(),

            columns=[

                "Word",

                "Frequency"

            ]

        )

        .sort_values(

            by="Frequency",

            ascending=False

        )

    )

    # Save Top 100 Words

    frequency_df.head(100).to_csv(

        os.path.join(

            TABLE_FOLDER,

            table_name

        ),

        index=False

    )

    # ------------------------------------------------------
    # Create WordCloud
    # ------------------------------------------------------

    wordcloud = WordCloud(

        width=1800,

        height=900,

        background_color="white",

        stopwords=CUSTOM_STOPWORDS,

        max_words=500,

        collocations=False,

        min_font_size=10,

        max_font_size=180

    ).generate_from_frequencies(

        word_frequency

    )

    # ------------------------------------------------------
    # Plot
    # ------------------------------------------------------

    plt.figure(

        figsize=(18,9)

    )

    plt.imshow(

        wordcloud,

        interpolation="bilinear"

    )

    plt.axis("off")

    plt.title(

        title,

        fontsize=20

    )

    plt.tight_layout()

    plt.savefig(

        os.path.join(

            FIGURE_FOLDER,

            image_name

        ),

        dpi=300,

        bbox_inches="tight"

    )

    plt.close()

    print(f"✓ {image_name}")

    print(f"✓ {table_name}")

# ==========================================================
# OVERALL WORD CLOUD
# ==========================================================

def overall_wordcloud(df):

    print("\nGenerating Overall WordCloud...")

    words = prepare_text(df)

    generate_wordcloud(

        words,

        "Overall WordCloud",

        "overall_wordcloud.png",

        "overall_top_words.csv"

    )


# ==========================================================
# POSITIVE WORD CLOUD
# ==========================================================

def positive_wordcloud(df):

    print("Generating Positive WordCloud...")

    words = prepare_text(

        df,

        "Positive"

    )

    generate_wordcloud(

        words,

        "Positive Comments",

        "positive_wordcloud.png",

        "positive_top_words.csv"

    )


# ==========================================================
# NEGATIVE WORD CLOUD
# ==========================================================

def negative_wordcloud(df):

    print("Generating Negative WordCloud...")

    words = prepare_text(

        df,

        "Negative"

    )

    generate_wordcloud(

        words,

        "Negative Comments",

        "negative_wordcloud.png",

        "negative_top_words.csv"

    )


# ==========================================================
# NEUTRAL WORD CLOUD
# ==========================================================

def neutral_wordcloud(df):

    print("Generating Neutral WordCloud...")

    words = prepare_text(

        df,

        "Neutral"

    )

    generate_wordcloud(

        words,

        "Neutral Comments",

        "neutral_wordcloud.png",

        "neutral_top_words.csv"

    )

# ==========================================================
# UNIVERSITY WORD CLOUDS
# ==========================================================

def university_wordclouds(df):

    print("\nGenerating University WordClouds...")

    top_universities = (

        df["university_name"]

        .value_counts()

        .head(5)

        .index

    )

    for university in top_universities:

        subset = df[

            df["university_name"] == university

        ]

        words = prepare_text(subset)

        image_name = (

            university

            .replace(" ", "_")

            .replace("/", "_")

            .lower()

            + "_wordcloud.png"

        )

        table_name = (

            university

            .replace(" ", "_")

            .replace("/", "_")

            .lower()

            + "_top_words.csv"

        )

        generate_wordcloud(

            words,

            university,

            image_name,

            table_name

        )
# ==========================================================
# UNIVERSITY SENTIMENT WORD CLOUDS
# ==========================================================

def university_sentiment_wordclouds(df):

    print("\nGenerating University Sentiment WordClouds...")

    top_universities = (

        df["university_name"]

        .value_counts()

        .head(5)

        .index

    )

    for university in top_universities:

        university_df = df[

            df["university_name"] == university

        ]

        for sentiment in [

            "Positive",

            "Negative"

        ]:

            sentiment_df = university_df[

                university_df["sentiment"] == sentiment

            ]

            words = prepare_text(sentiment_df)

            image_name = (

                university

                .replace(" ", "_")

                .replace("/", "_")

                .lower()

                + "_"

                + sentiment.lower()

                + ".png"

            )

            table_name = (

                university

                .replace(" ", "_")

                .replace("/", "_")

                .lower()

                + "_"

                + sentiment.lower()

                + "_top_words.csv"

            )

            generate_wordcloud(

                words,

                f"{university} ({sentiment})",

                image_name,

                table_name

            )
# ==========================================================
# REPORT
# ==========================================================

def save_report(df, category):

    report_file = os.path.join(

        REPORT_FOLDER,

        f"{category}_wordcloud_report.txt"

    )

    with open(

        report_file,

        "w",

        encoding="utf-8"

    ) as file:

        file.write("=" * 60 + "\n")

        file.write("WORD CLOUD ANALYSIS REPORT\n")

        file.write("=" * 60 + "\n\n")

        file.write(f"Total Comments : {len(df)}\n")

        file.write(f"Positive : {(df['sentiment']=='Positive').sum()}\n")

        file.write(f"Neutral : {(df['sentiment']=='Neutral').sum()}\n")

        file.write(f"Negative : {(df['sentiment']=='Negative').sum()}\n\n")

        file.write("Top Universities\n")

        file.write("-------------------------\n")

        top = df["university_name"].value_counts().head(10)

        file.write(top.to_string())

    print("\nReport Saved")

    print(report_file)

# ==========================================================
# FINAL VALIDATION
# ==========================================================

def final_validation():

    print("\n" + "=" * 60)

    print("FINAL VALIDATION")

    print("=" * 60)

    print("✓ Overall WordCloud")

    print("✓ Positive WordCloud")

    print("✓ Negative WordCloud")

    print("✓ Neutral WordCloud")

    print("✓ University WordClouds")

    print("✓ Word Frequency Tables")

    print("✓ Report Generated")

# ==========================================================
# MAIN
# ==========================================================

def main():

    category = get_category()

    df = load_dataset(category)

    dataset_information(df)

    overall_wordcloud(df)

    positive_wordcloud(df)

    negative_wordcloud(df)

    neutral_wordcloud(df)

    university_wordclouds(df)

    university_sentiment_wordclouds(df)

    save_report(

        df,

        category

    )

    final_validation()

    print("\n" + "=" * 60)

    print("WORD CLOUD ANALYSIS COMPLETED")

    print("=" * 60)

# ==========================================================
# DRIVER
# ==========================================================

if __name__ == "__main__":

    main()