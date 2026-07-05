"""
============================================================
04_tfidf.py

TF-IDF Keyword Analysis for YouTube Comments

Input:
    cleaned_data_comments/<category>_comment_sentiment.csv

Outputs:
    analysis_comments/tables/
    analysis_comments/figures/
    analysis_comments/reports/

Author : Ananya Upadhyay
============================================================
"""

import os
import sys

# ==========================================================
# ADD PROJECT ROOT TO PYTHON PATH
# ==========================================================

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

from sklearn.feature_extraction.text import TfidfVectorizer

from resources.text_normalization import (
    normalize_text,
    ALL_STOPWORDS
)


# ==========================================================
# FOLDERS
# ==========================================================

INPUT_FOLDER = "cleaned_data_comments"

TABLE_FOLDER = os.path.join(
    "analysis_comments",
    "tables"
)

FIGURE_FOLDER = os.path.join(
    "analysis_comments",
    "figures"
)

REPORT_FOLDER = os.path.join(
    "analysis_comments",
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
    print("TF-IDF KEYWORD ANALYSIS")
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
# PREPARE DOCUMENTS
# ==========================================================

def prepare_documents(df, sentiment=None):

    if sentiment is None:

        subset = df.copy()

    else:

        subset = df[
            df["sentiment"] == sentiment
        ].copy()

    documents = []

    for comment in subset["combined_comment"].fillna(""):

        cleaned = normalize_text(comment)

        if cleaned:

            documents.append(cleaned)

    return documents


# ==========================================================
# CALCULATE TF-IDF
# ==========================================================

def calculate_tfidf(

    documents,

    top_n=50

):

    vectorizer = TfidfVectorizer(

        stop_words=list(ALL_STOPWORDS),
        lowercase=True,

        max_features=1000,

        ngram_range=(1,2),

        min_df=5,

        max_df=0.90,

        sublinear_tf=True

    )

    tfidf_matrix = vectorizer.fit_transform(documents)

    feature_names = vectorizer.get_feature_names_out()

    scores = tfidf_matrix.mean(axis=0).A1

    tfidf_table = pd.DataFrame({

        "Keyword": feature_names,

        "TFIDF Score": scores

    })

    tfidf_table = tfidf_table.sort_values(

        by="TFIDF Score",

        ascending=False

    )

    return tfidf_table.head(top_n)


# ==========================================================
# SAVE TABLE
# ==========================================================

def save_table(

    table,

    filename

):

    table.to_csv(

        os.path.join(

            TABLE_FOLDER,

            filename

        ),

        index=False

    )


# ==========================================================
# PLOT BAR CHART
# ==========================================================

def plot_tfidf(

    table,

    title,

    filename

):

    plt.figure(

        figsize=(12,8)

    )

    plt.barh(

        table["Keyword"][::-1],

        table["TFIDF Score"][::-1]

    )

    plt.title(

        title,

        fontsize=16

    )

    plt.xlabel(

        "Average TF-IDF Score"

    )

    plt.tight_layout()

    plt.savefig(

        os.path.join(

            FIGURE_FOLDER,

            filename

        ),

        dpi=300,

        bbox_inches="tight"

    )

    plt.close()
# ==========================================================
# OVERALL TF-IDF
# ==========================================================

def overall_tfidf(df):

    print("\nGenerating Overall TF-IDF...")

    documents = prepare_documents(df)

    table = calculate_tfidf(documents)

    save_table(
        table,
        "overall_tfidf.csv"
    )

    plot_tfidf(
        table,
        "Overall TF-IDF Keywords",
        "overall_tfidf.png"
    )

    print("✓ Overall TF-IDF Completed")

    return table


# ==========================================================
# POSITIVE TF-IDF
# ==========================================================

def positive_tfidf(df):

    print("\nGenerating Positive TF-IDF...")

    documents = prepare_documents(
        df,
        "Positive"
    )

    table = calculate_tfidf(documents)

    save_table(
        table,
        "positive_tfidf.csv"
    )

    plot_tfidf(
        table,
        "Positive TF-IDF Keywords",
        "positive_tfidf.png"
    )

    print("✓ Positive TF-IDF Completed")

    return table


# ==========================================================
# NEGATIVE TF-IDF
# ==========================================================

def negative_tfidf(df):

    print("\nGenerating Negative TF-IDF...")

    documents = prepare_documents(
        df,
        "Negative"
    )

    table = calculate_tfidf(documents)

    save_table(
        table,
        "negative_tfidf.csv"
    )

    plot_tfidf(
        table,
        "Negative TF-IDF Keywords",
        "negative_tfidf.png"
    )

    print("✓ Negative TF-IDF Completed")

    return table


# ==========================================================
# NEUTRAL TF-IDF
# ==========================================================

def neutral_tfidf(df):

    print("\nGenerating Neutral TF-IDF...")

    documents = prepare_documents(
        df,
        "Neutral"
    )

    table = calculate_tfidf(documents)

    save_table(
        table,
        "neutral_tfidf.csv"
    )

    plot_tfidf(
        table,
        "Neutral TF-IDF Keywords",
        "neutral_tfidf.png"
    )

    print("✓ Neutral TF-IDF Completed")

    return table


# ==========================================================
# DISPLAY TOP KEYWORDS
# ==========================================================

def print_top_keywords(title, table):

    print("\n" + "=" * 60)

    print(title)

    print("=" * 60)

    print(table.head(20).to_string(index=False))

# ==========================================================
# UNIVERSITY TF-IDF
# ==========================================================

def university_tfidf(df):

    print("\nGenerating University-wise TF-IDF...")

    top_universities = (

        df["university_name"]

        .value_counts()

        .head(10)

        .index

    )

    summary = []

    for university in top_universities:

        print(f"\nProcessing : {university}")

        subset = df[
            df["university_name"] == university
        ]

        documents = prepare_documents(subset)

        table = calculate_tfidf(documents)

        filename = (

            university

            .replace(" ", "_")

            .replace("/", "_")

            .lower()

            + "_tfidf.csv"

        )

        figure = (

            university

            .replace(" ", "_")

            .replace("/", "_")

            .lower()

            + "_tfidf.png"

        )

        save_table(
            table,
            filename
        )

        plot_tfidf(
            table,
            f"{university} TF-IDF",
            figure
        )

        top_keyword = table.iloc[0]["Keyword"]

        top_score = table.iloc[0]["TFIDF Score"]

        summary.append({

            "University": university,

            "Top Keyword": top_keyword,

            "TFIDF Score": round(top_score,4)

        })

    summary = pd.DataFrame(summary)

    save_table(

        summary,

        "university_tfidf_summary.csv"

    )

    return summary


# ==========================================================
# SAVE REPORT
# ==========================================================

def save_report(

    category,

    overall,

    positive,

    negative,

    neutral,

    university_summary

):

    report = os.path.join(

        REPORT_FOLDER,

        f"{category}_tfidf_report.txt"

    )

    with open(

        report,

        "w",

        encoding="utf-8"

    ) as file:

        file.write("="*60+"\n")

        file.write("TF-IDF KEYWORD ANALYSIS REPORT\n")

        file.write("="*60+"\n\n")

        file.write("OVERALL TOP KEYWORDS\n")

        file.write("------------------------------\n")

        file.write(overall.to_string(index=False))

        file.write("\n\n")

        file.write("POSITIVE TOP KEYWORDS\n")

        file.write("------------------------------\n")

        file.write(positive.to_string(index=False))

        file.write("\n\n")

        file.write("NEGATIVE TOP KEYWORDS\n")

        file.write("------------------------------\n")

        file.write(negative.to_string(index=False))

        file.write("\n\n")

        file.write("NEUTRAL TOP KEYWORDS\n")

        file.write("------------------------------\n")

        file.write(neutral.to_string(index=False))

        file.write("\n\n")

        file.write("UNIVERSITY SUMMARY\n")

        file.write("------------------------------\n")

        file.write(

            university_summary.to_string(index=False)

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

    print("✓ Overall TF-IDF")

    print("✓ Positive TF-IDF")

    print("✓ Negative TF-IDF")

    print("✓ Neutral TF-IDF")

    print("✓ University TF-IDF")

    print("✓ CSV Tables")

    print("✓ Bar Charts")

    print("✓ Report")


# ==========================================================
# MAIN
# ==========================================================

def main():

    category = get_category()

    df = load_dataset(category)

    dataset_information(df)

    overall = overall_tfidf(df)

    positive = positive_tfidf(df)

    negative = negative_tfidf(df)

    neutral = neutral_tfidf(df)

    university_summary = university_tfidf(df)

    print_top_keywords(

        "OVERALL TOP KEYWORDS",

        overall

    )

    save_report(

        category,

        overall,

        positive,

        negative,

        neutral,

        university_summary

    )

    final_validation()

    print("\n"+"="*60)

    print("TF-IDF ANALYSIS COMPLETED")

    print("="*60)


# ==========================================================
# DRIVER
# ==========================================================

if __name__ == "__main__":

    main()


#.    python analysis_comments/04_tfidf.py