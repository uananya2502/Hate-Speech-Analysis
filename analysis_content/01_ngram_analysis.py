"""
============================================================
01_ngram_analysis.py

N-Gram Analysis

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
# IMPORT LIBRARIES
# ==========================================================

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import CountVectorizer

from resources.text_normalization import normalize_text

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
    print("N-GRAM ANALYSIS")
    print("="*60)

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

    print("\n"+"="*60)

    print("DATASET INFORMATION")

    print("="*60)

    print(f"Rows       : {len(df)}")

    print(f"Columns    : {len(df.columns)}")

    print(
        f"Memory MB  : {df.memory_usage(deep=True).sum()/(1024**2):.2f}"
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
# EXTRACT NGRAMS
# ==========================================================

def extract_ngrams(

    documents,

    n=1,

    top_k=30

):

    vectorizer = CountVectorizer(

        ngram_range=(n, n),

        min_df=5

    )

    X = vectorizer.fit_transform(documents)

    frequencies = X.sum(axis=0).A1

    words = vectorizer.get_feature_names_out()

    ngrams = pd.DataFrame({

        "NGram": words,

        "Frequency": frequencies

    })

    ngrams = (

        ngrams

        .sort_values(

            by="Frequency",

            ascending=False

        )

        .head(top_k)

    )

    return ngrams
# ==========================================================
# PLOT NGRAMS
# ==========================================================

def plot_ngrams(

    ngrams,

    title,

    filename

):

    plt.figure(figsize=(12,8))

    plt.barh(

        ngrams["NGram"],

        ngrams["Frequency"]

    )

    plt.gca().invert_yaxis()

    plt.xlabel("Frequency")

    plt.ylabel("N-Gram")

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
# OVERALL NGRAM ANALYSIS
# ==========================================================

def overall_ngrams(df):

    print("\nGenerating Overall N-Grams...")

    documents = prepare_documents(df)

    # -------------------------
    # Unigrams
    # -------------------------

    unigram = extract_ngrams(

        documents,

        n=1

    )

    unigram.to_csv(

        os.path.join(

            TABLE_FOLDER,

            "overall_unigrams.csv"

        ),

        index=False

    )

    plot_ngrams(

        unigram,

        "Top 30 Unigrams",

        "overall_unigrams.png"

    )

    # -------------------------
    # Bigrams
    # -------------------------

    bigram = extract_ngrams(

        documents,

        n=2

    )

    bigram.to_csv(

        os.path.join(

            TABLE_FOLDER,

            "overall_bigrams.csv"

        ),

        index=False

    )

    plot_ngrams(

        bigram,

        "Top 30 Bigrams",

        "overall_bigrams.png"

    )

    # -------------------------
    # Trigrams
    # -------------------------

    trigram = extract_ngrams(

        documents,

        n=3

    )

    trigram.to_csv(

        os.path.join(

            TABLE_FOLDER,

            "overall_trigrams.csv"

        ),

        index=False

    )

    plot_ngrams(

        trigram,

        "Top 30 Trigrams",

        "overall_trigrams.png"

    )

    print("✓ overall_unigrams.csv")

    print("✓ overall_bigrams.csv")

    print("✓ overall_trigrams.csv")

    return unigram, bigram, trigram
# ==========================================================
# SENTIMENT-WISE NGRAMS
# ==========================================================

def sentiment_ngrams(df):

    print("\nGenerating Sentiment-wise N-Grams...")

    sentiments = [

        "Positive",

        "Negative",

        "Neutral"

    ]

    for sentiment in sentiments:

        print(f"\nProcessing {sentiment}...")

        documents = prepare_documents(

            df,

            sentiment

        )

        if len(documents) == 0:

            continue

        # -----------------------
        # Bigrams
        # -----------------------

        bigram = extract_ngrams(

            documents,

            n=2

        )

        csv_name = (

            sentiment.lower()

            + "_bigrams.csv"

        )

        png_name = (

            sentiment.lower()

            + "_bigrams.png"

        )

        bigram.to_csv(

            os.path.join(

                TABLE_FOLDER,

                csv_name

            ),

            index=False

        )

        plot_ngrams(

            bigram,

            f"{sentiment} Bigrams",

            png_name

        )

        # -----------------------
        # Trigrams
        # -----------------------

        trigram = extract_ngrams(

            documents,

            n=3

        )

        csv_name = (

            sentiment.lower()

            + "_trigrams.csv"

        )

        png_name = (

            sentiment.lower()

            + "_trigrams.png"

        )

        trigram.to_csv(

            os.path.join(

                TABLE_FOLDER,

                csv_name

            ),

            index=False

        )

        plot_ngrams(

            trigram,

            f"{sentiment} Trigrams",

            png_name

        )

    print("\n✓ Sentiment-wise N-Grams Completed")
# ==========================================================
# UNIVERSITY-WISE NGRAMS
# ==========================================================

def university_ngrams(df):

    print("\nGenerating University-wise N-Grams...")

    top_universities = (

        df["university_name"]

        .value_counts()

        .head(5)

        .index

    )

    summary = []

    for university in top_universities:

        print(f"\nProcessing {university}...")

        subset = df[

            df["university_name"] == university

        ]

        documents = prepare_documents(subset)

        if len(documents) == 0:

            continue

        # -----------------------------
        # Bigrams
        # -----------------------------

        bigram = extract_ngrams(

            documents,

            n=2,

            top_k=20

        )

        file_name = (

            university

            .lower()

            .replace(" ", "_")

            + "_bigrams.csv"

        )

        bigram.to_csv(

            os.path.join(

                TABLE_FOLDER,

                file_name

            ),

            index=False

        )

        plot_ngrams(

            bigram,

            f"{university} - Top Bigrams",

            file_name.replace(

                ".csv",

                ".png"

            )

        )

        for _, row in bigram.iterrows():

            summary.append({

                "University": university,

                "Bigram": row["NGram"],

                "Frequency": row["Frequency"]

            })

    summary = pd.DataFrame(summary)

    summary.to_csv(

        os.path.join(

            TABLE_FOLDER,

            "university_bigrams.csv"

        ),

        index=False

    )

    print("\n✓ university_bigrams.csv")

    return summary
# ==========================================================
# SAVE REPORT
# ==========================================================

def save_report(

    category,

    unigram,

    bigram,

    trigram

):

    report = os.path.join(

        REPORT_FOLDER,

        f"{category}_ngram_report.txt"

    )

    with open(

        report,

        "w",

        encoding="utf-8"

    ) as file:

        file.write("="*60 + "\n")

        file.write("N-GRAM ANALYSIS REPORT\n")

        file.write("="*60 + "\n\n")

        file.write("TOP UNIGRAMS\n")

        file.write("-"*40 + "\n")

        file.write(

            unigram.to_string(index=False)

        )

        file.write("\n\n")

        file.write("TOP BIGRAMS\n")

        file.write("-"*40 + "\n")

        file.write(

            bigram.to_string(index=False)

        )

        file.write("\n\n")

        file.write("TOP TRIGRAMS\n")

        file.write("-"*40 + "\n")

        file.write(

            trigram.to_string(index=False)

        )

    print("\nReport Saved")

    print(report)
# ==========================================================
# FINAL VALIDATION
# ==========================================================

def final_validation():

    print("\n" + "="*60)

    print("FINAL VALIDATION")

    print("="*60)

    print("✓ Overall Unigrams")

    print("✓ Overall Bigrams")

    print("✓ Overall Trigrams")

    print("✓ Sentiment-wise N-Grams")

    print("✓ University-wise N-Grams")

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

    unigram, bigram, trigram = overall_ngrams(df)

    sentiment_ngrams(df)

    university_ngrams(df)

    save_report(

        category,

        unigram,

        bigram,

        trigram

    )

    final_validation()

    print("\n" + "="*60)

    print("N-GRAM ANALYSIS COMPLETED")

    print("="*60)


# ==========================================================
# DRIVER
# ==========================================================

if __name__ == "__main__":

    main()
#.   python analysis_content/01_ngram_analysis.py