"""
============================================================
05_topic_modeling.py

Topic Modeling using LDA

Input:
    cleaned_data_comments/<category>_comment_sentiment.csv

Outputs

analysis_comments/
    tables/
    figures/
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

from gensim import corpora
from gensim.models import LdaModel

import pyLDAvis
import pyLDAvis.gensim_models

from resources.text_normalization import normalize_text


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
    print("TOPIC MODELING")
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

        f"Memory MB  : "

        f"{df.memory_usage(deep=True).sum()/(1024**2):.2f}"

    )

# ==========================================================
# PREPARE DOCUMENTS
# ==========================================================

def prepare_documents(df):

    print("\nPreparing documents...")

    documents = []

    for comment in df["combined_comment"].fillna(""):

        cleaned = normalize_text(comment)

        words = cleaned.split()

        if len(words) >= 3:

            documents.append(words)

    print(f"Documents Used : {len(documents)}")

    return documents


# ==========================================================
# CREATE DICTIONARY
# ==========================================================

def create_dictionary(documents):

    print("\nCreating dictionary...")

    dictionary = corpora.Dictionary(documents)

    # Remove very rare words
    # Remove very common words

    dictionary.filter_extremes(

        no_below=10,
        no_above=0.40

    )

    print(

        f"Vocabulary Size : {len(dictionary)}"

    )

    return dictionary


# ==========================================================
# CREATE CORPUS
# ==========================================================

def create_corpus(

    documents,

    dictionary

):

    print("\nCreating corpus...")

    corpus = [

        dictionary.doc2bow(doc)

        for doc in documents

    ]

    print(

        f"Corpus Size : {len(corpus)}"

    )

    return corpus

# ==========================================================
# TRAIN LDA MODEL
# ==========================================================

def train_lda_model(

    corpus,

    dictionary,

    num_topics=10

):

    print("\nTraining LDA Model...")

    lda_model = LdaModel(

        corpus=corpus,

        id2word=dictionary,

        num_topics=num_topics,

        random_state=42,

        chunksize=2000,

        passes=20,

        iterations=400,

        alpha="auto",

        eta="auto",

        per_word_topics=True

    )

    print("✓ LDA Model Trained Successfully")

    return lda_model
# ==========================================================
# EXTRACT TOPICS
# ==========================================================

def extract_topics(

    lda_model,

    num_words=15

):

    print("\n" + "=" * 60)

    print("DISCOVERED TOPICS")

    print("=" * 60)

    topic_rows = []

    for topic_id in range(

        lda_model.num_topics

    ):

        words = lda_model.show_topic(

            topic_id,

            topn=num_words

        )

        keywords = ", ".join(

            [

                word

                for word, score in words

            ]

        )

        print(f"\nTopic {topic_id + 1}")

        print(keywords)

        topic_rows.append({

            "Topic": topic_id + 1,

            "Keywords": keywords

        })

    topics_df = pd.DataFrame(

        topic_rows

    )

    topics_df.to_csv(

        os.path.join(

            TABLE_FOLDER,

            "overall_topics.csv"

        ),

        index=False

    )

    print("\n✓ overall_topics.csv Saved")

    return topics_df
# ==========================================================
# ASSIGN DOMINANT TOPIC
# ==========================================================

def assign_topics(

    df,

    lda_model,

    corpus

):

    print("\nAssigning Topics to Comments...")

    dominant_topics = []

    topic_probability = []

    for document in corpus:

        topics = lda_model.get_document_topics(

            document,

            minimum_probability=0

        )

        dominant = max(

            topics,

            key=lambda x: x[1]

        )

        dominant_topics.append(

            dominant[0] + 1

        )

        topic_probability.append(

            round(

                dominant[1],

                4

            )

        )

    df = df.iloc[:len(corpus)].copy()

    df["dominant_topic"] = dominant_topics

    df["topic_probability"] = topic_probability

    output_file = os.path.join(

        TABLE_FOLDER,

        "comments_with_topics.csv"

    )

    df.to_csv(

        output_file,

        index=False

    )

    print("✓ comments_with_topics.csv Saved")

    return df


# ==========================================================
# TOPIC DISTRIBUTION
# ==========================================================

def topic_distribution(df):

    print("\nGenerating Topic Distribution...")

    distribution = (

        df["dominant_topic"]

        .value_counts()

        .sort_index()

        .reset_index()

    )

    distribution.columns = [

        "Topic",

        "Comments"

    ]

    distribution["Percentage"] = (

        distribution["Comments"]

        / distribution["Comments"].sum()

        * 100

    ).round(2)

    distribution.to_csv(

        os.path.join(

            TABLE_FOLDER,

            "topic_distribution.csv"

        ),

        index=False

    )

    print(distribution)

    return distribution
# ==========================================================
# TOPIC VS UNIVERSITY
# ==========================================================

def topic_vs_university(df):

    print("\nGenerating Topic vs University...")

    table = pd.crosstab(

        df["university_name"],

        df["dominant_topic"]

    )

    table.to_csv(

        os.path.join(

            TABLE_FOLDER,

            "topic_vs_university.csv"

        )

    )

    print("✓ topic_vs_university.csv Saved")

    return table


# ==========================================================
# TOPIC VS SENTIMENT
# ==========================================================

def topic_vs_sentiment(df):

    print("\nGenerating Topic vs Sentiment...")

    table = pd.crosstab(

        df["sentiment"],

        df["dominant_topic"]

    )

    table.to_csv(

        os.path.join(

            TABLE_FOLDER,

            "topic_vs_sentiment.csv"

        )

    )

    print("✓ topic_vs_sentiment.csv Saved")

    return table
# ==========================================================
# TOPIC DISTRIBUTION PLOT
# ==========================================================

def plot_topic_distribution(distribution):

    print("\nGenerating Topic Distribution Plot...")

    plt.figure(figsize=(10,6))

    plt.bar(

        distribution["Topic"].astype(str),

        distribution["Comments"]

    )

    plt.xlabel("Topic")

    plt.ylabel("Number of Comments")

    plt.title("Topic Distribution")

    plt.tight_layout()

    plt.savefig(

        os.path.join(

            FIGURE_FOLDER,

            "topic_distribution.png"

        ),

        dpi=300

    )

    plt.close()

    print("✓ topic_distribution.png Saved")


# ==========================================================
# PYLDAVIS VISUALIZATION
# ==========================================================

def create_pyldavis(

    lda_model,

    corpus,

    dictionary

):

    print("\nGenerating Interactive Topic Visualization...")

    visualization = pyLDAvis.gensim_models.prepare(

        lda_model,

        corpus,

        dictionary

    )

    output_file = os.path.join(

        FIGURE_FOLDER,

        "topic_visualization.html"

    )

    pyLDAvis.save_html(

        visualization,

        output_file

    )

    print("✓ topic_visualization.html Saved")

# ==========================================================
# SAVE REPORT
# ==========================================================

def save_report(

    category,

    topics_df,

    distribution

):

    report = os.path.join(

        REPORT_FOLDER,

        f"{category}_topic_modeling_report.txt"

    )

    with open(

        report,

        "w",

        encoding="utf-8"

    ) as file:

        file.write("="*60+"\n")

        file.write("TOPIC MODELING REPORT\n")

        file.write("="*60+"\n\n")

        file.write("DISCOVERED TOPICS\n")

        file.write("---------------------------\n")

        file.write(

            topics_df.to_string(index=False)

        )

        file.write("\n\n")

        file.write("TOPIC DISTRIBUTION\n")

        file.write("---------------------------\n")

        file.write(

            distribution.to_string(index=False)

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

    print("✓ LDA Model")

    print("✓ Topics Extracted")

    print("✓ Topic Distribution")

    print("✓ Topic vs University")

    print("✓ Topic vs Sentiment")

    print("✓ CSV Tables")

    print("✓ Figures")

    print("✓ Interactive HTML")

    print("✓ Report")

# ==========================================================
# MAIN
# ==========================================================

def main():

    category = get_category()

    df = load_dataset(category)

    dataset_information(df)

    documents = prepare_documents(df)

    dictionary = create_dictionary(documents)

    corpus = create_corpus(

        documents,

        dictionary

    )

    lda_model = train_lda_model(

        corpus,

        dictionary,

        num_topics=10

    )

    topics_df = extract_topics(

        lda_model

    )

    df = assign_topics(

        df,

        lda_model,

        corpus

    )

    distribution = topic_distribution(df)

    topic_vs_university(df)

    topic_vs_sentiment(df)

    plot_topic_distribution(

        distribution

    )

    create_pyldavis(

        lda_model,

        corpus,

        dictionary

    )

    save_report(

        category,

        topics_df,

        distribution

    )

    final_validation()

    print("\n"+"="*60)

    print("TOPIC MODELING COMPLETED")

    print("="*60)


# ==========================================================
# DRIVER
# ==========================================================

if __name__ == "__main__":

    main()

#.   python analysis_comments/05_topic_modeling.py