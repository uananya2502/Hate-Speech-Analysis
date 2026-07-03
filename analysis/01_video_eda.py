"""
============================================================
01_video_eda.py

Exploratory Data Analysis
YouTube University Dataset

Input:
    analysis_data/<category>_feature_engineered.csv

Outputs:
    results/<category>/

        figures/
        tables/
        summary_report.txt

Author : Ananya Upadhyay
============================================================
"""

# ==========================================================
# IMPORT LIBRARIES
# ==========================================================

import os

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt


# ==========================================================
# CONFIGURATION
# ==========================================================

ANALYSIS_FOLDER = "analysis_data"

RESULT_FOLDER = "results"


# ==========================================================
# CREATE OUTPUT FOLDERS
# ==========================================================

def create_output_folders(category):

    base = os.path.join(
        RESULT_FOLDER,
        category
    )

    figures = os.path.join(
        base,
        "figures"
    )

    tables = os.path.join(
        base,
        "tables"
    )

    os.makedirs(
        figures,
        exist_ok=True
    )

    os.makedirs(
        tables,
        exist_ok=True
    )

    return base, figures, tables


# ==========================================================
# DATASET SELECTION
# ==========================================================

def get_category():

    print("=" * 60)
    print("VIDEO EXPLORATORY DATA ANALYSIS")
    print("=" * 60)

    print("\nAvailable Categories\n")

    print("1. infrastructure")

    print("2. controversies")

    print("3. faculty_research")

    print("4. rankings")

    choice = input(
        "\nChoose category : "
    ).strip()

    mapping = {

        "1": "infrastructure",

        "2": "controversies",

        "3": "faculty_research",

        "4": "rankings"

    }

    if choice not in mapping:

        print("\nInvalid Choice.")

        exit()

    return mapping[choice]


# ==========================================================
# LOAD DATASET
# ==========================================================

def load_dataset(category):

    file_path = os.path.join(

        ANALYSIS_FOLDER,

        f"{category}_feature_engineered.csv"

    )

    if not os.path.exists(file_path):

        print("\nDataset not found.")

        print(file_path)

        exit()

    df = pd.read_csv(file_path)

    print("\nDataset Loaded Successfully.")

    print(f"Rows    : {len(df)}")

    print(f"Columns : {len(df.columns)}")

    return df


# ==========================================================
# SAVE TABLE
# ==========================================================

def save_table(df, filename, table_folder):

    output = os.path.join(

        table_folder,

        filename

    )

    df.to_csv(

        output,

        index=False

    )

    print(f"Saved : {filename}")


# ==========================================================
# SAVE FIGURE
# ==========================================================

def save_figure(filename, figure_folder):

    output = os.path.join(

        figure_folder,

        filename

    )

    plt.tight_layout()

    plt.savefig(

        output,

        dpi=300,

        bbox_inches="tight"

    )

    plt.close()

    print(f"Saved : {filename}")

# ==========================================================
# DATASET OVERVIEW
# ==========================================================

def dataset_overview(df):

    print("\n" + "=" * 60)
    print("DATASET OVERVIEW")
    print("=" * 60)

    print(f"Rows              : {df.shape[0]}")
    print(f"Columns           : {df.shape[1]}")
    print(f"Memory Usage (MB) : {round(df.memory_usage(deep=True).sum()/1024**2,2)}")

    print("\nColumn Names\n")

    for column in df.columns:

        print(column)


# ==========================================================
# DATA TYPES
# ==========================================================

def datatype_summary(df, table_folder):

    print("\n" + "=" * 60)
    print("DATA TYPES")
    print("=" * 60)

    datatype = pd.DataFrame({

        "Column": df.columns,

        "Datatype": df.dtypes.astype(str)

    })

    print(datatype)

    save_table(

        datatype,

        "data_types.csv",

        table_folder

    )


# ==========================================================
# MISSING VALUE ANALYSIS
# ==========================================================

def missing_value_analysis(df, table_folder):

    print("\n" + "=" * 60)
    print("MISSING VALUE ANALYSIS")
    print("=" * 60)

    missing = pd.DataFrame({

        "Column": df.columns,

        "Missing Values": df.isnull().sum(),

        "Missing %": (

            df.isnull().sum()

            / len(df)

            * 100

        ).round(2)

    })
    missing = missing[
        missing["Missing Values"] > 0
    ]

    print(missing)

    save_table(

        missing,

        "missing_values.csv",

        table_folder

    )


# ==========================================================
# DESCRIPTIVE STATISTICS
# ==========================================================

def descriptive_statistics(df, table_folder):

    print("\n" + "=" * 60)
    print("DESCRIPTIVE STATISTICS")
    print("=" * 60)

    numeric = df.select_dtypes(

        include=np.number

    )

    stats = numeric.describe().T

    stats["median"] = numeric.median()

    stats["variance"] = numeric.var()

    stats = stats.round(2)

    print(stats)

    save_table(

        stats,

        "descriptive_statistics.csv",

        table_folder

    )


# ==========================================================
# SUMMARY REPORT
# ==========================================================

def write_basic_summary(

        df,

        category,

        report_file

):

    with open(

        report_file,

        "w",

        encoding="utf-8"

    ) as file:

        file.write(

            "=" * 60 + "\n"

        )

        file.write(

            "VIDEO EDA SUMMARY REPORT\n"

        )

        file.write(

            "=" * 60 + "\n\n"

        )

        file.write(

            f"Category : {category}\n\n"

        )

        file.write(

            f"Rows : {len(df)}\n"

        )

        file.write(

            f"Columns : {len(df.columns)}\n"

        )

        file.write(

            f"Memory Usage : "

            f"{round(df.memory_usage(deep=True).sum()/1024**2,2)} MB\n\n"

        )

        file.write(

            f"Unique Universities : "

            f"{df['university_name'].nunique()}\n"

        )

        if "channel_name" in df.columns:

            file.write(

                f"Unique Channels : "

                f"{df['channel_name'].nunique()}\n"

            )

        file.write("\n")

        file.write("Missing Values\n")

        file.write("------------------------\n")

        file.write(

            str(

                df.isnull().sum()

            )

        )

        file.write("\n\n")

    print("\nSummary report created.")

# ==========================================================
# UNIVERSITY ANALYSIS
# ==========================================================

def university_analysis(df, table_folder):

    print("\n" + "=" * 60)
    print("UNIVERSITY ANALYSIS")
    print("=" * 60)

    university = (

        df["university_name"]

        .value_counts()

        .reset_index()

    )

    university.columns = [

        "University",

        "Videos"

    ]

    print(university.head(20))

    save_table(

        university,

        "top_universities.csv",

        table_folder

    )


# ==========================================================
# CHANNEL ANALYSIS
# ==========================================================

def channel_analysis(df, table_folder):

    if "channel_name" not in df.columns:

        return

    print("\n" + "=" * 60)
    print("CHANNEL ANALYSIS")
    print("=" * 60)

    channels = (

        df["channel_name"]

        .value_counts()

        .reset_index()

    )

    channels.columns = [

        "Channel",

        "Videos"

    ]

    print(channels.head(20))

    save_table(

        channels,

        "top_channels.csv",

        table_folder

    )


# ==========================================================
# TOP VIEWED VIDEOS
# ==========================================================

def top_viewed(df, table_folder):

    print("\n" + "=" * 60)
    print("TOP VIEWED VIDEOS")
    print("=" * 60)

    top = (

        df.sort_values(

            "views",

            ascending=False

        )

        .head(20)

    )

    print(

        top[

            [

                "title",

                "university_name",

                "views"

            ]

        ]

    )

    save_table(

        top,

        "top_viewed.csv",

        table_folder

    )


# ==========================================================
# TOP LIKED VIDEOS
# ==========================================================

def top_liked(df, table_folder):

    print("\n" + "=" * 60)
    print("TOP LIKED VIDEOS")
    print("=" * 60)

    top = (

        df.sort_values(

            "likes",

            ascending=False

        )

        .head(20)

    )

    print(

        top[

            [

                "title",

                "university_name",

                "likes"

            ]

        ]

    )

    save_table(

        top,

        "top_liked.csv",

        table_folder

    )


# ==========================================================
# TOP COMMENTED VIDEOS
# ==========================================================

def top_commented(df, table_folder):

    print("\n" + "=" * 60)
    print("TOP COMMENTED VIDEOS")
    print("=" * 60)

    top = (

        df.sort_values(

            "comment_count",

            ascending=False

        )

        .head(20)

    )

    print(

        top[

            [

                "title",

                "university_name",

                "comment_count"

            ]

        ]

    )

    save_table(

        top,

        "top_commented.csv",

        table_folder

    )


# ==========================================================
# HIGHEST ENGAGEMENT
# ==========================================================

def highest_engagement(df, table_folder):

    if "engagement_rate" not in df.columns:

        return

    print("\n" + "=" * 60)
    print("HIGHEST ENGAGEMENT VIDEOS")
    print("=" * 60)

    top = (

        df.sort_values(

            "engagement_rate",

            ascending=False

        )

        .head(20)

    )

    print(

        top[

            [

                "title",

                "university_name",

                "engagement_rate"

            ]

        ]

    )

    save_table(

        top,

        "highest_engagement.csv",

        table_folder

    )


# ==========================================================
# UPDATE SUMMARY REPORT
# ==========================================================

def append_dataset_summary(

        df,

        report_file

):

    with open(

        report_file,

        "a",

        encoding="utf-8"

    ) as file:

        file.write("=" * 60 + "\n")

        file.write("DATASET SUMMARY\n")

        file.write("=" * 60 + "\n\n")

        file.write(

            f"Total Universities : "

            f"{df['university_name'].nunique()}\n"

        )

        if "channel_name" in df.columns:

            file.write(

                f"Total Channels : "

                f"{df['channel_name'].nunique()}\n"

            )

        file.write(

            f"\nAverage Views : "

            f"{round(df['views'].mean(),2)}\n"

        )

        file.write(

            f"Average Likes : "

            f"{round(df['likes'].mean(),2)}\n"

        )

        file.write(

            f"Average Comments : "

            f"{round(df['comment_count'].mean(),2)}\n"

        )

        if "engagement_rate" in df.columns:

            file.write(

                f"Average Engagement : "

                f"{round(df['engagement_rate'].mean(),5)}\n"

            )

        file.write("\n")
# ==========================================================
# UNIVERSITY DISTRIBUTION
# ==========================================================

def plot_university_distribution(df, figure_folder):

    university = (
        df["university_name"]
        .value_counts()
        .head(20)
        .sort_values()
    )

    plt.figure(figsize=(12,6))

    university.plot(kind="bar")

    plt.title("Top 20 Universities by Number of Videos")

    plt.xlabel("University")

    plt.ylabel("Number of Videos")

    plt.xticks(rotation=75)

    save_figure(
        "university_distribution.png",
        figure_folder
    )


# ==========================================================
# CHANNEL DISTRIBUTION
# ==========================================================

def plot_channel_distribution(df, figure_folder):

    if "channel_name" not in df.columns:
        return

    channels = (
        df["channel_name"]
        .value_counts()
        .head(20)
        .sort_values()
    )

    plt.figure(figsize=(12,6))

    channels.plot(kind="bar")

    plt.title("Top 20 Channels")

    plt.xlabel("Channel")

    plt.ylabel("Videos")

    plt.xticks(rotation=75)

    save_figure(
        "channel_distribution.png",
        figure_folder
    )


# ==========================================================
# HISTOGRAM
# ==========================================================

def histogram(df, column, title, filename, figure_folder):

    if column not in df.columns:
        return

    plt.figure(figsize=(10,6))

    data = np.log1p(df[column])

    plt.hist(
        data,
        bins=40
    )

    plt.title(title)

    plt.xlabel(f"Log(1 + {column})")

    plt.ylabel("Frequency")

    save_figure(
        filename,
        figure_folder
    )


# ==========================================================
# BOXPLOT
# ==========================================================

def boxplot(df, column, title, filename, figure_folder):

    if column not in df.columns:
        return

    plt.figure(figsize=(8,5))

    plt.boxplot(df[column].dropna())

    plt.title(title)

    save_figure(
        filename,
        figure_folder
    )


# ==========================================================
# UPLOAD TREND
# ==========================================================

def upload_trend(df, figure_folder):

    if "upload_date" not in df.columns:
        return

    temp = df.copy()

    temp["upload_date"] = pd.to_datetime(
        temp["upload_date"],
        errors="coerce"
    )

    yearly = (
        temp["upload_date"]
        .dt.year
        .value_counts()
        .sort_index()
    )

    plt.figure(figsize=(10,6))

    yearly.plot(marker="o")

    plt.title("Video Upload Trend")

    plt.xlabel("Year")

    plt.ylabel("Number of Videos")

    save_figure(
        "upload_trend.png",
        figure_folder
    )

# ==========================================================
# CORRELATION HEATMAP
# ==========================================================

def correlation_heatmap(df, figure_folder):

    numeric = df.select_dtypes(include=np.number)

    corr = numeric.corr()

    fig, ax = plt.subplots(figsize=(12,10))

    im = ax.imshow(corr)

    ax.set_xticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=90)

    ax.set_yticks(range(len(corr.columns)))
    ax.set_yticklabels(corr.columns)

    for i in range(len(corr.columns)):
        for j in range(len(corr.columns)):
            ax.text(
                j,
                i,
                f"{corr.iloc[i,j]:.2f}",
                ha="center",
                va="center",
                fontsize=7
            )

    plt.colorbar(im)

    plt.title("Correlation Heatmap")

    save_figure(
        "correlation_heatmap.png",
        figure_folder
    )


# ==========================================================
# ALL HISTOGRAMS
# ==========================================================

def create_all_histograms(df, figure_folder):

    histogram(
        df,
        "views",
        "Views Distribution",
        "views_histogram.png",
        figure_folder
    )

    histogram(
        df,
        "likes",
        "Likes Distribution",
        "likes_histogram.png",
        figure_folder
    )

    histogram(
        df,
        "comment_count",
        "Comments Distribution",
        "comments_histogram.png",
        figure_folder
    )

    histogram(
        df,
        "engagement_rate",
        "Engagement Distribution",
        "engagement_histogram.png",
        figure_folder
    )

    histogram(
        df,
        "combined_word_count",
        "Word Count Distribution",
        "word_count_histogram.png",
        figure_folder
    )

    histogram(
        df,
        "combined_char_count",
        "Character Count Distribution",
        "character_count_histogram.png",
        figure_folder
    )

    histogram(
        df,
        "emoji_count",
        "Emoji Distribution",
        "emoji_histogram.png",
        figure_folder
    )

    histogram(
        df,
        "hashtag_count",
        "Hashtag Distribution",
        "hashtag_histogram.png",
        figure_folder
    )


# ==========================================================
# ALL BOXPLOTS
# ==========================================================

def create_all_boxplots(df, figure_folder):

    boxplot(
        df,
        "views",
        "Views Boxplot",
        "views_boxplot.png",
        figure_folder
    )

    boxplot(
        df,
        "likes",
        "Likes Boxplot",
        "likes_boxplot.png",
        figure_folder
    )

    boxplot(
        df,
        "comment_count",
        "Comments Boxplot",
        "comments_boxplot.png",
        figure_folder
    )

# ==========================================================
# FINAL SUMMARY REPORT
# ==========================================================

def final_summary(df, report_file):

    with open(report_file, "a", encoding="utf-8") as file:

        file.write("\n")
        file.write("=" * 70 + "\n")
        file.write("KEY INSIGHTS\n")
        file.write("=" * 70 + "\n\n")

        # ---------------------------------------------
        # Most Viewed
        # ---------------------------------------------

        top_view = df.loc[df["views"].idxmax()]

        file.write("Most Viewed Video\n")
        file.write("-----------------------------\n")
        file.write(f"Title : {top_view['title']}\n")
        file.write(f"University : {top_view['university_name']}\n")
        file.write(f"Views : {top_view['views']}\n\n")

        # ---------------------------------------------
        # Most Liked
        # ---------------------------------------------

        top_like = df.loc[df["likes"].idxmax()]

        file.write("Most Liked Video\n")
        file.write("-----------------------------\n")
        file.write(f"Title : {top_like['title']}\n")
        file.write(f"University : {top_like['university_name']}\n")
        file.write(f"Likes : {top_like['likes']}\n\n")

        # ---------------------------------------------
        # Most Commented
        # ---------------------------------------------

        top_comment = df.loc[df["comment_count"].idxmax()]

        file.write("Most Commented Video\n")
        file.write("-----------------------------\n")
        file.write(f"Title : {top_comment['title']}\n")
        file.write(f"University : {top_comment['university_name']}\n")
        file.write(f"Comments : {top_comment['comment_count']}\n\n")

        # ---------------------------------------------
        # Highest Engagement
        # ---------------------------------------------

        if "engagement_rate" in df.columns:

            top_engagement = df.loc[df["engagement_rate"].idxmax()]

            file.write("Highest Engagement Video\n")
            file.write("-----------------------------\n")
            file.write(f"Title : {top_engagement['title']}\n")
            file.write(f"University : {top_engagement['university_name']}\n")
            file.write(
                f"Engagement : {round(top_engagement['engagement_rate'],5)}\n\n"
            )

        # ---------------------------------------------
        # General Statistics
        # ---------------------------------------------

        file.write("=" * 70 + "\n")
        file.write("GENERAL STATISTICS\n")
        file.write("=" * 70 + "\n\n")

        file.write(
            f"Average Views : {round(df['views'].mean(),2)}\n"
        )

        file.write(
            f"Median Views : {round(df['views'].median(),2)}\n"
        )

        file.write(
            f"Average Likes : {round(df['likes'].mean(),2)}\n"
        )

        file.write(
            f"Average Comments : {round(df['comment_count'].mean(),2)}\n"
        )

        if "engagement_rate" in df.columns:

            file.write(
                f"Average Engagement : {round(df['engagement_rate'].mean(),5)}\n"
            )

        file.write(
            f"Unique Universities : {df['university_name'].nunique()}\n"
        )

        if "channel_name" in df.columns:

            file.write(
                f"Unique Channels : {df['channel_name'].nunique()}\n"
            )


# ==========================================================
# EXECUTION SUMMARY
# ==========================================================

def execution_summary(category):

    print("\n")
    print("=" * 70)
    print("VIDEO EDA COMPLETED SUCCESSFULLY")
    print("=" * 70)

    print(f"\nCategory : {category}")

    print("\nGenerated Tables")

    print("----------------------------")

    print("✓ data_types.csv")

    print("✓ missing_values.csv")

    print("✓ descriptive_statistics.csv")

    print("✓ top_universities.csv")

    print("✓ top_channels.csv")

    print("✓ top_viewed.csv")

    print("✓ top_liked.csv")

    print("✓ top_commented.csv")

    print("✓ highest_engagement.csv")

    print("\nGenerated Figures")

    print("----------------------------")

    print("✓ university_distribution.png")

    print("✓ channel_distribution.png")

    print("✓ views_histogram.png")

    print("✓ likes_histogram.png")

    print("✓ comments_histogram.png")

    print("✓ engagement_histogram.png")

    print("✓ word_count_histogram.png")

    print("✓ character_count_histogram.png")

    print("✓ emoji_histogram.png")

    print("✓ hashtag_histogram.png")

    print("✓ upload_trend.png")

    print("✓ correlation_heatmap.png")

    print("✓ views_boxplot.png")

    print("✓ likes_boxplot.png")

    print("✓ comments_boxplot.png")

    print("\nSummary Report")

    print("----------------------------")

    print("✓ summary_report.txt")

    print("\nAll outputs saved inside results folder.")

    print("=" * 70)

# ==========================================================
# MAIN
# ==========================================================

def main():

    # ---------------------------------------------
    # Select Dataset
    # ---------------------------------------------

    category = get_category()

    # ---------------------------------------------
    # Create Output Folders
    # ---------------------------------------------

    base, figures, tables = create_output_folders(category)

    report = os.path.join(
        base,
        "summary_report.txt"
    )

    # ---------------------------------------------
    # Load Dataset
    # ---------------------------------------------

    df = load_dataset(category)

    # ---------------------------------------------
    # Dataset Overview
    # ---------------------------------------------

    dataset_overview(df)

    datatype_summary(df, tables)

    missing_value_analysis(df, tables)

    descriptive_statistics(df, tables)

    write_basic_summary(
        df,
        category,
        report
    )

    # ---------------------------------------------
    # Dataset Analysis
    # ---------------------------------------------

    university_analysis(df, tables)

    channel_analysis(df, tables)

    top_viewed(df, tables)

    top_liked(df, tables)

    top_commented(df, tables)

    highest_engagement(df, tables)

    append_dataset_summary(
        df,
        report
    )

    # ---------------------------------------------
    # Visualizations
    # ---------------------------------------------

    plot_university_distribution(
        df,
        figures
    )

    plot_channel_distribution(
        df,
        figures
    )

    create_all_histograms(
        df,
        figures
    )

    create_all_boxplots(
        df,
        figures
    )

    upload_trend(
        df,
        figures
    )

    correlation_heatmap(
        df,
        figures
    )

    # ---------------------------------------------
    # Final Report
    # ---------------------------------------------

    final_summary(
        df,
        report
    )

    execution_summary(
        category
    )


# ==========================================================
# DRIVER
# ==========================================================

if __name__ == "__main__":

    main()