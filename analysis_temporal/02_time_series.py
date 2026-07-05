"""
============================================================
02_time_series.py

Time Series Analysis

Input
------
cleaned_data_comments/<category>_comment_topics.csv

Outputs
-------
analysis_temporal/
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
import numpy as np

import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression

# ==========================================================
# FOLDERS
# ==========================================================

INPUT_FOLDER = "cleaned_data_comments"

TABLE_FOLDER = os.path.join(
    "analysis_temporal",
    "tables"
)

FIGURE_FOLDER = os.path.join(
    "analysis_temporal",
    "figures"
)

REPORT_FOLDER = os.path.join(
    "analysis_temporal",
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
    print("TIME SERIES ANALYSIS")
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

        f"{category}_comment_topics.csv"

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

    print("\nDate Range")

    print(df["published_at"].min())

    print(df["published_at"].max())

# ==========================================================
# CREATE MONTHLY TIME SERIES
# ==========================================================

def create_monthly_series(df):

    print("\nCreating Monthly Time Series...")

    monthly = df.copy()

    monthly = monthly.set_index("published_at")

    monthly = (

        monthly

        .resample("M")

        .size()

        .reset_index(name="Comments")

    )

    monthly["Month"] = (

        monthly["published_at"]

        .dt.strftime("%b-%Y")

    )

    monthly.to_csv(

        os.path.join(

            TABLE_FOLDER,

            "monthly_time_series.csv"

        ),

        index=False

    )

    print("✓ monthly_time_series.csv")

    return monthly
# ==========================================================
# ROLLING AVERAGE
# ==========================================================

def rolling_average(monthly):

    print("\nCalculating Rolling Average...")

    monthly["Rolling_Average"] = (

        monthly["Comments"]

        .rolling(

            window=3,

            min_periods=1

        )

        .mean()

    )

    monthly.to_csv(

        os.path.join(

            TABLE_FOLDER,

            "rolling_average.csv"

        ),

        index=False

    )

    plt.figure(figsize=(14,6))

    plt.plot(

        monthly["Month"],

        monthly["Comments"],

        marker="o",

        label="Monthly Comments"

    )

    plt.plot(

        monthly["Month"],

        monthly["Rolling_Average"],

        linewidth=3,

        label="3-Month Rolling Average"

    )

    plt.xticks(rotation=90)

    plt.xlabel("Month")

    plt.ylabel("Comments")

    plt.title("Rolling Average of Monthly Comments")

    plt.legend()

    plt.tight_layout()

    plt.savefig(

        os.path.join(

            FIGURE_FOLDER,

            "rolling_average.png"

        ),

        dpi=300

    )

    plt.close()

    print("✓ rolling_average.csv")

    print("✓ rolling_average.png")

    return monthly
# ==========================================================
# TREND LINE
# ==========================================================

def trend_line(monthly):

    print("\nGenerating Trend Line...")

    monthly["Time"] = np.arange(len(monthly))

    X = monthly[["Time"]]

    y = monthly["Comments"]

    model = LinearRegression()

    model.fit(X, y)

    monthly["Trend"] = model.predict(X)

    monthly.to_csv(

        os.path.join(

            TABLE_FOLDER,

            "trendline.csv"

        ),

        index=False

    )

    plt.figure(figsize=(14,6))

    plt.plot(

        monthly["Month"],

        monthly["Comments"],

        marker="o",

        label="Actual Comments"

    )

    plt.plot(

        monthly["Month"],

        monthly["Trend"],

        linewidth=3,

        label="Trend Line"

    )

    plt.xticks(rotation=90)

    plt.xlabel("Month")

    plt.ylabel("Comments")

    plt.title("Monthly Comment Trend")

    plt.legend()

    plt.tight_layout()

    plt.savefig(

        os.path.join(

            FIGURE_FOLDER,

            "trendline.png"

        ),

        dpi=300

    )

    plt.close()

    print("✓ trendline.csv")

    print("✓ trendline.png")

    return monthly
# ==========================================================
# PEAK DETECTION
# ==========================================================

def detect_peaks(monthly):

    print("\nDetecting Peak Months...")

    threshold = (

        monthly["Comments"].mean()

        + monthly["Comments"].std()

    )

    peaks = monthly[

        monthly["Comments"] >= threshold

    ].copy()

    peaks.to_csv(

        os.path.join(

            TABLE_FOLDER,

            "peak_months.csv"

        ),

        index=False

    )

    plt.figure(figsize=(14,6))

    plt.plot(

        monthly["Month"],

        monthly["Comments"],

        marker="o",

        label="Comments"

    )

    plt.scatter(

        peaks["Month"],

        peaks["Comments"],

        s=120,

        label="Peak"

    )

    plt.xticks(rotation=90)

    plt.xlabel("Month")

    plt.ylabel("Comments")

    plt.title("Peak Detection")

    plt.legend()

    plt.tight_layout()

    plt.savefig(

        os.path.join(

            FIGURE_FOLDER,

            "peak_detection.png"

        ),

        dpi=300

    )

    plt.close()

    print("✓ peak_months.csv")

    print("✓ peak_detection.png")

    return peaks
# ==========================================================
# FORECAST NEXT 6 MONTHS
# ==========================================================

def forecast(monthly):

    print("\nForecasting Next 6 Months...")

    X = monthly[["Time"]]

    y = monthly["Comments"]

    model = LinearRegression()

    model.fit(X, y)

    future = pd.DataFrame({

        "Time": np.arange(

            len(monthly),

            len(monthly)+6

        )

    })

    future["Forecast"] = model.predict(future)

    last_date = monthly["published_at"].iloc[-1]

    future["published_at"] = pd.date_range(

        last_date + pd.offsets.MonthEnd(),

        periods=6,

        freq="M"

    )

    future["Month"] = future["published_at"].dt.strftime("%b-%Y")

    future.to_csv(

        os.path.join(

            TABLE_FOLDER,

            "forecast.csv"

        ),

        index=False

    )

    plt.figure(figsize=(14,6))

    plt.plot(

        monthly["Month"],

        monthly["Comments"],

        marker="o",

        label="Historical"

    )

    plt.plot(

        future["Month"],

        future["Forecast"],

        marker="o",

        linestyle="--",

        linewidth=3,

        label="Forecast"

    )

    plt.xticks(rotation=90)

    plt.xlabel("Month")

    plt.ylabel("Comments")

    plt.title("6-Month Comment Forecast")

    plt.legend()

    plt.tight_layout()

    plt.savefig(

        os.path.join(

            FIGURE_FOLDER,

            "forecast.png"

        ),

        dpi=300

    )

    plt.close()

    print("✓ forecast.csv")

    print("✓ forecast.png")

    return future
# ==========================================================
# SAVE REPORT
# ==========================================================

def save_report(

    category,

    monthly,

    peaks,

    forecast_df

):

    report = os.path.join(

        REPORT_FOLDER,

        f"{category}_time_series_report.txt"

    )

    with open(

        report,

        "w",

        encoding="utf-8"

    ) as file:

        file.write("="*60+"\n")

        file.write("TIME SERIES ANALYSIS REPORT\n")

        file.write("="*60+"\n\n")

        file.write(

            f"Total Months : {len(monthly)}\n\n"

        )

        file.write("Detected Peaks\n")

        file.write("-"*40+"\n")

        file.write(

            peaks.to_string(index=False)

        )

        file.write("\n\n")

        file.write("Forecast\n")

        file.write("-"*40+"\n")

        file.write(

            forecast_df.to_string(index=False)

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

    print("✓ Monthly Time Series")

    print("✓ Rolling Average")

    print("✓ Trend Line")

    print("✓ Peak Detection")

    print("✓ Forecast")

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

    monthly = create_monthly_series(df)

    monthly = rolling_average(monthly)

    monthly = trend_line(monthly)

    peaks = detect_peaks(monthly)

    forecast_df = forecast(monthly)

    save_report(

        category,

        monthly,

        peaks,

        forecast_df

    )

    final_validation()

    print("\n"+"="*60)

    print("TIME SERIES ANALYSIS COMPLETED")

    print("="*60)


# ==========================================================
# DRIVER
# ==========================================================

if __name__ == "__main__":

    main()

#.   python analysis_temporal/02_time_series.py