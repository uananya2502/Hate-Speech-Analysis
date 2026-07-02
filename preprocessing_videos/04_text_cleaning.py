"""
===========================================================
04_text_cleaning.py

Text Cleaning for YouTube Dataset

Creates:
    clean_title
    clean_description
    combined_text

Author : Ananya Upadhyay
===========================================================
"""

# ==========================================================
# IMPORT LIBRARIES
# ==========================================================

import os
import re
import html
import pandas as pd

# ==========================================================
# TEXT CLEANING FUNCTIONS
# ==========================================================

def remove_html_tags(text):
    """
    Remove HTML tags like <br>, <p>, etc.
    """

    if pd.isna(text):
        return ""

    text = str(text)

    return re.sub(r"<.*?>", " ", text)


# ==========================================================
# REMOVE URLS
# ==========================================================

def remove_urls(text):
    """
    Remove all web URLs.
    """

    text = str(text)

    return re.sub(
        r"http\S+|www\S+|https\S+",
        "",
        text
    )


# ==========================================================
# REMOVE EMAILS
# ==========================================================

def remove_emails(text):
    """
    Remove email addresses.
    """

    text = str(text)

    return re.sub(
        r"\S+@\S+",
        "",
        text
    )


# ==========================================================
# REMOVE NEW LINES
# ==========================================================

def remove_newlines(text):

    text = str(text)

    text = text.replace("\n", " ")

    text = text.replace("\r", " ")

    text = text.replace("\t", " ")

    return text


# ==========================================================
# DECODE HTML ENTITIES
# ==========================================================

def decode_html(text):

    return html.unescape(str(text))


# ==========================================================
# REMOVE EXTRA SPACES
# ==========================================================

def normalize_spaces(text):

    return re.sub(
        r"\s+",
        " ",
        str(text)
    ).strip()


# ==========================================================
# REDUCE REPEATED PUNCTUATION
# ==========================================================

def reduce_punctuation(text):

    text = re.sub(r"!{2,}", "!", text)

    text = re.sub(r"\?{2,}", "?", text)

    text = re.sub(r"\.{2,}", ".", text)

    return text
# ==========================================================
# COMPLETE TEXT CLEANING FUNCTION
# ==========================================================

def clean_text(text):
    """
    Complete text cleaning pipeline.

    Keeps:
    - Emojis
    - Hashtags
    - Numbers
    - Mentions

    Removes:
    - HTML
    - URLs
    - Emails
    - Extra spaces
    """

    if pd.isna(text):
        return ""

    text = str(text)

    # Decode HTML entities
    text = decode_html(text)

    # Remove HTML tags
    text = remove_html_tags(text)

    # Remove URLs
    text = remove_urls(text)

    # Remove email addresses
    text = remove_emails(text)

    # Remove new lines
    text = remove_newlines(text)

    # Reduce repeated punctuation
    text = reduce_punctuation(text)

    # Normalize spaces
    text = normalize_spaces(text)

    return text


# ==========================================================
# MAIN PROGRAM
# ==========================================================

def main():

    print("=" * 60)
    print("TEXT CLEANING")
    print("=" * 60)

    category = input(
        "\nEnter category "
        "(infrastructure, controversies, faculty_research, rankings): "
    ).strip().lower()

    INPUT_FILE = f"cleaned_data/{category}_relevant.csv"

    OUTPUT_FILE = f"cleaned_data/{category}_text_cleaned.csv"

    if not os.path.exists(INPUT_FILE):
        print("\nInput file not found.")
        return

    print("\nLoading dataset...")

    df = pd.read_csv(INPUT_FILE)

    print(f"Loaded {len(df)} rows.")

    # ------------------------------------------------------
    # Fill missing values
    # ------------------------------------------------------

    df["title"] = df["title"].fillna("")
    df["description"] = df["description"].fillna("")

    # ------------------------------------------------------
    # Clean title
    # ------------------------------------------------------

    print("Cleaning titles...")

    df["clean_title"] = df["title"].apply(clean_text)

    # ------------------------------------------------------
    # Clean description
    # ------------------------------------------------------

    print("Cleaning descriptions...")

    df["clean_description"] = df["description"].apply(clean_text)

    # ------------------------------------------------------
    # Combined text
    # ------------------------------------------------------

    print("Creating combined text...")

    df["combined_text"] = (
        df["clean_title"]
        + " "
        + df["clean_description"]
    )

    df["combined_text"] = df["combined_text"].apply(
        normalize_spaces
    )

    # ------------------------------------------------------
    # Save
    # ------------------------------------------------------

    df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("\n" + "=" * 60)
    print("TEXT CLEANING REPORT")
    print("=" * 60)

    print(f"Category : {category}")
    print(f"Rows     : {len(df)}")
    print(f"Columns  : {len(df.columns)}")

    print("\nNew Columns Added")

    print("✓ clean_title")
    print("✓ clean_description")
    print("✓ combined_text")

    print("\nSaved File")

    print(OUTPUT_FILE)

    print("\nSample Clean Text\n")

    print(df["combined_text"].head(5))

    print("\n✅ Text Cleaning Completed Successfully.")


# ==========================================================
# DRIVER CODE
# ==========================================================

if __name__ == "__main__":
    main()