"""
===========================================================
02_relevance_filter.py

Generic Relevance Filter

Works for:
1. Infrastructure
2. Controversies
3. Faculty Research
4. Rankings

Author : Ananya Upadhyay
===========================================================
"""

# ===========================================================
# IMPORT LIBRARIES
# ===========================================================

import os
import re
import pandas as pd

try:
    from rapidfuzz import fuzz
except ImportError:
    raise SystemExit(
        "\nRapidFuzz is not installed.\n"
        "Run:\n"
        "pip install rapidfuzz"
    )

# ===========================================================
# CATEGORY KEYWORDS
# ===========================================================

CATEGORY_KEYWORDS = {

    # -------------------------------------------------------
    # Infrastructure
    # -------------------------------------------------------

    "infrastructure": [

        # Campus
        "campus",
        "campus tour",
        "campus life",
        "campus vlog",

        # Hostel
        "hostel",
        "hostel tour",
        "hostel life",
        "room tour",

        # Buildings
        "building",
        "block",
        "classroom",
        "lecture hall",

        # Labs
        "lab",
        "laboratory",

        # Library
        "library",
        "library tour",

        # Food
        "canteen",
        "mess",
        "mess food",
        "food court",

        # Sports
        "sports",
        "gym",
        "ground",

        # Facilities
        "facility",
        "facilities",
        "wifi",
        "transport",
        "auditorium",
        "infrastructure",

        # Reviews
        "college review",
        "campus review",
        "student review",
        "honest review",

        # Daily Life
        "day in my life"

    ],

    # -------------------------------------------------------
    # Controversies
    # -------------------------------------------------------

    "controversies": [

        "controversy",
        "scam",
        "fraud",
        "fake",
        "ragging",
        "harassment",
        "sexual harassment",
        "student death",
        "suicide",
        "protest",
        "viral",
        "court",
        "police",
        "investigation",
        "complaint"

    ],

    # -------------------------------------------------------
    # Faculty Research
    # -------------------------------------------------------

    "faculty_research": [

        "research",
        "research paper",
        "publication",
        "faculty",
        "professor",
        "teacher",
        "journal",
        "citation",
        "plagiarism",
        "research ethics",
        "predatory journal",
        "paper retraction",
        "fake research",
        "academic misconduct"

    ],

    # -------------------------------------------------------
    # Rankings
    # -------------------------------------------------------

    "rankings": [

        "nirf",
        "naac",
        "ranking",
        "rank",
        "accreditation",
        "reputation"

    ]
}

# ===========================================================
# IMPORTANT PHRASES
# (Very strong relevance indicators)
# ===========================================================

IMPORTANT_PHRASES = [

    "campus tour",
    "hostel tour",
    "library tour",
    "room tour",
    "campus review",
    "college review",
    "official campus tour",
    "hostel review"

]

# ===========================================================
# NEGATIVE KEYWORDS
# ===========================================================

NEGATIVE_KEYWORDS = [

    "movie",
    "movies",
    "music",
    "lyrics",
    "anime",
    "minecraft",
    "pubg",
    "free fire",
    "valorant",
    "roblox",
    "gaming",
    "cartoon",
    "trailer",
    "reaction",
    "live stream"

]

# ===========================================================
# UNIVERSITY ALIASES
# ===========================================================

ALIASES = {

    "Indian Institute of Technology": [
        "iit"
    ],

    "National Institute of Technology": [
        "nit"
    ],

    "Birla Institute of Technology and Science": [
        "bits"
    ],

    "Delhi University": [
        "du"
    ],

    "Jawaharlal Nehru University": [
        "jnu"
    ],

    "Vellore Institute of Technology": [
        "vit"
    ],

    "SRM Institute of Science and Technology": [
        "srm"
    ],

    "Lovely Professional University": [
        "lpu"
    ]

}

# ===========================================================
# OFFICIAL CHANNEL WORDS
# ===========================================================

OFFICIAL_CHANNEL_WORDS = [

    "official",
    "university",
    "institute"

]
# ===========================================================
# HELPER FUNCTIONS
# ===========================================================

def normalize_text(text):
    """
    Convert text to lowercase and safely handle missing values.
    """
    if pd.isna(text):
        return ""

    return str(text).lower().strip()


# ===========================================================
# WHOLE WORD MATCH
# ===========================================================

def contains(text, phrase):
    """
    Returns True if the complete keyword/phrase exists.
    Prevents matching:
        lab -> collaboration
    """

    text = normalize_text(text)
    phrase = normalize_text(phrase)

    pattern = r"\b" + re.escape(phrase) + r"\b"

    return re.search(pattern, text) is not None


# ===========================================================
# RAPIDFUZZ SIMILARITY
# ===========================================================

def similarity_score(query, title):
    """
    Calculate similarity between search query
    and video title.
    """

    query = normalize_text(query)
    title = normalize_text(title)

    if query == "" or title == "":
        return 0

    similarity = fuzz.token_set_ratio(query, title)

    # Convert 0–100 to maximum 8 points
    return min(round(similarity / 10), 8)


# ===========================================================
# UNIVERSITY MATCH SCORE
# ===========================================================

def university_score(title,
                     description,
                     channel,
                     university):

    score = 0

    university = normalize_text(university)

    if university == "":
        return score

    # Exact university match

    if university in title:
        score += 5

    if university in description:
        score += 3

    if university in channel:
        score += 2

    # Alias match

    for full_name, alias_list in ALIASES.items():

        if full_name.lower() in university:

            for alias in alias_list:

                if contains(title, alias):
                    score += 4

                if contains(description, alias):
                    score += 2

                if contains(channel, alias):
                    score += 2

    return score


# ===========================================================
# OFFICIAL CHANNEL BONUS
# ===========================================================

def official_channel_score(channel):

    channel = normalize_text(channel)

    for word in OFFICIAL_CHANNEL_WORDS:

        if word in channel:
            return 2

    return 0


# ===========================================================
# CATEGORY KEYWORD SCORE
# ===========================================================

def keyword_score(title,
                  description,
                  category):

    score = 0

    keywords = CATEGORY_KEYWORDS.get(category, [])

    for keyword in keywords:

        if contains(title, keyword):
            score += 4

        if contains(description, keyword):
            score += 2

    return score


# ===========================================================
# IMPORTANT PHRASE BONUS
# ===========================================================

def phrase_bonus(title):

    score = 0

    for phrase in IMPORTANT_PHRASES:

        if contains(title, phrase):
            score += 6

    return score


# ===========================================================
# NEGATIVE KEYWORD PENALTY
# ===========================================================

def negative_penalty(title,
                     description,
                     category):

    positive = any(
        contains(title, word)
        for word in CATEGORY_KEYWORDS.get(category, [])
    )

    if positive:
        return 0

    penalty = 0

    for word in NEGATIVE_KEYWORDS:

        if contains(title, word):
            penalty += 3

        if contains(description, word):
            penalty += 2

    return penalty
# ===========================================================
# FINAL RELEVANCE SCORING
# ===========================================================

def score_row(row, category):
    """
    Calculate the final relevance score for a video.
    """

    title = normalize_text(row.get("title", ""))
    description = normalize_text(row.get("description", ""))
    channel = normalize_text(row.get("channel_name", ""))
    university = normalize_text(row.get("university_name", ""))
    search_query = normalize_text(row.get("search_query", ""))

    score = 0

    # -------------------------------------------------------
    # University Match
    # -------------------------------------------------------

    score += university_score(
        title,
        description,
        channel,
        university
    )

    # -------------------------------------------------------
    # Official Channel Bonus
    # -------------------------------------------------------

    score += official_channel_score(channel)

    # -------------------------------------------------------
    # Category Keyword Score
    # -------------------------------------------------------

    score += keyword_score(
        title,
        description,
        category
    )

    # -------------------------------------------------------
    # Important Phrase Bonus
    # -------------------------------------------------------

    score += phrase_bonus(title)

    # -------------------------------------------------------
    # Search Query Similarity
    # -------------------------------------------------------

    score += similarity_score(
        search_query,
        title
    )

    # -------------------------------------------------------
    # Negative Keyword Penalty
    # -------------------------------------------------------

    score -= negative_penalty(
        title,
        description,
        category
    )

    # -------------------------------------------------------
    # Count Keyword Matches
    # -------------------------------------------------------

    keyword_hits = 0

    for keyword in CATEGORY_KEYWORDS.get(category, []):

        if contains(title, keyword):
            keyword_hits += 1

        if contains(description, keyword):
            keyword_hits += 1

    # -------------------------------------------------------
    # Bonus for Multiple Matches
    # -------------------------------------------------------

    if keyword_hits >= 5:
        score += 4

    elif keyword_hits >= 3:
        score += 2

    # -------------------------------------------------------
    # Ensure Score is Never Negative
    # -------------------------------------------------------

    score = max(score, 0)

    return score


# ===========================================================
# CLASSIFICATION FUNCTION
# ===========================================================

def classify(score):
    """
    Convert numerical score into relevance label.
    """

    if score >= 10:
        return "Relevant"

    elif score >= 4:
        return "Review"

    else:
        return "Removed"
# ===========================================================
# MAIN PROGRAM
# ===========================================================

def main():

    print("=" * 60)
    print("YOUTUBE DATASET RELEVANCE FILTER")
    print("=" * 60)

    # -------------------------------------------------------
    # Category Input
    # -------------------------------------------------------

    category = input(
        "\nEnter category "
        "(infrastructure, controversies, faculty_research, rankings): "
    ).strip().lower()

    if category not in CATEGORY_KEYWORDS:

        print("\n❌ Invalid category.")
        return

    # -------------------------------------------------------
    # File Paths
    # -------------------------------------------------------

    INPUT_FILE = f"cleaned_data/{category}_clean.csv"

    RELEVANT_FILE = f"cleaned_data/{category}_relevant.csv"

    REVIEW_FILE = f"cleaned_data/{category}_review.csv"

    REMOVED_FILE = f"cleaned_data/{category}_removed.csv"

    # -------------------------------------------------------
    # Check File Exists
    # -------------------------------------------------------

    if not os.path.exists(INPUT_FILE):

        print(f"\n❌ File not found:\n{INPUT_FILE}")
        return

    # -------------------------------------------------------
    # Load Dataset
    # -------------------------------------------------------

    print("\nLoading dataset...")

    df = pd.read_csv(INPUT_FILE)

    print(f"✓ Loaded {len(df)} videos")

    # -------------------------------------------------------
    # Calculate Scores
    # -------------------------------------------------------

    print("Calculating relevance scores...")

    df["relevance_score"] = df.apply(
        lambda row: score_row(row, category),
        axis=1
    )

    df["relevance_label"] = df["relevance_score"].apply(
        classify
    )

    # -------------------------------------------------------
    # Split Dataset
    # -------------------------------------------------------

    relevant_df = df[
        df["relevance_label"] == "Relevant"
    ].copy()

    review_df = df[
        df["relevance_label"] == "Review"
    ].copy()

    removed_df = df[
        df["relevance_label"] == "Removed"
    ].copy()

    # -------------------------------------------------------
    # Sort by Score
    # -------------------------------------------------------

    relevant_df.sort_values(
        "relevance_score",
        ascending=False,
        inplace=True
    )

    review_df.sort_values(
        "relevance_score",
        ascending=False,
        inplace=True
    )

    removed_df.sort_values(
        "relevance_score",
        ascending=True,
        inplace=True
    )

    # -------------------------------------------------------
    # Save Files
    # -------------------------------------------------------

    relevant_df.to_csv(
        RELEVANT_FILE,
        index=False
    )

    review_df.to_csv(
        REVIEW_FILE,
        index=False
    )

    removed_df.to_csv(
        REMOVED_FILE,
        index=False
    )

    # -------------------------------------------------------
    # Statistics
    # -------------------------------------------------------

    total = len(df)

    relevant = len(relevant_df)

    review = len(review_df)

    removed = len(removed_df)

    print("\n" + "=" * 60)
    print("RELEVANCE FILTER REPORT")
    print("=" * 60)

    print(f"Category        : {category}")
    print(f"Total Videos    : {total}")
    print(f"Relevant        : {relevant}")
    print(f"Review          : {review}")
    print(f"Removed         : {removed}")

    print(f"\nAverage Score   : {df['relevance_score'].mean():.2f}")
    print(f"Highest Score   : {df['relevance_score'].max()}")
    print(f"Lowest Score    : {df['relevance_score'].min()}")

    print("\nFiles Saved")

    print(f"✓ {RELEVANT_FILE}")
    print(f"✓ {REVIEW_FILE}")
    print(f"✓ {REMOVED_FILE}")

    # -------------------------------------------------------
    # Top Relevant Videos
    # -------------------------------------------------------

    print("\nTop 10 Relevant Videos")

    print(
        relevant_df[
            [
                "title",
                "university_name",
                "relevance_score"
            ]
        ].head(10)
    )

    print("\n✅ Relevance filtering completed successfully.")


# ===========================================================
# DRIVER CODE
# ===========================================================

if __name__ == "__main__":
    main()
    