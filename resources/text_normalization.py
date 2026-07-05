"""
============================================================
text_normalization.py

Reusable Text Normalization Module

Used By

- WordCloud
- TF-IDF
- Topic Modeling
- N-Grams

============================================================
"""

# ==========================================================
# IMPORT LIBRARIES
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

import os
import re

from nltk.stem import WordNetLemmatizer

import nltk

# Download once
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)

lemmatizer = WordNetLemmatizer()

# ==========================================================
# LOAD STOPWORDS
# ==========================================================

STOPWORD_FOLDER = os.path.join(

    "resources",

    "stopwords"

)


def load_stopwords(filename):

    filepath = os.path.join(

        STOPWORD_FOLDER,

        filename

    )

    words = set()

    if os.path.exists(filepath):

        with open(

            filepath,

            "r",

            encoding="utf-8"

        ) as file:

            for line in file:

                word = line.strip().lower()

                if word:

                    words.add(word)

    return words


ENGLISH_STOPWORDS = load_stopwords(

    "english_stopwords.txt"

)

HINGLISH_STOPWORDS = load_stopwords(

    "hinglish_stopwords.txt"

)

YOUTUBE_STOPWORDS = load_stopwords(

    "youtube_stopwords.txt"

)

UNIVERSITY_STOPWORDS = load_stopwords(

    "university_stopwords.txt"

)

PROFANITY_WORDS = load_stopwords(

    "profanity_words.txt"

)

ALL_STOPWORDS = (

    ENGLISH_STOPWORDS

    | HINGLISH_STOPWORDS

    | YOUTUBE_STOPWORDS

    | UNIVERSITY_STOPWORDS

)

# ==========================================================
# REMOVE URLS
# ==========================================================

def remove_urls(text):

    return re.sub(

        r"http\S+|https\S+|www\S+",

        " ",

        text

    )


# ==========================================================
# REMOVE EMAILS
# ==========================================================

def remove_emails(text):

    return re.sub(

        r"\S+@\S+",

        " ",

        text

    )


# ==========================================================
# REMOVE HTML TAGS
# ==========================================================

def remove_html(text):

    return re.sub(

        r"<.*?>",

        " ",

        text

    )


# ==========================================================
# REMOVE EMOJIS
# ==========================================================

def remove_emojis(text):

    emoji_pattern = re.compile(

        "["

        "\U0001F600-\U0001F64F"

        "\U0001F300-\U0001F5FF"

        "\U0001F680-\U0001F6FF"

        "\U0001F1E0-\U0001F1FF"

        "\U00002700-\U000027BF"

        "\U000024C2-\U0001F251"

        "]+",

        flags=re.UNICODE

    )

    return emoji_pattern.sub(

        " ",

        text

    )


# ==========================================================
# REMOVE NUMBERS
# ==========================================================

def remove_numbers(text):

    return re.sub(

        r"\d+",

        " ",

        text

    )


# ==========================================================
# REMOVE PUNCTUATION
# ==========================================================

def remove_punctuation(text):

    return re.sub(

        r"[^a-zA-Z\u0900-\u097F\s]",

        " ",

        text

    )


# ==========================================================
# REMOVE EXTRA SPACES
# ==========================================================

def normalize_spaces(text):

    return re.sub(

        r"\s+",

        " ",

        text

    ).strip()

# ==========================================================
# LEMMATIZATION
# ==========================================================

def lemmatize_text(words):

    output = []

    for word in words:

        output.append(

            lemmatizer.lemmatize(word)

        )

    return output


# ==========================================================
# REMOVE STOPWORDS
# ==========================================================

def remove_stopwords(words):

    cleaned = []

    for word in words:

        if word not in ALL_STOPWORDS:

            if len(word) > 2:

                cleaned.append(word)

    return cleaned

# ==========================================================
# NORMALIZATION DICTIONARY
# ==========================================================

NORMALIZATION_MAP = {

    # Hindi

    "nhi": "nahi",
    "nai": "nahi",
    "nah": "nahi",
    "na": "nahi",

    "mai": "main",
    "mein": "main",
    "me": "main",

    "toh": "to",
    "tho": "to",

    "bohot": "bahut",
    "bahot": "bahut",
    "bohut": "bahut",

    "acha": "accha",
    "achha": "accha",

    "kr": "kar",
    "krna": "karna",
    "krte": "karte",

    "hn": "haan",
    "ha": "haan",

    "yr": "yaar",

    # English

    "u": "you",
    "ur": "your",

}
# ==========================================================
# NORMALIZE TEXT
# ==========================================================

def normalize_words(words):

    normalized = []

    for word in words:

        normalized.append(

            NORMALIZATION_MAP.get(

                word,

                word

            )

        )

    return normalized

def normalize_text(text):

    text = str(text).lower()

    text = remove_urls(text)

    text = remove_emails(text)

    text = remove_html(text)

    text = remove_emojis(text)

    text = remove_numbers(text)

    text = remove_punctuation(text)

    text = normalize_spaces(text)

    words = text.split()

    words = normalize_words(words)

    words = lemmatize_text(words)

    words = remove_stopwords(words)

    return " ".join(words)

