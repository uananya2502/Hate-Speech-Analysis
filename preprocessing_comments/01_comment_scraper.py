"""
============================================================
01_comment_scraper.py

Scrapes ALL YouTube comments (including replies)
from validated videos.

Input:
    analysis_data/<category>_feature_engineered.csv

Output:
    comments/<category>_comments.csv

Features
--------
✓ Resume support
✓ Incremental CSV writing
✓ Automatic checkpoint
✓ Retry mechanism
✓ Reply scraping
✓ Progress tracking
✓ Failed video logging

Author : Ananya Upadhyay
============================================================
"""

# ==========================================================
# IMPORT LIBRARIES
# ==========================================================

import os
import json
import time
import random
import pandas as pd

from datetime import datetime

from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from config import API_KEY


# ==========================================================
# CONFIGURATION
# ==========================================================

ANALYSIS_DATA = "analysis_data"

COMMENTS_FOLDER = "comments"

LOG_FOLDER = "logs"

PROGRESS_FOLDER = "progress"

SAVE_INTERVAL = 20

MAX_RETRIES = 3


# ==========================================================
# CREATE REQUIRED FOLDERS
# ==========================================================

os.makedirs(COMMENTS_FOLDER, exist_ok=True)

os.makedirs(LOG_FOLDER, exist_ok=True)

os.makedirs(PROGRESS_FOLDER, exist_ok=True)


# ==========================================================
# INITIALIZE YOUTUBE API
# ==========================================================

youtube = build(
    "youtube",
    "v3",
    developerKey=API_KEY
)


# ==========================================================
# LOAD CATEGORY DATASET
# ==========================================================

def load_dataset(category):

    input_file = os.path.join(
        ANALYSIS_DATA,
        f"{category}_feature_engineered.csv"
    )

    if not os.path.exists(input_file):

        raise FileNotFoundError(
            f"\nFile not found:\n{input_file}"
        )

    df = pd.read_csv(input_file)

    print(f"\nLoaded {len(df)} videos.")

    return df


# ==========================================================
# OUTPUT FILES
# ==========================================================

def get_comment_file(category):

    return os.path.join(
        COMMENTS_FOLDER,
        f"{category}_comments.csv"
    )


def get_progress_file(category):

    return os.path.join(
        PROGRESS_FOLDER,
        f"{category}_progress.json"
    )


def get_failed_file(category):

    return os.path.join(
        LOG_FOLDER,
        f"{category}_failed_videos.csv"
    )


def get_log_file(category):

    return os.path.join(
        LOG_FOLDER,
        f"{category}_scraping_log.csv"
    )


# ==========================================================
# CSV HEADER
# ==========================================================

COMMENT_COLUMNS = [

    "comment_id",

    "parent_comment_id",

    "video_id",

    "university_name",

    "category",

    "video_title",

    "author_name",

    "author_channel_id",

    "author_profile_url",

    "author_thumbnail",

    "comment_text",

    "published_at",

    "updated_at",

    "likes",

    "reply_count",

    "is_reply",

    "moderation_status",

    "comment_length",

    "word_count"

]


# ==========================================================
# CREATE OUTPUT CSV
# ==========================================================

def initialize_csv(category):

    output_file = get_comment_file(category)

    if not os.path.exists(output_file):

        pd.DataFrame(
            columns=COMMENT_COLUMNS
        ).to_csv(
            output_file,
            index=False,
            encoding="utf-8"
        )

        print("\nCreated comment dataset.")


# ==========================================================
# PROGRESS FUNCTIONS
# ==========================================================

def load_progress(category):

    progress_file = get_progress_file(category)

    if not os.path.exists(progress_file):

        return {

            "last_video_index": 0,

            "videos_processed": 0,

            "comments_scraped": 0,

            "replies_scraped": 0

        }

    with open(progress_file, "r") as file:

        return json.load(file)


def save_progress(

        category,

        last_video_index,

        videos_processed,

        comments_scraped,

        replies_scraped

):

    progress = {

        "last_video_index": last_video_index,

        "videos_processed": videos_processed,

        "comments_scraped": comments_scraped,

        "replies_scraped": replies_scraped,

        "last_updated": str(datetime.now())

    }

    with open(

            get_progress_file(category),

            "w"

    ) as file:

        json.dump(

            progress,

            file,

            indent=4

        )


# ==========================================================
# APPEND COMMENTS TO CSV
# ==========================================================

def append_comments(category, rows):

    if len(rows) == 0:

        return

    output_file = get_comment_file(category)

    pd.DataFrame(rows).to_csv(

        output_file,

        mode="a",

        header=False,

        index=False,

        encoding="utf-8"

    )

# ==========================================================
# LOG FAILED VIDEOS
# ==========================================================

def log_failed_video(category, video_id, reason):

    failed_file = get_failed_file(category)

    row = pd.DataFrame([{

        "video_id": video_id,

        "reason": reason,

        "timestamp": datetime.now()

    }])

    if os.path.exists(failed_file):

        row.to_csv(

            failed_file,

            mode="a",

            header=False,

            index=False

        )

    else:

        row.to_csv(

            failed_file,

            index=False

        )


# ==========================================================
# SCRAPING LOG
# ==========================================================

def write_scraping_log(

        category,

        videos_processed,

        comments_scraped,

        replies_scraped,

        runtime

):

    log_file = get_log_file(category)

    row = pd.DataFrame([{

        "timestamp": datetime.now(),

        "category": category,

        "videos_processed": videos_processed,

        "comments_scraped": comments_scraped,

        "replies_scraped": replies_scraped,

        "runtime_seconds": runtime

    }])

    if os.path.exists(log_file):

        row.to_csv(

            log_file,

            mode="a",

            header=False,

            index=False

        )

    else:

        row.to_csv(

            log_file,

            index=False

        )


# ==========================================================
# SAFE API REQUEST
# ==========================================================

def safe_execute(request):

    """
    Executes YouTube API request safely
    with retry mechanism.
    """

    retries = 0

    while retries < MAX_RETRIES:

        try:

            return request.execute()

        except HttpError as e:

            print(f"\nAPI Error: {e}")

            retries += 1

            wait = (2 ** retries) + random.random()

            print(f"Retrying in {round(wait,2)} sec...")

            time.sleep(wait)

        except Exception as e:

            print(e)

            retries += 1

            time.sleep(2)

    return None


# ==========================================================
# GET COMMENT THREADS
# ==========================================================

def get_comment_threads(video_id, page_token=None):

    request = youtube.commentThreads().list(

        part="snippet,replies",

        videoId=video_id,

        maxResults=100,

        pageToken=page_token,

        textFormat="plainText"

    )

    return safe_execute(request)


# ==========================================================
# GET REPLIES
# ==========================================================

def get_replies(parent_comment_id, page_token=None):

    request = youtube.comments().list(

        part="snippet",

        parentId=parent_comment_id,

        maxResults=100,

        pageToken=page_token,

        textFormat="plainText"

    )

    return safe_execute(request)


# ==========================================================
# COUNT WORDS
# ==========================================================

def word_count(text):

    if pd.isna(text):

        return 0

    return len(str(text).split())


# ==========================================================
# COUNT CHARACTERS
# ==========================================================

def comment_length(text):

    if pd.isna(text):

        return 0

    return len(str(text))

# ==========================================================
# SCRAPE SINGLE VIDEO
# ==========================================================

def scrape_video_comments(video_row, category):

    video_id = str(video_row["video_id"])

    university = str(video_row["university_name"])

    title = str(video_row["title"])

    rows = []

    comment_counter = 0

    reply_counter = 0

    next_page = None

    while True:

        response = get_comment_threads(
            video_id,
            next_page
        )

        if response is None:
            break

        items = response.get("items", [])

        for item in items:

            snippet = item["snippet"]["topLevelComment"]["snippet"]

            comment_id = item["snippet"]["topLevelComment"]["id"]

            text = snippet.get("textDisplay", "")

            row = {

                "comment_id": comment_id,

                "parent_comment_id": "",

                "video_id": video_id,

                "university_name": university,

                "category": category,

                "video_title": title,

                "author_name":
                    snippet.get(
                        "authorDisplayName",
                        ""
                    ),

                "author_channel_id":
                    snippet.get(
                        "authorChannelId",
                        {}
                    ).get(
                        "value",
                        ""
                    ),

                "author_profile_url":
                    snippet.get(
                        "authorChannelUrl",
                        ""
                    ),

                "author_thumbnail":
                    snippet.get(
                        "authorProfileImageUrl",
                        ""
                    ),

                "comment_text": text,

                "published_at":
                    snippet.get(
                        "publishedAt",
                        ""
                    ),

                "updated_at":
                    snippet.get(
                        "updatedAt",
                        ""
                    ),

                "likes":
                    snippet.get(
                        "likeCount",
                        0
                    ),

                "reply_count":
                    item["snippet"].get(
                        "totalReplyCount",
                        0
                    ),

                "is_reply": False,

                "moderation_status":
                    snippet.get(
                        "moderationStatus",
                        ""
                    ),

                "comment_length":
                    comment_length(text),

                "word_count":
                    word_count(text)

            }

            rows.append(row)

            comment_counter += 1

            # ==========================================
            # SCRAPE REPLIES
            # ==========================================

            if item["snippet"]["totalReplyCount"] > 0:

                reply_page = None

                while True:

                    replies = get_replies(
                        comment_id,
                        reply_page
                    )

                    if replies is None:
                        break

                    reply_items = replies.get(
                        "items",
                        []
                    )

                    for reply in reply_items:

                        rs = reply["snippet"]

                        reply_text = rs.get(
                            "textDisplay",
                            ""
                        )

                        reply_row = {

                            "comment_id":
                                reply["id"],

                            "parent_comment_id":
                                comment_id,

                            "video_id":
                                video_id,

                            "university_name":
                                university,

                            "category":
                                category,

                            "video_title":
                                title,

                            "author_name":
                                rs.get(
                                    "authorDisplayName",
                                    ""
                                ),

                            "author_channel_id":
                                rs.get(
                                    "authorChannelId",
                                    {}
                                ).get(
                                    "value",
                                    ""
                                ),

                            "author_profile_url":
                                rs.get(
                                    "authorChannelUrl",
                                    ""
                                ),

                            "author_thumbnail":
                                rs.get(
                                    "authorProfileImageUrl",
                                    ""
                                ),

                            "comment_text":
                                reply_text,

                            "published_at":
                                rs.get(
                                    "publishedAt",
                                    ""
                                ),

                            "updated_at":
                                rs.get(
                                    "updatedAt",
                                    ""
                                ),

                            "likes":
                                rs.get(
                                    "likeCount",
                                    0
                                ),

                            "reply_count":
                                0,

                            "is_reply":
                                True,

                            "moderation_status":
                                rs.get(
                                    "moderationStatus",
                                    ""
                                ),

                            "comment_length":
                                comment_length(reply_text),

                            "word_count":
                                word_count(reply_text)

                        }

                        rows.append(reply_row)

                        reply_counter += 1

                    reply_page = replies.get(
                        "nextPageToken"
                    )

                    if not reply_page:
                        break

        next_page = response.get("nextPageToken")

        if not next_page:
            break

    append_comments(category, rows)

    return comment_counter, reply_counter

# ==========================================================
# MAIN
# ==========================================================

def main():

    print("=" * 60)
    print("YOUTUBE COMMENT SCRAPER")
    print("=" * 60)

    category = input(
        "\nEnter category "
        "(infrastructure, controversies, faculty_research, rankings): "
    ).strip().lower()

    # ------------------------------------------
    # Load dataset
    # ------------------------------------------

    df = load_dataset(category)

    initialize_csv(category)

    progress = load_progress(category)

    start_index = progress["last_video_index"]

    videos_processed = progress["videos_processed"]

    comments_scraped = progress["comments_scraped"]

    replies_scraped = progress["replies_scraped"]

    start_time = time.time()

    print(f"\nResuming from video index : {start_index}")
    print(f"Videos already processed : {videos_processed}")
    print(f"Comments already scraped : {comments_scraped}")
    print(f"Replies already scraped  : {replies_scraped}")

    # ------------------------------------------
    # Loop through videos
    # ------------------------------------------

    for index in range(start_index, len(df)):

        row = df.iloc[index]

        video_id = row["video_id"]

        title = row["title"]

        print("\n" + "=" * 60)

        print(
            f"Video {index+1}/{len(df)}"
        )

        print(title)

        try:

            comments, replies = scrape_video_comments(
                row,
                category
            )

            comments_scraped += comments

            replies_scraped += replies

            videos_processed += 1

            print(f"Comments : {comments}")

            print(f"Replies  : {replies}")

        except Exception as e:

            print(f"\nFailed : {video_id}")

            print(e)

            log_failed_video(
                category,
                video_id,
                str(e)
            )

        # --------------------------------------
        # Save progress
        # --------------------------------------

        if (index + 1) % SAVE_INTERVAL == 0:

            save_progress(

                category,

                index + 1,

                videos_processed,

                comments_scraped,

                replies_scraped

            )

            print("\nProgress Saved.")

    # ------------------------------------------
    # Final Save
    # ------------------------------------------

    save_progress(

        category,

        len(df),

        videos_processed,

        comments_scraped,

        replies_scraped

    )

    runtime = round(

        time.time() - start_time,

        2

    )

    write_scraping_log(

        category,

        videos_processed,

        comments_scraped,

        replies_scraped,

        runtime

    )

    print("\n" + "=" * 60)
    print("SCRAPING COMPLETED")
    print("=" * 60)

    print(f"Videos Processed : {videos_processed}")

    print(f"Comments Scraped : {comments_scraped}")

    print(f"Replies Scraped  : {replies_scraped}")

    print(f"Runtime          : {runtime} sec")

    print("\nOutput File")

    print(get_comment_file(category))

    print("\nProgress File")

    print(get_progress_file(category))

    print("\nFailed Videos")

    print(get_failed_file(category))

    print("\nLog File")

    print(get_log_file(category))

    print("\nAll comments successfully scraped.")


# ==========================================================
# DRIVER
# ==========================================================

if __name__ == "__main__":

    main()