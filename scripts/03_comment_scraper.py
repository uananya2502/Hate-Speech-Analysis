# =====================================
# IMPORTS
# =====================================

import os
import sys
import time
import pandas as pd

from googleapiclient.discovery import build
from googleapiclient.errors import HttpError


# =====================================
# ALLOW ROOT IMPORT
# =====================================

sys.path.append(
    os.path.dirname(
        os.path.dirname(__file__)
    )
)


import config


from scripts.utils import (
    append_csv,
    log_error,
    log_activity
)


# =====================================
# YOUTUBE CONNECTION
# =====================================

youtube = build(
    "youtube",
    "v3",
    developerKey=config.API_KEY
)


# =====================================
# LOAD VIDEOS OF CATEGORY
# =====================================

def load_videos(category):

    video_file = (
        config.get_video_file(category)
    )


    if not os.path.exists(
        video_file
    ):

        return pd.DataFrame()


    return pd.read_csv(
        video_file
    )


# =====================================
# CHECK EXISTING COMMENTS
# Prevent duplicate comments
# =====================================

def get_existing_comments(category):

    comment_file = (
        config.get_comment_file(category)
    )


    if not os.path.exists(
        comment_file
    ):

        return set()


    try:

        df = pd.read_csv(
            comment_file,
            usecols=[
                "comment_id"
            ]
        )


        return set(
            df["comment_id"]
        )


    except Exception as e:

        log_error(
            config.ERROR_FILE,
            f"""
Comment loading error

Category:
{category}

Error:
{str(e)}
"""
        )

        return set()


# =====================================
# FORMAT COMMENT DATA
# =====================================

def create_comment_row(
    comment_id,
    parent_id,
    video_id,
    university,
    author,
    text,
    date,
    likes,
    reply_count,
    is_reply
):

    return {

        "comment_id":
        comment_id,


        "parent_comment_id":
        parent_id,


        "video_id":
        video_id,


        "university_name":
        university,


        "author_name":
        author,


        "comment_text":
        text,


        "published_at":
        date,


        "likes":
        likes,


        "reply_count":
        reply_count,


        "is_reply":
        is_reply,


        "comment_length":
        len(text),


        "word_count":
        len(
            text.split()
        )
    }
# =====================================
# GET COMMENTS FROM A VIDEO
# Parent comments only
# =====================================

def get_video_comments(
    video_id,
    university,
    existing_comments
):

    comments = []

    next_page = None


    while True:

        try:

            request = youtube.commentThreads().list(

                part="snippet",

                videoId=video_id,

                maxResults=100,

                pageToken=next_page,

                textFormat="plainText"

            )


            response = request.execute()


            for item in response["items"]:


                comment = (
                    item["snippet"]
                    ["topLevelComment"]
                    ["snippet"]
                )


                comment_id = (
                    item["snippet"]
                    ["topLevelComment"]
                    ["id"]
                )


                # Skip duplicates
                if comment_id in existing_comments:
                    continue


                row = create_comment_row(

                    comment_id=comment_id,

                    parent_id=None,

                    video_id=video_id,

                    university=university,

                    author=comment.get(
                        "authorDisplayName",
                        "Unknown"
                    ),

                    text=comment.get(
                        "textDisplay",
                        ""
                    ),

                    date=comment.get(
                        "publishedAt",
                        ""
                    ),

                    likes=comment.get(
                        "likeCount",
                        0
                    ),

                    reply_count=item["snippet"].get(
                        "totalReplyCount",
                        0
                    ),

                    is_reply=False
                )


                comments.append(row)


                existing_comments.add(
                    comment_id
                )


            next_page = response.get(
                "nextPageToken"
            )


            if not next_page:
                break


            time.sleep(
                config.SLEEP_SECONDS
            )


        except HttpError as e:


            log_error(

                config.ERROR_FILE,

                f"""
Comment API Error

Video ID:
{video_id}

Error:
{str(e)}
"""

            )


            break


    return comments

# =====================================
# GET REPLIES TO PARENT COMMENTS
# Child comments
# =====================================

def get_comment_replies(
    parent_comment_id,
    video_id,
    university,
    existing_comments
):

    replies = []

    next_page = None


    while True:

        try:

            request = youtube.comments().list(

                part="snippet",

                parentId=parent_comment_id,

                maxResults=100,

                pageToken=next_page,

                textFormat="plainText"

            )


            response = request.execute()


            for item in response["items"]:


                reply = item["snippet"]


                reply_id = item["id"]


                # Skip duplicates
                if reply_id in existing_comments:
                    continue


                row = create_comment_row(

                    comment_id=reply_id,

                    parent_id=parent_comment_id,

                    video_id=video_id,

                    university=university,

                    author=reply.get(
                        "authorDisplayName",
                        "Unknown"
                    ),

                    text=reply.get(
                        "textDisplay",
                        ""
                    ),

                    date=reply.get(
                        "publishedAt",
                        ""
                    ),

                    likes=reply.get(
                        "likeCount",
                        0
                    ),

                    reply_count=0,

                    is_reply=True
                )


                replies.append(
                    row
                )


                existing_comments.add(
                    reply_id
                )


            next_page = response.get(
                "nextPageToken"
            )


            if not next_page:
                break


            time.sleep(
                config.SLEEP_SECONDS
            )


        except HttpError as e:


            log_error(

                config.ERROR_FILE,

                f"""
Reply API Error

Parent Comment:
{parent_comment_id}

Video ID:
{video_id}

Error:
{str(e)}
"""

            )


            break


    return replies

# =====================================
# PROCESS ALL COMMENTS OF A CATEGORY
# =====================================

def process_category(category):

    print(
        "\n================================"
    )

    print(
        f"COMMENT CATEGORY: {category.upper()}"
    )

    print(
        "================================"
    )


    videos = load_videos(category)


    if videos.empty:

        print(
            "No videos found in this category."
        )

        return


    existing_comments = get_existing_comments(
        category
    )


    print(
        f"Existing comments found: "
        f"{len(existing_comments)}"
    )


    comment_file = config.get_comment_file(
        category
    )


    batch = []

    total_videos = len(videos)


    for index, video in videos.iterrows():

        video_id = video["video_id"]

        university = video["university_name"]


        print(
            f"\n[{index + 1}/{total_videos}] "
            f"Fetching comments: {video_id}"
        )


        # ------------------------------
        # Get parent comments
        # ------------------------------

        parents = get_video_comments(
            video_id,
            university,
            existing_comments
        )


        batch.extend(
            parents
        )


        # ------------------------------
        # Get replies of each parent
        # ------------------------------

        for parent in parents:


            if parent["reply_count"] > 0:


                replies = get_comment_replies(

                    parent["comment_id"],

                    video_id,

                    university,

                    existing_comments

                )


                batch.extend(
                    replies
                )


        # ------------------------------
        # Save every 200 comments
        # ------------------------------

        if len(batch) >= 200:


            append_csv(
                comment_file,
                batch
            )


            print(
                f"Saved {len(batch)} comments"
            )


            batch = []


        time.sleep(
            config.SLEEP_SECONDS
        )


    # ----------------------------------
    # Save remaining comments
    # ----------------------------------

    if batch:


        append_csv(
            comment_file,
            batch
        )


        print(
            f"Saved remaining {len(batch)} comments"
        )


# =====================================
# MAIN FUNCTION
# =====================================

def main():

    print(
        "\n========== COMMENT SCRAPER STARTED ==========\n"
    )


    total_categories = len(
        config.ACTIVE_CATEGORIES
    )


    for index, category in enumerate(
        config.ACTIVE_CATEGORIES,
        start=1
    ):


        print(
            f"\n[{index}/{total_categories}] "
            f"Processing category: {category}"
        )


        try:


            process_category(
                category
            )


            log_activity(
                config.LOG_FILE,
                "comment_scraping",
                "COMPLETED",
                f"Category completed: {category}"
            )


        except Exception as e:


            log_error(
                config.ERROR_FILE,
                f"""
Comment scraper failed

Category:
{category}

Error:
{str(e)}
"""
            )


            log_activity(
                config.LOG_FILE,
                "comment_scraping",
                "FAILED",
                f"Category failed: {category}"
            )


    print(
        "\n========== ALL COMMENT SCRAPING COMPLETED =========="
    )


# =====================================
# RUN SCRIPT
# =====================================

if __name__ == "__main__":

    main()