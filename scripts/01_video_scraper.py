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
# ALLOW IMPORT FROM ROOT DIRECTORY
# =====================================

sys.path.append(
    os.path.dirname(
        os.path.dirname(__file__)
    )
)

import config as config

from scripts.utils import (
    append_csv,
    load_progress,
    save_progress,
    log_activity,
    log_error
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
# LOAD UNIVERSITIES
# =====================================

def get_universities():

    df = pd.read_csv(
        "universities.csv"
    )

    return (
        df["university_name"]
        .dropna()
        .tolist()
    )


# =====================================
# GENERATE CATEGORY-WISE QUERIES
# =====================================

def generate_queries():

    queries = []

    universities = get_universities()

    for category in config.ACTIVE_CATEGORIES:

        topics = config.TOPIC_CATEGORIES[
            category
        ]

        for topic in topics:

            for university in universities:

                queries.append({

                    "category":
                    category,

                    "topic":
                    topic,

                    "university":
                    university,

                    "query":
                    f"{university} {topic}"

                })

    return queries


# =====================================
# CHECK DAILY QUOTA
# =====================================

def can_continue(progress):

    return (
        progress["quota_used"]
        <
        config.STOP_BEFORE_QUOTA
    )


# =====================================
# SEARCH VIDEOS
# =====================================

def search_videos(
    query,
    category,
    topic,
    university,
    progress
):

    videos = []

    next_page = None

    while True:

        try:

            request = youtube.search().list(

                q=query,

                part="snippet",

                type="video",

                maxResults=50,

                pageToken=next_page,

                publishedAfter=config.START_DATE

            )

            response = request.execute()

            # Search costs 100 quota units
            progress["quota_used"] += (
                config.SEARCH_COST
            )

            for item in response["items"]:

                videos.append({

                    "video_id":
                        item["id"]["videoId"],

                    "university_name":
                        university,

                    "category":
                        category,

                    "topic":
                        topic,

                    "search_query":
                        query,

                    "title":
                        item["snippet"]["title"],

                    "description":
                        item["snippet"]["description"],

                    "channel_name":
                        item["snippet"]["channelTitle"],

                    "upload_date":
                        item["snippet"]["publishedAt"],

                    "language":
                        item["snippet"].get(
                            "defaultLanguage",
                            "unknown"
                        )

                })

            next_page = response.get(
                "nextPageToken"
            )

            if (
                not next_page
                or
                len(videos)
                >= config.MAX_VIDEOS_PER_QUERY
            ):
                break

            time.sleep(
                config.SLEEP_SECONDS
            )

        except HttpError as e:

            log_error(

                config.ERROR_FILE,

                f"""
Search Error

Query:
{query}

Error:
{str(e)}
"""

            )

            break

    return videos


# =====================================
# GET VIDEO DETAILS
# Views, Likes, Comments, Duration
# =====================================

def get_video_details(video_ids):

    details = []

    for i in range(
        0,
        len(video_ids),
        50
    ):

        batch = video_ids[i:i+50]

        try:

            request = youtube.videos().list(

                part="statistics,contentDetails",

                id=",".join(batch)

            )

            response = request.execute()

            for item in response["items"]:

                details.append({

                    "video_id":
                    item["id"],

                    "views":
                    item["statistics"].get(
                        "viewCount",
                        0
                    ),

                    "likes":
                    item["statistics"].get(
                        "likeCount",
                        0
                    ),

                    "comment_count":
                    item["statistics"].get(
                        "commentCount",
                        0
                    ),

                    "duration":
                    item["contentDetails"]["duration"]

                })

            time.sleep(
                config.SLEEP_SECONDS
            )

        except HttpError as e:

            log_error(

                config.ERROR_FILE,

                f"""
Video Detail Error

Video IDs:
{batch}

Error:
{str(e)}
"""

            )

    return details

# =====================================
# MERGE VIDEO DETAILS
# =====================================

def merge_video_details(
    videos,
    details
):

    detail_map = {}

    for item in details:

        detail_map[
            item["video_id"]
        ] = item

    final_videos = []

    for video in videos:

        video.update(

            detail_map.get(

                video["video_id"],

                {}

            )

        )

        final_videos.append(
            video
        )

    return final_videos


# =====================================
# LOAD EXISTING VIDEOS FOR CATEGORY
# Prevent duplicate storage
# =====================================

def get_existing_video_ids(category):

    video_file = config.get_video_file(
        category
    )

    if not os.path.exists(
        video_file
    ):
        return set()

    try:

        df = pd.read_csv(

            video_file,

            usecols=[
                "video_id"
            ]

        )

        return set(
            df["video_id"]
        )

    except Exception as e:

        log_error(

            config.ERROR_FILE,

            f"""
Existing Video Loading Error

Category:
{category}

Error:
{str(e)}
"""

        )

        return set()


# =====================================
# PREPARE FINAL VIDEO DATASET
# =====================================

def prepare_video_rows(
    videos,
    university,
    existing_ids
):

    rows = []

    for video in videos:

        # Skip duplicates
        if (
            video["video_id"]
            in existing_ids
        ):
            continue

        row = {

            "video_id":
            video["video_id"],

            "university_name":
            university,

            "category":
            video.get(
                "category",
                ""
            ),

            "topic":
            video.get(
                "topic",
                ""
            ),

            "search_query":
            video.get(
                "search_query",
                ""
            ),

            "title":
            video.get(
                "title",
                ""
            ),

            "description":
            video.get(
                "description",
                ""
            ),

            "channel_name":
            video.get(
                "channel_name",
                ""
            ),

            "upload_date":
            video.get(
                "upload_date",
                ""
            ),

            "upload_year":
            video.get(
                "upload_date",
                ""
            )[:4],

            "views":
            video.get(
                "views",
                0
            ),

            "likes":
            video.get(
                "likes",
                0
            ),

            "comment_count":
            video.get(
                "comment_count",
                0
            ),

            "duration":
            video.get(
                "duration",
                ""
            ),

            "language":
            video.get(
                "language",
                "unknown"
            ),

            "video_url":
            (
                "https://www.youtube.com/watch?v="
                +
                video["video_id"]
            )

        }

        rows.append(
            row
        )

        # Add to duplicate checker
        existing_ids.add(
            video["video_id"]
        )

    return rows

# =====================================
# MAIN SCRAPER
# =====================================

def main():

    print("\n========== VIDEO SCRAPER STARTED ==========\n")

    total_queries = 0

    # Loop through each active category
    for category in config.ACTIVE_CATEGORIES:

        print(
            f"\n=============================="
            f"\nCATEGORY: {category.upper()}"
            f"\n=============================="
        )

        # Category progress file
        progress_file = config.get_progress_file(
            category
        )

        progress = load_progress(
            progress_file
        )

        print(
            f"Quota used today: "
            f"{progress['quota_used']}"
        )

        # Load duplicate checker
        existing_ids = get_existing_video_ids(
            category
        )

        queries = []

        universities = get_universities()

        # Create category queries
        for topic in config.TOPIC_CATEGORIES[
            category
        ]:

            for university in universities:

                queries.append({

                    "topic": topic,

                    "university": university,

                    "query":
                    f"{university} {topic}"

                })

        start_index = (
            progress["queries_completed"]
        )

        print(
            f"Starting from query: "
            f"{start_index + 1}"
        )

        # Start scraping queries
        for i in range(
            start_index,
            len(queries)
        ):

            if not can_continue(progress):

                print(
                    "\nDaily quota reached!"
                )

                save_progress(
                    progress_file,
                    progress
                )

                log_activity(
                    config.LOG_FILE,
                    "video_scraping",
                    "STOPPED",
                    (
                        f"Category: {category}, "
                        f"Queries completed: "
                        f"{progress['queries_completed']}, "
                        f"Quota: "
                        f"{progress['quota_used']}"
                    )
                )

                return

            current = queries[i]

            university = (
                current["university"]
            )

            query = (
                current["query"]
            )

            print(
                f"\n[{i+1}/{len(queries)}] "
                f"Searching: {query}"
            )

            # =====================================
            # SEARCH VIDEOS
            # =====================================

            videos = search_videos(

                query,

                category,

                current["topic"],

                university,

                progress

            )

            if not videos:

                progress["queries_completed"] += 1

                save_progress(
                    progress_file,
                    progress
                )

                continue

            # =====================================
            # GET VIDEO DETAILS
            # =====================================

            details = get_video_details(

                [
                    video["video_id"]
                    for video in videos
                ]

            )

            videos = merge_video_details(

                videos,

                details

            )

            # =====================================
            # PREPARE FINAL ROWS
            # =====================================

            rows = prepare_video_rows(

                videos,

                university,

                existing_ids

            )

            if rows:

                append_csv(

                    config.get_video_file(
                        category
                    ),

                    rows

                )

                print(
                    f"Saved {len(rows)} videos"
                )

            else:

                print(
                    "No new videos found"
                )

            # =====================================
            # UPDATE PROGRESS
            # =====================================

            progress["queries_completed"] += 1

            save_progress(

                progress_file,

                progress

            )

            total_queries += 1

            time.sleep(
                config.SLEEP_SECONDS
            )

        print(
            f"\nCompleted category: "
            f"{category}"
        )

        log_activity(

            config.LOG_FILE,

            "video_scraping",

            "COMPLETED",

            (
                f"Finished category: "
                f"{category}"
            )

        )

    print(
        "\n========== ALL VIDEO SCRAPING COMPLETED =========="
    )


# =====================================
# RUN SCRIPT
# =====================================

if __name__ == "__main__":

    main()