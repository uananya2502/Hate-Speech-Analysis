# =====================================
# IMPORTS
# =====================================

import os
import json
import pandas as pd
from datetime import datetime


# =====================================
# CREATE DIRECTORY IF NOT EXISTS
# =====================================

def ensure_directory(path):

    directory = os.path.dirname(path)

    if directory and not os.path.exists(directory):
        os.makedirs(directory)


# =====================================
# APPEND DATA TO CSV
# =====================================

def append_csv(file_path, data):

    if not data:
        return


    # Create folder automatically
    ensure_directory(file_path)


    df = pd.DataFrame(data)


    # If file does not exist create it
    if not os.path.exists(file_path):

        df.to_csv(
            file_path,
            index=False,
            encoding="utf-8"
        )

    else:

        df.to_csv(
            file_path,
            mode="a",
            header=False,
            index=False,
            encoding="utf-8"
        )


# =====================================
# LOAD PROGRESS
# =====================================

def load_progress(progress_file):

    ensure_directory(progress_file)


    if not os.path.exists(progress_file):

        default_progress = {

            "last_category": None,
            "last_topic": None,
            "last_university": None,

            "quota_used": 0,

            "queries_completed": 0,

            "date": str(datetime.now().date())

        }


        with open(
            progress_file,
            "w"
        ) as f:

            json.dump(
                default_progress,
                f,
                indent=4
            )


        return default_progress


    with open(
        progress_file,
        "r"
    ) as f:

        progress = json.load(f)


    # Reset quota on a new day
    today = str(datetime.now().date())


    if progress["date"] != today:

        progress["quota_used"] = 0

        progress["date"] = today


    return progress


# =====================================
# SAVE PROGRESS
# =====================================

def save_progress(
    progress_file,
    progress
):

    ensure_directory(progress_file)


    with open(
        progress_file,
        "w"
    ) as f:

        json.dump(
            progress,
            f,
            indent=4
        )


# =====================================
# SCRAPING ACTIVITY LOG
# =====================================

def log_activity(
    log_file,
    task,
    status,
    details
):

    ensure_directory(log_file)


    row = {

        "timestamp": str(datetime.now()),

        "task": task,

        "status": status,

        "details": details
    }


    df = pd.DataFrame(
        [row]
    )


    if not os.path.exists(log_file):

        df.to_csv(
            log_file,
            index=False
        )

    else:

        df.to_csv(
            log_file,
            mode="a",
            header=False,
            index=False
        )


# =====================================
# ERROR LOGGING
# =====================================

def log_error(
    error_file,
    message
):

    ensure_directory(error_file)


    with open(
        error_file,
        "a",
        encoding="utf-8"
    ) as f:

        f.write(
            "\n"
            + "=" * 50
            + "\n"
        )

        f.write(
            str(datetime.now())
            + "\n"
        )

        f.write(
            message
            + "\n"
        )


# =====================================
# CHECK EXISTING CSV
# =====================================

def file_exists(file_path):

    return os.path.exists(file_path)