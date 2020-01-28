import datetime
import os
import json
import time as Time
import subprocess
from subprocess import Popen, PIPE


def upload_json_to_s3(paths: dict,
                      save_name: str,
                      selected_files: list):
    """Save json file to s3
    Args:
        paths (dict): dictionary of paths from yml file
        save_name (str): name of json to be saved
        selected_files (list): list of file paths to be stored in json
    """
    # Upload selected file names to s3
    filepath = os.path.join(paths["video_names"], save_name + '.json')
    with open(filepath, "w") as f:
        json.dump(selected_files, f)
    try:
        res = subprocess.call(["aws", "s3", 'cp',
                               filepath,
                               's3://air-pollution-uk/' +
                               paths['s3_video_names'],
                               '--profile',
                               'dssg'])
    except:
        print('JSON video name upload failed!')
    # Delete the json from local
    os.remove(filepath)


def generate_dates(from_date: datetime.datetime,
                   to_date: datetime.datetime) -> list:
    """ Generate a list of dates between two dates
    Args:
        from_date: starting date
        to_date: end date

    Returns:
        dates: list of dates between the two dates specified
    """
    dates = []
    while from_date <= to_date:
        dates.append(from_date)
        from_date += datetime.timedelta(days=1)
    return dates
