import datetime
import subprocess
import os
import json
import time as Time
from subprocess import Popen, PIPE


def upload_json_to_s3(paths, save_name, selected_files):
    """ save json file to s3
                Args:
                    paths (dict): dictionary of paths from yml file
                    save_name (str): name of json to be saved
                    selected_files (list): list of file paths to be stored in json

                Returns:

    """
    # Upload selected file names to s3
    filepath = os.path.join(paths["video_names"], save_name + ".json")
    with open(filepath, "w") as f:
        json.dump(selected_files, f)
    try:
        res = subprocess.call(["aws", "s3", 'cp',
                               filepath,
                               's3://air-pollution-uk/' + paths['s3_video_names'],
                               '--profile',
                               'dssg'])
    except:
        print("JSON video name upload failed!")
    # Delete the json from local
    os.remove(filepath)

    return


def generate_dates(from_date, to_date):
    """ Generate a list of dates between two dates
                Args:
                    from_date (datetime): starting date
                    to_date (datetime): end date

                Returns:
                    dates (list): list of dates between the two dates specified
    """
    dates = []
    while from_date <= to_date:
        dates.append(from_date)
        from_date += datetime.timedelta(days=1)
    return dates


def get_names_of_folder_content_from_s3(bucket_name, prefix, s3_profile):

    start = Time.time()
    ls = Popen(["aws", "s3", 'ls', 's3://%s/%s' % (bucket_name, prefix),
                '--profile',
                s3_profile], stdout=PIPE)
    p1 = Popen(['awk', '{$1=$2=$3=""; print $0}'],
               stdin=ls.stdout, stdout=PIPE)
    p2 = Popen(['sed', 's/^[ \t]*//'], stdin=p1.stdout, stdout=PIPE)
    ls.stdout.close()
    p1.stdout.close()
    output = p2.communicate()[0]
    p2.stdout.close()
    files = output.decode("utf-8").split("\n")
    end = Time.time()
    elapsed_time = end - start

    assert (len(files) == 0) or (files[0] != ""), "set your aws credentials"

    return elapsed_time, files
