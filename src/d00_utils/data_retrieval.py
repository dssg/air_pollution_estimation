import boto3
import cv2
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import shutil
import time
import datetime
import glob
import json
from subprocess import Popen, PIPE


def retrieve_video_names_from_s3(paths, from_date='2019-06-01', to_date=str(datetime.datetime.now())[:10],
                                 from_time='00-00-00', to_time='23-59-59', camera_list=None, save_to_file: bool = True):
    """Retrieve names of jamcam videos from the s3 bucket based on the dates specified.

        Args:
            paths: dictionary containing temp_video, raw_video, s3_profile and bucket_name paths
            from_date: start date (inclusive) for retrieving videos, if None then will retrieve from 2019-06-01 onwards
            to_date: end date (inclusive) for retrieving vidoes, if None then will retrieve up to current day
            from_time: start time for retrieving videos, if None then will retrieve from the start of the day
            to_time: end time for retrieving videos, if None then will retrieve up to the end of the day
            camera_list: list of cameras to retrieve from, if None then retrieve from all cameras
            save_to_file: boolean for keeping the downloaded file names to the local folder for raw videos
        Returns:
            selected_files: A list of filenames of videos in s3 that satisfies the date and time conditions
        Raises:

    """
    save_folder = 'temp_video'
    bucket_name = paths['bucket_name']
    s3_profile = paths['s3_profile']
    s3_video = paths['s3_video']
    delete_and_recreate_dir(paths[save_folder])
    from_date = datetime.datetime.strptime(from_date, '%Y-%m-%d').date()
    to_date = datetime.datetime.strptime(to_date, '%Y-%m-%d').date()
    from_time = datetime.datetime.strptime(from_time, '%H-%M-%S').time()
    to_time = datetime.datetime.strptime(to_time, '%H-%M-%S').time()
    dates = []
    selected_files = []

    # Generate the list of dates
    dates = generate_dates(from_date, to_date)
    for date in dates:
        date = date.strftime('%Y-%m-%d')
        prefix = "%s%s/" % (s3_video, date)
        start = time.time()

        # fetch video filenames
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
        end = time.time()
        print(end - start, len(files))
        if not files:
            break
        for filename in files:
            if filename:
                res = filename.split('_')
                camera_id = res[-1][:-4]
                time_of_day = res[0].split(".")[0]
                time_of_day = datetime.datetime.strptime(
                    time_of_day, '%Y-%m-%d %H:%M:%S').time()
                if from_time <= time_of_day <= to_time and (not camera_list or camera_id in camera_list):
                    selected_files.append("%s%s" % (prefix, filename))
        if save_to_file:
            filepath = os.path.join(paths["raw_video"], "searched_videos")
            with open(filepath, "w") as f:
                json.dump(selected_files, f)
    return selected_files


def generate_dates(from_date, to_date):
    dates = []
    while(from_date <= to_date):
        dates.append(from_date)
        from_date += datetime.timedelta(days=1)
    return dates


def load_video_names(paths):
    save_folder = 'raw_video'
    filepath = os.path.join(paths[save_folder], "searched_videos")
    with open(filepath, "r") as f:
        videos = list(json.load(f))
        return videos


def load_videos_into_np(folder):
     # Load files into a list of numpy arrays using opencv
    videos = []
    names = []
    for filename in glob.glob(os.path.join(folder, '*.mp4')):
        try:
            videos.append(mp4_to_npy(filename))
            names.append(filename.split('/')[-1])
        except:
            print("Could not convert " + filename + " to numpy array")
    return videos, names


def download_video_and_convert_to_numpy(local_folder, s3_profile, bucket, filenames: list):
    """Downloads videos from s3 to a local temp directory and then loads them into numpy arrays, before
    deleting the temp directory (default behavior).

        Args:
            local_folder: local folder to temporarily download the videos to.
            s3_profile: s3 profile on amazon aws
            bucket: bucket name where the videos are stored
            filenames: list of filenames to download from s3
        Returns:
            videos: list of numpy arrays containing all the jamcam videos between the selected dates
            names: list of video filenames
        Raises:

    """
    my_bucket = connect_to_bucket(s3_profile, bucket)

    # Download the files
    for filename in filenames:
        try:
            my_bucket.download_file(filename, local_folder + filename.split('/')[-1].replace(
                ':', '-').replace(" ", "_"))
        except:
            print("Could not download " + filename)
    return load_videos_into_np(local_folder)


def delete_and_recreate_dir(temp_dir):
    """
    Creates an empty local directory for downloading from s3.
    Will wipe local_dir if already a directory
    """
    if (os.path.isdir(temp_dir)):
        shutil.rmtree(temp_dir)
    os.mkdir(temp_dir)
    return


def mp4_to_npy(local_mp4_path):
    """Load mp4 filename from the local directory and turn it into a numpy array"""
    cap = cv2.VideoCapture(local_mp4_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    buf = np.empty((frame_count, frame_height, frame_width, 3),
                   np.dtype('uint8'))
    fc = 0
    ret = True
    while (fc < frame_count and ret):
        ret, buf[fc] = cap.read()
        fc += 1
    cap.release()

    if(buf.size == 0):
        raise Exception('Numpy array is empty')

    return buf


def connect_to_bucket(profile_dir, bucket_name):
    """Connects to the s3 bucket"""
    # Set up boto3 session
    s3_session = boto3.Session(profile_name=profile_dir)
    s3_resource = s3_session.resource('s3')
    my_bucket = s3_resource.Bucket(bucket_name)

    return my_bucket


def retrieve_tims_from_s3():

    return


def retrieve_cam_details_from_database():

    return


def describe_s3_bucket(paths):
    """Plot the number of videos in the s3 bucket for each date.
    Plot is saved locally under plots/01_exploratory.

            Args:

            Returns:

            Raises:

        """
    my_bucket = connect_to_bucket(paths['s3_profile'], paths['bucket_name'])

    # Get list of all dates in the s3 bucket
    objects = my_bucket.objects.filter(Prefix="raw/video_data_new/")
    files = [obj.key for obj in objects]
    dates = []
    for filename in files:
        res = re.search("([0-9]{4}\-[0-9]{2}\-[0-9]{2})", filename)
        dates.append(res.group())

    # Plot a bar chart of the dates in the s3 bucket
    unique_dates, counts = np.unique(dates, return_counts=True)
    plt.figure()
    plt.bar(np.arange(unique_dates.shape[0]), counts)
    plt.xticks(np.arange(unique_dates.shape[0]),
               unique_dates, rotation='vertical')
    plt.xlabel('Date')
    plt.ylabel('Number of Videos')
    plt.tight_layout()
    plt.savefig(paths['plots'] + '01_exploratory/s3_description.pdf')
    plt.close()

    return


def append_to_csv(filepath: str, df: pd.DataFrame, columns: list, dtype: dict):
    if df.empty:
        return
    # check if filepath exists
    if not os.path.exists(filepath):
        df_main = pd.DataFrame(columns=columns)
        df_main.to_csv(filepath)
    df_main = pd.read_csv(filepath, dtype=dtype)
    df_main = df_main.append(df)
    df_main.to_csv(filepath, columns=columns, index=False)
