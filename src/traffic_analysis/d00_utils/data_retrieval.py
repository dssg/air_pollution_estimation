import boto3
import cv2
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import shutil
import glob
import json


def load_video_names(paths):
    save_folder = 'raw_video'
    filepath = os.path.join(paths[save_folder], "searched_videos")
    with open(filepath, "r") as f:
        videos = list(json.load(f))
        return videos


def load_videos_into_np(folder):
    # Load files into a dict of numpy arrays using opencv
    video_dict = {}
    for file in glob.glob(folder + '*.mp4'):
        try:
            video_dict[file.split('/')[-1]] = mp4_to_npy(file)
        except:
            print("Could not convert " + file + " to numpy array")

    return video_dict


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


def load_videos_from_local(paths):
    """Load video data from the local raw jamcam folder and return it as a list of numpy arrays

            Args:
                paths: dictionary containing raw_video, s3_profile and bucket_name paths
            Returns:
                video_dict: dict of numpy arrays containing all the jamcam videos from the local raw jamcam folder
            Raises:

        """
    video_dict = {}
    files = glob.glob(paths['raw_video'] + '*.mp4')
    for file in files:
        try:
            video_dict[file.split('/')[-1]] = mp4_to_npy(file)
        except:
            print("Could not convert " + file + " to numpy array")

    return video_dict
