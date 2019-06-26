import boto3
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import re
import shutil
import datetime


def retrieve_single_video(camera, date, time, paths, bool_keep_data=True):
    """Retrieve one jamcam video from the s3 bucket based on the details specified.
        Downloads to a local directory and then loads into a numpy array.

            Args:
                camera: camera number as a string (XXXXX)
                date: date as a string (YYYY-MM-DD)
                time: time as a string (HH:MM:SS.MsMsMsMsMsMs)
                paths: dictionary containing local_video, s3_video, s3_profile and bucket_name paths
                bool_keep_data: boolean for keeping the downloaded data in the local folder
            Returns:
                numpy array containing the jamcam video
            Raises:

        """
    # 2019-06-20 13:25:31.321785_00001.03675.mp4

    create_local_dir(paths['local_video'])
    timestamp = date + " " + time + "_00001." + camera
    s3_vid_key = paths['s3_video'] + timestamp + '.mp4'
    s3_bucket = connect_to_bucket(paths['s3_profile'], paths['bucket_name'])

    file_dir = paths['local_video'] + date + "_" + time + "_" + camera + '.mp4'
    s3_bucket.download_file(s3_vid_key, file_dir)
    buf = mp4_to_npy(file_dir)

    if (not bool_keep_data):
        shutil.rmtree(paths['local_video'])

    return buf


def retrieve_videos_based_on_dates(paths, from_date='2019-06-01', to_date=str(datetime.datetime.now())[:10], bool_keep_data=True):
    """Retrieve jamcam videos from the s3 bucket based on the dates specified.
    Downloads to a local directory and then loads them into numpy arrays.

        Args:
            paths: dictionary containing local_video, s3_profile and bucket_name paths
            from_date: start date (inclusive) for retrieving videos, if None then will retrieve from 2019-06-01 onwards
            to_date: end date (inclusive) for retrieving vidoes, if None then will retrieve up to current day
            bool_keep_data: boolean for keeping the downloaded data in the local folder
        Returns:
            list of numpy arrays containing all the jamcam videos between the selected dates
        Raises:

    """
    create_local_dir(paths['local_video'])
    my_bucket = connect_to_bucket(paths['s3_profile'], paths['bucket_name'])

    # Get list of files in s3 based on dates provided
    selected_files = []
    objects = my_bucket.objects.filter(Prefix="raw/video_data_new/")

    for obj in objects:
        file = obj.key
        try:
            date = re.search("([0-9]{4}\-[0-9]{2}\-[0-9]{2})", file).group()
            if (date >= from_date and date <= to_date):
                selected_files.append(file)
        except:
            print('Could not find date for: ' + file)

    # Download the selected files
    for file in selected_files:
        my_bucket.download_file(file, paths['local_video'] + file.split('/')[-1])

    # Load files into a list of numpy arrays using opencv
    data = []
    for file in os.listdir(paths['local_video']):
        data.append(mp4_to_npy(paths['local_video'] + file))

    # Delete local data unless specified
    if(not bool_keep_data):
        shutil.rmtree(paths['local_video'])

    return data


def create_local_dir(local_dir):
    """Creates the local directory for downloading from s3"""
    # Set local directory for downloading data, will overwrite whatever is currently there
    if (os.path.isdir(local_dir)):
        shutil.rmtree(local_dir)
    os.mkdir(local_dir)
    return


def mp4_to_npy(local_mp4_path):
    """Load mp4 file from the local directory and turn it into a numpy array"""
    cap = cv2.VideoCapture(local_mp4_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    buf = np.empty((frame_count, frame_height, frame_width, 3), np.dtype('uint8'))
    fc = 0
    ret = True
    while (fc < frame_count and ret):
        ret, buf[fc] = cap.read()
        fc += 1
    cap.release()

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
    for file in files:
        res = re.search("([0-9]{4}\-[0-9]{2}\-[0-9]{2})", file)
        dates.append(res.group())

    # Plot a bar chart of the dates in the s3 bucket
    unique_dates, counts = np.unique(dates, return_counts=True)
    plt.figure()
    plt.bar(np.arange(unique_dates.shape[0]), counts)
    plt.xticks(np.arange(unique_dates.shape[0]), unique_dates, rotation='vertical')
    plt.xlabel('Date')
    plt.ylabel('Number of Videos')
    plt.tight_layout()
    plt.savefig(paths['plots'] + '01_exploratory/s3_description.pdf')
    plt.close()

    return