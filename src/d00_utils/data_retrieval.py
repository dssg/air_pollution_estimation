import boto3
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import re
import shutil
import datetime
import glob
import threading


def retrieve_single_video_s3_to_np(camera, date, time, paths, bool_keep_data=True):
    """Retrieve one jamcam video from the s3 bucket based on the details specified.
        Downloads to a local directory and then loads into a numpy array.

            Args:
                camera: camera number as a string (XXXXX)
                date: date as a string (YYYY-MM-DD)
                time: time as a string (HH:MM:SS.MsMsMsMsMsMs)
                paths: dictionary containing raw_video, s3_video, s3_profile and bucket_name paths
                bool_keep_data: boolean for keeping the downloaded data in the local folder
            Returns:
                numpy array containing the jamcam video
            Raises:

        """
    # 2019-06-20 13:25:31.321785_00001.03675.mp4

    create_local_dir(paths['raw_video'])
    timestamp = date + " " + time + "_00001." + camera
    s3_vid_key = paths['s3_video'] + timestamp + '.mp4'
    s3_bucket = connect_to_bucket(paths['s3_profile'], paths['bucket_name'])

    file_dir = paths['raw_video'] + date + "_" + time + "_" + camera + '.mp4'
    s3_bucket.download_file(s3_vid_key, file_dir)
    buf = mp4_to_npy(file_dir)

    if (not bool_keep_data):
        shutil.rmtree(paths['raw_video'])

    return buf


def retrieve_videos_s3_to_np(paths, from_date='2019-06-01', to_date=str(datetime.datetime.now())[:10],
                             from_time='00-00-00', to_time='23-59-59', camera_list=None, bool_keep_data=True):
    """Retrieve jamcam videos from the s3 bucket based on the dates specified.
    Downloads to a local directory and then loads them into numpy arrays.

        Args:
            paths: dictionary containing raw_video, s3_profile and bucket_name paths
            from_date: start date (inclusive) for retrieving videos, if None then will retrieve from 2019-06-01 onwards
            to_date: end date (inclusive) for retrieving vidoes, if None then will retrieve up to current day
            from_time: start time for retrieving videos, if None then will retrieve from the start of the day
            to_time: end time for retrieving videos, if None then will retrieve up to the end of the day
            camera_list: list of cameras to retrieve from, if None then retrieve from all cameras
            bool_keep_data: boolean for keeping the downloaded data in the local folder
        Returns:
            videos: list of numpy arrays containing all the jamcam videos between the selected dates
            names: list of video file names
        Raises:

    """
    create_local_dir(paths['raw_video'])
    my_bucket = connect_to_bucket(paths['s3_profile'], paths['bucket_name'])

    from_date = datetime.datetime.strptime(from_date, '%Y-%m-%d').date()
    to_date = datetime.datetime.strptime(to_date, '%Y-%m-%d').date()
    from_time = datetime.datetime.strptime(from_time, '%H-%M-%S').time()
    to_time = datetime.datetime.strptime(to_time, '%H-%M-%S').time()

    dates = []

    # Generate the list of dates
    while(from_date <= to_date):
        dates.append(from_date)
        from_date += datetime.timedelta(days=1)

    # Download the files in each of the date folders on s3
    for date in dates:
        date = date.strftime('%Y-%m-%d')
        objects = my_bucket.objects.filter(Prefix="raw/videos/" + date + "/")

        selected_files = []

        for obj in objects:
            file = obj.key
            time = re.search("([0-9]{2}\:[0-9]{2}\:[0-9]{2})", file).group()
            time = datetime.datetime.strptime(time, '%H:%M:%S').time()
            camera_id = file.split('_')[-1][:-4]
            if(not camera_list):
                if(time >= from_time and time <= to_time):
                    selected_files.append(file)
            else:
                if (time >= from_time and time <= to_time and camera_id in camera_list):
                    selected_files.append(file)

        for file in selected_files:
            try:
                my_bucket.download_file(file, paths['raw_video'] + file.split('/')[-1].replace(
                    ':', '-').replace(" ", "_"))
            except:
                print("Could not download " + file)

    # Load files into a list of numpy arrays using opencv
    videos = []
    names = []
    for file in os.listdir(paths['raw_video']):
        videos.append(mp4_to_npy(paths['raw_video'] + file))
        names.append(file.split('/')[-1])

    # Delete local data unless specified
    if(not bool_keep_data):
        shutil.rmtree(paths['raw_video'])

    return videos, names


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

def load_videos_from_local(paths):
    """Load video data from the local raw jamcam folder and return it as a list of numpy arrays

            Args:
                paths: dictionary containing raw_video, s3_profile and bucket_name paths
            Returns:
                videos: list of numpy arrays containing all the jamcam videos from the local raw jamcam folder
                names: list of video file names
            Raises:

        """
    files = glob.glob(paths['raw_video'] + '*.mp4')
    names = [vals.split('/')[-1] for vals in files]
    videos = []
    for file in files:
        videos.append(mp4_to_npy(file))

    return videos, names