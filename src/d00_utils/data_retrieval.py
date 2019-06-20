import boto3
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import re
import shutil
import datetime


def retrieve_videos_from_s3(bool_keep_data=True, from_date=None, to_date=None):
    """Retrieve jamcam videos from the s3 bucket based on the dates specified.
    Downloads to a local directory and then loads them into numpy arrays.

        Args:
            bool_keep_data: boolean for keeping the downloaded data in the local folder
            from_date: start date (inclusive) for retrieving videos, if None then will retrieve from 2019-06-01 onwards
            to_date: end date (inclusive) for retrieving vidoes, if None then will retrieve up to current day
        Returns:
            list of numpy arrays containing all the jamcam videos between the selected dates
        Raises:

    """

    # Deal with the cases when a date isn't specified
    if (from_date == None):
        from_date = '2019-06-01'
    if (to_date == None):
        to_date = str(datetime.datetime.now())[:10]

    # Set local directory for downloading data, will overwrite whatever is currently there
    local_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                             '..', '..', 'data/01_raw/jamcams/')
    if (os.path.isdir(local_dir)):
        shutil.rmtree(local_dir)
    os.mkdir(local_dir)

    # Set up boto3 session
    s3_session = boto3.Session(profile_name='dssg')
    bucket_name = 'air-pollution-uk'
    s3_resource = s3_session.resource('s3')
    my_bucket = s3_resource.Bucket(bucket_name)

    # Get list of files in s3 based on dates provided
    selected_files = []
    objects = my_bucket.objects.filter(Prefix="raw/video_data_new/")
    files = [obj.key for obj in objects]
    for file in files:
        date = re.search("([0-9]{4}\-[0-9]{2}\-[0-9]{2})", file).group()
        if (date >= from_date and date <= to_date):
            selected_files.append(file)

    # Download the selected files
    data = []
    for file in selected_files:
        my_bucket.download_file(file, local_dir + file.split('/')[-1])

    # Load files into a list of numpy arrays using opencv
    for file in os.listdir(local_dir):
        cap = cv2.VideoCapture(local_dir + file)

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
        data.append(buf)

    # Delete local data unless specified
    if(not bool_keep_data):
        shutil.rmtree(local_dir)

    return data


def retrieve_tims_from_s3():



    return


def retrieve_cam_details_from_database():




    return


def describe_s3_bucket():
    """Plot the number of videos in the s3 bucket for each date.
    Plot is saved locally under plots/01_exploratory.

            Args:

            Returns:

            Raises:

        """
    # Set up boto3 session
    s3_session = boto3.Session(profile_name='dssg')
    bucket_name = 'air-pollution-uk'
    s3_resource = s3_session.resource('s3')
    my_bucket = s3_resource.Bucket(bucket_name)

    # Get list of all dates in the s3 bucket
    objects = my_bucket.objects.filter(Prefix="raw/video_data_new/")
    files = [obj.key for obj in objects]
    dates = []
    print(len(files))
    for file in files:
        # print(file)
        res = re.search("([0-9]{4}\-[0-9]{2}\-[0-9]{2})", file)
        if not res:
            res = re.search("([0-9]+\.[0-9]+)", file)
            timestamp = datetime.datetime.fromtimestamp(res.group()).strftime('%Y-%m-%d %H:%M:%S')
        else:
            timestamp = res.group()
            
        dates.append(timestamp)

    # Plot a bar chart of the dates in the s3 bucket
    unique_dates, counts = np.unique(dates, return_counts=True)
    plt.figure()
    plt.bar(np.arange(unique_dates.shape[0]), counts)
    plt.xticks(np.arange(unique_dates.shape[0]), unique_dates, rotation='vertical')
    plt.xlabel('Date')
    plt.ylabel('Number of Videos')
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                      '..', '..', 'plots/01_exploratory/s3_description.pdf'))
    plt.close()

    return

if __name__ == '__main__':
    describe_s3_bucket()