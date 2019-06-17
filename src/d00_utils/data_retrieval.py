import boto3
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import re
import shutil
import datetime

def retrieve_videos_from_s3(bDownload=True, from_date=None, to_date=None):

    if (from_date == None):
        from_date = '2019-06-01'
    if (to_date == None):
        to_date = str(datetime.datetime.now())[:10]

    s3_session = boto3.Session(profile_name='dssg')
    bucket_name = 'air-pollution-uk'
    local_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                             '..', '..', 'data/01_raw/jamcams/')
    shutil.rmtree(local_dir)
    os.mkdir(local_dir)
    s3_resource = s3_session.resource('s3')
    my_bucket = s3_resource.Bucket(bucket_name)

    selected_files = []
    objects = my_bucket.objects.filter(Prefix="raw/video_data_new/")
    files = [obj.key for obj in objects]
    for file in files:
        date = re.search("([0-9]{4}\-[0-9]{2}\-[0-9]{2})", file).group()
        if (date >= from_date and date <= to_date):
            selected_files.append(file)

    data = []

    if(bDownload):

        for file in selected_files:
            my_bucket.download_file(file, local_dir + file.split('/')[-1])

        for file in os.listdir(local_dir):
            cap = cv2.VideoCapture(local_dir + file)

            frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            buf = np.empty((frameCount, frameHeight, frameWidth, 3), np.dtype('uint8'))
            fc = 0
            ret = True
            while (fc < frameCount and ret):
                ret, buf[fc] = cap.read()
                fc += 1
            cap.release()
            data.append(buf)

    else:
        ### NOT WORKING ###
        #TODO find a way to get videos directly into computer memory
        for file in selected_files:
            obj = s3_resource.Object(bucket_name, file)
            response = obj.get()
            result = response['Body'].read()
            video = cv2.VideoCapture.open(BytesIO(result))
            data.append(video)

    return data


def retireve_tims_from_database():



    return


def retrieve_cam_details_from_database():




    return


def describe_s3_bucket():
    s3_session = boto3.Session(profile_name='dssg')
    bucket_name = 'air-pollution-uk'
    s3_resource = s3_session.resource('s3')
    my_bucket = s3_resource.Bucket(bucket_name)

    objects = my_bucket.objects.filter(Prefix="raw/video_data_new/")
    files = [obj.key for obj in objects]

    dates = []
    for file in files:
        res = re.search("([0-9]{4}\-[0-9]{2}\-[0-9]{2})", file)
        dates.append(res.group())

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

retrieve_videos_from_s3(from_date='2019-06-06', to_date='2019-06-07')
#describe_s3_bucket()