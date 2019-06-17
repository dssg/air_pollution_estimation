# requirements
import boto3
import cv2
import numpy as np
import os


def s3_to_local_mp4(camera, date, time, extension, local_mp4_path):
    """ download mp4 to working directory """

    # create s3 file path based on desired camera
    timestamp = date[:4] + "-" + date[4:6] + "-" + date[6:] + "_" + time[:2] + '.' + time[
                                                                                     2:]  # assumes datetime = date as yyyymmdd + time as hhmm
    s3_vid_key = "raw" + "/" + "video_data" + "/" + camera + "/" + timestamp + extension

    # convert to s3 bucket
    s3_session = boto3.Session(profile_name='dssg')
    s3_resource = s3_session.resource('s3')
    bucket_name = 'air-pollution-uk'
    s3_bucket = s3_resource.Bucket(bucket_name)

    # download to working directory of choice
    s3_bucket.download_file(s3_vid_key, local_mp4_path)


def mp4_to_npy(local_mp4_path):
    """ create np array file from mp4 file in same directory """

    # use cv2 to read in mp4 file
    vid_mp4 = cv2.VideoCapture(local_mp4_path)

    # setup vid_array np skeleton according to captured file
    frame_count = int(vid_mp4.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(vid_mp4.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(vid_mp4.get(cv2.CAP_PROP_FRAME_HEIGHT))
    vid_array = np.empty((frame_count, frame_height, frame_width, 3), np.dtype('uint8'))

    # feed video data into vid_array
    fc = 0
    ret = True
    while fc < frame_count and ret:
        ret, vid_array[fc] = vid_mp4.read()
        fc += 1

    # not sure what this does
    vid_mp4.release()
    cv2.waitKey(0)

    # save file to mp4 directory as .npy file
    pre, ext = os.path.splitext(local_mp4_path)
    np.save(pre, vid_array)

    return vid_array
