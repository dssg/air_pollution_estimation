# requirements
import boto3
import cv2
import numpy as np
import os
import yaml

os.chdir(".")
with open('../../conf/base/paths.yml') as f:
    path = yaml.safe_load(f)['s3_paths']


def s3_to_local_mp4(camera, date, time, local_vid_dir, extension='mp4'):
    """ download mp4 to working directory
        Args:
            camera (str): camera name
            date (str): date of selected clip in YYYY-MM-DD
            time (str): time of selected clip in HH:MM
            local_vid_dir (str): local directory where the video will be stored
            extension (str): file extension

        Returns:
            local_vid_path (str): local path to the video file
    """

    # create s3 file path based on desired camera
    timestamp = date + "_" + time[:2] + '.' + time[3:]
    s3_vid_key = path['video_path'] + "/" + camera + "/" + timestamp + '.' + extension

    # convert to s3 bucket
    s3_session = boto3.Session(profile_name=path['profile_name'])
    s3_resource = s3_session.resource('s3')
    s3_bucket = s3_resource.Bucket(path['bucket'])

    # download to working directory of choice
    local_vid_path = local_vid_dir + camera + timestamp + '.' + extension
    s3_bucket.download_file(s3_vid_key, local_vid_path)

    return local_vid_path


def mp4_to_npy(local_vid_path):
    """ create np array file from mp4 file in same directory
        Args:
            local_vid_path (str): path to the local video

        Returns:
            vid_array (np.array): 4-D array of floats of a video loop in time, width, length, and RBG values
    """

    # use cv2 to read in mp4 file
    cap = cv2.VideoCapture(local_vid_path)

    # setup vid_array np skeleton according to captured file
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    vid_array = np.empty((frame_count, frame_height, frame_width, 3), np.dtype('uint8'))

    # feed video data into vid_array
    fc = 0
    ret = True
    while fc < frame_count and ret:
        ret, vid_array[fc] = cap.read()
        fc += 1

    # not sure what this does
    vid_mp4.release()
    cv2.waitKey(0)

    # save file to mp4 directory as .npy file
    pre, ext = os.path.splitext(local_vid_path)
    np.save(pre, vid_array)

    return vid_array
