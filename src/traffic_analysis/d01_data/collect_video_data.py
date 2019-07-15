import urllib.request
import datetime
import os
import subprocess
import time
import json
from collections import defaultdict
from email_service import send_email_warning
import datetime


def download_camera_data(tfl_cam_api: str = "https://api.tfl.gov.uk/Place/Type/jamcams",
                         cam_file: str = "data/00_ref/cam_file.json"):
    '''
    Gets a list of camera ids and info from tfl api
    '''
    # get the traffic cameras data
    res = urllib.request.urlopen(tfl_cam_api)
    data = json.loads(res.read())
    camera_list = {val["id"]: val for val in data}

    # save camera info to file
    with open(cam_file, "w") as f:
        json.dump(camera_list, f)


def collect_camera_videos(local_video_dir: str,
                          download_url: str = "https://s3-eu-west-1.amazonaws.com/jamcams.tfl.gov.uk/",
                          cam_file: str = "data/00_ref/cam_file.json",
                          iterations: int = None,
                          delay: int = 3):
    '''
    This function was created to download videos from cameras using the tfl api.
        local_video_dir: local directly to download the videos in

        download_url: the tfl api to download traffic camera videos

        cam_file: stores the last time the camera was modified. The file is checked in ordere to download new videos

        iterations: number of times the download should run. The video are downloaded continuously if no value is supplied

        delay: amount of time (minutes) to wait for before downloading new data

    '''
    # check if the local directory exists.
    if not os.path.exists(local_video_dir):
        os.makedirs(local_video_dir)

    # get all the data in the cam_file
    video_urls_dict = {}
    if not os.path.exists(cam_file):
        download_camera_data()
    with open(cam_file, 'r') as f:
        video_urls_dict = dict(json.loads(f.read()))
    iteration = 0
    while True:
        count = 0
        # download videos for camera
        for camera_id in video_urls_dict.keys():
            count += 1
            camera_id = camera_id.replace("JamCams_", "")
            filename = camera_id + ".mp4"
            file_path = os.path.join(download_url, filename)
            timestamp = str(datetime.datetime.now())
            local_path = os.path.join(
                local_video_dir, "%s_%s" % (timestamp, filename))

            # download video
            print("Downloading videos to ", file_path)
            try:
                urllib.request.urlretrieve(file_path, local_path)
            except Exception as e:
                send_email_warning(str(e), "Video download failed!")
        iteration += 1
        if iteration == iterations:
            break
        if delay:
            time.sleep(delay * 60)
 
def upload_videos(local_video_dir: str, iterations=None, delay: int = None):
    '''
    This function uploads the video in the local_video_dir to S3. Each video is deleted after an upload.
        local_video_dir: local directly where the videos are stored

        iterations: number of times the upload should run. The local video directory is checked continuously for new videos if no value is supplied

        delay: amount of time (minutes) to wait for before downloading new data
    '''
    if not os.path.exists(local_video_dir):
        os.makedirs(local_video_dir)

    iteration = 0
    while True:
        try:
            res = subprocess.call(["aws", "s3", 'mv',
                                   local_video_dir,
                                   's3://air-pollution-uk/raw/video_data_new/',
                                   '--recursive',
                                   '--profile',
                                   'dssg'])
        except Exception as e:
            send_email_warning(str(e), "Video upload failed.")
        iteration += 1
        if iteration == iterations:
            break
        if delay:
            time.sleep(delay * 60)
