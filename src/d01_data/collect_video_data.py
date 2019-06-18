import urllib.request
import datetime
import os
import subprocess
import time
import json
from collections import defaultdict


def collect_camera_videos(local_video_dir: str,
                          website: str = "https://api.tfl.gov.uk/Place/Type/JamCam",
                          cam_file: str = "data/00_ref/cam_file.json",
                          check_if_video_is_available=False):
    '''
    This function was created to download videos from cameras using the tfl api.
        local_video_dir: local directly to download the videos in.

        website: The tfl api to download traffic camera videos

        cam_file: Stores the last time the camera was modified. The file is checked in ordere to download new videos.

        check_if_video_is_available: if set to True, check the "available" of the camera data to see if it's set to "true" before downloading the video. The json data returned by tfl api contains a key, in the "additionalProperties" field, called "available". Our assumption is that the "available" property means the camera is available and working if the value is "true" and not available otherwise. However, after going through the data, we discovered that some cameras are working when the "available" property is "false".We plan to investigate this before removing the check_if_video_is_available argument.
    '''
    # check if the local directory exists.
    if not os.path.exists(local_video_dir):
        os.makedirs(local_video_dir)

    # get the traffic cameras data
    res = urllib.request.urlopen(website)
    data = json.loads(res.read())
    new_video_urls = defaultdict()

    # get all the data in the cam_file to check the last time the video data were modified
    if not os.path.exists(cam_file):
        video_urls_dict = defaultdict(str)
    else:
        with open(cam_file, 'r') as f:
            video_urls_dict = dict(json.loads(f.read()))

    # parse data
    for item in data:
        additional_properties = item['additionalProperties']
        properties = {val["key"]: val for val in additional_properties}
        available_prop = properties["available"]
        if not check_if_video_is_available or available_prop["value"] == "true":
            video_prop = properties["videoUrl"]
            video_url = video_prop['value']
            filename = video_url.split('/')[-1]
            timestamp = video_prop['modified']
            file_path = os.path.join(local_video_dir, timestamp+"_"+filename)

            # check if the video data has been modified
            print("Checking if video already exist")
            if filename in video_urls_dict and video_urls_dict[filename] == timestamp:
                print("Video already exist")
                continue

            # download video
            print("Downloading videos to ", file_path)
            urllib.request.urlretrieve(video_prop['value'], file_path)
            new_video_urls[filename] = video_prop['modified']

            # store the modified time of video in cam_file
            with open(cam_file, 'w') as f:
                json.dump(new_video_urls, f)
        else:
            print("%s video not available." % (item['id']))


def upload_videos(local_video_dir: str):
    '''
    This function uploads the video in the local_video_dir to S3. Each video is deleted after an upload.
    '''
    if not os.path.exists(local_video_dir):
        os.makedirs(local_video_dir)
    for filename in os.listdir(local_video_dir):
        filepath = os.path.join(local_video_dir, filename)
        f = open(filepath, 'r')
        res = subprocess.call(["aws", "s3", 'cp',
                               filepath,
                               's3://air-pollution-uk/raw/video_data_new/',
                               '--profile',
                               'dssg'])

        # delete file if it was successfully uploaded
        if res == 0:
            # delete file
            res = subprocess.call(["rm",
                                   filepath
                                   ])
