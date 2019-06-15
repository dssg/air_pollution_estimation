import urllib.request
import datetime
import os
import subprocess
import time
import json
from collections import defaultdict

def collect_camera_videos(local_video_dir:str,
    website:str = "https://api.tfl.gov.uk/Place/Type/JamCam",
    cam_file:str="cam_file.json"):

    # check if api is working
    if not os.path.exists(local_video_dir):
        os.makedirs(local_video_dir)
    res = urllib.request.urlopen(website)
    data = json.loads(res.read())
    new_video_urls = defaultdict()
    if not os.path.exists(cam_file):
        video_urls_dict = defaultdict(str)
    else:
        with open(cam_file, 'r') as f:
            video_urls_dict = dict(json.loads(f.read()))

    # parse data
    for item in data:
        additionalProperties = item['additionalProperties']
        for prop in additionalProperties:
            video_url = prop['value']
            if  prop['key'] == 'videoUrl':
                filename = prop['value'].split('/')[-1]
                timestamp = prop['modified']
                file_path = os.path.join(local_video_dir, timestamp+"_"+filename)
                print("Checking if video already exist")

                # check if video already exist
                if filename in video_urls_dict and video_urls_dict[filename] == timestamp:
                    print("Video already exist")
                    continue

                # download video
                print("Downloading videos to ", file_path)
                urllib.request.urlretrieve(prop['value'], file_path)
                new_video_urls[filename] = prop['modified']
                with open(cam_file, 'w') as f:
                    json.dump(new_video_urls, f)

def collect_available_camera_videos(local_video_dir:str,
    num_iterations:int = None,
    website:str = "https://api.tfl.gov.uk/Place/Type/JamCam",
    cam_file:str="cam_file.json"):
    '''
    This function was created to download videos from cameras that are marked as available in th tfl data. The json data returned by tfl api contains a key, in the "additionalProperties" field, called "available". Our assumption was that the "available" property means that the camera is available if the value is "true" and not available otherwise. 
    However, after going through the data, we discovered that somee cameras are working when the "available" property is "false".

    We plan to investigate this before removing this function.
    '''
    # check if api is working
    if not os.path.exists(local_video_dir):
        os.makedirs(local_video_dir)
    res = urllib.request.urlopen(website)
    data = json.loads(res.read())
    new_video_urls = defaultdict()
    if not os.path.exists(cam_file):
        video_urls_dict = defaultdict(str)
    else:
        with open(cam_file, 'r') as f:
            video_urls_dict = dict(json.loads(f.read()))

    # parse data
    for item in data:
        additionalProperties = item['additionalProperties']
        properties = {val["key"]:val for val in additionalProperties}
        available_prop = properties["available"]
        if available_prop["value"] == "true":
            video_prop = properties["videoUrl"]
            video_url = video_prop['value']
            filename = video_url.split('/')[-1]
            timestamp = video_prop['modified']
            file_path = os.path.join(local_video_dir, timestamp+"_"+filename)
            print("Checking if video already exist")

            # check if video already exist
            if filename in video_urls_dict and video_urls_dict[filename] == timestamp:
                print("Video already exist")
                continue

            # download video
            print("Downloading videos to ", file_path)
            urllib.request.urlretrieve(prop['value'], file_path)
            new_video_urls[filename] = prop['modified']

            with open(cam_file, 'w') as f:
                json.dump(new_video_urls, f)
        
def upload_videos(local_video_dir:str):
    if not os.path.exists(local_video_dir):
        os.makedirs(local_video_dir)
    for filename in os.listdir(local_video_dir):
        filepath = os.path.join(local_video_dir, filename)
        f = open(filepath, 'r')
        res = subprocess.call(["aws", "s3", 'cp',
                            filepath,
                            's3://air-pollution-uk/raw/video_data_new/',
                            # '--recursive',
                            '--profile',
                            'dssg'])
        print(type(res),res)

        # delete file if it was successfully uploaded
        if res == 0:
            # delete file
            res = subprocess.call(["rm", 
                            filepath
                           ])