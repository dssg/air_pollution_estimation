import urllib.request
import datetime
import os
import subprocess
import time
import json
from collections import defaultdict


def download_jam_cams(website, camera, extension, video_dir):

    now = datetime.datetime.now()
    timestamp = now.strftime("%Y-%m-%d_%H.%M")
    url = website + camera + extension
    file_path = video_dir + "/" + camera + "/" + timestamp + extension
    urllib.request.urlretrieve(url, file_path)


def create_folder_structure(local_video_dir: str,
                            camera_list: list):
    if not os.path.exists(local_video_dir):
        os.makedirs(local_video_dir)
    for camera in camera_list:
        this_camera_dir = os.path.join(local_video_dir, camera)
        if not os.path.exists(this_camera_dir):
            os.makedirs(this_camera_dir)


def get_videos_and_upload_to_s3(local_video_dir: str,
                                camera_list: list,
                                website: str = "http://jamcams.tfl.gov.uk/00001.",
                                extension: str ='.mp4',
                                ):

    create_folder_structure(local_video_dir=local_video_dir,
                            camera_list=camera_list)

    for camera in camera_list:
        download_jam_cams(website, camera, extension, local_video_dir)

    time.sleep(1 * 60)
    res = subprocess.call(["aws", "s3", 'cp',
                           local_video_dir,
                           's3://air-pollution-uk/raw/video_data/',
                           '--recursive',
                           '--profile',
                           'dssg'])

    print(res)
    res = subprocess.call(["rm", "-r",
                           local_video_dir
                           ])


def collect_video_data(local_video_dir: str,
                       camera_list: list,
                       num_iterations: int = None,
                       website: str = "http://jamcams.tfl.gov.uk/00001.",):
    upload_num = 0

    if num_iterations is None:
        print('Starting infinite data collection.')
        while True:
            get_videos_and_upload_to_s3(local_video_dir=local_video_dir,
                                        camera_list=camera_list,
                                        website=website)
            upload_num += 1
            print('Completed {} iterations'.format(upload_num))
            time.sleep(3 * 60)

    else:
        print('Starting data collection.')
        while upload_num < num_iterations:
            get_videos_and_upload_to_s3(local_video_dir=local_video_dir,
                                        camera_list=camera_list,
                                        website=website)
            upload_num += 1
            print('Completed {}/{} iterations'.format(upload_num,
                                                      num_iterations))
            time.sleep(3 * 60)

def collect_camera_videos(local_video_dir:str,
    num_iterations:int = None,
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
        
def upload_videos(local_video_dir:str, chunk_size:int = 5):

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

