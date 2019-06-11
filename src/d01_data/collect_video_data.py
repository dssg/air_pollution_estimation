import urllib.request
import datetime
import os
import subprocess
import time


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
                           'air-quality'])
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
            print('Completed Iteration')
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






