import collect_tims_data
from collect_video_data import collect_camera_videos
from collect_tims_data import get_tims_data_and_upload_to_s3
import os
import sys
import configparser
import ast
import time
from email_service import send_email

if __name__ == "__main__":
    setup_dir = os.path.join(os.getcwd(),'.')
    print(os.path.join(setup_dir, 'conf', 'base', 'parameters.yml'))

    # local data folder
    video_dir = os.path.join(setup_dir, 'data', '01_raw', 'video_data')
    tims_dir = os.path.join(setup_dir, 'data', '01_raw', 'tims')
    uploaded_file = os.path.join(setup_dir, 'data', '01_raw', 'uploaded_file.json')
    while True:
        try:
            send_email()
            collect_camera_videos(local_video_dir=video_dir)
            time.sleep(2 * 60)
            send_email()
        except Exception as e:
            send_email()