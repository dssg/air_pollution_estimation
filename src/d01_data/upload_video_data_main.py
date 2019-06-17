from collect_video_data import upload_videos
import os
import sys
import configparser
import time
from email_service import send_email_warning


if __name__ == "__main__":
    setup_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
    # credentials
    config = configparser.ConfigParser()
    config.read(os.path.join(setup_dir, 'conf', 'base', 'parameters.yml'))

    # local data folder
    video_dir = os.path.join(setup_dir, 'data', '01_raw', 'video_data')
    while True:
        try:
            upload_videos(local_video_dir=video_dir)
            time.sleep(2 * 60)
        except Exception as e:
            send_email_warning(str(e))
