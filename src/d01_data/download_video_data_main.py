from collect_video_data import collect_camera_videos
import os
import sys
import configparser
import time
from email_service import send_email_warning

if __name__ == "__main__":
    # local data folder
    setup_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
    video_dir = os.path.join(setup_dir, 'data', '01_raw', 'video_data')
    cam_file = os.path.join(setup_dir, 'data', '00_ref', 'cam_file.json')
    while True:
        try:
            collect_camera_videos(
                local_video_dir=video_dir,
                cam_file=cam_file,
                check_if_video_is_available=False)
            time.sleep(2 * 60)
        except Exception as e:
            send_email_warning(str(e))
