import collect_tims_data
from collect_video_data import collect_video_data, collect_camera_videos, upload_videos
from collect_tims_data import get_tims_data_and_upload_to_s3
import os
import sys
import configparser
import ast
import time


if __name__ == "__main__":
    setup_dir = os.path.join(os.getcwd(),'.')
    src_dir = os.path.join(setup_dir, 'src')
    sys.path.append(src_dir)
    print(os.path.join(setup_dir, 'conf', 'base', 'parameters.yml'))

    # credentials
    config = configparser.ConfigParser()
    config.read(os.path.join(setup_dir, 'conf', 'local', 'credentials.yml'))
    config.read(os.path.join(setup_dir, 'conf', 'base', 'parameters.yml'))

    # local data folder
    video_dir = os.path.join(setup_dir, 'data', '01_raw', 'video_data')
    tims_dir = os.path.join(setup_dir, 'data', '01_raw', 'tims')

    uploaded_file = os.path.join(setup_dir, 'data', '01_raw', 'uploaded_file.json')

    # # videos
    # collect_video_data(local_video_dir=video_dir,
    #                camera_list=ast.literal_eval(config.get('DATA_COLLECTION','camera_list')), 
    #                website = ast.literal_eval(config.get('DATA_COLLECTION', 'jam_cam_website')),
    #                num_iterations = 2)

    # tims data
    # get_tims_data_and_upload_to_s3(local_tims_dir = tims_dir,
    #                           file_website = ast.literal_eval(config.get('DATA_COLLECTION', 'tims_file_website')),
    #                           download_website = ast.literal_eval(config.get('DATA_COLLECTION', 'tims_download_website')), 
    #                           uploaded_file = uploaded_file)    
    
    while True:
        collect_camera_videos(local_video_dir=video_dir)
        time.sleep(2 * 60)

  