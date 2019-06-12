import collect_tims_data
from collect_video_data import collect_video_data
from collect_tims_data import get_tims_data_and_upload_to_s3
import os
import sys
import configparser
import ast

if __name__ == "__main__":

    setup_dir = os.path.join(os.getcwd(),'..', '..')
    config = configparser.ConfigParser()
    config.read(os.path.join(setup_dir, 'conf', 'base', 'parameters.yml'))
    video_dir = os.path.join(setup_dir, 'data', '01_raw', 'video_data')

    collect_video_data(local_video_dir=video_dir,
                   camera_list=ast.literal_eval(config.get('DATA_COLLECTION','camera_list')), 
                   website = ast.literal_eval(config.get('DATA_COLLECTION', 'jam_cam_website')),
                   num_iterations = None)