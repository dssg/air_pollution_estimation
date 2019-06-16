from collect_tims_data import get_tims_data_and_upload_to_s3
import os
import sys
import configparser
import ast

if __name__ == "__main__":
    setup_dir = os.path.join(os.getcwd(), '.')

    # credentials
    config = configparser.ConfigParser()
    config.read(os.path.join(setup_dir, 'conf', 'base', 'parameters.yml'))

    # local data folder
    tims_dir = os.path.join(setup_dir, 'data', '01_raw', 'tims')
    uploaded_file = os.path.join(
        setup_dir, 'data', '01_raw', 'uploaded_file.txt')

    # tims data
    get_tims_data_and_upload_to_s3(local_tims_dir=tims_dir,
                                            file_website=ast.literal_eval(config.get(
                                                'DATA_COLLECTION', 'tims_file_website')),
                                            download_website=ast.literal_eval(config.get(
                                                'DATA_COLLECTION', 'tims_download_website')),
                                            uploaded_file=uploaded_file)
