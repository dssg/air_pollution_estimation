import urllib.request
import os
import subprocess
from bs4 import BeautifulSoup
import requests


def get_tims_data_and_upload_to_s3(local_tims_dir: str,
                                   file_website: str = 'https://s3-eu-west-1.amazonaws.com/roads.data.tfl.gov.uk',
                                   download_website: str = "http://roads.data.tfl.gov.uk/TIMS/",
                                   tims_data_file: str = "tims_data_file.txt",
                                   chunk_size=5
                                   ):

    '''
    This function downloads tims data by crawling the links on the tims website. The function first checks if the file has already been downloaded by checking a text file that contains names of downloaded files.

        local_tims_dir: the local filepath used to save the tims data before uploading to S3.

        file_website: url containing names of files in tims data

        download_website: url where the files are stored for download

        tims_data_file: a text file containing the names of files that have been downloaded from tims website and uploaded to s3

        chunk_size: number of files to download at a single time and upload to s3
    '''
    # get html file from website
    r = requests.get(file_website)
    data = r.text
    soup = BeautifulSoup(data, features="html.parser")

    # extract the list of available file names
    keys = list(soup.find_all('key'))
    keys = [str(key) for key in keys]
    files = [key.replace('<key>', '').replace('</key>', '').replace('TIMS/', '') for key in keys
             if 'TIMS' in key and '.csv' in key]

    # create tims directory if it doesn't exist
    if not os.path.exists(local_tims_dir):
        os.makedirs(local_tims_dir)

    # get a list of uploaded tims data, create an empty list if the tims data file does not exist
    if not os.path.exists(tims_data_file):
        uploaded_files_list = []
    else:
        with open(tims_data_file, 'r') as f:
            uploaded_files_list = f.read().splitlines()

    # remove files that have already been downloaded to avoid duplicate download
    files = [filename for filename in files if filename not in uploaded_files_list]
    print(uploaded_files_list)
    while files:
        # delete local tims directory
        res = subprocess.call(["rm", "-r",
                               local_tims_dir
                               ])

        # create tims directory if it doesn't exist
        if not os.path.exists(local_tims_dir):
            os.makedirs(local_tims_dir)

        chunk = files[:chunk_size]
        resp = [urllib.request.urlretrieve(
            download_website + filename, local_tims_dir + "/" + filename) for filename in chunk]

        res = subprocess.call(["aws", "s3", 'cp',
                               local_tims_dir,
                               's3://air-pollution-uk/raw/tims_data/',
                               '--recursive',
                               '--profile',
                               'dssg'])

        # append recently saved file to the tims data file
        with open(tims_data_file, "a") as f:
            print("\n".join(chunk), file=f)
            print("Saved ", chunk)
        files = files[chunk_size:]
