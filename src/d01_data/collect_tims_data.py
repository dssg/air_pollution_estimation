import urllib.request
import os
import subprocess
from bs4 import BeautifulSoup
import requests


def get_tims_data_and_upload_to_s3(local_tims_dir: str,
                                   file_website: str = 'https://s3-eu-west-1.amazonaws.com/roads.data.tfl.gov.uk',
                                   download_website: str = "http://roads.data.tfl.gov.uk/TIMS/"):
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

    for file in files:
        url = download_website + file
        file_path = local_tims_dir + "/" + file
        urllib.request.urlretrieve(url, file_path)

        res = subprocess.call(["aws", "s3", 'cp',
                               local_tims_dir,
                               's3://air-pollution-uk/raw/tims_data/',
                               '--recursive',
                               '--profile',
                               'dssg'])
        print(res)

    # delete local tims directory
    res = subprocess.call(["rm", "-r",
                           local_tims_dir
                           ])

    
