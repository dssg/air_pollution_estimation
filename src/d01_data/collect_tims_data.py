import urllib.request
import os
import subprocess
from bs4 import BeautifulSoup
import requests


def get_tims_data_and_upload_to_s3(local_tims_dir: str,
                                   file_website: str = 'https://s3-eu-west-1.amazonaws.com/roads.data.tfl.gov.uk',
                                   download_website: str = "http://roads.data.tfl.gov.uk/TIMS/",
                                   uploaded_file: str = "uploaded_file.txt",
                                   chunk_size = 5
                                   ):
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


    if not os.path.exists(uploaded_file):
        uploaded_files_list = []
    else:
        with open(uploaded_file, 'r') as f:
            uploaded_files_list = f.read().splitlines() 

    files = [filename for filename in  files if filename not in uploaded_files_list]
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
        resp = [urllib.request.urlretrieve( download_website + filename, local_tims_dir + "/" + filename) for filename in chunk]

        res = subprocess.call(["aws", "s3", 'cp',
                                local_tims_dir,
                                's3://air-pollution-uk/raw/tims_data/',
                                '--recursive',
                                '--profile',
                                'dssg'])

        with open(uploaded_file, "a") as f:
            print("\n".join(chunk), file=f)
            print("Saved ", chunk)

        files = files[chunk_size:]


    
