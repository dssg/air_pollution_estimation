import urllib.request
import os
import subprocess
import requests
import shutil
import datetime
import time
from email_service import send_email_warning
import threading


def get_tims_data_and_upload_to_s3():
    """Retrieve TIMS data from tfl website and upload to s3 bucket.
        Downloads any backlogs and then continually checks for new files every 15 minutes
            Args:

            Returns:

            Raises:
    """
    website = "http://roads.data.tfl.gov.uk/TIMS/"

    # Set up the local directory for temporary downloading
    local_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                             '..', '..', 'data/01_raw/tims/')
    if (os.path.isdir(local_dir)):
    if os.path.isdir(local_dir):
        shutil.rmtree(local_dir)
    os.makedirs(local_dir)

    # Check every second of the datetime stored in 'date' for a csv file
    # If a csv file exists then we are not up to date and we should check the next 15 minutes
    # If we are up to date then just wait 15 minutes before checking again
    date = datetime.datetime(2019, 1, 1, 0, 0)
    counter = 0

    while True:
        counter += 1
        for i in range(60):
            # File to check for
            name = (
                "detdata"
                + date.strftime("%d%m%Y-%H%M")
                + "{num:02d}".format(num=i)
                + ".csv"
            )
            url = website + name
            if requests.head(url).status_code == requests.codes.ok:
                # Download
                t = threading.Thread(
                    name="upload_to_s3",
                    target=download_from_site(url, local_dir + name),
                )
                t.start()
                t.join()
                # Upload
                t = threading.Thread(
                    name="upload_to_s3", target=upload_to_s3(local_dir + name)
                )
                t.start()
                t.join()
                # Delete the file
                os.remove(local_dir + name)
                print("Processed TIMS file: " + name)
                break

        # Up to date so wait 15 minutes before checking again
        if date > datetime.datetime.now():
            time.sleep(900)
        else:
            date += datetime.timedelta(minutes=15)

        # Every 24 hours send an email to say we are up to date
        if counter % 96 == 0:
            send_email_warning(
                msg="TIMS data successfully collected in the last 24 hours",
                subj="TIMS Data Collection Successful",
            )
            counter = 0

    return


def download_from_site(url, file):
    urllib.request.urlretrieve(url, file)


def upload_to_s3(file):
    res = subprocess.call(
        [
            "aws",
            "s3",
            "cp",
            file,
            "s3://air-pollution-uk/raw/tims_data/",
            "--profile",
            "dssg",
        ]
    )
