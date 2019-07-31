import urllib.request
import os
import subprocess
from subprocess import PIPE, Popen
import time
import json
import datetime
import sys
import dateutil.parser
from traffic_analysis.d00_utils.data_loader_s3 import DataLoaderS3
from traffic_analysis.d00_utils.email_service import send_email_warning


def download_camera_meta_data(tfl_camera_api: str, paths: dict, s3_credentials: dict):
    """
    Gets a list of camera ids and info from tfl api
    """

    dl = DataLoaderS3(s3_credentials, bucket_name=paths["bucket_name"])

    camera_meta_data_path = paths["s3_camera_details"]

    if dl.file_exists(camera_meta_data_path):
        return

    # get the traffic cameras data
    res = urllib.request.urlopen(tfl_camera_api)
    data = json.loads(res.read())
    camera_list = {val["id"]: val for val in data}

    # save camera info to file
    dl.save_json(data=camera_list, file_path=camera_meta_data_path)


def collect_camera_videos(
    download_url: dict,
    s3_credentials: dict,
    paths: dict,
    iterations: int = None,
    delay: int = 3,
):
    """
    This function was created to download videos from cameras using the tfl api.
        download_url: the tfl api to download traffic camera videos
        s3_credentials: s3 credentials
        paths:
        cam_file: stores the last time the camera was modified. The file is checked in order to download new videos
        iterations: number of times the download should run. The video are downloaded continuously if no value is supplied
        delay: amount of time (minutes) to wait for before downloading new data
    """

    dl = DataLoaderS3(s3_credentials, bucket_name=paths["bucket_name"])

    video_urls_dict = dict(dl.read_json(paths["s3_camera_details"]))

    iteration = 0
    while True:
        count = 0
        # download videos for camera
        for camera_id in video_urls_dict.keys():
            count += 1

            camera_id = camera_id.replace("JamCams_", "")
            filename = camera_id + ".mp4"
            file_path = os.path.join(download_url, filename)
            datetime_obj = datetime.datetime.now()
            timestamp = datetime_obj.strftime("%Y%m%d-%H%M%S")
            local_video_dir_date = "%s/%s/" % (
                paths["temp_raw_video"],
                datetime_obj.date(),
            )

            # check if the local directory exists.
            if not os.path.exists(local_video_dir_date):
                os.makedirs(local_video_dir_date)
            local_path = os.path.join(
                local_video_dir_date, "%s_%s" % (timestamp, filename)
            )

            # download video
            print("Downloading videos from ", file_path)
            try:
                urllib.request.urlretrieve(file_path, local_path)
            except Exception as e:
                send_email_warning(str(e), "Video download failed!")
        iteration += 1
        if iteration == iterations:
            break
        if delay:
            time.sleep(delay * 60)


def upload_videos(paths: dict, iterations=None, delay: int = None):
    """
    This function uploads the video in the local_video_dir to S3. Each video is deleted after an upload.
    Args:
        local_video_dir: local directly where the videos are stored
        paths: Contains the s3 folder to save the videos, bucket name, and s3 profile
        iterations: number of times the upload should run. The local video directory is checked continuously for new videos if no value is supplied
        delay: amount of time (minutes) to wait for before downloading new data
    """

    local_video_dir = paths["temp_raw_video"]
    if not os.path.exists(local_video_dir):
        return
    s3_folder = paths["s3_video"]
    bucket_name = paths["bucket_name"]
    s3_profile = paths["s3_profile"]
    target_dir_on_s3 = "s3://%s/%s" % (bucket_name, s3_folder)

    iteration = 0
    while True:
        try:
            res = subprocess.call(
                [
                    "aws",
                    "s3",
                    "mv",
                    local_video_dir,
                    target_dir_on_s3,
                    "--recursive",
                    "--profile",
                    s3_profile,
                ]
            )
        except Exception as e:
            send_email_warning(str(e), "Video upload failed.")
        iteration += 1
        if iteration == iterations:
            break
        print(delay)
        if delay:
            time.sleep(delay * 60)


def rename_videos(paths, params, chunk_size=100):
    bucket_name = paths["bucket_name"]
    s3_profile = paths["s3_profile"]
    s3_folder = "s3://%s/%s" % (bucket_name, params["old_path"])
    date_format = params["date_format"]

    if len(sys.argv) > 1:
        chunk_size = sys.argv[1]
    while True:
        start = time.time()
        ls = Popen(
            [
                "aws",
                "s3",
                "ls",
                s3_folder,
                "--summarize",
                "--recursive",
                "--profile",
                s3_profile,
            ],
            stdout=PIPE,
        )
        p1 = Popen(["awk", '{$1=$2=$3=""; print $0}'], stdin=ls.stdout, stdout=PIPE)
        p2 = Popen(["head", "-n " + str(chunk_size)], stdin=p1.stdout, stdout=PIPE)
        ls.stdout.close()
        p1.stdout.close()
        output = p2.communicate()[0]
        p2.stdout.close()
        files = output.decode("utf-8").split("\n")
        files = list(filter(lambda x: x.strip(), files))
        if not files:
            break
        for full_path in files:
            full_path = full_path.strip()
            try:
                if full_path:
                    old_filename = "s3://%s/%s" % (bucket_name, full_path)
                    filename = full_path.split("/")[-1]
                    filename = filename.strip()
                    res = filename.split("_")
                    datetime_obj = dateutil.parser.parse(res[0])
                    timestamp = datetime_obj.strftime(date_format)
                    new_filename = "_".join([timestamp, res[1]])
                    new_filename = "s3://%s/%s/%s/%s" % (
                        bucket_name,
                        params["new_path"],
                        str(datetime_obj.date()),
                        new_filename,
                    )
                    res = subprocess.call(
                        [
                            "aws",
                            "s3",
                            "mv",
                            old_filename,
                            new_filename,
                            "--profile",
                            s3_profile,
                        ]
                    )
            except Exception as e:
                print(e)
        end = time.time()
        print(end - start)
