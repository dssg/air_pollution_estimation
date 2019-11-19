import urllib.request
import os
import time
import json
import datetime
import sys
import subprocess
from subprocess import PIPE, Popen
import dateutil.parser

from traffic_analysis.d00_utils.data_loader_blob import DataLoaderBlob
from traffic_analysis.d00_utils.email_service import send_email_warning
from traffic_analysis.d00_utils.load_confs import load_paths
from traffic_analysis.d00_utils.video_helpers import parse_video_or_annotation_name


paths = load_paths()


def download_camera_meta_data(tfl_camera_api: str,
                              blob_credentials: dict):
    """
    Gets a list of camera ids and info from tfl api
    """

    dl = DataLoaderBlob(blob_credentials)

    camera_meta_data_path = paths['blob_camera_details']

    if dl.file_exists(camera_meta_data_path):
        return

    # get the traffic cameras data
    res = urllib.request.urlopen(tfl_camera_api)
    data = json.loads(res.read())
    camera_list = {val["id"]: val for val in data}

    # save camera info to file
    dl.save_json(data=camera_list, file_path=camera_meta_data_path)


def collect_camera_videos(download_url: dict,
                          blob_credentials: dict,
                          iterations: int = None,
                          delay: int = 3):
    """
    This function was created to download videos from cameras using the tfl api.
    Args:
        download_url: the tfl api to download traffic camera videos
        blob_credentials: blob credentials
        paths:
        cam_file: stores the last time the camera was modified. The file is checked in order to download new videos
        iterations: number of times the download should run. The video are downloaded continuously if no value is supplied
        delay: amount of time (minutes) to wait for before downloading new data
    """

    dl = DataLoaderBlob(blob_credentials)
    video_urls_dict = dict(dl.read_json(paths['blob_camera_details']))

    # continuously download videos if iterations is None, otherwise stop after n iterations
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
            local_video_dir_date = "%s/%s/" % (paths['temp_raw_video'],
                                               datetime_obj.date())

            # check if the local directory exists.
            if not os.path.exists(local_video_dir_date):
                os.makedirs(local_video_dir_date)
            local_path = os.path.join(
                local_video_dir_date, "%s_%s" % (timestamp, filename))

            # download video
            print("Downloading videos from ", file_path)
            try:
                urllib.request.urlretrieve(file_path, local_path)
            except Exception as e:
                send_email_warning(str(e), f"Video download failed for {file_path}!" )
        iteration += 1
        if iteration == iterations:
            break
        if delay:
            time.sleep(delay * 60)


def upload_videos(blob_credentials: dict,
                  iterations: int =None,
                  delay: int = None):
    """
    This function uploads the video in the local_video_dir to S3. Each video is deleted after an upload.
    Args:
        blob_credentials: blob credentials
        iterations: number of times the upload should run. The local video directory is checked continuously for new videos if no value is supplied
        delay: amount of time (minutes) to wait for before downloading new data
    """

    local_video_dir = paths['temp_raw_video']
    if not os.path.exists(local_video_dir):
        return
    blob_folder = paths['blob_video']

    dl = DataLoaderBlob(blob_credentials)

    iteration = 0
    while True:
        try:
            for r, d, f in os.walk(local_video_dir):
                if f:
                    for file in f:
                        file_path_on_azure = os.path.join(blob_folder, file)
                        file_path_on_local = os.path.join(r, file)
                        dl.upload_blob(path_of_file_to_upload=file_path_on_local, path_to_upload_file_to=file_path_on_azure)
                        os.remove(file_path_on_local)
                        print('Uploaded video to ' + file_path_on_azure)

        except Exception as e:
            send_email_warning(str(e), "Video upload failed.")
        iteration += 1
        if iteration == iterations:
            break
        print(delay)
        if delay:
            time.sleep(delay * 60)

    return


# TODO: remove unused function
def rename_videos(old_path, new_path, date_format, chunk_size=100):
    bucket_name = paths['bucket_name']
    s3_profile = paths['s3_profile']
    s3_folder = "s3://%s/%s" % (bucket_name, old_path)

    if len(sys.argv) > 1:
        chunk_size = sys.argv[1]
    while True:
        start = time.time()
        ls = Popen(["aws", "s3", 'ls',
                    s3_folder,
                    '--summarize',
                    '--recursive',
                    '--profile',
                    s3_profile], stdout=PIPE)
        p1 = Popen(["awk", '{$1=$2=$3=""; print $0}'],
                   stdin=ls.stdout, stdout=PIPE)
        p2 = Popen(["head", "-n " + str(chunk_size)],
                   stdin=p1.stdout, stdout=PIPE)
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
                    filename = full_path.split('/')[-1]
                    filename = filename.strip()
                    res = filename.split("_")
                    datetime_obj = dateutil.parser.parse(res[0])
                    timestamp = datetime_obj.strftime(date_format)
                    new_filename = "_".join([timestamp, res[1]])
                    new_filename = "s3://%s/%s/%s/%s" % (
                        bucket_name, new_path, str(datetime_obj.date()), new_filename)
                    res = subprocess.call(["aws", "s3", 'mv',
                                           old_filename,
                                           new_filename,
                                           '--profile',
                                           s3_profile])
            except Exception as e:
                print(e)
        end = time.time()
        print(end - start)
