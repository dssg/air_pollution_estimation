import os
import sys
import pandas as pd
from collections import OrderedDict
from flask_caching import Cache

src_dir = os.path.join(os.getcwd(), "..", "src")
data_dir = os.path.join(os.getcwd(), "..", "data")
sys.path.append(src_dir)

from traffic_analysis.d00_utils.data_loader_s3 import DataLoaderS3
from traffic_analysis.d00_utils.load_confs import (
    load_app_parameters,
    load_paths,
    load_credentials,
)
from traffic_viz.d06_visualisation.dash_object_detection.server import server

TIMEOUT = 60
cache = Cache(
    server, config={"CACHE_TYPE": "filesystem", "CACHE_DIR": "cache-directory"}
)

params = load_app_parameters()
paths = load_paths()
creds = load_credentials()
s3_credentials = creds[paths["s3_creds"]]
DEBUG = params["debug"]


def get_cams():
    dl = DataLoaderS3(s3_credentials, bucket_name=paths["bucket_name"])

    camera_meta_data_path = paths["s3_camera_details"]

    data = dict(dl.read_json(camera_meta_data_path))
    values = data.values()
    cam_dict = {item["id"]: item["commonName"] for item in values}
    # [{'label': item['commonName'],  'value': item['id']}
    # for item in values]
    cam_dict = OrderedDict(sorted(cam_dict.items(), key=lambda x: x[1]))
    return cam_dict


def load_camera_statistics(camera_id):
    output = pd.DataFrame()

    if DEBUG:
        filepath = os.path.join(data_dir, paths["s3_video_level_stats"])
        if not os.path.exists(filepath):
            print(filepath)

            return output
        df = pd.read_csv(filepath, dtype={"camera_id": "category"})
        df["datetime"] = pd.to_datetime(df.date) + pd.to_timedelta(df.time, unit="h")
        output = df[df.camera_id == camera_id]
        return output

    dl = DataLoaderS3(s3_credentials, bucket_name=paths["bucket_name"])

    camera_meta_data_path = paths["s3_video_level_stats"]
    if not dl.file_exists(camera_meta_data_path):
        return output

    data = dict(dl.read_json(camera_meta_data_path))
    df = pd.DataFrame(data, dtype={"camera_id": "category"})
    df["datetime"] = pd.to_datetime(df.date) + pd.to_timedelta(df.time, unit="h")
    output = df[df.camera_id == camera_id]
    return output


def load_object_statistics(df, object_type, start_date, end_date):
    df_object = df.pivot_table(object_type, ["video_upload_datetime"], "metric")
    df_object.sort_values("video_upload_datetime", inplace=True)
    df_object = df_object[
        ((start_date <= df_object.index) & (df_object.index <= end_date))
    ]
    return df_object
