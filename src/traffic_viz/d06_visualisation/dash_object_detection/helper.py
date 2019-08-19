import sys
import os
from collections import OrderedDict

from flask_caching import Cache
import pandas as pd

from traffic_analysis.d00_utils.data_loader_s3 import DataLoaderS3
from traffic_analysis.d00_utils.data_loader_sql import DataLoaderSQL
from traffic_analysis.d00_utils.load_confs import load_app_parameters, load_parameters, load_paths, load_credentials
from traffic_analysis.d00_utils.get_project_directory import get_project_directory
from traffic_viz.d06_visualisation.dash_object_detection.server import server

project_dir = get_project_directory()
src_dir = f"{project_dir}/src"
sys.path.append(src_dir)
TIMEOUT = 60
cache = Cache(
    server, config={"CACHE_TYPE": "filesystem", "CACHE_DIR": "cache-directory"}
)

app_params = load_app_parameters()
params = load_parameters()
paths = load_paths()
creds = load_credentials()
s3_credentials = creds[paths["s3_creds"]]
DEBUG = app_params["debug"]


def get_vehicle_types():
    return params["selected_labels"]


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
    dl = DataLoaderSQL(creds, paths)
    sql = f"select * from {paths['db_video_level']} where camera_id = '{camera_id}';"
    df = dl.select_from_table(sql)
    if df is None:
        return output

    df["video_upload_datetime"] = pd.to_datetime(df.video_upload_datetime)
    output = df[df.camera_id == camera_id]
    return output


def load_vehicle_type_statistics(df, vehicle_type, start_date, end_date):
    df_vehicle_type = df[df.vehicle_type == vehicle_type]
    df_vehicle_type.sort_values("video_upload_datetime", inplace=True)
    df_vehicle_type = df_vehicle_type[
        ((start_date <= df_vehicle_type.video_upload_datetime)
         & (df_vehicle_type.video_upload_datetime <= end_date))
    ]
    return df_vehicle_type
