import json
import os
import sys

import numpy as np
import pandas as pd
import os
import sys
from collections import OrderedDict

src_dir = os.path.join(os.getcwd(), '..', 'src')
data_dir = os.path.join(os.getcwd(), '..', 'data')
sys.path.append(src_dir)

from traffic_analysis.d00_utils.data_loader_s3 import DataLoaderS3
from traffic_analysis.d00_utils.load_confs import (load_app_parameters,
                                                   load_paths, load_credentials)

print(sys.path)


params = load_app_parameters()
paths = load_paths()
creds = load_credentials()
s3_credentials = creds[paths["s3_creds"]]
LOCAL = True

def get_cams():
    dl = DataLoaderS3(s3_credentials,
                      bucket_name=paths['bucket_name'])

    camera_meta_data_path = paths['s3_camera_details']

    data = dict(dl.read_json(camera_meta_data_path))
    values = data.values()
    cam_dict = {item['id']: item['commonName'] for item in values} 
    # [{'label': item['commonName'],  'value': item['id']}
                # for item in values]
    cam_dict = OrderedDict( sorted(cam_dict.items(), key=lambda x: x[1]))
    print(cam_dict)

    return cam_dict

def load_camera_statistics(camera_id):
    output = pd.DataFrame()

    if LOCAL:
        filepath = os.path.join(data_dir,paths['s3_video_level_stats'])
        if not os.path.exists(filepath):
            print(filepath)

            return output
        df = pd.read_csv(filepath, dtype={'camera_id': 'category'})
        df['datetime'] = pd.to_datetime(
            df.date) + pd.to_timedelta(df.time, unit='h')
        output = df[df.camera_id == camera_id]
        return output

    dl = DataLoaderS3(s3_credentials,
                      bucket_name=paths['bucket_name'])

    camera_meta_data_path = paths['s3_video_level_stats'] 
    if not dl.file_exists(camera_meta_data_path):
        return output
    
    data = dict(dl.read_json(camera_meta_data_path))
    df = pd.DataFrame(data, dtype={'camera_id': 'category'})
    df['datetime'] = pd.to_datetime(
        df.date) + pd.to_timedelta(df.time, unit='h')
    output = df[df.camera_id == camera_id]
    return output


def load_objects(df):
    # remove_columns = ["date", "metric", "time", "camera_id", "datetime"]
    load_columns = ["bus", "car", "motorcycle", "truck"]
    # columns = set(df.columns).inter(load_columns)
    return load_columns


def load_object_statistics(df, object_type, start_date, end_date):
    df_object = df.pivot_table(object_type, ["datetime"], "metric")
    df_object.sort_values("datetime", inplace=True)
    print(df_object.head().index)
    df_object = df_object[((start_date <= df_object.index) & (df_object.index <= end_date))]
    return df_object
