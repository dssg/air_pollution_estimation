import json
import os
import sys

import numpy as np
import pandas as pd
import os
import sys

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
    cam_list = [{'label': item['commonName'],  'value': item['id']}
                for item in values]
    cam_list.sort(key=lambda x: x['label'])
    return cam_list


def load_data(path):
    """Load data about a specific footage (given by the path). It returns a dictionary of useful variables such as
    the dataframe containing all the detection and bounds localization, the number of classes inside that footage,
    the matrix of all the classes in string, the given class with padding, and the root of the number of classes,
    rounded."""
    # Load the dataframe containing all the processed object detections inside the video
    video_info_df = pd.read_csv(path)

    # The list of classes, and the number of classes
    classes_list = video_info_df["obj_classification"].value_counts(
    ).index.tolist()
    n_classes = len(classes_list)

    # Gets the smallest value needed to add to the end of the classes list to get a square matrix
    root_round = np.ceil(np.sqrt(len(classes_list)))
    total_size = root_round ** 2
    padding_value = int(total_size - n_classes)
    classes_padded = np.pad(classes_list, (0, padding_value), mode='constant')

    # The padded matrix containing all the classes inside a matrix
    classes_matrix = np.reshape(
        classes_padded, (int(root_round), int(root_round)))

    # Flip it for better looks
    classes_matrix = np.flip(classes_matrix, axis=0)

    data_dict = {
        "video_info_df": video_info_df,
        "n_classes": n_classes,
        "classes_matrix": classes_matrix,
        "classes_padded": classes_padded,
        "root_round": root_round
    }
    return data_dict


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


def load_object_statistics(df, object_type):
    df_object = df.pivot_table(object_type, ["datetime"], "metric")
    return df_object
