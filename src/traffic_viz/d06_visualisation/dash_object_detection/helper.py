import sys
import os
import json
import pandas as pd
import numpy as np

from src.traffic_analysis.d00_utils.load_confs import load_app_parameters, load_paths

params = load_app_parameters()
paths = load_paths()


def get_cams():
    filepath = paths['cameras']

    data = dict(json.loads(open(filepath, 'r').read()))
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
    # TODO: Change path to s3
    filepath = paths['jamcam_stats']
    output = pd.DataFrame()
    if not os.path.exists(filepath):
        return output
        
    df = pd.read_csv(filepath, dtype={'camera_id': 'category'})
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
