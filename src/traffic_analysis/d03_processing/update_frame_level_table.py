import datetime

import pandas as pd

from traffic_analysis.d00_utils.data_loader_s3 import DataLoaderS3
from traffic_analysis.d00_utils.data_loader_sql import DataLoaderSQL
from traffic_analysis.d00_utils.data_retrieval import (delete_and_recreate_dir,
                                                       load_videos_into_np)


def update_frame_level_table(analyser,
                             file_names: list,
                             paths: dict,
                             creds: dict):
    """ Update the frame level table on s3 (pq) based on the videos in the files list
                Args:
                    analyser: analyser object for doing traffic analysis
                    file_names (list): list of s3 filepaths for the videos to be processed
                    paths (dict): dictionary of paths from yml file
                    creds (dict): dictionary of credentials from yml file

                Returns:

    """
    s3_credentials = creds[paths['s3_creds']]
    dl = DataLoaderS3(s3_credentials,
                      bucket_name=paths['bucket_name'])

    delete_and_recreate_dir(paths["temp_video"])
    # Download the video file_names using the file list
    for filename in file_names:
        try:
            path_to_download_file_to = paths["temp_video"] + \
                filename.split('/')[-1].replace(':', '-').replace(" ", "_")
            dl.download_file(path_of_file_to_download=filename,
                             path_to_download_file_to=path_to_download_file_to)
        except:
            print("Could not download " + filename)

    video_dict = load_videos_into_np(paths["temp_video"])
    delete_and_recreate_dir(paths["temp_video"])

    frame_level_df = analyser.construct_frame_level_df(video_dict)
    if frame_level_df.empty:
        return None
    frame_level_df.dropna(how='any', inplace=True)
    frame_level_df = frame_level_df.astype(
        {'frame_id': 'int64',
         'vehicle_id': 'int64'})

    frame_level_sql_df = pd.DataFrame.copy(frame_level_df)
    x, y, w, h = [], [], [], []
    for vals in frame_level_sql_df['bboxes'].values:
        if isinstance(vals, list) and len(vals) > 3:
            x.append(int(vals[0]))
            y.append(int(vals[1]))
            w.append(int(vals[2]))
            h.append(int(vals[3]))
    frame_level_sql_df['bbox_x'] = x
    frame_level_sql_df['bbox_y'] = y
    frame_level_sql_df['bbox_w'] = w
    frame_level_sql_df['bbox_h'] = h
    frame_level_sql_df.drop('bboxes', axis=1, inplace=True)
    frame_level_sql_df['creation_datetime'] = datetime.datetime.now()

    db_obj = DataLoaderSQL(creds=creds, paths=paths)
    db_obj.add_to_sql(df=frame_level_sql_df,
                      table_name=paths['db_frame_level'])

    return frame_level_df
