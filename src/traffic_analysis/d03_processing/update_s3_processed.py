import os
import subprocess
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import sqlalchemy

from traffic_analysis.d00_utils.data_retrieval import connect_to_bucket, load_videos_into_np, delete_and_recreate_dir
from traffic_analysis.d04_modelling.classify_objects import classify_objects
# from traffic_analysis.d05_reporting.report_yolo import yolo_report_count_stats


def update_frame_level_table(file_names, paths, params, creds):
    """ Update the frame level table on s3 (pq) based on the videos in the files list
                Args:
                    file_names (list): list of s3 filepaths for the videos to be processed
                    paths (dict): dictionary of paths from yml file
                    params (dict): dictionary of parameters from yml file

                Returns:

    """
    my_bucket = connect_to_bucket(paths['s3_profile'], paths['bucket_name'])

    delete_and_recreate_dir(paths["temp_video"])
    # Download the video file_names using the file list
    for file in file_names:
        try:
            my_bucket.download_file(file, paths["temp_video"] + file.split('/')[-1].replace(
                ':', '-').replace(" ", "_"))
        except:
            print("Could not download " + file)

    video_dict = load_videos_into_np(paths["temp_video"])
    delete_and_recreate_dir(paths["temp_video"])

    frame_level_df = classify_objects(video_dict=video_dict,
                                      params=params,
                                      paths=paths,
                                      vid_time_length=10,
                                      make_videos=False)

    # update_s3_parquet(file="frame_table",
    #                   df=frame_level_df,
    #                   paths=paths)

    append_to_database(df=frame_level_df,
                       table='frame_level',
                       creds=creds)

    return


# def update_video_level_table(paths, params):
#    # TODO needs to be implemented
#    frame_level_df = load_s3_parquet("frame_table", paths)
#    video_level_df = yolo_report_count_stats(frame_level_df)
#
#    return


def append_to_database(df, table, creds):

    x, y, w, h = [], [], [], []
    for vals in df['obj_bounds'].values:
        x.append(vals[0])
        y.append(vals[1])
        w.append(vals[2])
        h.append(vals[3])
    df['box_x'] = x
    df['box_y'] = y
    df['box_w'] = w
    df['box_h'] = h
    df.drop('obj_bounds', axis=1, inplace=True)

    db_host = creds['postgres']['host']
    db_name = creds['postgres']['name']
    db_user = creds['postgres']['user']
    db_pass = creds['postgres']['passphrase']

    conn = sqlalchemy.create_engine('postgresql://%s:%s@%s/%s' %
                                    (db_user, db_pass, db_host, db_name),
                                    encoding='latin1',
                                    echo=True)

    dtypes = {'obj_ind': sqlalchemy.types.INTEGER(),
             'camera_id': sqlalchemy.types.String(),
             'frame_id': sqlalchemy.types.INTEGER(),
             'datetime': sqlalchemy.DateTime(),
             'obj_classification': sqlalchemy.types.String(),
             'confidence': sqlalchemy.types.Float(precision=3, asdecimal=True),
             'video_id': sqlalchemy.types.INTEGER(),
             'box_x': sqlalchemy.types.INTEGER(),
             'box_y': sqlalchemy.types.INTEGER(),
             'box_w': sqlalchemy.types.INTEGER(),
             'box_h': sqlalchemy.types.INTEGER()}

    df.to_sql(name=table, con=conn, if_exists='append', dtype=dtypes)

    return

