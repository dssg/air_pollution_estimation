import os
import subprocess
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from traffic_analysis.d00_utils.data_retrieval import connect_to_bucket, load_videos_into_np, delete_and_recreate_dir
from traffic_analysis.d04_modelling.classify_objects import classify_objects
from traffic_analysis.d05_reporting.report_yolo import yolo_report_count_stats


def update_frame_level_table(file_names, paths, params):
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

    update_s3_parquet(file="frame_table",
                      df=frame_level_df,
                      paths=paths)

    return


def update_video_level_table(paths, params):
    # TODO needs to be implemented
    frame_level_df = load_s3_parquet("frame_table", paths)
    video_level_df = yolo_report_count_stats(frame_level_df)

    return


def update_s3_parquet(file, df, paths):
    """ Append to parquet file on s3
                Args:
                    file (str): name of parquet file to be appended to
                    df (df): dataframe containing the data to be appended
                    paths (dict): dictionary of paths from yml file

                Returns:

    """

    if df.empty:
        print('Dataframe in update_s3_parquet() is empty!')
        return

    # Download the pq file
    my_bucket = connect_to_bucket(paths['s3_profile'], paths['bucket_name'])

    delete_and_recreate_dir(paths["temp_frame_level"])
    local_path = os.path.join(paths['temp_frame_level'], file + '.parquet')

    file_to_download = paths['s3_frame_level'] + file + '.parquet'
    try:
        my_bucket.download_file(file_to_download, local_path)

    except:
        print("Could not download " + file_to_download)
        print("Creating new file instead...")

    # fix data types
    df['obj_classification'] = df['obj_classification'].astype(str)
    df['camera_id'] = df['camera_id'].astype(str)
    df['confidence'] = df['confidence'].astype('float64')

    table = pa.Table.from_pandas(df)
    pqwriter = pq.ParquetWriter(local_path, table.schema)
    pqwriter.write_table(table)

    # close the parquet writer
    if pqwriter:
        pqwriter.close()

    # upload back to S3
    try:
        res = subprocess.call(["aws", "s3", 'cp',
                               local_path,
                               's3://air-pollution-uk/' + paths['s3_frame_level'],
                               '--profile',
                               'dssg'])
    except:
        print('Parquet upload failed!')

    os.remove(local_path)

    return


def load_s3_parquet(file, paths):
    """ Load parquet file from s3 into memory
            Args:
                file (str): name of parquet file to be loaded
                paths (dict): dictionary of paths from yml file

            Returns:

    """

    # Download the pq file
    my_bucket = connect_to_bucket(paths['s3_profile'], paths['bucket_name'])
    local_path = os.path.join(paths['temp_parquet'], file + '.parquet')

    df = None

    try:
        my_bucket.download_file(paths['s3_processed_jamcam'] + file + '.parquet', local_path)
        df = pd.read_parquet(local_path, engine='pyarrow')
        os.remove(local_path)

    except:
        print("Could not download " + paths['s3_processed_jamcam'] + file + '.parquet')

    return df
