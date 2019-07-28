import pandas as pd

from traffic_analysis.d00_utils.data_retrieval import load_videos_into_np, delete_and_recreate_dir
from traffic_analysis.d03_processing.add_to_table_sql import add_to_table_sql
from traffic_analysis.d00_utils.data_loader_s3 import DataLoaderS3


def update_frame_level_table(analyzer,
                             file_names: list,
                             paths: dict,
                             params: dict,
                             creds: dict,
                             s3_credentials: dict):
    """ Update the frame level table on s3 (pq) based on the videos in the files list
                Args:
                    analyzer: analyzer object for doing traffic analysis
                    file_names (list): list of s3 filepaths for the videos to be processed
                    paths (dict): dictionary of paths from yml file
                    params (dict): dictionary of parameters from yml file
                    creds (dict): dictionary of credentials from yml file
                    creds: generic credentials dictionary
                    s3_credentials: S3 credentials

                Returns:

    """
    dl = DataLoaderS3(s3_credentials,
                      bucket_name=paths['bucket_name'])

    delete_and_recreate_dir(paths["temp_video"])
    # Download the video file_names using the file list
    for filename in file_names:
        try:
            path_to_download_file_to = (paths["temp_video"]
                                        + filename.split('/')[-1].replace(':', '-').replace(" ", "_")
                                        )
            dl.download_file(path_of_file_to_download=filename,
                             path_to_download_file_to=path_to_download_file_to)
        except:
            print("Could not download " + filename)

    video_dict = load_videos_into_np(paths["temp_video"])
    delete_and_recreate_dir(paths["temp_video"])

    frame_level_df = analyzer.construct_frame_level_df(video_dict)
    frame_level_df.dropna(how='any', inplace=True)
    frame_level_df = frame_level_df.astype(
        {'frame_id': 'int64', 'vehicle_id': 'int64'})
    x, y, w, h = [], [], [], []
    for vals in frame_level_df['bboxes'].values:
        if isinstance(vals, list) and len(vals) > 3:
            x.append(vals[0])
            y.append(vals[1])
            w.append(vals[2])
            h.append(vals[3])
    frame_level_df['bbox_x'] = x
    frame_level_df['bbox_y'] = y
    frame_level_df['bbox_w'] = w
    frame_level_df['bbox_h'] = h
    frame_level_df.drop('bboxes', axis=1, inplace=True)
    print(frame_level_df.head(3))
    add_to_table_sql(df=frame_level_df,
                     table='frame_stats',
                     creds=creds,
                     paths=paths)

    return
