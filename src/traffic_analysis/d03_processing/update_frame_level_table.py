import pandas as pd

from traffic_analysis.d00_utils.data_retrieval import connect_to_bucket, load_videos_into_np, delete_and_recreate_dir
from traffic_analysis.d03_processing.add_to_table_sql import add_to_table_sql


def update_frame_level_table(analyzer, file_names, paths, params, creds):
    """ Update the frame level table on the database based on the videos in the files list
                Args:
                    file_names (list): list of s3 filepaths for the videos to be processed
                    paths (dict): dictionary of paths from yml file
                    params (dict): dictionary of parameters from yml file
                    creds (dict): dictionary of credentials from yml file

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

    frame_level_df = analyzer.construct_frame_level_df(video_dict)
    
    x, y, w, h = [], [], [], []
    for vals in frame_level_df['bboxes'].values:
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
