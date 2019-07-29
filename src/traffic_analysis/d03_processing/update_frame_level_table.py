from traffic_analysis.d00_utils.data_retrieval import connect_to_bucket, load_videos_into_np, delete_and_recreate_dir
from traffic_analysis.d04_modelling.classify_objects import classify_objects
from traffic_analysis.d00_utils.data_loader_sql import DataLoaderSQL


def update_frame_level_table(file_names, paths, params, creds):
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

    frame_level_df = classify_objects(video_dict=video_dict,
                                      params=params,
                                      paths=paths,
                                      vid_time_length=10,
                                      make_videos=False)

    db_obj = DataLoaderSQL(creds=creds, paths=paths)
    db_obj.add_to_sql(df=frame_level_df, table_name='frame_stats')

    return
