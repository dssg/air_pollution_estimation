from traffic_analysis.d00_utils.data_retrieval import load_videos_into_np, delete_and_recreate_dir
from traffic_analysis.d04_modelling.classify_objects import classify_objects
from traffic_analysis.d00_utils.data_loader_sql import DataLoaderSQL
from traffic_analysis.d00_utils.data_loader_s3 import DataLoaderS3


def update_frame_level_table(file_names: list,
                             paths: dict,
                             params: dict,
                             creds: dict):
    """ Update the frame level table on s3 (pq) based on the videos in the files list
                Args:
                    file_names (list): list of s3 filepaths for the videos to be processed
                    paths (dict): dictionary of paths from yml file
                    params (dict): dictionary of parameters from yml file
                    creds (dict): dictionary of credentials from yml file

                Returns:

    """
    s3_credentials = creds[paths['s3_creds']]
    dl = DataLoaderS3(s3_credentials,
                      bucket_name=paths['bucket_name'])

    delete_and_recreate_dir(paths["temp_video"])
    # Download the video file_names using the file list
    for file in file_names:
        try:
            path_to_download_file_to = (paths["temp_video"]
                                        + file.split('/')[-1].replace(':', '-').replace(" ", "_")
                                        )
            dl.download_file(path_of_file_to_download=file,
                             path_to_download_file_to=path_to_download_file_to)
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

    return frame_level_df
