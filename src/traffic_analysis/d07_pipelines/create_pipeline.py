
from traffic_analysis.d00_utils.data_retrieval import delete_and_recreate_dir
from traffic_analysis.d00_utils.load_confs import (load_credentials,
                                                   load_parameters, load_paths)
from traffic_analysis.d02_ref.load_video_names_from_s3 import \
    load_video_names_from_s3
from traffic_analysis.d02_ref.retrieve_and_upload_video_names_to_s3 import \
    retrieve_and_upload_video_names_to_s3
from traffic_analysis.d03_processing.update_frame_level_table import \
    update_frame_level_table
from traffic_analysis.d03_processing.update_video_level_table import \
    update_video_level_table
from traffic_analysis.d04_modelling.tracking.tracking_analyser import \
    TrackingAnalyser
from traffic_analysis.d00_utils.data_loader_s3 import DataLoaderS3


def create_pipeline(output_file_name,
                    from_date,
                    to_date,
                    from_time,
                    to_time,
                    camera_list,
                    chunk_size,
                    move_to_processed_folder=False,
                    delete_processed_videos=False,
                    load_ref_file=False):

    params = load_parameters()
    paths = load_paths()
    creds = load_credentials()
    s3_credentials = creds[paths['s3_creds']]
    data_loader_s3 = DataLoaderS3(
        s3_credentials=s3_credentials, bucket_name=paths['bucket_name'])
    if load_ref_file:
        retrieve_and_upload_video_names_to_s3(output_file_name=output_file_name,
                                              paths=paths,
                                              from_date=from_date, to_date=to_date,
                                              from_time=from_time, to_time=to_time,
                                              s3_credentials=s3_credentials,
                                              camera_list=camera_list)

    # Start from here if video names already specified
    selected_videos = load_video_names_from_s3(ref_file=output_file_name,
                                               paths=paths,
                                               s3_credentials=s3_credentials)

    analyser = TrackingAnalyser(
        params=params, paths=paths, s3_credentials=s3_credentials)

    # select chunks of videos and classify objects
    while selected_videos:
        file_names = selected_videos[:chunk_size]
        frame_level_df = update_frame_level_table(analyser=analyser,
                                                  file_names=file_names,
                                                  paths=paths,
                                                  creds=creds)
        update_video_level_table(analyser=analyser,
                                 frame_level_df=frame_level_df,
                                 file_names=selected_videos[:chunk_size],
                                 paths=paths,
                                 creds=creds,
                                 return_data=False)

        # move processed videos to processed folder
        if move_to_processed_folder:
            for filename in file_names:
                new_filename = filename.replace(
                    paths['s3_video'], paths['s3_processed_videos'])
                data_loader_s3.move_file(filename, new_filename)

        # delete processed videos if true
        if delete_processed_videos is True:
            data_loader_s3.delete_folder(paths['s3_processed_videos'])

        # Move on to next chunk
        selected_videos = selected_videos[chunk_size:]
        delete_and_recreate_dir(paths["temp_video"])
