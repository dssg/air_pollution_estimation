
from traffic_analysis.d00_utils.data_retrieval import delete_and_recreate_dir
from traffic_analysis.d00_utils.load_confs import (load_credentials,
                                                   load_parameters, load_paths)
from traffic_analysis.d02_ref.load_video_names_from_blob import \
    load_video_names_from_blob
from traffic_analysis.d02_ref.retrieve_and_upload_video_names_to_s3 import \
    retrieve_and_upload_video_names_to_s3
from traffic_analysis.d03_processing.update_frame_level_table import \
    update_frame_level_table
from traffic_analysis.d03_processing.update_video_level_table import \
    update_video_level_table
from traffic_analysis.d04_modelling.tracking.tracking_analyser import \
    TrackingAnalyser
from traffic_analysis.d00_utils.data_loader_blob import DataLoaderBlob


def create_pipeline(output_file_name,
                    from_date,
                    to_date,
                    from_time,
                    to_time,
                    camera_list,
                    chunk_size,
                    move_to_processed_folder=False,
                    delete_processed_videos=False,
                    construct_ref_file=False,
                    make_video=False):

    params = load_parameters()
    paths = load_paths()
    creds = load_credentials()
    blob_credentials = creds[paths['blob_creds']]
    dl = DataLoaderBlob(blob_credentials=blob_credentials)
    if construct_ref_file:
        retrieve_and_upload_video_names_to_s3(output_file_name=output_file_name,
                                              paths=paths,
                                              from_date=from_date, to_date=to_date,
                                              from_time=from_time, to_time=to_time,
                                              blob_credentials=blob_credentials,
                                              camera_list=camera_list)

    # Start from here if video names already specified
    selected_videos = load_video_names_from_blob(ref_file=output_file_name,
                                                 paths=paths,
                                                 blob_credentials=blob_credentials)

    analyser = TrackingAnalyser(
        params=params, paths=paths, blob_credentials=blob_credentials)

    # select chunks of videos and classify objects
    while selected_videos:
        file_names = selected_videos[:chunk_size]
        success, frame_level_df, runtime_list, lost_tracking = update_frame_level_table(analyser=analyser,
                                                  file_names=file_names,
                                                  paths=paths,
                                                  creds=creds,
                                                  make_video=make_video,
                                                  db_frame_level_name=paths['db_frame_level'])
        update_video_level_table(analyser=analyser,
                                 frame_level_df=frame_level_df,
                                 file_names=selected_videos[:chunk_size],
                                 paths=paths,
                                 creds=creds,
                                 return_data=False,
                                 db_video_level_name=paths['db_video_level'])

        # move processed videos to processed folder
        if move_to_processed_folder:
            delete_and_recreate_dir(paths["temp_video"])

            for filename in file_names:
                dl.copy_blob(file_to_move=filename, new_file=filename.replace(
                    paths['blob_video'], paths['blob_processed_videos']), paths=paths)

            dl.delete_blobs(blobs=file_names)
            delete_and_recreate_dir(paths["temp_video"])

        # delete processed videos if true
        if delete_processed_videos is True:
            blobs, elapsed_time = dl.list_blobs(prefix=paths['blob_processed_videos'])
            dl.delete_blobs(blobs=blobs)

        # Move on to next chunk
        selected_videos = selected_videos[chunk_size:]
        delete_and_recreate_dir(paths["temp_video"])
