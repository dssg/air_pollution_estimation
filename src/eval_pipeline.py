from traffic_analysis.d00_utils.load_confs import load_parameters, load_paths, load_credentials
from traffic_analysis.d04_modelling.trackinganalyser.trackinganalyser import TrackingAnalyser

params = load_parameters()
paths = load_paths()
creds = load_credentials()
s3_credentials = creds[paths['s3_creds']]
traffic_analyser = eval(params["traffic_analyser"])(params=params, paths=paths)


# If running first time:
# creates the test_seach_json. Change the camera list and output file name for full run

# get annotation xmls from s3 saves json on s3 containing to corresponding video filepaths
upload_annotation_names_to_s3(paths=paths,
							  output_file_name=params['eval_ref_name'],
                              s3_credentials=s3_credentials)

# Start from here if video names already specified
selected_videos = load_video_names_from_s3(ref_file= params['eval_ref_name'],
                                           paths=paths,
                                           s3_credentials=s3_credentials)

# select chunks of videos and classify objects
chunk_size = params['eval_chunk_size']
while selected_videos:
    frame_level_df = update_frame_level_table(file_names=selected_videos[:chunk_size],
                                              paths=paths,
                                              params=params,
                                              creds=creds)

    # evaluate_frame_level_table

    update_video_level_table(frame_level_df=frame_level_df,
                             file_names=selected_videos[:chunk_size],
                             paths=paths,
                             creds=creds,
                             params=params)

    # evaluate_video_level_table

    # Move on to next chunk
    selected_videos = selected_videos[chunk_size:]
    delete_and_recreate_dir(paths["temp_video"])
