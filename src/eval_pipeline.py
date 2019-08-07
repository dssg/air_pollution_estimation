from traffic_analysis.d00_utils.load_confs import load_parameters, load_paths, load_credentials
from traffic_analysis.d00_utils.create_sql_tables import create_sql_tables
from traffic_analysis.d04_modelling.trackinganalyser.trackinganalyser import TrackingAnalyser

params = load_parameters()
paths = load_paths()
creds = load_credentials()
s3_credentials = creds[paths['s3_creds']]

# initialize traffic analysers with various tracker types 
traffic_analysers = {}
for tracker_type in params["eval_tracker_types"]:
    traffic_analysers[tracker_type] = eval(params["traffic_analyser"])(params=params, 
                                                                         paths=paths, 
                                                                         tracker_type = tracker_type)

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

for tracker_type, traffic_analyser in traffic_analysers.items():
    #wipe and recreate eval tables for tracker types  
    create_sql_tables(db_frame_level_name=f"{paths['eval_db_frame_level']}_{tracker_type}", 
                      db_video_level_name=f"{paths['eval_db_video_level']}_{tracker_type}",
                      drop=True)

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
