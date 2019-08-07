from traffic_analysis.d00_utils.data_retrieval import delete_and_recreate_dir
from traffic_analysis.d00_utils.load_confs import load_parameters, load_paths, load_credentials
from traffic_analysis.d00_utils.create_sql_tables import create_primary_sql_tables, create_eval_sql_tables
from traffic_analysis.d02_ref.load_video_names_from_s3 import load_video_names_from_s3
from traffic_analysis.d02_ref.upload_annotation_names_to_s3 import upload_annotation_names_to_s3
from traffic_analysis.d03_processing.update_frame_level_table import update_frame_level_table
from traffic_analysis.d03_processing.update_video_level_table import update_video_level_table
from traffic_analysis.d03_processing.update_eval_tables import update_eval_tables
from traffic_analysis.d04_modelling.tracking.tracking_analyser import TrackingAnalyser

params = load_parameters()
paths = load_paths()
creds = load_credentials()
s3_credentials = creds[paths['s3_creds']]

# initialize traffic analysers with various tracker types 
traffic_analysers = {}
for tracker_type in params["eval_tracker_types"]:
    analyser_name = params["traffic_analyser"].lower()
    traffic_analysers[ f"{analyser_name}_{tracker_type}"] = \
        eval(params["traffic_analyser"])(params=params, 
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

# wipe and recreate eval tables 
create_eval_sql_tables(creds=creds,
                       paths=paths, 
                       drop=False)

for tracker_type, traffic_analyser in traffic_analysers.items():
    db_frame_level_name = f"{paths['eval_db_frame_level_prefix']}_{tracker_type}"
    db_video_level_name = f"{paths['eval_db_video_level_prefix']}_{tracker_type}"

    #wipe and recreate stats tables for tracker types
    create_primary_sql_tables(db_frame_level_name=db_frame_level_name, 
                      db_video_level_name=db_video_level_name,
                      drop=True)

    # select chunks of videos and classify objects
    chunk_size = params['eval_chunk_size']
    while selected_videos:
        frame_level_df = update_frame_level_table(analyser=traffic_analyser,
                                                  file_names=selected_videos[:chunk_size],
                                                  db_frame_level_name = db_frame_level_name,
                                                  paths=paths,
                                                  creds=creds)

        video_level_df = update_video_level_table(analyser=analyser,
                                 db_video_level_name=db_video_level_name,
                                 frame_level_df=frame_level_df,
                                 file_names=selected_videos[:chunk_size],
                                 paths=paths,
                                 creds=creds,
                                 return_data=True)

        # Move on to next chunk
        selected_videos = selected_videos[chunk_size:]
        delete_and_recreate_dir(paths["temp_video"])

    # append to table 
    update_eval_tables(db_frame_level_name=db_frame_level_name, 
                       db_video_level_name=db_video_level_name,
                       params=params,
                       creds = creds,
                       paths=paths,
                       analyser_type=tracker_type
                       )
