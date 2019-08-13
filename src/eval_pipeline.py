import numpy as np

from traffic_analysis.d00_utils.data_retrieval import delete_and_recreate_dir
from traffic_analysis.d00_utils.load_confs import load_parameters, load_paths, load_credentials
from traffic_analysis.d00_utils.create_sql_tables import create_primary_sql_tables, create_eval_sql_tables
from traffic_analysis.d02_ref.load_video_names_from_s3 import load_video_names_from_s3
from traffic_analysis.d02_ref.upload_annotation_names_to_s3 import upload_annotation_names_to_s3
from traffic_analysis.d03_processing.update_frame_level_table import update_frame_level_table
from traffic_analysis.d03_processing.update_video_level_table import update_video_level_table
from traffic_analysis.d03_processing.update_eval_tables import update_eval_tables
from traffic_analysis.d03_processing.create_traffic_analysers import create_traffic_analysers

params = load_parameters()
paths = load_paths()
creds = load_credentials()
s3_credentials = creds[paths['s3_creds']]

# settings
verbose = True

# pipeline start
traffic_analysers = create_traffic_analysers(params=params,
                                             paths=paths,
                                             s3_credentials=s3_credentials,
                                             verbose=verbose)
# If running first time:
# get annotation xmls from s3 saves json on s3 containing to corresponding video filepaths
upload_annotation_names_to_s3(paths=paths,
                              output_file_name=params['eval_ref_name'],
                              s3_credentials=s3_credentials,
                              verbose=verbose)

selected_videos = load_video_names_from_s3(ref_file=params['eval_ref_name'],
                                           paths=paths,
                                           s3_credentials=s3_credentials)

if verbose:
    print("Successfully loaded selected videos")

# create eval tables if they don't exist
create_eval_sql_tables(creds=creds,
                       paths=paths,
                       drop=True)

if verbose:
    print("Running evaluation for traffic analysers: ", traffic_analysers.keys())

selected_videos_master = selected_videos.copy()
for i, (analyser_name, (traffic_analyser, param_set)) in enumerate(traffic_analysers.items()):
    if verbose: 
        print(f"Now evaluating {analyser_name} with param set: {param_set}")
    
    db_frame_level_name = f"{paths['eval_db_frame_level_prefix']}_{analyser_name}_{i}"
    db_video_level_name = f"{paths['eval_db_video_level_prefix']}_{analyser_name}_{i}"

    # wipe and recreate stats tables for tracker types
    create_primary_sql_tables(db_frame_level_name=db_frame_level_name,
                              db_video_level_name=db_video_level_name,
                              drop=True)

    # select chunks of videos and classify objects
    chunk_size = params['eval_chunk_size']
    chunk_counter = 0
    analyser_runtime = []

    # regenerate selected videos
    selected_videos = selected_videos_master
    while selected_videos:
        success, frame_level_df, runtime_list = update_frame_level_table(analyser=traffic_analyser,
                                                                         file_names=selected_videos[:chunk_size],
                                                                         db_frame_level_name=db_frame_level_name,
                                                                         paths=paths,
                                                                         creds=creds)
        analyser_runtime += runtime_list

        if success:
            video_level_df = update_video_level_table(analyser=traffic_analyser,
                                                      db_video_level_name=db_video_level_name,
                                                      frame_level_df=frame_level_df,
                                                      file_names=selected_videos[:chunk_size],
                                                      paths=paths,
                                                      creds=creds,
                                                      return_data=True)

            if verbose:
                print(f"Successfully processed chunk {chunk_counter}")

        else:
            print("Analysing current chunk failed. Continuing to next chunk.")

        chunk_counter += 1
        if chunk_counter = 1: 
          break
        selected_videos = selected_videos[chunk_size:]
        delete_and_recreate_dir(paths["temp_video"])

    avg_runtime = np.mean(np.array(analyser_runtime))
    
    if verbose:
        print(f"Successfully processed videos for traffic analyser: {analyser_name}")
        print(f"Avg runtime of one video for tracking type {analyser_name}: {avg_runtime}")

    # append to table
    update_eval_tables(db_frame_level_name=db_frame_level_name,
                       db_video_level_name=db_video_level_name,
                       params=params,
                       creds=creds,
                       paths=paths,
                       avg_runtime=avg_runtime,
                       evaluated_params=param_set
                       )

    if verbose:
        print("Successfully evaluated videos for one tracking type")
