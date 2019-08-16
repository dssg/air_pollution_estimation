import numpy as np

from traffic_analysis.d00_utils.data_retrieval import delete_and_recreate_dir
from traffic_analysis.d00_utils.load_confs import load_parameters, load_paths, load_credentials
from traffic_analysis.d00_utils.create_sql_tables import create_primary_sql_tables, create_eval_sql_tables
from traffic_analysis.d02_ref.load_video_names_from_s3 import load_video_names_from_s3
from traffic_analysis.d02_ref.upload_annotation_names_to_s3 import upload_annotation_names_to_s3
from traffic_analysis.d03_processing.update_frame_level_table import update_frame_level_table
from traffic_analysis.d03_processing.update_video_level_table import update_video_level_table
from traffic_analysis.d03_processing.update_eval_tables import update_eval_tables
from traffic_analysis.d03_processing.create_traffic_analyser import create_traffic_analyser, initialize_param_sets

params = load_parameters()
paths = load_paths()
creds = load_credentials()
s3_credentials = creds[paths['s3_creds']]

# TODO: make it so that you can read in the videos/annotations from local

# settings
verbose = True

# pipeline start

# get param grids
traffic_analysers_params = initialize_param_sets(params=params)

# If running first time:
# get annotation xmls from s3 saves json on s3 containing to corresponding video filepaths
#upload_annotation_names_to_s3(paths=paths,
#                              output_file_name=params['eval_ref_name'],
#                              s3_credentials=s3_credentials,
#                              verbose=verbose)

selected_videos = load_video_names_from_s3(ref_file=params['eval_ref_name'],
                                           paths=paths,
                                           s3_credentials=s3_credentials)

if verbose:
    print("Successfully loaded selected videos")
    print(selected_videos)

# create eval tables if they don't exist
create_eval_sql_tables(creds=creds,
                       paths=paths,
                       drop=False)

if verbose:
    print("Running evaluation for traffic analysers: ", traffic_analysers_params.keys())

selected_videos_master = selected_videos.copy()
for i, (analyser_name, params_to_set_dict) in enumerate(traffic_analysers_params.items()):
    if verbose:
        print(f"Now evaluating {analyser_name} with param set: {params_to_set_dict}")

    # initialize db names
    db_frame_level_name = f"{paths['eval_db_frame_level_prefix']}_{analyser_name}_{i}"
    db_video_level_name = f"{paths['eval_db_video_level_prefix']}_{analyser_name}_{i}"

    # wipe and recreate video/frame level stats tables for tracker types
    create_primary_sql_tables(db_frame_level_name=db_frame_level_name,
                              db_video_level_name=db_video_level_name,
                              drop=True)

    # create traffic analyser
    traffic_analyser = create_traffic_analyser(params_to_set=params_to_set_dict,
                                               params=params,
                                               paths=paths,
                                               s3_credentials=s3_credentials,
                                               verbose=verbose)

    # select chunks of videos and classify objects
    chunk_size = params['eval_chunk_size']
    chunk_counter = 0
    analyser_runtime = []

    # regenerate selected videos
    selected_videos = selected_videos_master
    while selected_videos:
        try:
            success, frame_level_df, runtime_list, lost_tracking = update_frame_level_table(analyser=traffic_analyser,
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
                                                          lost_tracking=lost_tracking,
                                                          paths=paths,
                                                          creds=creds,
                                                          return_data=True)

                if verbose:
                    print(f"Successfully processed chunk {chunk_counter}")
        except Exception as e:
            print(e)
            print("Analysing current chunk failed. Continuing to next chunk.")
            pass

        chunk_counter += 1
        selected_videos = selected_videos[chunk_size:]
        delete_and_recreate_dir(paths["temp_video"])

    avg_runtime = np.mean(np.array(analyser_runtime))
    
    if verbose:
        print(f"Successfully processed videos for traffic analyser: {analyser_name}")
        print(f"Avg runtime of one video for tracking type {analyser_name}: {avg_runtime}")

    traffic_analyser.cleanup_on_finish()
    
    # append to table
    update_eval_tables(db_frame_level_name=db_frame_level_name,
                       db_video_level_name=db_video_level_name,
                       params=params,
                       creds=creds,
                       paths=paths,
                       avg_runtime=avg_runtime,
                       evaluated_params=params_to_set_dict
                       )

    if verbose:
        print("Successfully evaluated videos for one tracking type")
