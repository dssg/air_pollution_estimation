import numpy as np

from traffic_analysis.d00_utils.data_retrieval import delete_and_recreate_dir
from traffic_analysis.d00_utils.load_confs import load_parameters, load_paths, load_credentials
from traffic_analysis.d00_utils.create_sql_tables import create_primary_sql_tables, create_eval_sql_tables
from traffic_analysis.d02_ref.load_video_names_from_blob import load_video_names_from_blob
from traffic_analysis.d02_ref.upload_annotation_names_to_blob import upload_annotation_names_to_blob
from traffic_analysis.d03_processing.update_frame_level_table import update_frame_level_table
from traffic_analysis.d03_processing.update_video_level_table import update_video_level_table
from traffic_analysis.d03_processing.update_eval_tables import update_eval_tables
from traffic_analysis.d03_processing.create_traffic_analyser import create_traffic_analyser, initialize_param_sets

params = load_parameters()
paths = load_paths()
creds = load_credentials()
blob_credentials = creds[paths['blob_creds']]

# pipeline start

# get param grids
traffic_analysers_params, num_params = initialize_param_sets(params=params)

# If running first time:
# get annotation xmls from s3 saves json on s3 containing to corresponding video filepaths
upload_annotation_names_to_blob(paths=paths,
                              output_file_name=params['eval_ref_name'],
                              blob_credentials=blob_credentials,
                              verbose=params["eval_verbosity"])

selected_videos = load_video_names_from_blob(ref_file=params['eval_ref_name'],
                                           paths=paths,
                                           blob_credentials=blob_credentials)

if params["eval_verbosity"]:
    print("Successfully loaded selected videos")

# create eval tables if they don't exist
create_eval_sql_tables(creds=creds,
                       paths=paths,
                       drop=True)

if params["eval_verbosity"]:
    print("Running evaluation for traffic analysers: ", traffic_analysers_params.keys())

selected_videos_master = selected_videos.copy()
param_counter = 1

for analyser_name, params_subgrid in traffic_analysers_params.items():
    # iterate thru tuneable params for each combo of detection model/tracking type
    for params_to_set_dict in params_subgrid:
        if params["eval_verbosity"]:
            print(f"Now evaluating param set {param_counter}/{num_params}")

        # initialize db names
        db_frame_level_name = f"{paths['eval_db_frame_level_prefix']}_{analyser_name}_test"
        db_video_level_name = f"{paths['eval_db_video_level_prefix']}_{analyser_name}_test"

        # wipe and recreate video/frame level stats tables for tracker types
        create_primary_sql_tables(db_frame_level_name=db_frame_level_name,
                                  db_video_level_name=db_video_level_name,
                                  drop=True)

        # create traffic analyser
        traffic_analyser = create_traffic_analyser(params_to_set=params_to_set_dict,
                                                   params=params,
                                                   paths=paths,
                                                   blob_credentials=blob_credentials,
                                                   verbose=params["eval_verbosity"])

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
                                                                                 creds=creds, make_video=False)
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

                    if params["eval_verbosity"]:
                        print(f"Successfully processed chunk {chunk_counter}")
            except Exception as e:
                print(e)
                print("Analysing current chunk failed. Continuing to next chunk.")
                pass

            chunk_counter += 1
            selected_videos = selected_videos[chunk_size:]
            delete_and_recreate_dir(paths["temp_video"])

        avg_runtime = np.mean(np.array(analyser_runtime))
        
        if params["eval_verbosity"]:
            print(f"Successfully processed videos for traffic analyser {analyser_name}, param set {params_to_set_dict}")
            print(f"Avg runtime of one video: {avg_runtime}")
            print(f"Now evaluating param set {param_counter}/{num_params}")


        traffic_analyser.cleanup_on_finish()
        
        # append to table
        try:
            update_eval_tables(db_frame_level_name=db_frame_level_name,
                             db_video_level_name=db_video_level_name,
                             params=params,
                             creds=creds,
                             paths=paths,
                             avg_runtime=avg_runtime,
                             evaluated_params=params_to_set_dict
                             )
        except Exception as e: 
            print(e)
            print(f"Failed to evaluate videos for traffic_analyser {analyser_name}, param set {params_to_set_dict}")
            continue 

        if params["eval_verbosity"]:
            print(f"Successfully evaluated videos for traffic_analyser {analyser_name}, param set {params_to_set_dict}")

        param_counter += 1
