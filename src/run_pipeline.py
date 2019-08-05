from traffic_analysis.d00_utils.data_retrieval import delete_and_recreate_dir
from traffic_analysis.d00_utils.load_confs import load_parameters, load_paths, load_credentials
from traffic_analysis.d02_ref.load_video_names_from_s3 import load_video_names_from_s3
from traffic_analysis.d02_ref.retrieve_and_upload_video_names_to_s3 import retrieve_and_upload_video_names_to_s3
from traffic_analysis.d03_processing.update_frame_level_table import update_frame_level_table
from traffic_analysis.d03_processing.update_video_level_table import update_video_level_table
from traffic_analysis.d04_modelling.tracking.tracking_analyser import TrackingAnalyser

params = load_parameters()
paths = load_paths()
creds = load_credentials()
s3_credentials = creds[paths['s3_creds']]

analyzer = TrackingAnalyser(params=params, paths=paths)

# If running first time:
# creates the test_seach_json. Change the camera list and output file name for full run

retrieve_and_upload_video_names_to_s3(ouput_file_name='sam_test',
                                      paths=paths,
                                      from_date='2019-07-17', to_date='2019-07-17',
                                      from_time= '13-00-00', to_time='14-00-00',
                                      s3_credentials=s3_credentials,
                                      camera_list=['00001.03604', '00001.02262'])
"""
upload_annotation_names_to_s3(paths=paths,
                              s3_credentials=s3_credentials)
"""
# Start from here if video names already specified
selected_videos = load_video_names_from_s3(ref_file='sam_test',
                                           paths=paths,
                                           s3_credentials=s3_credentials)

# load annotation file names from s3
annotation_videos = load_video_names_from_s3(ref_file='annotation',
                                           paths=paths,
                                           s3_credentials=s3_credentials)
selected_videos =  selected_videos + annotation_videos

# select chunks of videos and classify objects
chunk_size = params['chunk_size']
while selected_videos:
    frame_level_df = update_frame_level_table(analyzer=analyzer,
                             file_names=selected_videos[:chunk_size],
                             paths=paths,
                             params=params,
                             creds=creds)

    # evaluate_frame_level_table

    update_video_level_table(frame_level_df=frame_level_df,
                             analyzer=analyzer,
                             file_names=selected_videos[:chunk_size],
                             paths=paths,
                             creds=creds,
                             params=params)

    # evaluate_video_level_table

    # Move on to next chunk
    selected_videos = selected_videos[chunk_size:]
    delete_and_recreate_dir(paths["temp_video"])
