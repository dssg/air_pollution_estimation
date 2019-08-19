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

# TODO: possibly mkae it so that retrieve_upload video_names_to_s3 always happens? 
# If running first time:
# creates the test_seach_json. Change the camera list and output file name for full run
output_file_name = params['ref_file_name']

if(params['load_ref_file']):
    retrieve_and_upload_video_names_to_s3(output_file_name=output_file_name,
                                          paths=paths,
                                          from_date=params['from_date'], to_date=params['to_date'],
                                          from_time=params['from_time'], to_time=params['to_time'],
                                          s3_credentials=s3_credentials,
                                          camera_list=params['camera_list'])

# Start from here if video names already specified
selected_videos = load_video_names_from_s3(ref_file=output_file_name,
                                           paths=paths,
                                           s3_credentials=s3_credentials)

analyzer = TrackingAnalyser(
    params=params, paths=paths, s3_credentials=s3_credentials)

# select chunks of videos and classify objects
chunk_size = params['chunk_size']
while selected_videos:
    frame_level_df = update_frame_level_table(analyzer=analyzer,
                                              file_names=selected_videos[:chunk_size],
                                              paths=paths,
                                              creds=creds)

    update_video_level_table(analyzer=analyzer,
                             frame_level_df=frame_level_df,
                             file_names=selected_videos[:chunk_size],
                             paths=paths,
                             creds=creds,
                             return_data=False)

    # Move on to next chunk
    selected_videos = selected_videos[chunk_size:]
    delete_and_recreate_dir(paths["temp_video"])
