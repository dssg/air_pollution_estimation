import time
from datetime import datetime

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
from traffic_analysis.d00_utils.move_file_within_s3_bucket import move_file

params = load_parameters()
paths = load_paths()
creds = load_credentials()
s3_credentials = creds[paths['s3_creds']]

# If running first time:
# creates the test_seach_json. Change the camera list and output file name for full run
current_date = datetime.now().date()
output_file_name = f'{current_date}'
from_date = str(current_date)
to_date = str(current_date)

# retrieve_and_upload_video_names_to_s3(output_file_name=output_file_name,
#                                       paths=paths,
#                                       from_date=from_date, to_date=to_date,
#                                       s3_credentials=s3_credentials,
#                                       camera_list=params['tims_camera_list'])

# Start from here if video names already specified
selected_videos = load_video_names_from_s3(ref_file=output_file_name,
                                           paths=paths,
                                           s3_credentials=s3_credentials)

analyzer = TrackingAnalyser(
    params=params, paths=paths, s3_credentials=s3_credentials)

# select chunks of videos and classify objects
chunk_size = params['chunk_size']
while selected_videos:
    start = time.time()
    file_names = selected_videos[:chunk_size]
    frame_level_df = update_frame_level_table(analyzer=analyzer,
                                              file_names=file_names,
                                              paths=paths,
                                              creds=creds)

    # evaluate_frame_level_table

    update_video_level_table(analyzer=analyzer,
                             frame_level_df=frame_level_df,
                             file_names=selected_videos[:chunk_size],
                             paths=paths,
                             creds=creds,
                             return_data=False)

    # evaluate_video_level_table

    # move processed videos to processed folder
    for filename in file_names:
        new_filename = filename.replace(
            paths['s3_video'], paths['s3_processed_videos'])
        print(filename, new_filename)
        move_file(filename, new_filename)

    # Move on to next chunk
    selected_videos = selected_videos[chunk_size:]
    delete_and_recreate_dir(paths["temp_video"])
    print(f"Processed {chunk_size} in {time.time() - start}")
