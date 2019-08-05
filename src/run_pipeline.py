from traffic_analysis.d00_utils.data_retrieval import delete_and_recreate_dir
from traffic_analysis.d00_utils.load_confs import load_parameters, load_paths, load_credentials
from traffic_analysis.d02_ref.load_video_names_from_s3 import load_video_names_from_s3
from traffic_analysis.d02_ref.retrieve_and_upload_video_names_to_s3 import retrieve_and_upload_video_names_to_s3
from traffic_analysis.d02_ref.upload_annotation_names_to_s3 import upload_annotation_names_to_s3
from traffic_analysis.d03_processing.update_frame_level_table import update_frame_level_table
from traffic_analysis.d03_processing.update_video_level_table import update_video_level_table
from traffic_analysis.d04_modelling.trackinganalyser.trackinganalyser import TrackingAnalyser
#############
import sys
#############
params = load_parameters()
paths = load_paths()
creds = load_credentials()
s3_credentials = creds[paths['s3_creds']]

traffic_analyser = eval(params["traffic_analyser"])(params=params, paths=paths)

print(params)
# If running first time:
# creates the test_seach_json. Change the camera list and output file name for full run

retrieve_and_upload_video_names_to_s3(output_file_name= params['dataset_ref_name'],
                                      paths=paths,
                                      from_date='2019-07-17', to_date='2019-07-17',
                                      s3_credentials=s3_credentials,
                                      camera_list=['00001.03604', '00001.02262'])
"""
upload_annotation_names_to_s3(paths=paths,
                              s3_credentials=s3_credentials)
"""
# Start from here if video names already specified
selected_videos = load_video_names_from_s3(ref_file= params['dataset_ref_name'],
                                           paths=paths,
                                           s3_credentials=s3_credentials)

# select chunks of videos and classify objects
chunk_size = params['chunk_size']
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


"""
from d05_evaluation.report_yolo import yolo_output_df, yolo_report_count_stats
from d04_modelling.evaluation import parse_annotations, report_count_differences
# Load Annotations and Evaluate
yolo_df = pd.read_csv(os.path.join(paths['processed_video'], 'JamCamYolo.csv'))
annotations_df = parse_annotations(paths['annotations'], bool_print_summary=False)
count_differences_df = report_count_differences(annotations_df, yolo_df)
count_differences_df.to_csv(paths['processed_video'] + 'JamCamCountDifferences.csv')
"""









