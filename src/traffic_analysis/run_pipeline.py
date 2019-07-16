from traffic_analysis.d00_utils.data_retrieval import delete_and_recreate_dir
from traffic_analysis.d00_utils.load_confs import load_parameters, load_paths
from traffic_analysis.d02_intermediate.update_s3_ref import load_video_names_from_s3, upload_video_names_to_s3, \
    upload_annotation_names_to_s3
from traffic_analysis.d03_processing.update_s3_processed import update_frame_level_table

params = load_parameters()
paths = load_paths()

upload_video_names_to_s3('test_search',
    paths, from_date='2019-06-30', to_date='2019-06-30',
    from_time='13-00-00', to_time='13-05-00',
    camera_list=params['tims_camera_list'][:2])

upload_annotation_names_to_s3(paths)

selected_videos = load_video_names_from_s3('test_search', paths)

# select chunks of videos and classify objects
chunk_size = params['chunk_size']
while selected_videos:

    update_frame_level_table(selected_videos[:chunk_size], paths, params)

    # evaluate_frame_level_table

    # update_video_level_table

    # evaluate_video_level_table

    # Move on to next chunk
    selected_videos = selected_videos[chunk_size:]
    delete_and_recreate_dir(paths["temp_video"])


"""
from d05_reporting.report_yolo import yolo_output_df, yolo_report_count_stats
from d04_modelling.evaluation import parse_annotations, report_count_differences
# Load Annotations and Evaluate
yolo_df = pd.read_csv(os.path.join(paths['processed_video'], 'JamCamYolo.csv'))
annotations_df = parse_annotations(paths['annotations'], bool_print_summary=False)
count_differences_df = report_count_differences(annotations_df, yolo_df)
count_differences_df.to_csv(paths['processed_video'] + 'JamCamCountDifferences.csv')
"""









