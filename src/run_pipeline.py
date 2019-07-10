import os
import pandas as pd

from src.d00_utils.data_retrieval import upload_video_names_to_s3, append_to_csv, load_video_names, download_video_and_convert_to_numpy, delete_and_recreate_dir
from src.d00_utils.load_confs import load_parameters, load_paths
from src.d04_modelling.classify_objects import classify_objects
from src.d05_reporting.report_yolo import yolo_output_df, yolo_report_count_stats
from src.d04_modelling.evaluation import parse_annotations, report_count_differences

params = load_parameters()
paths = load_paths()
print(params['dtype'])
print("Saving video filenames")


upload_video_names_to_s3('test_search',
    paths, from_date='2019-06-30', to_date='2019-06-30',
    from_time='13-00-00', to_time='13-05-00',
    camera_list=params['tims_camera_list'])




##########################

selected_videos = load_video_names(paths)
print("Finished saving %s video filenames " % (len(selected_videos)))

# select chunks of videos and classify objects
chunk_size = params['chunk_size']
while selected_videos:

    # download chunks of videos
    print("Downloading video files")
    video_dict = download_video_and_convert_to_numpy(
        paths['temp_video'], paths['s3_profile'], paths['bucket_name'], selected_videos[:chunk_size])
    print("Finished downloading video files")

    print("Classifying objects in video files")
    # classify objects in videos
    yolo_df = classify_objects(video_dict, params, paths,
                                 vid_time_length=10, make_videos=False)
    print(yolo_df.head())

    # append to csv
    filepath = os.path.join(paths['processed_video'], 'JamCamYolo.csv')
    append_to_csv(filepath, yolo_df, params['yolo_columns'], params['dtype'])
    print("Appended yolo df to big csv")

    # Get the stats dataframe
    stats_df = yolo_report_count_stats(yolo_df)
    print("Generated stats for some video files")

    # append to csv
    filepath = os.path.join(paths['processed_video'], 'JamCamStats.csv')
    append_to_csv(filepath, stats_df, params['stats_columns'], params['dtype'])
    print("Appended stats to big csv")

    # Move on to next chunk
    selected_videos = selected_videos[5:]
    delete_and_recreate_dir(paths["temp_video"])

# Load Annotations and Evaluate
yolo_df = pd.read_csv(os.path.join(paths['processed_video'], 'JamCamYolo.csv'))
annotations_df = parse_annotations(paths['annotations'], bool_print_summary=False)
count_differences_df = report_count_differences(annotations_df, yolo_df)
count_differences_df.to_csv(paths['processed_video'] + 'JamCamCountDifferences.csv')









