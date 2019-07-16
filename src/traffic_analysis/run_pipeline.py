from src.traffic_analysis.d00_utils.data_retrieval import retrieve_video_names_from_s3, append_to_csv, load_video_names, download_video_and_convert_to_numpy, delete_and_recreate_dir
from src.traffic_analysis.d00_utils.load_confs import load_parameters, load_paths
from src.traffic_analysis.d04_modelling.classify_objects import classify_objects
from src.traffic_analysis.d05_reporting.report_yolo import yolo_output_df, yolo_report_stats
import os

params = load_parameters()
paths = load_paths()

print("Saving video filenames")
retrieve_video_names_from_s3(
    paths, from_date='2019-06-05', to_date='2019-06-18',
    from_time='00-00-00', to_time='23-59-59',
    camera_list=params['tims_camera_list'],
    save_to_file=True)

selected_videos = load_video_names(paths)
print("Finished saving %s video filenames " % (len(selected_videos)))

# select chunks of videos and classify objects
chunk_size = params['chunk_size']
while selected_videos:
    print("Downloading video files")

    # download chunks of videos
    videos, names = download_video_and_convert_to_numpy(
        local_folder=paths['temp_video'], s3_profile=paths['s3_profile'], bucket=paths['bucket_name'], filenames=selected_videos[:chunk_size])
    print("Finished downloading video files")

    print("Classifying objects in video files")
    # classify objects in videos
    yolo_dict = classify_objects(
        videos=videos, names=names, params=params, paths=paths, vid_time_length=10, make_videos=False)
    yolo_df = yolo_output_df(yolo_dict)
    print(yolo_df.head())
    stats_df = yolo_report_stats(yolo_df)
    print("Generated stats for some video files")

    # append to csv
    filename = os.path.join(paths['processed_video'], 'JamCamStats.csv')
    append_to_csv(filename, stats_df, params['columns'], params['dtype'])
    print("Appended stats to big csv")
    selected_videos = selected_videos[5:]
    delete_and_recreate_dir(paths["temp_video"])
