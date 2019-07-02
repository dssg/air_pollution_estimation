import pandas as pd


from d00_utils.data_retrieval import retrieve_videos_s3_to_np, \
    load_videos_from_local
from d00_utils.load_confs import load_parameters, load_paths

from d04_modelling.classify_objects import classify_objects
from d05_reporting.report_yolo import yolo_output_df, yolo_report_stats

params = load_parameters()
paths = load_paths()

jamcam_tims_overlap = ['00001.03601', '00001.07591', '00001.01252', '00001.06597']

print('Downloading videos...')
videos, names = retrieve_videos_s3_to_np(paths, from_date='2019-06-24', to_date='2019-07-01',
                                         camera_list=jamcam_tims_overlap,
                                         bool_keep_data=True)

#videos, names = load_videos_from_local(paths)

print('Classifying objects...')
yolo_dict = classify_objects(videos, names, params, paths,
                                vid_time_length=10, make_videos=True)

print('Gathering statistics...')
yolo_df = yolo_output_df(yolo_dict)
stats_df = yolo_report_stats(yolo_df)
stats_df.to_csv(paths['processed_video'] + 'JamCamStats.csv')


