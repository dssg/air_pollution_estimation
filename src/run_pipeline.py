import pandas as pd

from src.d00_utils.data_retrieval import retrieve_videos_s3_to_np, \
    load_videos_from_local
from src.d00_utils.load_confs import load_parameters, load_paths

from src.d04_modelling.classify_objects import classify_objects
from src.d05_reporting.report_yolo import yolo_output_df, yolo_report_stats

params = load_parameters()
paths = load_paths()

videos, names = retrieve_videos_s3_to_np(paths, from_date='2019-06-19', to_date='2019-06-19',
                                         from_time='20-20-00', to_time='20-20-02',
                                         bool_keep_data=False)
#videos, names = load_videos_from_local(paths)
yolo_dict = classify_objects(videos, names, params, paths,
                                vid_time_length=10, make_videos=True)
yolo_df = yolo_output_df(yolo_dict)
stats_df = yolo_report_stats(yolo_df)
stats_df.to_csv(paths['processed_video'] + 'JamCamStats.csv')
