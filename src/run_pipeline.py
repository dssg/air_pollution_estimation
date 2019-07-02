import pandas as pd

from src.d00_utils.data_retrieval import retrieve_videos_s3_to_np, \
    load_videos_from_local
from src.d00_utils.load_confs import load_parameters, load_paths

from src.d04_modelling.classify_objects import classify_objects
from src.d05_reporting.report_yolo import yolo_output_df, yolo_report_stats

params = load_parameters()
paths = load_paths()

jamcam_tims_overlap = ['00001.03601', '00001.07591', '00001.01252', '00001.06597',
                       '00001.08853', '00001.06510', '00001.04280', '00001.04534',
                       '00001.06590', '00001.07382', '00001.04328', '00001.06514',
                       '00001.03604', '00001.06501', '00001.05900', '00001.03490',
                       '00001.08926', '00001.07355', '00001.04336', '00001.09560']

#videos, names = retrieve_videos_s3_to_np(paths, from_date='2019-06-30', to_date='2019-06-30',
#                                         from_time='13-00-00', to_time='13-05-00',
#                                         camera_list=jamcam_tims_overlap,
#                                         bool_keep_data=True)

videos, names = load_videos_from_local(paths)
yolo_dict = classify_objects(videos, names, params, paths,
                                vid_time_length=10, make_videos=True)
yolo_df = yolo_output_df(yolo_dict)
stats_df = yolo_report_stats(yolo_df)
stats_df.to_csv(paths['processed_video'] + 'JamCamStats.csv')


