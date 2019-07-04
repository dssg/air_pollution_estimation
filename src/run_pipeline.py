from traffic_estimation.d00_utils.data_retrieval import retrieve_videos_s3_to_np
from traffic_estimation.d00_utils.load_confs import load_parameters, load_paths

from traffic_estimation.d04_modelling.classify_objects import classify_objects
from traffic_estimation.d05_reporting.report_yolo import yolo_output_df, yolo_report_stats

params = load_parameters()
paths = load_paths()


### select data for model validation runs

videos, names = retrieve_videos_s3_to_np(paths, from_date='2019-06-30', to_date='2019-06-30',
                                         from_time='13-00-00', to_time='13-05-00',
                                         camera_list=params['tims_camera_list'],
                                         bool_keep_data=True)

##### put boxes around objects at frame level

#videos, names = load_videos_from_local(paths)
yolo_dict = classify_objects(videos, names, params, paths,
                                vid_time_length=10, make_videos=True)
yolo_df = yolo_output_df(yolo_dict)

###### count cars per video

stats_df = yolo_report_stats(yolo_df)
stats_df.to_csv(paths['processed_video'] + 'JamCamStats.csv')



##### summarise model performance



