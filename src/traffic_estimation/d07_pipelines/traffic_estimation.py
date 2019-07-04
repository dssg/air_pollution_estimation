from kedro.pipeline import Pipeline, node
from traffic_estimation.d02_intermediate.create_int_videos import create_int_videos
from traffic_estimation.d04_modelling.create_out_frames_with_boxes import create_out_video_with_boxes
from traffic_estimation.d05_reporting.report_yolo import  yolo_report_stats


traffic_estimation_pipeline = Pipeline([
    node(
        func=create_int_videos,
        inputs='raw_videos',
        outputs='int_videos',
        name='create_int_videos'),
    # TODO: int videos are pandas data frames at the moment, not numpy arrays. add .values in the right place
    node(
        func=create_out_video_with_boxes,
        inputs=['int_videos', 'paths'],
        outputs='out_frames_with_boxes',
        name='create_frames_with_boxes'),
    node(
        func=yolo_report_stats,
        inputs='out_frames_with_boxes',
        outputs='out_car_counts',
        name='create_out_car_counts')
    ])
