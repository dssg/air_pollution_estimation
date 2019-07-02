from traffic_estimation.d04_modelling.classify_objects import classify_objects
from traffic_estimation.d05_reporting.report_yolo import yolo_output_df


def create_out_video_with_boxes(int_videos: dict,
                                params: dict,
                                paths: dict):

    yolo_dict = classify_objects(int_videos, params, paths,
                                 vid_time_length=10, make_videos=True)
    yolo_df = yolo_output_df(yolo_dict)

    return yolo_df
