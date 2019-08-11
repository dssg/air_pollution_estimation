import pandas as pd
import re

from traffic_analysis.d00_utils.video_helpers import parse_video_or_annotation_name
from traffic_analysis.d05_evaluation.video_level_evaluator import VideoLevelEvaluator
from traffic_analysis.d05_evaluation.frame_level_evaluator import FrameLevelEvaluator
from traffic_analysis.d00_utils.data_loader_s3 import DataLoaderS3


class ChunkEvaluator:
    """
    Purpose of this class is to evaluate a chunk of videos, given
    a list of the annotation_xml_paths, and lists of the corresponding
    video_level_dfs and frame_level_dfs.
    """
    def __init__(self,
                 annotation_xml_paths: list,
                 annotations_videos_name_mapper: dict,
                 selected_labels: list,
                 video_level_df: pd.DataFrame = None,
                 frame_level_df: pd.DataFrame = None,
                 data_loader_s3=None,
                 video_level_column_order: list=None):

        annotations_available = {}
        for xml_path in annotation_xml_paths:
            xml_name = re.split(r"\\|/", xml_path)[-1]
            camera_id, video_upload_datetime = parse_video_or_annotation_name(xml_name)
            try:
                corresp_video_path = annotations_videos_name_mapper[xml_path]
            except:
                corresp_video_path = None
            annotations_available[xml_path] = [camera_id, video_upload_datetime, corresp_video_path]

        annotations_available = (pd.DataFrame
                                 .from_dict(annotations_available,
                                            orient='index',
                                            columns=['camera_id', 'video_upload_datetime', 'video_path'])
                                 .reset_index()
                                 .rename(columns={'index': 'xml_path'}))

        if video_level_df is not None:
            video_level_videos_to_eval = video_level_df[['camera_id', 'video_upload_datetime']].drop_duplicates()

            # evaluate only those videos for which we have annotations
#            print("number annotations available: ", len(annotations_available))
#            print("number videos available: ", len(video_level_videos_to_eval))
#            print("annotations: ", annotations_available)
#            print("annotation datetimes:",annotations_available[["camera_id", "video_upload_datetime"]])
#            print(video_level_videos_to_eval)
            self.video_level_videos_to_eval = pd.merge(left=annotations_available,
                                                       right=video_level_videos_to_eval,
                                                       on=['camera_id', 'video_upload_datetime'],
                                                       how='inner')
            
            self.num_video_level_videos = len(self.video_level_videos_to_eval)

            assert self.num_video_level_videos > 0
            self.video_level_df = video_level_df

            assert video_level_column_order is not None
            self.video_level_column_order = video_level_column_order

        if frame_level_df is not None: 
            frame_level_videos_to_eval = frame_level_df[['camera_id', 'video_upload_datetime']].drop_duplicates()

            # evaluate only those frame level videos for which we have annotations
            self.frame_level_videos_to_eval = pd.merge(left=annotations_available,
                                                       right=frame_level_videos_to_eval,
                                                       on=['camera_id', 'video_upload_datetime'],
                                                       how='inner')
            self.num_frame_level_videos = len(self.video_level_videos_to_eval)

            assert self.num_frame_level_videos > 0
            self.frame_level_df = frame_level_df

        self.selected_labels = selected_labels

        if data_loader_s3 is not None:
            self.data_loader_s3 = data_loader_s3


    def evaluate_video_level(self) -> (pd.DataFrame, pd.DataFrame):
        """This function evaluates a chunk of videos with the VideoLevelEvaluator object
        """
        video_level_evaluator = VideoLevelEvaluator(videos_to_eval=self.video_level_videos_to_eval,
                                                    video_level_df=self.video_level_df,
                                                    video_level_column_order=self.video_level_column_order,
                                                    selected_labels=self.selected_labels,
                                                    data_loader_s3=self.data_loader_s3)

        video_level_performance, video_level_diff = video_level_evaluator.evaluate()
        return video_level_performance, video_level_diff

    def evaluate_frame_level(self) -> pd.DataFrame:
        """This function evaluates mean average precision for a chunk of videos 
        """
        frame_level_evaluator = FrameLevelEvaluator(videos_to_eval=self.frame_level_videos_to_eval,
                                                    frame_level_df=self.frame_level_df,
                                                    selected_labels=self.selected_labels,
                                                    data_loader_s3=self.data_loader_s3)
        frame_level_mAP = frame_level_evaluator.evaluate()
        return frame_level_mAP
