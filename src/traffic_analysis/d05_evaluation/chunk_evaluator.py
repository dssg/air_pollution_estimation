from traffic_analysis.d00_utils.video_helpers import parse_video_or_annotation_name
from traffic_analysis.d05_evaluation.video_level_evaluator import VideoLevelEvaluator
import pandas as pd
import re 


class ChunkEvaluator:
    """
    Purpose of this class is to evaluate a chunk of videos, given 
    a list of the annotation_xml_paths, and lists of the corresponding 
    video_level_dfs and frame_level_dfs. 
    """
    def __init__(self,
                 annotation_xml_paths: list,
                 params: dict,
                 frame_level_dfs: list = None,
                 video_level_df: list = None):

        annotations_available = {}
        for xml_path in annotation_xml_paths:
            xml_name = re.split(r"\\|/", xml_path)[-1]
            camera_id, video_upload_datetime = parse_video_or_annotation_name(xml_name)
            annotations_available[xml_path] = [camera_id, video_upload_datetime]

        annotations_available = (pd.DataFrame
                                 .from_dict(annotations_available,
                                            orient='index',
                                            columns=['camera_id', 'video_upload_datetime'])
                                 .reset_index()
                                 .rename(columns={'index': 'xml_path'}))

        if video_level_df is not None:
            videos_to_eval = video_level_df[['camera_id', 'video_upload_datetime']].drop_duplicates()

            # evaluate only those videos for which we have annotations
            self.videos_to_eval = pd.merge(left=annotations_available,
                                           right=videos_to_eval,
                                           on=['camera_id', 'video_upload_datetime'],
                                           how='inner')

            self.num_videos = len(self.videos_to_eval)

            assert self.num_videos > 0
            self.video_level_df = video_level_df

            self.video_level_column_order = params['video_level_column_order']
            self.selected_labels = params['selected_labels']

        self.params = params
        
    def evaluate_video_level(self)-> pd.DataFrame:
        """This function evaluates a chunk of videos utilizing multiple SingleEvaluator
           objects.
        """

        video_level_evaluator = VideoLevelEvaluator(videos_to_eval=self.videos_to_eval,
                                                    video_level_df=self.video_level_df,
                                                    video_level_column_order=self.video_level_column_order,
                                                    selected_labels=self.selected_labels)

        video_level_performance = video_level_evaluator.evaluate()
        return video_level_performance
