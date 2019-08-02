import pandas as pd 
import numpy as np

from traffic_analysis.d05_evaluation.parse_annotation import parse_annotation


class FrameLevelEvaluator:
    """
    Purpose of this class is to conduct video level evaluation for one video.
    """
    def __init__(self,
                 videos_to_eval: pd.DataFrame,
                 video_level_df: pd.DataFrame,
                 video_level_column_order: list,
                 selected_labels: list):

        # data frames to work with
        self.videos_to_eval = videos_to_eval
        self.video_level_df = video_level_df
        self.video_level_ground_truth = pd.DataFrame({})

        # parameters
        self.video_level_column_order = video_level_column_order
        self.selected_labels = selected_labels
        self.stats_to_evaluate = ['counts', 'starts', 'stops']
