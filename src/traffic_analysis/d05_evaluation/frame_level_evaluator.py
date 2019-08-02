import pandas as pd 
import numpy as np

from traffic_analysis.d05_evaluation.parse_annotation import parse_annotation


class FrameLevelEvaluator:
    """
    Purpose of this class is to conduct video level evaluation for one video.
    """
    def __init__(self,
                 videos_to_eval: pd.DataFrame,
                 frame_level_df: pd.DataFrame,
                 frame_level_column_order: list,
                 selected_labels: list):

        # data frames to work with
        self.videos_to_eval = videos_to_eval
        self.frame_level_df = frame_level_df
        self.video_level_ground_truth = pd.DataFrame({})

        # parameters
        self.frame_level_column_order = frame_level_column_order
        self.selected_labels = selected_labels
