from traffic_analysis.d05_evaluation.single_evaluator import SingleEvaluator


class FrameLevelEvaluator(SingleEvaluator): 
    """
    Not yet implemented. 
    """
    def __init__(self, xml_root, xml_name, frame_level_df: pd.DataFrame, params):
        super().__init__(xml_root, xml_name, params)
        self.ground_truth_df = super().parse_annotations()
        self.frame_level_df = frame_level_df

    def evaluate_video(self): 
        pass

    def plot_video(self): 
        pass
