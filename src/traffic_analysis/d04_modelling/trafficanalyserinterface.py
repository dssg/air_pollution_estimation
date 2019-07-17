from abc import ABC, abstractmethod
import pandas as pd

# an abstract base class
class TrafficAnalyserInterface(ABC): 
    """
    All video models should inherit from this interface.
    Should handle multiple videos for input 
    """
    def __init__(self, video_dict: dict, params: dict, paths: dict):
        """
        Keyword arguments

        video_dict -- dict with video names as keys and videos encoded as numpy arrays 
                        for values
        """
        super().__init__()   
        self.video_dict = video_dict 
        self.params = params
        self.paths = paths

    @abstractmethod
    def construct_frame_level_df(self, video_dict: dict) -> pd.DataFrame: pass
        """Should use the video_dict to create frame level df for multiple videos with columns 
        as specified in the project data tables schema
        """
        #assertion code and reordering for df dataframes 
        

    @abstractmethod
    def construct_video_level_df(self, frame_level_df: pd.DataFrame) -> pd.DataFrame: pass
        """Should return video level pandas dataframe for multiple videos with columns as specified in the project
        data tables schema
        """