from abc import ABC, abstractmethod
import pandas as pd

# an abstract base class
class TrafficAnalyserInterface(ABC): 
    """
    All video models should inherit from this interface.
    Should handle multiple videos for input 
    """
    def __init__(self, video_dict,params,paths):
        super().__init__()   
        self.video_dict = video_dict 
        self.params = params
        self.paths = paths

    @abstractmethod
    def construct_frame_level_df(self, video_dict) -> pd.DataFrame:
        """
        Should be for multiple videos. 
        """
        # possibly require this method return the frame level table? 
        #assertion code and reordering for df dataframes 
        return 

    @abstractmethod
    def construct_video_level_df(self,frame_level_df) -> pd.DataFrame: 
        """
        Should be for multiple videos 
        """
        # possibly require this method return video level table? 
        return 
