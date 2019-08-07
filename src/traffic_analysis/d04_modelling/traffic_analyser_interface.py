from abc import ABC, abstractmethod
import pandas as pd

# abc is abstract base class


class TrafficAnalyserInterface(ABC):
    """
    All video models should inherit from this interface.
    Should handle multiple videos for input 
    """

    def __init__(self, params: dict, paths: dict):
        """ 
        Keyword arguments

        params -- yaml with modelling parameters
        paths -- yaml with paths 
        """
        super().__init__()

        self.params = params
        self.paths = paths

    @abstractmethod
    def construct_frame_level_df(self, video_dict) -> pd.DataFrame:
        """Should use the video_dict to create frame level df for multiple videos with columns 
        as specified in the project data tables schema
        """
        pass

    @abstractmethod
    def construct_video_level_df(self, frame_level_df: pd.DataFrame) -> pd.DataFrame:
        """Should return video level pandas dataframe for multiple videos with columns as specified in the project
        data tables schema

        Keyword arguments: 

        frame_level_df -- df returned by above function
        """
        pass
