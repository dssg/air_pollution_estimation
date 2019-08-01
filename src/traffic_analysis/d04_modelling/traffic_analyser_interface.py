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

        video_dict -- dict with video names as keys and videos encoded as numpy arrays 
                        for values
        params -- yaml with modelling parameters
        paths -- yaml with paths 
        """

        self.video_dict = None
        self.params = params
        self.paths = paths

    def check_video_dict(self, video_dict: dict):
        # Check that video doesn't come from in-use camera (some are)
        for video_name in list(video_dict.keys()):
            n_frames = video_dict[video_name].shape[0]
            if n_frames < 75:
                del video_dict[video_name]
                print("Video ", video_name,
                      " has been removed from processing because it may be invalid")

        self.video_dict = video_dict

        return

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
