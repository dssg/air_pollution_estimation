import datetime
import pandas as pd


def update_video_level_local(analyser,
                             frame_level_df: pd.DataFrame=None,
                             lost_tracking=None):
    """ Update the video level table on the database based on the videos in the files list
    Args:
        analyser: TrafficAnalyser object
        frame_level_df: dataframe containing the frame level stats. If None then
        this is loaded from the database using the file names

    Returns:
        video_level_df: if return_data flag is True, will return this df. Contains
                        video level information returned by analyser
    """

    # Create video level table and add to database
    video_level_df = analyser.construct_video_level_df(frame_level_df, lost_tracking)

    if video_level_df.empty:
        return
    video_level_df['creation_datetime'] = datetime.datetime.now()

    return video_level_df
