import datetime
import pandas as pd

from traffic_analysis.d00_utils.video_helpers import parse_video_or_annotation_name
from traffic_analysis.d00_utils.data_loader_sql import DataLoaderSQL


def update_hour_level_table(db_hour_level_name,
                            video_level_df: pd.DataFrame=None,
                            paths: dict=None,
                            creds: dict=None):
    """ Update the video level table on the database based on the videos in the files list
    Args:
        analyser: TrafficAnalyser object
        db_video_level_name: name of database table to write to
        db_frame_level_name: name of database table to read from
        frame_level_df: dataframe containing the frame level stats. If None then
        this is loaded from the database using the file names
        file_names: list of s3 filepaths for the videos to be processed
        paths: dictionary of paths from yml file
        creds: dictionary of credentials from yml file
        return_data: For debugging it might be useful to return the video level df

    Returns:
        video_level_df: if return_data flag is True, will return this df. Contains
                        video level information returned by analyser
    """

    db_obj = DataLoaderSQL(creds=creds, paths=paths)

    hour_level_df = video_level_df.copy()

    date_times = hour_level_df['video_upload_datetime'].values
    new_date_times = []

    for date_time in date_times:
        new_date_times.append(date_time.astype('datetime64[h]'))

    hour_level_df['video_upload_datetime'] = new_date_times
    hour_level_df = hour_level_df.drop(['creation_datetime'], axis=1)

    #db_obj.add_to_sql(df=hour_level_df, table_name=db_hour_level_name)

    return hour_level_df
