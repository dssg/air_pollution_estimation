import datetime
import re
import pandas as pd

from traffic_analysis.d00_utils.data_retrieval import mp4_to_npy


def update_frame_level_local(analyser,
                             file_names: list,
                             paths: dict,
                             creds: dict, make_video):
    """ Update the frame level table on PSQL based on the videos in the files list
    Args:
        analyser: analyser object for doing traffic analysis
        file_names: list of s3 filepaths for the videos to be processed
        paths: dictionary of paths from yml file
        creds: dictionary of credentials from yml file

    Returns:
        frame_level_df: dataframe of frame level information returned by
                        analyser object
    """

    video_dict = {}
    for filename in file_names:
        try:
            video_name = re.split(r"\\|/", filename)[-1]
            video_dict[video_name] = mp4_to_npy(filename)
        except Exception as e:
            print(f"Could not convert {filename}  to numpy array due to {e}")

    frame_level_df, runtime_list, lost_tracking = analyser.construct_frame_level_df(video_dict, make_video)
    if frame_level_df.empty:
        return None
    frame_level_df.dropna(how='any', inplace=True)
    frame_level_df = frame_level_df.astype(
        {'frame_id': 'int64',
         'vehicle_id': 'int64'})

    success = True

    return success, frame_level_df, runtime_list, lost_tracking
