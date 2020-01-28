import imageio
import numpy as np
import os
import dateutil
import datetime
import re


def write_mp4(local_mp4_dir: str, mp4_name: str, video: np.ndarray, fps: int):
    """Write provided video to provided path

    Args: 
        local_mp4_dir -- path to directory to store vids in 
        mp4_name -- desired name for video. Please include .mp4 extension 
        fps -- provide the frames per second of the video 
    Returns:
    Raises:
    """
    name = mp4_name.split('.')[0] + '.mp4'
    imageio.mimwrite(name, video, fps=fps)
    print('Video Saved to ' + name)


def parse_video_or_annotation_name(video_name: str) -> (str, datetime.datetime):
    """Helper function to parse the jamcam video/annotation names into camera_id and 
       upload datetime, in the types we need them in 

    Args:
        video_name -- can handle format is CVATid_YYYY-mm-dd_HH-mm-seconds_camera_id, where id is sometimes
                        not present and seconds is sometimes to integer precision sometimes to decimal 
                        precision; can also handle if the entire path name is passed in 
    Returns: 
        camera_id: string in format borough_id.camera_number
        video_upload_datetime: datetime with no milliseconds
    Raises:
    """
    video_name = re.split(
        r"_|\\|/", video_name.replace(".mp4", "").replace(".xml", ""))
    if len(video_name) > 3:
        # remove id which is sometimes added by cvat
        # or remove folder names which are sometimes getting included
        # in video name
        video_name = video_name[-3:]

    date_str, time_str, camera_id = (video_name[-3], video_name[-2], video_name[-1]) if len(
        video_name) > 2 else (video_name[-2], " ", video_name[-1])

    video_upload_datetime = "%s %s" % (date_str, time_str.replace("-", ":"))
    video_upload_datetime = dateutil.parser.parse(
        video_upload_datetime.strip())
    return camera_id, video_upload_datetime
