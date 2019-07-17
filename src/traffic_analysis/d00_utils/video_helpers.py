import imageio
import numpy as np
import os
import datetime


def write_mp4(local_mp4_dir:str, mp4_name:str,video:np.ndarray,fps:int):
    """Write provided video to provided path

    Keyword arguments 

    local_mp4_dir -- path to directory to store vids in 
    mp4_name -- desired name for video. Please include .mp4 extension 
    fps -- provide the frames per second of the video 
    """
    local_mp4_path_out = os.path.join(local_mp4_dir, mp4_name)
    imageio.mimwrite(local_mp4_path_out, video, fps=fps)


def parse_video_name(video_name:str) -> (str, datetime.datetime):
    """Helper function to parse the jamcam video names into camera_id and 
       upload datetime, in the types we need them in 
      
    Keyword arguments 
    video_name -- can handle format is id_YYYY-mm-dd_HH-mm-seconds_camera_id, where id is sometimes
                    not present and seconds is sometimes to integer precision sometimes to decimal 
                    precision 
    """ 
    video_name = video_name.replace(".mp4", "").replace(".xml", "").split("_")
    if len(video_name) == 4: 
        # remove id which is sometimes added by cvat
        video_name = video_name[1:]

    YYYYmmdd, time, camera_id = video_name
    YYYY, mm, dd = YYYYmmdd.split("-")
    hh, mm, seconds = time.split("-")
    # only want seconds to integer precision
    ss = seconds.split('.')[0] 
    video_upload_datetime = datetime.datetime(year = int(YYYY), month = int(mm), day = int(dd),
                                              hour = int(hh), minute = int(mm), second = int(ss))
    return camera_id, video_upload_datetime
    